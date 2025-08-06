#!/usr/bin/env python3
"""
Polymarket User Activity Monitor
Fetches user activities from Polymarket Data API
"""

import requests
import json
import time
import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'  # Up/Yes
    RED = '\033[91m'    # Down/No
    BLUE = '\033[94m'   # Buy
    YELLOW = '\033[93m' # Other actions
    RESET = '\033[0m'   # Reset color
    BOLD = '\033[1m'    # Bold text

class PolymarketMonitor:
    def __init__(self, username: str, user_address: str, log_to_json: bool = False, show_colors: bool = True, market_filter: str = None, exact_market: bool = False, num_activities: int = 200):
        self.username = username
        self.user_address = user_address
        self.log_to_json = log_to_json
        self.show_colors = show_colors
        self.market_filter = market_filter.lower() if market_filter else None
        self.exact_market = exact_market
        self.num_activities = num_activities
        self.api_base = "https://data-api.polymarket.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        self.last_seen_hash = None
        self.debug_mode = False
    
    def get_user_address(self) -> Optional[str]:
        """Get user wallet address from username - this would need to be implemented"""
        # For now, return the provided address
        # In a real implementation, you'd need to resolve username to wallet address
        return self.user_address
    
    def colorize_outcome(self, outcome: str) -> str:
        """Add color formatting to outcome text"""
        if not self.show_colors:
            return outcome
        
        outcome_lower = outcome.lower()
        if outcome_lower in ['up', 'yes']:
            return f"{Colors.GREEN}{Colors.BOLD}{outcome}{Colors.RESET}"
        elif outcome_lower in ['down', 'no']:
            return f"{Colors.RED}{Colors.BOLD}{outcome}{Colors.RESET}"
        else:
            return f"{Colors.YELLOW}{outcome}{Colors.RESET}"
    
    def colorize_side(self, side: str) -> str:
        """Add color formatting to trade side"""
        if not self.show_colors:
            return side
        
        if side.upper() == 'BUY':
            return f"{Colors.BLUE}{side}{Colors.RESET}"
        else:
            return f"{Colors.YELLOW}{side}{Colors.RESET}"

    def fetch_activities(self) -> List[Dict]:
        """Fetch activities using Polymarket Data API"""
        user_address = self.get_user_address()
        if not user_address:
            print("Error: No user address provided")
            return []
        
        activities = []
        try:
            print(f"Fetching activities for user: {user_address}")
            
            # API endpoint for user activities
            url = f"{self.api_base}/activity"
            params = {
                'user': user_address,
                'limit': self.num_activities,
                'sortBy': 'TIMESTAMP',
                'sortDirection': 'DESC'
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                activities = data if isinstance(data, list) else data.get('data', [])
                print(f"✓ Found {len(activities)} activities from API")
            else:
                print(f"✗ API request failed: {response.status_code}")
                if response.text:
                    print(f"Error: {response.text}")
                    
        except Exception as e:
            print(f"Error fetching activities: {e}")
        
        return activities
    
    def filter_new_activities(self, activities: List[Dict]) -> List[Dict]:
        """Filter activities to show only new ones since last check"""
        if not activities:
            return []
        
        if self.last_seen_hash is None:
            # First run, show only the most recent activity
            self.last_seen_hash = activities[0].get('transactionHash', '')
            return [activities[0]] if activities else []
        
        # Find new activities since last seen hash
        new_activities = []
        for activity in activities:
            if activity.get('transactionHash', '') == self.last_seen_hash:
                break
            new_activities.append(activity)
        
        # Update last seen hash if we found new activities
        if new_activities:
            self.last_seen_hash = new_activities[0].get('transactionHash', '')
        
        return new_activities
    
    def filter_by_market(self, activities: List[Dict]) -> List[Dict]:
        """Filter activities by market name if market_filter is set"""
        if not self.market_filter or not activities:
            return activities
        
        filtered_activities = []
        for activity in activities:
            market_name = activity.get('title', '').lower()
            
            if self.exact_market:
                # Exact matching
                if market_name == self.market_filter:
                    filtered_activities.append(activity)
            else:
                # Partial matching (original behavior)
                if self.market_filter in market_name:
                    filtered_activities.append(activity)
        
        return filtered_activities
    
    def get_exact_market_activities(self, activities: List[Dict], target_market_name: str) -> List[Dict]:
        """Filter activities to only those from the exact market title"""
        if not target_market_name or not activities:
            return activities
        
        exact_activities = []
        target_lower = target_market_name.lower().strip()
        
        for activity in activities:
            activity_market = activity.get('title', '').lower().strip()
            if activity_market == target_lower:
                exact_activities.append(activity)
        
        return exact_activities
    

    def get_market_price(self, asset_id: str, side: str = 'buy') -> float:
        """Get current market price for asset using CLOB price API"""
        try:
            url = f"https://clob.polymarket.com/price"
            params = {
                'token_id': asset_id,
                'side': side  # 'buy' for bid price, 'sell' for ask price
            }
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    return float(data['price'])
                        
        except Exception as e:
            if self.debug_mode:
                print(f"Error fetching market price: {e}")
        
        return 0.5  # Default fallback
    
    def calculate_position_value(self, activities: List[Dict]) -> Dict:
        """Calculate user's current position value and profit/loss"""
        positions = {}  # outcome -> {'shares': float, 'cost': float, 'asset_ids': set}
        
        # Calculate net positions and collect asset IDs
        for activity in activities:
            outcome = activity.get('outcome', '').strip().lower()
            if not outcome:
                continue
                
            side = activity.get('side', '').upper()
            shares = float(activity.get('size', 0))
            cost = float(activity.get('usdcSize', 0))
            asset_id = activity.get('asset', '')
            condition_id = activity.get('conditionId', '')
            outcome_index = activity.get('outcomeIndex', 0)
            
            if outcome not in positions:
                positions[outcome] = {
                    'shares': 0.0, 
                    'cost': 0.0, 
                    'asset_ids': set(),
                    'condition_id': condition_id,
                    'outcome_index': outcome_index
                }
            
            # Track all asset IDs for this outcome
            if asset_id:
                positions[outcome]['asset_ids'].add(asset_id)
            
            if side == 'BUY':
                positions[outcome]['shares'] += shares
                positions[outcome]['cost'] += cost
            elif side == 'SELL':
                positions[outcome]['shares'] -= shares
                positions[outcome]['cost'] -= cost
        
        # Calculate current values and P&L
        pnl_summary = {
            'total_cost': 0.0,
            'total_current_value': 0.0,
            'total_pnl': 0.0,
            'positions': {}
        }
        
        for outcome, position in positions.items():
            if position['shares'] == 0:
                continue
                
            cost = position['cost']
            shares = position['shares']
            current_price = 0.5  # Default fallback
            
            # Get current market price (use sell price)
            asset_ids = position['asset_ids']
            if asset_ids:
                asset_id = list(asset_ids)[0]
                # Get sell price (what you could sell for)
                sell_price = self.get_market_price(asset_id, 'sell')
                
                if self.debug_mode:
                    print(f"Asset {asset_id[-20:]}: sell_price={sell_price}")
                
                # Use sell price if it's a real market price (not fallback)
                if sell_price != 0.5:
                    current_price = sell_price
                
                if self.debug_mode:
                    print(f"Current price for {outcome.upper()}: {current_price}")
            
            current_value = shares * current_price
            pnl = current_value - cost
            
            pnl_summary['positions'][outcome] = {
                'shares': shares,
                'cost': cost,
                'current_price': current_price,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_pct': (pnl / abs(cost) * 100) if cost != 0 else 0,
                'asset_ids': list(asset_ids)  # For debugging
            }
            
            pnl_summary['total_cost'] += cost
            pnl_summary['total_current_value'] += current_value
        
        pnl_summary['total_pnl'] = pnl_summary['total_current_value'] - pnl_summary['total_cost']
        pnl_summary['total_pnl_pct'] = (pnl_summary['total_pnl'] / abs(pnl_summary['total_cost']) * 100) if pnl_summary['total_cost'] != 0 else 0
        
        return pnl_summary
    
    def get_pnl_display(self, pnl_summary: Dict) -> str:
        """Format P&L information for display"""
        if not pnl_summary['positions']:
            return ""
        
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("💰 PROFIT & LOSS")
        lines.append("=" * 70)
        
        # Individual positions
        for outcome, pos in pnl_summary['positions'].items():
            if pos['shares'] == 0:
                continue
                
            outcome_display = outcome.upper()
            outcome_color = self.colorize_outcome(outcome_display) if self.show_colors else outcome_display
            
            pnl_color = ""
            pnl_reset = ""
            if self.show_colors:
                if pos['pnl'] > 0:
                    pnl_color = Colors.GREEN
                    pnl_reset = Colors.RESET
                elif pos['pnl'] < 0:
                    pnl_color = Colors.RED
                    pnl_reset = Colors.RESET
            
            shares_display = f"{pos['shares']:,.0f}" if pos['shares'] == int(pos['shares']) else f"{pos['shares']:,.2f}"
            
            lines.append(f"{outcome_color:<15} Shares: {shares_display:>12} | Cost: ${pos['cost']:>8.2f}")
            lines.append(f"{'':15} Price:  ${pos['current_price']:>12.3f} | Value: ${pos['current_value']:>8.2f}")
            lines.append(f"{'':15} P&L: {pnl_color}${pos['pnl']:>+8.2f} ({pos['pnl_pct']:>+6.1f}%){pnl_reset}")
            
            # Show asset IDs in debug mode
            if self.debug_mode and 'asset_ids' in pos and pos['asset_ids']:
                asset_info = ', '.join(pos['asset_ids'][:2])  # Show first 2 asset IDs
                if len(pos['asset_ids']) > 2:
                    asset_info += f" +{len(pos['asset_ids'])-2} more"
                lines.append(f"{'':15} Assets: {asset_info}")
            
            lines.append("-" * 70)
        
        # Total P&L
        total_pnl_color = ""
        total_pnl_reset = ""
        if self.show_colors:
            if pnl_summary['total_pnl'] > 0:
                total_pnl_color = Colors.GREEN + Colors.BOLD
                total_pnl_reset = Colors.RESET
            elif pnl_summary['total_pnl'] < 0:
                total_pnl_color = Colors.RED + Colors.BOLD
                total_pnl_reset = Colors.RESET
        
        lines.append(f"TOTAL COST:      ${pnl_summary['total_cost']:>10.2f}")
        lines.append(f"CURRENT VALUE:   ${pnl_summary['total_current_value']:>10.2f}")
        lines.append(f"TOTAL P&L:  {total_pnl_color}${pnl_summary['total_pnl']:>+10.2f} ({pnl_summary['total_pnl_pct']:>+6.1f}%){total_pnl_reset}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def get_market_choices(self, activities: List[Dict]) -> Dict[str, int]:
        """Get all unique markets from activities with their activity counts"""
        market_counts = {}
        for activity in activities:
            title = activity.get('title', '').strip()
            if title:
                market_counts[title] = market_counts.get(title, 0) + 1
        return market_counts
    
    def prompt_market_selection(self, market_counts: Dict[str, int]) -> str:
        """Prompt user to select which market to monitor"""
        markets = list(market_counts.keys())
        
        print(f"\n🎯 Found {len(markets)} different markets:")
        print("=" * 60)
        
        for i, market in enumerate(markets, 1):
            count = market_counts[market]
            print(f"{i:>2}. {market} ({count} activities)")
        
        print("=" * 60)
        
        while True:
            try:
                choice = input(f"Select market to monitor (1-{len(markets)}): ").strip()
                if not choice:
                    continue
                    
                choice_num = int(choice)
                if 1 <= choice_num <= len(markets):
                    selected_market = markets[choice_num - 1]
                    print(f"\n✅ Selected: {selected_market}")
                    return selected_market
                else:
                    print(f"Please enter a number between 1 and {len(markets)}")
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nMonitoring cancelled by user")
                exit(0)
    
    def calculate_market_stats(self, activities: List[Dict]) -> Dict:
        """Calculate statistics for activities (Up/Down counts and amounts)"""
        stats = {
            'total_trades': len(activities),
            'up_trades': 0,
            'down_trades': 0,
            'up_amount': 0.0,
            'down_amount': 0.0,
            'up_shares': 0.0,
            'down_shares': 0.0,
            'other_trades': 0,
            'other_amount': 0.0,
            'outcomes': {},
            'debug_info': []
        }
        
        for activity in activities:
            outcome = activity.get('outcome', '').strip().lower()
            activity_type = activity.get('type', '')
            
            # For activities without outcome, try to use the activity type
            original_outcome = outcome
            if not outcome:
                if activity_type in ['MERGE', 'REDEEM', 'SPLIT']:
                    outcome = activity_type.lower()
                else:
                    outcome = 'unknown'
                    
                    # Debug info for unknown activities
                    if self.debug_mode:
                        debug_entry = {
                            'type': activity_type,
                            'original_outcome': original_outcome,
                            'amount': float(activity.get('usdcSize', 0)),
                            'timestamp': activity.get('timestamp'),
                            'side': activity.get('side', ''),
                            'raw_keys': list(activity.keys())
                        }
                        stats['debug_info'].append(debug_entry)
            
            cash = float(activity.get('usdcSize', 0))
            shares = float(activity.get('size', 0))
            
            # Count by outcome and sum amounts
            if outcome not in stats['outcomes']:
                stats['outcomes'][outcome] = {'count': 0, 'amount': 0.0, 'shares': 0.0}
            
            stats['outcomes'][outcome]['count'] += 1
            stats['outcomes'][outcome]['amount'] += cash
            stats['outcomes'][outcome]['shares'] += shares
            
            # Also track Up/Down specifically
            if outcome == 'up':
                stats['up_trades'] += 1
                stats['up_amount'] += cash
                stats['up_shares'] += shares
            elif outcome == 'down':
                stats['down_trades'] += 1
                stats['down_amount'] += cash
                stats['down_shares'] += shares
            else:
                stats['other_trades'] += 1
                stats['other_amount'] += cash
        
        return stats
    
    def get_market_stats_display(self, stats: Dict, market_name: str = None) -> str:
        """Return market statistics as a formatted string"""
        lines = []
        lines.append("=" * 70)
        if market_name:
            lines.append(f"📊 MARKET STATS: {market_name}")
        else:
            lines.append("📊 MARKET STATS")
        lines.append("=" * 70)
        
        # Calculate total from all outcomes
        total_amount = sum(data['amount'] for data in stats['outcomes'].values())
        
        lines.append(f"Total Trades: {stats['total_trades']}")
        lines.append(f"Total Amount: ${total_amount:.2f}")
        lines.append("-" * 50)
        
        # Show all outcomes sorted by amount (descending)
        sorted_outcomes = sorted(stats['outcomes'].items(), key=lambda x: x[1]['amount'], reverse=True)
        
        up_down_total = 0
        up_down_shares_total = 0
        for outcome, data in sorted_outcomes:
            if data['count'] > 0:
                outcome_display = outcome.upper() if outcome else "UNKNOWN"
                outcome_color = self.colorize_outcome(outcome_display) if self.show_colors else outcome_display
                
                # Format shares display (show as integer if whole number, otherwise 2 decimals)
                shares = data.get('shares', 0)
                shares_display = f"{shares:,.0f}" if shares == int(shares) else f"{shares:,.2f}"
                
                lines.append(f"{outcome_color:<15} Trades: {data['count']:>3} | Amount: ${data['amount']:>10.2f} | Shares: {shares_display:>12}")
                
                # Track up/down for ratio calculation
                if outcome.lower() in ['up', 'down']:
                    up_down_total += data['amount']
                    up_down_shares_total += shares
        
        # Show UP/DOWN ratio if both exist
        if stats['up_trades'] > 0 and stats['down_trades'] > 0 and up_down_total > 0:
            lines.append("-" * 70)
            up_pct = (stats['up_amount'] / up_down_total) * 100
            down_pct = (stats['down_amount'] / up_down_total) * 100
            
            up_shares_pct = (stats['up_shares'] / up_down_shares_total) * 100 if up_down_shares_total > 0 else 0
            down_shares_pct = (stats['down_shares'] / up_down_shares_total) * 100 if up_down_shares_total > 0 else 0
            
            lines.append(f"UP/DOWN Trading Ratio: {up_pct:.1f}% / {down_pct:.1f}% (by amount)")
            lines.append(f"UP/DOWN Shares Ratio:  {up_shares_pct:.1f}% / {down_shares_pct:.1f}% (by shares)")
            lines.append(f"UP/DOWN Volume: ${up_down_total:.2f} (out of ${total_amount:.2f} total)")
            
            # Show share totals with proper formatting
            up_shares_display = f"{stats['up_shares']:,.0f}" if stats['up_shares'] == int(stats['up_shares']) else f"{stats['up_shares']:,.2f}"
            down_shares_display = f"{stats['down_shares']:,.0f}" if stats['down_shares'] == int(stats['down_shares']) else f"{stats['down_shares']:,.2f}"
            total_shares_display = f"{up_down_shares_total:,.0f}" if up_down_shares_total == int(up_down_shares_total) else f"{up_down_shares_total:,.2f}"
            
            lines.append(f"UP/DOWN Shares: {up_shares_display} / {down_shares_display} (total: {total_shares_display})")
        
        lines.append("=" * 70)
        
        # Add debug info if enabled and available
        if self.debug_mode and stats.get('debug_info'):
            lines.append("\n🔍 DEBUG INFO - Unknown Activities:")
            lines.append("-" * 50)
            for i, debug in enumerate(stats['debug_info'][:5]):  # Show first 5
                lines.append(f"#{i+1} Type: {debug['type']} | Side: {debug['side']} | Amount: ${debug['amount']:.2f}")
                lines.append(f"    Keys: {', '.join(debug['raw_keys'][:10])}...")  # First 10 keys
            if len(stats['debug_info']) > 5:
                lines.append(f"... and {len(stats['debug_info']) - 5} more")
            lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def parse_activity(self, activity_data: Dict) -> Optional[Dict]:
        """Parse activity data from API response"""
        try:
            parsed = {
                'timestamp': activity_data.get('timestamp', datetime.now().isoformat()),
                'type': activity_data.get('type', 'unknown'),
                'side': activity_data.get('side', ''),
                'market': activity_data.get('title', 'unknown'),
                'outcome': activity_data.get('outcome', 'unknown'),
                'tokens': activity_data.get('size', 0),
                'cash': activity_data.get('usdcSize', 0),
                'price': activity_data.get('price', 0),
                'transaction_hash': activity_data.get('transactionHash', ''),
                'raw_data': activity_data
            }
            
            # Convert timestamp if it's a unix timestamp
            if isinstance(parsed['timestamp'], (int, float)):
                parsed['timestamp'] = datetime.fromtimestamp(parsed['timestamp']).isoformat()
            
            return parsed
            
        except Exception as e:
            print(f"Error parsing activity: {e}")
            return None

    def log_to_file(self, data: Dict) -> None:
        """Log data to JSON file if enabled"""
        if self.log_to_json:
            with open(f'activities_{self.username}.json', 'a') as f:
                f.write(json.dumps(data) + '\n')

    def monitor_activity(self, interval: int = 60, stats_only: bool = False, max_activity_lines: int = 20) -> None:
        """Monitor user activities at specified intervals"""
        print(f"Starting activity monitoring for @{self.username}")
        print(f"User address: {self.user_address}")
        print(f"Check interval: {interval} seconds")
        print(f"JSON logging: {'Enabled' if self.log_to_json else 'Disabled'}")
        print(f"Color highlighting: {'Enabled' if self.show_colors else 'Disabled'}")
        print(f"Market filter: {self.market_filter if self.market_filter else 'None'}")
        print(f"Stats only: {'Enabled' if stats_only else 'Disabled'}")
        print("-" * 50)
        print("Press Ctrl+C to stop monitoring...")
        time.sleep(2)  # Give user time to see the startup info
        
        activity_log = []  # Keep track of recent activities
        seen_tx_hashes = set()  # Track transaction hashes to prevent duplicates
        
        while True:
            try:
                # Clear screen for refresh
                self.clear_screen()
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Get activities
                all_activities = self.fetch_activities()
                
                # Filter by market if specified
                market_filtered = self.filter_by_market(all_activities)
                
                # Use all activities for consistent results
                # Note: new_only filtering is designed for continuous monitoring, 
                # but causes inconsistent results when restarting the program
                activities = market_filtered
                
                # Show market stats at top if filtering by specific market
                if self.market_filter and market_filtered:
                    # Get the most common market name from filtered activities
                    market_titles = {}
                    for activity in market_filtered:
                        title = activity.get('title', '')
                        if title:
                            market_titles[title] = market_titles.get(title, 0) + 1
                    
                    if market_titles:
                        # Use the most frequent market title for exact stats
                        most_common_market = max(market_titles, key=market_titles.get)
                        exact_market_activities = self.get_exact_market_activities(market_filtered, most_common_market)
                        
                        if exact_market_activities:
                            stats = self.calculate_market_stats(exact_market_activities)
                            stats_display = self.get_market_stats_display(stats, most_common_market)
                            print(stats_display)
                            
                            # Calculate and show P&L
                            pnl_summary = self.calculate_position_value(exact_market_activities)
                            pnl_display = self.get_pnl_display(pnl_summary)
                            if pnl_display:
                                print(pnl_display)
                            
                            # Show how many different markets were found if more than one
                            if len(market_titles) > 1:
                                print(f"📝 Note: Found {len(market_titles)} different markets matching '{self.market_filter}'. Stats shown for: {most_common_market}")
                            print()  # Add spacing
                
                # Show activity feed
                print(f"🔄 ACTIVITY FEED - Last updated: {timestamp}")
                print("=" * 70)
                
                if not stats_only:
                    # Add new activities to log
                    if activities:
                        print(f"📝 Processing {len(activities)} activities from API...")
                        added_count = 0
                        
                        # Calculate dynamic column widths
                        parsed_activities = []
                        for activity in activities:
                            parsed = self.parse_activity(activity)
                            if parsed:
                                parsed_activities.append(parsed)
                        
                        if parsed_activities:
                            # Calculate max widths (account for display text, not colored text)
                            max_type_width = max(len(p['type']) for p in parsed_activities)
                            max_side_width = max(len(p.get('side', '')) for p in parsed_activities)
                            max_outcome_width = max(len(p.get('outcome', '')) for p in parsed_activities)
                            max_tokens_width = max(len(f"{float(p['tokens']):,.0f}" if float(p['tokens']) == int(float(p['tokens'])) else f"{float(p['tokens']):,.2f}") for p in parsed_activities)
                            max_cash_width = max(len(f"{float(p['cash']):,.2f}") for p in parsed_activities)
                            
                            # Set minimum widths for better alignment
                            max_type_width = max(max_type_width, 5)
                            max_side_width = max(max_side_width, 3)
                            max_outcome_width = max(max_outcome_width, 4)
                            max_tokens_width = max(max_tokens_width, 8)
                            max_cash_width = max(max_cash_width, 10)
                            
                            # Process activities with dynamic alignment
                            for parsed in parsed_activities:
                                # Format quantities and price
                                tokens = float(parsed['tokens'])
                                tokens_display = f"{tokens:,.0f}" if tokens == int(tokens) else f"{tokens:,.2f}"
                                price = float(parsed['price'])
                                cash_display = f"${float(parsed['cash']):,.2f}"
                                
                                # Handle MERGE activities differently (no side/outcome)
                                if parsed['type'] == 'MERGE':
                                    # For MERGE, use empty strings with proper spacing
                                    side_display = ""
                                    outcome_display = ""
                                    activity_line = f"[{parsed['timestamp'][11:19]}] {parsed['type']:<{max_type_width}} {side_display:<{max_side_width}} {outcome_display:<{max_outcome_width}} │ {tokens_display:>{max_tokens_width}} shares @ ${price:.3f} │ {cash_display:>12}"
                                else:
                                    # For regular trades, use colors but calculate spacing based on original text
                                    side_text = parsed['side']
                                    outcome_text = parsed['outcome']
                                    colored_side = self.colorize_side(side_text)
                                    colored_outcome = self.colorize_outcome(outcome_text)
                                    
                                    # Calculate padding needed (color codes don't count toward display width)
                                    side_padding = max_side_width - len(side_text)
                                    outcome_padding = max_outcome_width - len(outcome_text)
                                    
                                    activity_line = f"[{parsed['timestamp'][11:19]}] {parsed['type']:<{max_type_width}} {colored_side}{' ' * side_padding} {colored_outcome}{' ' * outcome_padding} │ {tokens_display:>{max_tokens_width}} shares @ ${price:.3f} │ {cash_display:>12}"
                                
                                # Check for duplicates using transaction hash
                                tx_hash = parsed.get('transaction_hash', '')

                                # Add activity if it has a unique tx_hash or no tx_hash at all
                                if not tx_hash or tx_hash not in seen_tx_hashes:
                                    # Store activity with full timestamp for sorting
                                    activity_log.append((parsed['timestamp'], activity_line))
                                    if tx_hash:  # Only track non-empty hashes
                                        seen_tx_hashes.add(tx_hash)
                                    added_count += 1
                                # Log to file if enabled
                                self.log_to_file(parsed)
                        
                        print(f"✅ Added {added_count} activities to display")
                    
                    # Sort activities by full timestamp (most recent first) and trim
                    activity_log.sort(key=lambda x: x[0], reverse=True)  # Sort by full timestamp
                    activity_log = activity_log[:max_activity_lines]
                    
                    # Display recent activities
                    if activity_log:
                        for timestamp, activity_line in activity_log:
                            print(f"  • {activity_line}")
                    else:
                        print("  No activities yet...")
                
                # Show status
                status_parts = []
                if activities and not stats_only:
                    status_parts.append(f"{len(activities)} activities from API")
                    status_parts.append(f"{len(activity_log)} displayed")
                
                if market_filtered:
                    status_parts.append(f"{len(market_filtered)} total in market")
                
                if status_parts:
                    print("-" * 70)
                    print(f"Status: {' | '.join(status_parts)}")
                
                print("=" * 70)
                print("Press Ctrl+C to stop monitoring...")
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Error during monitoring: {e}")
            
            time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description='Monitor Polymarket user activities')
    parser.add_argument('--address', required=True, help='User wallet address (required)')
    parser.add_argument('--interval', type=int, default=10, help='Check interval in seconds (default: 10)')
    parser.add_argument('--market', type=str, help='Filter activities by market name (partial match)')  
    parser.add_argument('--no-colors', action='store_true', help='Disable color highlighting')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug information')
    parser.add_argument('--num-activities', type=int, default=200, help='Number of activities to fetch (default: 200)')
    parser.add_argument('--max-activity-lines', type=int, default=20, help='Maximum activities to show (default: 20)')
    args = parser.parse_args()
    
    monitor = PolymarketMonitor(
        'LondonBridge',  # Default username
        args.address, 
        log_to_json=False,  # Simplified - no JSON logging by default
        show_colors=not args.no_colors,
        market_filter=args.market,
        exact_market=False,  # Always use partial matching
        num_activities=args.num_activities
    )
    monitor.debug_mode = not args.no_debug  # Debug enabled by default
    
    print("Polymarket Activity Monitor")
    print("=" * 30)
    
    # If market filter is specified, check for multiple markets and let user choose
    selected_market = None
    if args.market:
        print(f"Fetching activities to find markets matching '{args.market}'...")
        
        # Get initial activities to find available markets
        all_activities = monitor.fetch_activities()
        market_filtered = monitor.filter_by_market(all_activities)
        
        if not market_filtered:
            print(f"❌ No activities found matching '{args.market}'")
            return
        
        # Get unique markets
        market_counts = monitor.get_market_choices(market_filtered)
        
        if len(market_counts) > 1:
            # Multiple markets found - let user choose
            selected_market = monitor.prompt_market_selection(market_counts)
            # Update the monitor to use exact matching for the selected market
            monitor.market_filter = selected_market.lower()
            monitor.exact_market = True
        elif len(market_counts) == 1:
            # Only one market found
            selected_market = list(market_counts.keys())[0]
            print(f"\n✅ Found single market: {selected_market}")
    
    # Start monitoring
    try:
        print(f"\nStarting monitoring...")
        if selected_market:
            print(f"📊 Monitoring: {selected_market}")
    
        monitor.monitor_activity(args.interval, stats_only=False, max_activity_lines=args.max_activity_lines)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")

if __name__ == "__main__":
    main()