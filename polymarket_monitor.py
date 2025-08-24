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
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pytz

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'  # Up/Yes
    RED = '\033[91m'    # Down/No
    BLUE = '\033[94m'   # Buy
    YELLOW = '\033[93m' # Other actions
    RESET = '\033[0m'   # Reset color
    BOLD = '\033[1m'    # Bold text

class PolymarketMonitor:
    def __init__(self, user_address: str, log_to_json: bool = False, show_colors: bool = True, market_filter: str = None, exact_market: bool = False, num_activities: int = 200, currently_hourly_only: bool = False):
        self.user_address = user_address
        self.log_to_json = log_to_json
        self.show_colors = show_colors
        self.market_filter = market_filter.lower() if market_filter else None
        self.exact_market = exact_market
        self.num_activities = num_activities
        self.currently_hourly_only = currently_hourly_only
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
    
    def get_current_et_time(self) -> datetime:
        """Get current time in Eastern Time"""
        et_tz = pytz.timezone('US/Eastern')
        return datetime.now(et_tz)
    
    def parse_market_time(self, market_title: str) -> Optional[datetime]:
        """Parse time from market title like 'Bitcoin Up or Down - August 24, 2AM ET'"""
        # Pattern to match: "Month Day, HourAM/PM ET"
        pattern = r'(\w+)\s+(\d+),\s*(\d+)(AM|PM)\s*ET'
        
        match = re.search(pattern, market_title, re.IGNORECASE)
        if not match:
            return None
            
        try:
            month_name, day, hour, am_pm = match.groups()
            
            # Convert month name to number
            month_names = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            month = month_names.get(month_name.lower())
            if not month:
                return None
            
            # Convert to 24-hour format
            hour = int(hour)
            if am_pm.upper() == 'PM' and hour != 12:
                hour += 12
            elif am_pm.upper() == 'AM' and hour == 12:
                hour = 0
            
            # Create datetime in ET timezone
            current_et = self.get_current_et_time()
            et_tz = pytz.timezone('US/Eastern')
            market_time = et_tz.localize(datetime(current_et.year, month, int(day), hour, 0))
            
            return market_time
            
        except (ValueError, KeyError):
            return None
    
    def is_current_hourly_market(self, market_title: str) -> bool:
        """Check if market is for the current hour in ET"""
        market_time = self.parse_market_time(market_title)
        if not market_time:
            return False
        
        current_et = self.get_current_et_time()
        
        # Check if the market time is within the current hour
        return (market_time.date() == current_et.date() and 
                market_time.hour == current_et.hour)
    
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
        """Filter activities by market name if market_filter is set, or by current hourly markets if currently_hourly_only is enabled"""
        if not activities:
            return activities
        
        # If currently_hourly_only is enabled, filter by current hour markets
        if self.currently_hourly_only:
            filtered_activities = []
            for activity in activities:
                market_title = activity.get('title', '')
                if self.is_current_hourly_market(market_title):
                    filtered_activities.append(activity)
            return filtered_activities
        
        # Original market filter logic
        if not self.market_filter:
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
        positions = {}  # outcome -> {'shares': float, 'cost_basis': float, 'total_bought': float, 'total_sold': float, 'realized_pnl': float}
        
        # Sort activities by timestamp to process chronologically
        sorted_activities = sorted(activities, key=lambda x: x.get('timestamp', ''), reverse=False)
        
        # Calculate net positions and track realized P&L
        for activity in sorted_activities:
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
                    'cost_basis': 0.0,
                    'total_bought': 0.0,
                    'total_sold': 0.0,
                    'realized_pnl': 0.0,
                    'asset_ids': set(),
                    'condition_id': condition_id,
                    'outcome_index': outcome_index
                }
            
            # Track all asset IDs for this outcome
            if asset_id:
                positions[outcome]['asset_ids'].add(asset_id)
            
            if side == 'BUY':
                # Add to position
                positions[outcome]['shares'] += shares
                positions[outcome]['cost_basis'] += cost
                positions[outcome]['total_bought'] += cost
            elif side == 'SELL':
                # Calculate realized P&L for this sale
                if positions[outcome]['shares'] > 0:
                    # Calculate average cost per share of current position
                    avg_cost_per_share = positions[outcome]['cost_basis'] / positions[outcome]['shares']
                    # Cost basis of shares being sold
                    cost_of_sold_shares = shares * avg_cost_per_share
                    # Realized P&L = sale proceeds - cost of sold shares
                    realized_pnl = cost - cost_of_sold_shares
                    positions[outcome]['realized_pnl'] += realized_pnl
                    
                    # Reduce position
                    positions[outcome]['shares'] -= shares
                    positions[outcome]['cost_basis'] -= cost_of_sold_shares
                    positions[outcome]['total_sold'] += cost
                    
                    # Handle floating point precision - if position is very small, round to zero
                    if abs(positions[outcome]['shares']) < 0.001:
                        positions[outcome]['shares'] = 0.0
                    if abs(positions[outcome]['cost_basis']) < 0.01:
                        positions[outcome]['cost_basis'] = 0.0
                else:
                    # Short selling or selling more than we own
                    positions[outcome]['shares'] -= shares
                    positions[outcome]['cost_basis'] -= cost  # Negative cost basis for short position
                    positions[outcome]['total_sold'] += cost
        
        # Calculate current values and P&L
        total_realized_pnl = sum(pos['realized_pnl'] for pos in positions.values())
        total_invested = sum(pos['total_bought'] for pos in positions.values())
        total_sold = sum(pos['total_sold'] for pos in positions.values())
        
        pnl_summary = {
            'total_cost': 0.0,
            'total_current_value': 0.0,
            'total_pnl': 0.0,
            'realized_pnl': total_realized_pnl,
            'total_invested': total_invested,
            'total_sold': total_sold,
            'positions': {}
        }
        
        for outcome, position in positions.items():
            shares = position['shares']
            cost_basis = position['cost_basis']
            
            # Skip positions with zero or near-zero shares and no realized P&L
            if abs(shares) < 0.01 and abs(position['realized_pnl']) < 0.01:
                continue
                
            current_price = 0.5  # Default fallback
            
            # Get current market price (use sell price) for open positions
            if abs(shares) > 0.01:
                asset_ids = position['asset_ids']
                if asset_ids:
                    asset_id = list(asset_ids)[0]
                    # Get sell price (what you could sell for)
                    sell_price = self.get_market_price(asset_id, 'sell')
                    
                    # Use sell price if it's a real market price (not fallback)
                    if sell_price != 0.5:
                        current_price = sell_price
            
            current_value = shares * current_price
            unrealized_pnl = current_value - cost_basis
            total_pnl = unrealized_pnl + position['realized_pnl']
            
            # Calculate average cost per share for display
            avg_cost_per_share = 0
            if abs(shares) > 0.01:
                avg_cost_per_share = cost_basis / shares
            elif position['total_bought'] > 0:
                # For closed positions, show the average cost of all shares bought
                total_shares_bought = position['total_bought'] / (position['total_bought'] / max(position['total_bought'], 0.01))  # Rough estimate
                avg_cost_per_share = position['total_bought'] / max(total_shares_bought, 0.01)
            
            pnl_summary['positions'][outcome] = {
                'shares': shares,
                'cost_basis': cost_basis,
                'avg_cost_per_share': avg_cost_per_share,
                'current_price': current_price,
                'current_value': current_value,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': position['realized_pnl'],
                'total_pnl': total_pnl,
                'pnl_pct': (total_pnl / max(abs(cost_basis), 0.01) * 100) if cost_basis != 0 else 0,
                'asset_ids': list(position['asset_ids'])  # For debugging
            }
            
            pnl_summary['total_cost'] += cost_basis
            pnl_summary['total_current_value'] += current_value
        
        # Total P&L includes both unrealized (from current positions) and realized (from sales)
        unrealized_pnl = pnl_summary['total_current_value'] - pnl_summary['total_cost']
        pnl_summary['total_pnl'] = unrealized_pnl + total_realized_pnl
        pnl_summary['total_pnl_pct'] = (pnl_summary['total_pnl'] / total_invested * 100) if total_invested > 0 else 0
        
        return pnl_summary

    
    def get_pnl_display(self, pnl_summary: Dict) -> str:
        """Format P&L information for display"""
        if not pnl_summary['positions']:
            return ""
        
        lines = []
        lines.append("\n💰 P&L SUMMARY")
        lines.append("─" * 50)
        
        # Individual positions - compact format
        for outcome, pos in pnl_summary['positions'].items():
            outcome_display = outcome.upper()
            outcome_color = self.colorize_outcome(outcome_display) if self.show_colors else outcome_display
            
            # Show both realized and unrealized P&L
            total_pnl = pos['total_pnl']
            realized_pnl = pos['realized_pnl']
            unrealized_pnl = pos['unrealized_pnl']
            
            pnl_color = ""
            pnl_reset = ""
            if self.show_colors:
                if total_pnl > 0:
                    pnl_color = Colors.GREEN
                    pnl_reset = Colors.RESET
                elif total_pnl < 0:
                    pnl_color = Colors.RED
                    pnl_reset = Colors.RESET
            
            # Show current position if any
            if abs(pos['shares']) >= 0.01:
                shares_display = f"{pos['shares']:,.0f}" if pos['shares'] == int(pos['shares']) else f"{pos['shares']:,.2f}"
                avg_price = pos['avg_cost_per_share']
                
                # Format: OUTCOME: shares @ avg_price → current_price (Total P&L%)
                lines.append(f"{outcome_color}: {shares_display} @ ${avg_price:.3f} → ${pos['current_price']:.3f} {pnl_color}({pos['pnl_pct']:+.1f}%){pnl_reset}")
                
                # Show breakdown if both realized and unrealized exist
                if abs(realized_pnl) >= 0.01 and abs(unrealized_pnl) >= 0.01:
                    lines.append(f"  └─ Realized: ${realized_pnl:+.2f} | Unrealized: ${unrealized_pnl:+.2f}")
            else:
                # Closed position - show only realized P&L
                if abs(realized_pnl) >= 0.01:
                    lines.append(f"{outcome_color}: CLOSED {pnl_color}${realized_pnl:+.2f} ({pos['pnl_pct']:+.1f}%){pnl_reset}")
        
        # Total P&L - compact
        total_pnl_color = ""
        total_pnl_reset = ""
        if self.show_colors:
            if pnl_summary['total_pnl'] > 0:
                total_pnl_color = Colors.GREEN + Colors.BOLD
                total_pnl_reset = Colors.RESET
            elif pnl_summary['total_pnl'] < 0:
                total_pnl_color = Colors.RED + Colors.BOLD
                total_pnl_reset = Colors.RESET
        
        lines.append("─" * 50)
        
        # Show summary of total investment, current value, and P&L
        total_invested = pnl_summary.get('total_invested', 0)
        total_sold = pnl_summary.get('total_sold', 0)  # Cash received from sales
        current_positions_value = pnl_summary['total_current_value']  # Current value of open positions
        
        # Net investment = money put in - money taken out
        net_investment = total_invested - total_sold
        
        # Total current value = current position value + cash from sales
        total_current_value = current_positions_value + total_sold
        
        lines.append(f"Invested: ${total_invested:,.0f} | Sold: ${total_sold:,.0f} | Net: ${net_investment:,.0f}")
        lines.append(f"Current Value: ${total_current_value:,.0f} {total_pnl_color}({pnl_summary['total_pnl_pct']:+.1f}%){total_pnl_reset}")
        
        # Show breakdown of realized vs unrealized if significant
        if abs(pnl_summary.get('realized_pnl', 0)) >= 0.01:
            unrealized_total = pnl_summary['total_current_value'] - pnl_summary['total_cost']
            lines.append(f"P&L: ${pnl_summary['realized_pnl']:+.2f} realized + ${unrealized_total:+.2f} unrealized = ${pnl_summary['total_pnl']:+.2f}")
        
        lines.append("─" * 50)
        
        return "\n".join(lines)
    
    def get_market_choices(self, activities: List[Dict]) -> Dict[str, Dict]:
        """Get all unique markets from activities with their activity counts and most recent timestamps"""
        market_data = {}
        for activity in activities:
            title = activity.get('title', '').strip()
            if title:
                timestamp = activity.get('timestamp', 0)
                # Convert timestamp if it's a string
                if isinstance(timestamp, str):
                    try:
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
                    except:
                        timestamp = 0
                
                if title not in market_data:
                    market_data[title] = {'count': 0, 'latest_timestamp': timestamp}
                
                market_data[title]['count'] += 1
                # Keep track of the most recent timestamp
                if timestamp > market_data[title]['latest_timestamp']:
                    market_data[title]['latest_timestamp'] = timestamp
        
        return market_data
    
    def prompt_market_selection(self, market_data: Dict[str, Dict]) -> str:
        """Prompt user to select which market to monitor"""
        # Sort markets by most recent activity timestamp (most recent first)
        markets = sorted(market_data.keys(), key=lambda x: market_data[x]['latest_timestamp'], reverse=True)
        
        print(f"\n🎯 Found {len(markets)} different markets (ranked by most recent activity):")
        print("=" * 60)
        
        for i, market in enumerate(markets, 1):
            count = market_data[market]['count']
            timestamp = market_data[market]['latest_timestamp']
            
            # Format the timestamp for display
            if timestamp > 0:
                from datetime import datetime
                try:
                    dt = datetime.fromtimestamp(timestamp)
                    time_str = dt.strftime("%m/%d %H:%M")
                except:
                    time_str = "unknown"
            else:
                time_str = "unknown"
                
            print(f"{i:>2}. {market} ({count} activities, last: {time_str})")
        
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
            'up_buy_trades': 0,
            'up_sell_trades': 0,
            'down_buy_trades': 0,
            'down_sell_trades': 0,
            'up_buy_amount': 0.0,
            'up_sell_amount': 0.0,
            'down_buy_amount': 0.0,
            'down_sell_amount': 0.0,
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
            side = activity.get('side', '').upper()
            
            # Count by outcome and side, sum amounts for both BUY and SELL
            if outcome not in stats['outcomes']:
                stats['outcomes'][outcome] = {
                    'buy_count': 0, 'sell_count': 0,
                    'buy_amount': 0.0, 'sell_amount': 0.0,
                    'buy_shares': 0.0, 'sell_shares': 0.0
                }
            
            if side == 'BUY':
                stats['outcomes'][outcome]['buy_count'] += 1
                stats['outcomes'][outcome]['buy_amount'] += cash
                stats['outcomes'][outcome]['buy_shares'] += shares
            elif side == 'SELL':
                stats['outcomes'][outcome]['sell_count'] += 1
                stats['outcomes'][outcome]['sell_amount'] += cash
                stats['outcomes'][outcome]['sell_shares'] += shares
            
            # Also track Up/Down specifically
            if outcome == 'up':
                if side == 'BUY':
                    stats['up_buy_trades'] += 1
                    stats['up_buy_amount'] += cash
                    stats['up_shares'] += shares
                elif side == 'SELL':
                    stats['up_sell_trades'] += 1
                    stats['up_sell_amount'] += cash
            elif outcome == 'down':
                if side == 'BUY':
                    stats['down_buy_trades'] += 1
                    stats['down_buy_amount'] += cash
                    stats['down_shares'] += shares
                elif side == 'SELL':
                    stats['down_sell_trades'] += 1
                    stats['down_sell_amount'] += cash
            elif side == 'BUY':  # Other outcomes, only BUY trades for simplicity
                stats['other_trades'] += 1
                stats['other_amount'] += cash
        
        return stats
    
    def get_market_stats_display(self, stats: Dict, market_name: str = None) -> str:
        """Return market statistics as a formatted string"""
        lines = []
        
        # Compact header
        if market_name:
            # Truncate long market names
            display_name = market_name[:60] + "..." if len(market_name) > 60 else market_name
            lines.append(f"📊 {display_name}")
        else:
            lines.append("📊 MARKET STATS")
        lines.append("─" * 50)
        
        # Calculate total from all outcomes (both buy and sell)
        total_buy_amount = sum(data['buy_amount'] for data in stats['outcomes'].values())
        total_sell_amount = sum(data['sell_amount'] for data in stats['outcomes'].values())
        total_amount = total_buy_amount + total_sell_amount
        
        # Compact summary line
        lines.append(f"Trades: {stats['total_trades']} | Volume: ${total_amount:,.0f}")
        
        # Show outcomes in compact format with BUY/SELL breakdown
        sorted_outcomes = sorted(stats['outcomes'].items(), 
                               key=lambda x: x[1]['buy_amount'] + x[1]['sell_amount'], 
                               reverse=True)
        
        for outcome, data in sorted_outcomes:
            if data['buy_count'] > 0 or data['sell_count'] > 0:
                outcome_display = outcome.upper() if outcome else "UNKNOWN"
                outcome_color = self.colorize_outcome(outcome_display) if self.show_colors else outcome_display
                
                # Show BUY and SELL separately
                buy_info = f"{data['buy_count']} buys ${data['buy_amount']:,.0f}" if data['buy_count'] > 0 else ""
                sell_info = f"{data['sell_count']} sells ${data['sell_amount']:,.0f}" if data['sell_count'] > 0 else ""
                
                if buy_info and sell_info:
                    lines.append(f"{outcome_color}: {buy_info}, {sell_info}")
                elif buy_info:
                    lines.append(f"{outcome_color}: {buy_info}")
                elif sell_info:
                    lines.append(f"{outcome_color}: {sell_info}")
        
        # Show UP/DOWN ratios if both exist (buy volume only for position ratios)
        total_up_buy = stats['up_buy_amount']
        total_down_buy = stats['down_buy_amount']
        buy_total = total_up_buy + total_down_buy
        
        if stats['up_buy_trades'] > 0 and stats['down_buy_trades'] > 0 and buy_total > 0:
            # Buy trade ratio (by count)
            total_buy_trades = stats['up_buy_trades'] + stats['down_buy_trades']
            up_buy_pct = (stats['up_buy_trades'] / total_buy_trades) * 100
            down_buy_pct = (stats['down_buy_trades'] / total_buy_trades) * 100
            
            # Buy volume ratio (by amount) 
            up_buy_vol_pct = (total_up_buy / buy_total) * 100
            down_buy_vol_pct = (total_down_buy / buy_total) * 100
            
            lines.append(f"Buy Ratio: {up_buy_pct:.0f}% UP / {down_buy_pct:.0f}% DOWN")
            lines.append(f"Buy Volume: {up_buy_vol_pct:.0f}% UP / {down_buy_vol_pct:.0f}% DOWN")
        
        lines.append("─" * 50)
        
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
            with open(f'activities_{self.user_address}.json', 'a') as f:
                f.write(json.dumps(data) + '\n')

    def monitor_activity(self, interval: int = 60, stats_only: bool = False, max_activity_lines: int = 20) -> None:
        """Monitor user activities at specified intervals"""
        # print(f"Starting activity monitoring for @{self.username}")
        print(f"User address: {self.user_address}")
        print(f"Check interval: {interval} seconds")
        print(f"JSON logging: {'Enabled' if self.log_to_json else 'Disabled'}")
        print(f"Color highlighting: {'Enabled' if self.show_colors else 'Disabled'}")
        print(f"Market filter: {self.market_filter if self.market_filter else 'None'}")
        print(f"Currently hourly only: {'Enabled' if self.currently_hourly_only else 'Disabled'}")
        if self.currently_hourly_only:
            current_et = self.get_current_et_time()
            print(f"Current ET time: {current_et.strftime('%B %d, %I%p ET')}")
        print(f"Stats only: {'Enabled' if stats_only else 'Disabled'}")
        print("-" * 50)
        print("Press Ctrl+C to stop monitoring...")
        time.sleep(2)  # Give user time to see the startup info
        
        activity_log = []  # Keep track of recent activities
        seen_tx_hashes = set()  # Track transaction hashes to prevent duplicates
        first_run = True  # Track if this is the first run
        
        while True:
            try:
                # Get activities
                all_activities = self.fetch_activities()
                
                # Filter by market if specified
                market_filtered = self.filter_by_market(all_activities)
                
                # Check for new activities only
                new_activities = self.filter_new_activities(market_filtered)
                
                # Only refresh display if there are new activities or it's the first run
                if new_activities or first_run:
                    # Clear screen for refresh
                    self.clear_screen()
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Use all market-filtered activities for display (not just new ones)
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
                                
                                # Calculate market name width if no market filter is applied
                                max_market_width = 0
                                if not self.market_filter:
                                    max_market_width = max(len(p.get('market', '')[:40]) for p in parsed_activities)  # Limit to 40 chars
                                    max_market_width = max(max_market_width, 10)  # Minimum width
                                
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
                                    
                                    # Format market name if no filter is applied
                                    market_display = ""
                                    if not self.market_filter and max_market_width > 0:
                                        market_text = parsed.get('market', '')[:40]  # Truncate to 40 chars
                                        market_display = f" │ {market_text:<{max_market_width}}"
                                    
                                    # Handle MERGE activities differently (no side/outcome)
                                    if parsed['type'] == 'MERGE':
                                        # For MERGE, use empty strings with proper spacing
                                        side_display = ""
                                        outcome_display = ""
                                        activity_line = f"[{parsed['timestamp'][11:19]}] {parsed['type']:<{max_type_width}} {side_display:<{max_side_width}} {outcome_display:<{max_outcome_width}} │ {tokens_display:>{max_tokens_width}} shares @ ${price:.3f} │ {cash_display:>10}{market_display}"
                                    else:
                                        # For regular trades, use colors but calculate spacing based on original text
                                        side_text = parsed['side']
                                        outcome_text = parsed['outcome']
                                        colored_side = self.colorize_side(side_text)
                                        colored_outcome = self.colorize_outcome(outcome_text)
                                        
                                        # Calculate padding needed (color codes don't count toward display width)
                                        side_padding = max_side_width - len(side_text)
                                        outcome_padding = max_outcome_width - len(outcome_text)
                                        
                                        activity_line = f"[{parsed['timestamp'][11:19]}] {parsed['type']:<{max_type_width}} {colored_side}{' ' * side_padding} {colored_outcome}{' ' * outcome_padding} │ {tokens_display:>{max_tokens_width}} shares @ ${price:.3f} │ {cash_display:>10}{market_display}"
                                    
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
                        
                        # Display recent activities grouped by market
                        if activity_log:
                            # Group activities by market when no market filter is applied
                            if not self.market_filter:
                                market_groups = {}
                                for timestamp, activity_line in activity_log:
                                    # Extract market name from the activity line
                                    # Activity line format includes market at the end after the last │
                                    parts = activity_line.split('│')
                                    if len(parts) >= 3:
                                        market_part = parts[-1].strip()
                                        if market_part:
                                            market_name = market_part
                                        else:
                                            market_name = "Unknown Market"
                                    else:
                                        market_name = "Unknown Market"
                                    
                                    if market_name not in market_groups:
                                        market_groups[market_name] = []
                                    market_groups[market_name].append((timestamp, activity_line))
                                
                                # Display activities grouped by market (most recent first)
                                # Sort markets by their most recent activity timestamp
                                market_names = sorted(market_groups.keys(), 
                                                    key=lambda market: max(ts for ts, _ in market_groups[market]), 
                                                    reverse=True)
                                for i, market_name in enumerate(market_names):
                                    print("─" * 70)
                                    print(f"📈 {market_name}")
                                    print("─" * 70)
                                    for timestamp, activity_line in market_groups[market_name]:
                                        # Remove market name from activity line since it's now in the header
                                        line_parts = activity_line.split('│')
                                        if len(line_parts) >= 3:
                                            clean_line = '│'.join(line_parts[:-1])
                                        else:
                                            clean_line = activity_line
                                        print(f"  • {clean_line}")
                            else:
                                # When filtering by market, display normally without grouping
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
                    
                    first_run = False  # No longer first run after initial display
                    
                else:
                    # No new activities, don't clear screen or show any message
                    pass
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Error during monitoring: {e}")
            
            time.sleep(interval)

def load_address_book(filename='address_book.json'):
    """Load address book from JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                return data.get('addresses', [])
        else:
            print(f"Address book not found: {filename}")
            return []
    except Exception as e:
        print(f"Error loading address book: {e}")
        return []

def prompt_address_selection(addresses):
    """Prompt user to select an address from the address book"""
    if not addresses:
        print("No addresses available in address book")
        return None
    
    print("\n📋 ADDRESS BOOK")
    print("=" * 60)
    
    for i, addr in enumerate(addresses, 1):
        description = addr.get('description', '')
        desc_text = f" - {description}" if description else ""
        print(f"{i:>2}. {addr['name']}{desc_text}")
        print(f"     {addr['address']}")
    
    print("=" * 60)
    
    while True:
        try:
            choice = input(f"Select address to monitor (1-{len(addresses)}, or 'c' for custom): ").strip().lower()
            if not choice:
                continue
            
            if choice == 'c' or choice == 'custom':
                custom_address = input("Enter custom wallet address: ").strip()
                if custom_address:
                    return custom_address
                else:
                    print("Please enter a valid address")
                    continue
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(addresses):
                selected = addresses[choice_num - 1]
                print(f"\n✅ Selected: {selected['name']}")
                print(f"Address: {selected['address']}")
                return selected['address']
            else:
                print(f"Please enter a number between 1 and {len(addresses)}, or 'c' for custom")
                
        except ValueError:
            print("Please enter a valid number or 'c' for custom address")
        except KeyboardInterrupt:
            print("\nSelection cancelled by user")
            return None

def main():
    parser = argparse.ArgumentParser(description='Monitor Polymarket user activities')
    parser.add_argument('--address', type=str, help='User wallet address (if not provided, will prompt to select from address book)')
    parser.add_argument('--interval', type=int, default=10, help='Check interval in seconds (default: 10)')
    parser.add_argument('--market', type=str, help='Filter activities by market name (partial match)')  
    parser.add_argument('--current-hourly-only', action='store_true', help='Filter to show only markets for the current hour in ET (format: "August 24, 2AM ET")')
    parser.add_argument('--no-colors', action='store_true', help='Disable color highlighting')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug information')
    parser.add_argument('--num-activities', type=int, default=100, help='Number of activities to fetch (default: 100)')
    parser.add_argument('--max-activity-lines', type=int, default=20, help='Maximum activities to show (default: 20)')
    args = parser.parse_args()
    
    # If no address provided, prompt user to select from address book
    user_address = args.address
    if not user_address:
        print("Polymarket Activity Monitor")
        print("=" * 30)
        addresses = load_address_book()
        user_address = prompt_address_selection(addresses)
        if not user_address:
            print("No address selected. Exiting.")
            return
    else:
        print("Polymarket Activity Monitor")
        print("=" * 30)
    
    monitor = PolymarketMonitor(
        user_address, 
        log_to_json=False,  # Simplified - no JSON logging by default
        show_colors=not args.no_colors,
        market_filter=args.market,
        exact_market=False,  # Always use partial matching
        num_activities=args.num_activities,
        currently_hourly_only=args.current_hourly_only
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
        market_data = monitor.get_market_choices(market_filtered)
        
        if len(market_data) > 1:
            # Multiple markets found - let user choose
            selected_market = monitor.prompt_market_selection(market_data)
            # Update the monitor to use exact matching for the selected market
            monitor.market_filter = selected_market.lower()
            monitor.exact_market = True
        elif len(market_data) == 1:
            # Only one market found
            selected_market = list(market_data.keys())[0]
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