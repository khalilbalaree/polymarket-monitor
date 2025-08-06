#!/usr/bin/env python3
"""
Decision Analyzer Integration
Connects the Polymarket Monitor with the Markov Decision Process Reverse Engineering System
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from polymarket_monitor import PolymarketMonitor
from markov_decision_model import MarkovDecisionReverseEngineer, TradingStateEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse

class DecisionAnalyzer:
    """Integrates Polymarket monitoring with decision pattern analysis"""
    
    def __init__(self, user_address: str, model_type: str = 'lstm'):
        self.user_address = user_address
        self.monitor = PolymarketMonitor(user_address)
        self.mdp_system = MarkovDecisionReverseEngineer(model_type=model_type)
        self.activities_cache = []
        self.model_trained = False
        
    def collect_training_data(self, num_activities: int = 1000) -> List[Dict]:
        """Collect historical activities for training"""
        print(f"Collecting {num_activities} activities for training...")
        
        # Update monitor settings for maximum data collection
        self.monitor.num_activities = num_activities
        activities = self.monitor.fetch_activities()
        
        if not activities:
            print("No activities found!")
            return []
        
        print(f"Collected {len(activities)} activities")
        self.activities_cache = activities
        return activities
    
    def train_decision_model(self, activities: Optional[List[Dict]] = None, 
                           epochs: int = 100, save_model: bool = True) -> Dict:
        """Train the decision model on collected activities"""
        
        if activities is None:
            activities = self.activities_cache
        
        if not activities:
            raise ValueError("No activities available for training. Call collect_training_data() first.")
        
        print(f"Training decision model on {len(activities)} activities...")
        
        # Train the model
        self.mdp_system.train(activities, epochs=epochs)
        self.model_trained = True
        
        # Analyze patterns
        patterns = self.mdp_system.analyze_decision_patterns(activities)
        
        # Save model if requested
        if save_model:
            model_filename = f"decision_model_{self.user_address[-8:]}.pth"
            self.mdp_system.save_model(model_filename)
        
        return patterns
    
    def predict_next_decision(self, lookback_hours: int = 24) -> Dict:
        """Predict the next trading decision based on recent activity"""
        
        if not self.model_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get recent activities
        recent_activities = self.monitor.fetch_activities()
        
        if not recent_activities:
            return {'error': 'No recent activities found'}
        
        # Filter to recent activities within lookback window
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        filtered_activities = []
        for activity in recent_activities:
            timestamp = activity.get('timestamp', '')
            if timestamp:
                try:
                    activity_time = pd.to_datetime(timestamp)
                    if activity_time >= cutoff_time:
                        filtered_activities.append(activity)
                except:
                    continue
        
        if not filtered_activities:
            filtered_activities = recent_activities[:20]  # Fallback to last 20 activities
        
        # Make prediction
        prediction = self.mdp_system.predict_next_action(filtered_activities)
        
        # Add context information
        prediction['context'] = {
            'activities_analyzed': len(filtered_activities),
            'lookback_hours': lookback_hours,
            'latest_activity_time': filtered_activities[0].get('timestamp', 'unknown') if filtered_activities else 'unknown',
            'latest_market': filtered_activities[0].get('title', 'unknown') if filtered_activities else 'unknown'
        }
        
        return prediction
    
    def analyze_market_specific_patterns(self, market_name: str) -> Dict:
        """Analyze decision patterns for a specific market"""
        
        if not self.activities_cache:
            self.collect_training_data()
        
        # Filter activities for specific market
        market_activities = [
            activity for activity in self.activities_cache 
            if market_name.lower() in activity.get('title', '').lower()
        ]
        
        if not market_activities:
            return {'error': f'No activities found for market: {market_name}'}
        
        print(f"Analyzing {len(market_activities)} activities for market: {market_name}")
        
        # Analyze patterns
        patterns = self.mdp_system.analyze_decision_patterns(market_activities)
        
        # Add market-specific insights
        market_analysis = self._analyze_market_performance(market_activities)
        patterns['market_performance'] = market_analysis
        
        return patterns
    
    def _analyze_market_performance(self, activities: List[Dict]) -> Dict:
        """Analyze performance metrics for market activities"""
        
        buy_activities = [a for a in activities if a.get('side') == 'BUY']
        sell_activities = [a for a in activities if a.get('side') == 'SELL']
        
        analysis = {
            'total_trades': len(activities),
            'buy_trades': len(buy_activities),
            'sell_trades': len(sell_activities),
            'total_volume': sum(float(a.get('usdcSize', 0)) for a in activities),
            'avg_trade_size': 0,
            'price_range': {'min': 1.0, 'max': 0.0, 'avg': 0.5},
            'outcome_preference': {},
            'time_distribution': {}
        }
        
        if activities:
            volumes = [float(a.get('usdcSize', 0)) for a in activities]
            prices = [float(a.get('price', 0.5)) for a in activities]
            
            analysis['avg_trade_size'] = np.mean(volumes) if volumes else 0
            analysis['price_range'] = {
                'min': min(prices) if prices else 0,
                'max': max(prices) if prices else 1,
                'avg': np.mean(prices) if prices else 0.5
            }
            
            # Outcome preferences
            outcomes = [a.get('outcome', 'unknown') for a in activities]
            outcome_counts = pd.Series(outcomes).value_counts()
            analysis['outcome_preference'] = outcome_counts.to_dict()
            
            # Time distribution
            hours = []
            for activity in activities:
                timestamp = activity.get('timestamp', '')
                if timestamp:
                    try:
                        dt = pd.to_datetime(timestamp)
                        hours.append(dt.hour)
                    except:
                        continue
            
            if hours:
                hour_counts = pd.Series(hours).value_counts().sort_index()
                analysis['time_distribution'] = hour_counts.to_dict()
        
        return analysis
    
    def generate_decision_report(self, market_name: Optional[str] = None) -> str:
        """Generate a comprehensive decision analysis report"""
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("POLYMARKET DECISION ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"User: {self.user_address}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Overall patterns
        if self.activities_cache:
            patterns = self.mdp_system.analyze_decision_patterns(self.activities_cache)
            
            report_lines.append("OVERALL TRADING PATTERNS:")
            report_lines.append("-" * 30)
            report_lines.append(f"Total Activities: {patterns['total_activities']}")
            
            if 'action_distribution' in patterns:
                report_lines.append("Action Distribution:")
                for action, count in patterns['action_distribution'].items():
                    percentage = (count / patterns['total_activities']) * 100
                    report_lines.append(f"  {action}: {count} ({percentage:.1f}%)")
            
            if 'price_action_correlation' in patterns:
                price_corr = patterns['price_action_correlation']
                report_lines.append(f"Average Buy Price: ${price_corr.get('avg_buy_price', 0):.3f}")
                report_lines.append(f"Average Sell Price: ${price_corr.get('avg_sell_price', 0):.3f}")
            
            report_lines.append("")
        
        # Market-specific analysis
        if market_name:
            market_patterns = self.analyze_market_specific_patterns(market_name)
            if 'error' not in market_patterns:
                report_lines.append(f"MARKET ANALYSIS: {market_name}")
                report_lines.append("-" * 30)
                
                if 'market_performance' in market_patterns:
                    perf = market_patterns['market_performance']
                    report_lines.append(f"Total Trades: {perf['total_trades']}")
                    report_lines.append(f"Buy/Sell Ratio: {perf['buy_trades']}/{perf['sell_trades']}")
                    report_lines.append(f"Total Volume: ${perf['total_volume']:,.2f}")
                    report_lines.append(f"Average Trade Size: ${perf['avg_trade_size']:,.2f}")
                    
                    price_range = perf['price_range']
                    report_lines.append(f"Price Range: ${price_range['min']:.3f} - ${price_range['max']:.3f} (avg: ${price_range['avg']:.3f})")
                
                report_lines.append("")
        
        # Predictions
        if self.model_trained:
            try:
                prediction = self.predict_next_decision()
                if 'error' not in prediction:
                    report_lines.append("NEXT ACTION PREDICTION:")
                    report_lines.append("-" * 30)
                    report_lines.append(f"Predicted Action: {prediction['predicted_action']}")
                    report_lines.append(f"Confidence: {prediction['confidence']:.1%}")
                    
                    probs = prediction['action_probabilities']
                    report_lines.append("Action Probabilities:")
                    for action, prob in probs.items():
                        report_lines.append(f"  {action}: {prob:.1%}")
                    
                    report_lines.append(f"Estimated State Value: {prediction['estimated_value']:.3f}")
                    report_lines.append("")
            except Exception as e:
                report_lines.append(f"Prediction Error: {str(e)}")
                report_lines.append("")
        
        # Model performance
        if self.model_trained and self.mdp_system.training_history['accuracy']:
            final_accuracy = self.mdp_system.training_history['accuracy'][-1]
            final_val_accuracy = self.mdp_system.training_history['val_accuracy'][-1]
            
            report_lines.append("MODEL PERFORMANCE:")
            report_lines.append("-" * 30)
            report_lines.append(f"Final Training Accuracy: {final_accuracy:.1f}%")
            report_lines.append(f"Final Validation Accuracy: {final_val_accuracy:.1f}%")
            report_lines.append(f"Model Type: {self.mdp_system.model_type.upper()}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def visualize_decision_patterns(self, save_plots: bool = True):
        """Create visualizations of decision patterns"""
        
        if not self.activities_cache:
            print("No data available for visualization")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Decision Pattern Analysis - User: {self.user_address[-8:]}', fontsize=16)
        
        # Prepare data
        df = pd.DataFrame(self.activities_cache)
        
        # 1. Action distribution
        if 'side' in df.columns:
            action_counts = df['side'].value_counts()
            axes[0, 0].pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Action Distribution')
        
        # 2. Price distribution by action
        if 'price' in df.columns and 'side' in df.columns:
            df['price_numeric'] = pd.to_numeric(df['price'], errors='coerce')
            buy_prices = df[df['side'] == 'BUY']['price_numeric'].dropna()
            sell_prices = df[df['side'] == 'SELL']['price_numeric'].dropna()
            
            if not buy_prices.empty:
                axes[0, 1].hist(buy_prices, alpha=0.7, label='Buy', bins=20)
            if not sell_prices.empty:
                axes[0, 1].hist(sell_prices, alpha=0.7, label='Sell', bins=20)
            axes[0, 1].set_xlabel('Price')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Price Distribution by Action')
            axes[0, 1].legend()
        
        # 3. Trading activity over time
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df_time = df.dropna(subset=['datetime'])
            
            if not df_time.empty:
                df_time['hour'] = df_time['datetime'].dt.hour
                hourly_counts = df_time['hour'].value_counts().sort_index()
                axes[1, 0].bar(hourly_counts.index, hourly_counts.values)
                axes[1, 0].set_xlabel('Hour of Day')
                axes[1, 0].set_ylabel('Number of Trades')
                axes[1, 0].set_title('Trading Activity by Hour')
        
        # 4. Outcome preferences
        if 'outcome' in df.columns:
            outcome_counts = df['outcome'].value_counts()
            axes[1, 1].bar(outcome_counts.index, outcome_counts.values)
            axes[1, 1].set_xlabel('Outcome')
            axes[1, 1].set_ylabel('Number of Trades')
            axes[1, 1].set_title('Outcome Preferences')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'decision_patterns_{self.user_address[-8:]}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as {filename}")
        
        plt.show()
    
    def real_time_decision_monitoring(self, interval: int = 300):
        """Monitor and predict decisions in real-time"""
        
        if not self.model_trained:
            print("Training model first...")
            self.collect_training_data()
            self.train_decision_model()
        
        print(f"Starting real-time decision monitoring (updates every {interval} seconds)")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                print("\n" + "="*50)
                print(f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Get prediction
                prediction = self.predict_next_decision()
                
                if 'error' not in prediction:
                    print(f"🔮 Predicted Next Action: {prediction['predicted_action']}")
                    print(f"📊 Confidence: {prediction['confidence']:.1%}")
                    
                    probs = prediction['action_probabilities']
                    print("Action Probabilities:")
                    for action, prob in probs.items():
                        bar_length = int(prob * 20)  # Scale to 20 chars
                        bar = "█" * bar_length + "░" * (20 - bar_length)
                        print(f"  {action:4s}: {bar} {prob:.1%}")
                    
                    context = prediction.get('context', {})
                    print(f"📈 Latest Market: {context.get('latest_market', 'Unknown')}")
                    print(f"⏱️  Activities Analyzed: {context.get('activities_analyzed', 0)}")
                else:
                    print(f"❌ Prediction Error: {prediction['error']}")
                
                print("="*50)
                
                # Wait for next update
                import time
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Real-time monitoring stopped")


def main():
    parser = argparse.ArgumentParser(description='Analyze trading decision patterns using deep learning')
    parser.add_argument('--address', type=str, required=True, help='User wallet address')
    parser.add_argument('--model-type', type=str, default='lstm', choices=['lstm', 'transformer'], 
                       help='Model type for decision analysis')
    parser.add_argument('--train', action='store_true', help='Train the decision model')
    parser.add_argument('--predict', action='store_true', help='Make next action prediction')
    parser.add_argument('--analyze-market', type=str, help='Analyze patterns for specific market')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive report')
    parser.add_argument('--visualize', action='store_true', help='Create decision pattern visualizations')
    parser.add_argument('--monitor', action='store_true', help='Start real-time decision monitoring')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs (default: 50)')
    parser.add_argument('--num-activities', type=int, default=1000, help='Number of activities to collect (default: 1000)')
    
    args = parser.parse_args()
    
    print("🧠 Polymarket Decision Pattern Analyzer")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = DecisionAnalyzer(args.address, model_type=args.model_type)
    
    # Collect data
    if args.train or args.predict or args.analyze_market or args.report or args.visualize or args.monitor:
        print("📊 Collecting training data...")
        analyzer.collect_training_data(args.num_activities)
    
    # Train model
    if args.train:
        print("🎯 Training decision model...")
        patterns = analyzer.train_decision_model(epochs=args.epochs)
        print("✅ Model training completed!")
        
        # Show basic patterns
        print("\n📈 Basic Decision Patterns:")
        if 'action_distribution' in patterns:
            for action, count in patterns['action_distribution'].items():
                print(f"  {action}: {count} trades")
    
    # Make prediction
    if args.predict:
        if not analyzer.model_trained:
            print("🎯 Training model for prediction...")
            analyzer.train_decision_model(epochs=args.epochs)
        
        print("🔮 Predicting next action...")
        prediction = analyzer.predict_next_decision()
        
        if 'error' not in prediction:
            print(f"Predicted Action: {prediction['predicted_action']}")
            print(f"Confidence: {prediction['confidence']:.1%}")
            print("Action Probabilities:")
            for action, prob in prediction['action_probabilities'].items():
                print(f"  {action}: {prob:.1%}")
        else:
            print(f"Error: {prediction['error']}")
    
    # Analyze specific market
    if args.analyze_market:
        print(f"📊 Analyzing market: {args.analyze_market}")
        market_analysis = analyzer.analyze_market_specific_patterns(args.analyze_market)
        
        if 'error' not in market_analysis:
            print(f"Total activities in market: {market_analysis['total_activities']}")
            if 'market_performance' in market_analysis:
                perf = market_analysis['market_performance']
                print(f"Total trades: {perf['total_trades']}")
                print(f"Total volume: ${perf['total_volume']:,.2f}")
        else:
            print(f"Error: {market_analysis['error']}")
    
    # Generate report
    if args.report:
        print("📋 Generating comprehensive report...")
        report = analyzer.generate_decision_report(args.analyze_market)
        print(report)
        
        # Save report to file
        report_filename = f"decision_report_{args.address[-8:]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        print(f"\n📁 Report saved as {report_filename}")
    
    # Create visualizations
    if args.visualize:
        print("📊 Creating decision pattern visualizations...")
        analyzer.visualize_decision_patterns()
    
    # Start real-time monitoring
    if args.monitor:
        analyzer.real_time_decision_monitoring()


if __name__ == "__main__":
    main()
