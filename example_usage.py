#!/usr/bin/env python3
"""
Example Usage of the Markov Decision Process Reverse Engineering System
Demonstrates how to analyze and predict user trading decisions on Polymarket
"""

import sys
from datetime import datetime
from decision_analyzer import DecisionAnalyzer

def example_basic_analysis():
    """Example 1: Basic decision pattern analysis"""
    print("=" * 60)
    print("EXAMPLE 1: BASIC DECISION PATTERN ANALYSIS")
    print("=" * 60)
    
    # Replace with actual user address
    user_address = "0xfb1c3c1ab4fb2d0cbcbb9538c8d4d357dd95963e"
    
    # Initialize analyzer
    analyzer = DecisionAnalyzer(user_address, model_type='lstm')
    
    # Collect training data
    print("📊 Collecting training data...")
    activities = analyzer.collect_training_data(num_activities=500)
    
    if not activities:
        print("❌ No activities found for this user")
        return
    
    print(f"✅ Collected {len(activities)} activities")
    
    # Train the model
    print("🎯 Training decision model...")
    patterns = analyzer.train_decision_model(epochs=30)
    
    # Display basic patterns
    print("\n📈 Decision Patterns Found:")
    print(f"Total Activities: {patterns['total_activities']}")
    
    if 'action_distribution' in patterns:
        print("\nAction Distribution:")
        for action, count in patterns['action_distribution'].items():
            percentage = (count / patterns['total_activities']) * 100
            print(f"  {action}: {count} trades ({percentage:.1f}%)")
    
    if 'price_action_correlation' in patterns:
        price_corr = patterns['price_action_correlation']
        print(f"\nPrice Analysis:")
        print(f"  Average Buy Price: ${price_corr.get('avg_buy_price', 0):.3f}")
        print(f"  Average Sell Price: ${price_corr.get('avg_sell_price', 0):.3f}")
    
    # Make a prediction
    print("\n🔮 Predicting next action...")
    prediction = analyzer.predict_next_decision()
    
    if 'error' not in prediction:
        print(f"Predicted Action: {prediction['predicted_action']}")
        print(f"Confidence: {prediction['confidence']:.1%}")
        print("Action Probabilities:")
        for action, prob in prediction['action_probabilities'].items():
            print(f"  {action}: {prob:.1%}")
    else:
        print(f"Prediction Error: {prediction['error']}")


def example_market_specific_analysis():
    """Example 2: Market-specific decision analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: MARKET-SPECIFIC ANALYSIS")
    print("=" * 60)
    
    user_address = "0xfb1c3c1ab4fb2d0cbcbb9538c8d4d357dd95963e"
    analyzer = DecisionAnalyzer(user_address, model_type='transformer')
    
    # Collect data
    activities = analyzer.collect_training_data(num_activities=800)
    
    if not activities:
        print("❌ No activities found")
        return
    
    # Find markets this user has traded in
    markets = {}
    for activity in activities:
        market_name = activity.get('title', '')
        if market_name:
            markets[market_name] = markets.get(market_name, 0) + 1
    
    if not markets:
        print("❌ No market data found")
        return
    
    # Analyze the most traded market
    most_traded_market = max(markets, key=markets.get)
    print(f"🎯 Analyzing most traded market: {most_traded_market}")
    print(f"   ({markets[most_traded_market]} activities)")
    
    market_analysis = analyzer.analyze_market_specific_patterns(most_traded_market)
    
    if 'error' not in market_analysis:
        print(f"\n📊 Market Analysis Results:")
        print(f"Total Activities: {market_analysis['total_activities']}")
        
        if 'market_performance' in market_analysis:
            perf = market_analysis['market_performance']
            print(f"Buy Trades: {perf['buy_trades']}")
            print(f"Sell Trades: {perf['sell_trades']}")
            print(f"Total Volume: ${perf['total_volume']:,.2f}")
            print(f"Average Trade Size: ${perf['avg_trade_size']:,.2f}")
            
            if 'outcome_preference' in perf:
                print("\nOutcome Preferences:")
                for outcome, count in perf['outcome_preference'].items():
                    print(f"  {outcome}: {count} trades")
    else:
        print(f"❌ Analysis Error: {market_analysis['error']}")


def example_comprehensive_report():
    """Example 3: Generate comprehensive analysis report"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 60)
    
    user_address = "0xfb1c3c1ab4fb2d0cbcbb9538c8d4d357dd95963e"
    analyzer = DecisionAnalyzer(user_address, model_type='lstm')
    
    # Collect and train
    print("📊 Collecting data and training model...")
    activities = analyzer.collect_training_data(num_activities=1000)
    
    if not activities:
        print("❌ No activities found")
        return
    
    analyzer.train_decision_model(epochs=25)
    
    # Generate comprehensive report
    print("\n📋 Generating comprehensive report...")
    report = analyzer.generate_decision_report()
    
    print(report)
    
    # Save report to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"analysis_report_{timestamp}.txt"
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"\n📁 Report saved as: {filename}")


def example_visualization():
    """Example 4: Create decision pattern visualizations"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: DECISION PATTERN VISUALIZATION")
    print("=" * 60)
    
    user_address = "0xfb1c3c1ab4fb2d0cbcbb9538c8d4d357dd95963e"
    analyzer = DecisionAnalyzer(user_address)
    
    # Collect data
    activities = analyzer.collect_training_data(num_activities=600)
    
    if not activities:
        print("❌ No activities found")
        return
    
    print("📊 Creating decision pattern visualizations...")
    analyzer.visualize_decision_patterns(save_plots=True)
    print("✅ Visualizations created and saved!")


def example_model_comparison():
    """Example 5: Compare LSTM vs Transformer models"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: MODEL COMPARISON (LSTM vs TRANSFORMER)")
    print("=" * 60)
    
    user_address = "0xfb1c3c1ab4fb2d0cbcbb9538c8d4d357dd95963e"
    
    # Test LSTM model
    print("🔍 Testing LSTM Model...")
    lstm_analyzer = DecisionAnalyzer(user_address, model_type='lstm')
    activities = lstm_analyzer.collect_training_data(num_activities=800)
    
    if not activities:
        print("❌ No activities found")
        return
    
    lstm_patterns = lstm_analyzer.train_decision_model(epochs=20)
    lstm_prediction = lstm_analyzer.predict_next_decision()
    
    # Test Transformer model
    print("\n🔍 Testing Transformer Model...")
    transformer_analyzer = DecisionAnalyzer(user_address, model_type='transformer')
    transformer_analyzer.activities_cache = activities  # Use same data
    transformer_patterns = transformer_analyzer.train_decision_model(epochs=20)
    transformer_prediction = transformer_analyzer.predict_next_decision()
    
    # Compare results
    print("\n📊 Model Comparison Results:")
    print("-" * 40)
    
    if lstm_analyzer.model_trained and transformer_analyzer.model_trained:
        lstm_accuracy = lstm_analyzer.mdp_system.training_history['val_accuracy'][-1]
        transformer_accuracy = transformer_analyzer.mdp_system.training_history['val_accuracy'][-1]
        
        print(f"LSTM Validation Accuracy: {lstm_accuracy:.2f}%")
        print(f"Transformer Validation Accuracy: {transformer_accuracy:.2f}%")
        
        if 'error' not in lstm_prediction and 'error' not in transformer_prediction:
            print(f"\nLSTM Prediction: {lstm_prediction['predicted_action']} (confidence: {lstm_prediction['confidence']:.1%})")
            print(f"Transformer Prediction: {transformer_prediction['predicted_action']} (confidence: {transformer_prediction['confidence']:.1%})")
            
            if lstm_prediction['predicted_action'] == transformer_prediction['predicted_action']:
                print("✅ Both models agree on the prediction!")
            else:
                print("⚠️ Models disagree on the prediction")


def main():
    """Run all examples"""
    print("🧠 Polymarket Decision Analysis Examples")
    print("This script demonstrates the capabilities of the deep learning system")
    print("for reverse engineering trading decisions on Polymarket")
    print()
    
    try:
        # Run examples
        example_basic_analysis()
        example_market_specific_analysis()
        example_comprehensive_report()
        example_visualization()
        example_model_comparison()
        
        print("\n" + "=" * 60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext Steps:")
        print("1. Check the generated reports and visualizations")
        print("2. Try with your own user address using decision_analyzer.py")
        print("3. Experiment with different model types and parameters")
        print("4. Use the real-time monitoring feature for live analysis")
        
    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during examples: {str(e)}")
        print("This might be due to:")
        print("- Network connectivity issues")
        print("- No trading data for the test user")
        print("- Missing dependencies (run: pip install -r requirements.txt)")
        sys.exit(1)


if __name__ == "__main__":
    main()
