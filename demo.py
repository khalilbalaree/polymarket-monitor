#!/usr/bin/env python3
"""
Interactive Demo of the Polymarket Decision Analysis System
Demonstrates the deep learning-based reverse engineering of trading decisions
"""

import sys
import time
from datetime import datetime
from decision_analyzer import DecisionAnalyzer
from markov_decision_model import create_synthetic_activities

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a step in the demo"""
    print(f"\n🔹 Step {step_num}: {description}")
    print("-" * 40)

def wait_for_user():
    """Wait for user input to continue"""
    input("\nPress Enter to continue...")

def demo_with_synthetic_data():
    """Demo using synthetic data when no real data is available"""
    print_header("DEMO: SYNTHETIC DATA ANALYSIS")
    
    print("📝 This demo uses synthetic trading data to demonstrate the system capabilities.")
    print("   In real usage, the system would analyze actual Polymarket trading activities.")
    
    wait_for_user()
    
    print_step(1, "Creating Synthetic Trading Data")
    
    # Create synthetic activities
    activities = create_synthetic_activities(500)
    print(f"✅ Generated {len(activities)} synthetic trading activities")
    
    # Show sample activity
    sample_activity = activities[0]
    print(f"\nSample Activity:")
    print(f"  Timestamp: {sample_activity['timestamp']}")
    print(f"  Action: {sample_activity['side']}")
    print(f"  Market: {sample_activity['title']}")
    print(f"  Outcome: {sample_activity['outcome']}")
    print(f"  Price: ${sample_activity['price']:.3f}")
    print(f"  Volume: ${sample_activity['usdcSize']:.2f}")
    
    wait_for_user()
    
    print_step(2, "Initializing Decision Analysis System")
    
    # Initialize analyzer with synthetic data
    analyzer = DecisionAnalyzer("DEMO_USER", model_type='lstm')
    analyzer.activities_cache = activities
    
    print("✅ Decision analyzer initialized")
    print(f"   Model Type: LSTM")
    print(f"   Training Data: {len(activities)} activities")
    
    wait_for_user()
    
    print_step(3, "Training Deep Learning Model")
    print("🧠 Training neural network to learn decision patterns...")
    
    # Train with fewer epochs for demo speed
    patterns = analyzer.train_decision_model(epochs=20)
    
    print("✅ Model training completed!")
    
    # Show basic patterns
    print(f"\n📊 Discovered Decision Patterns:")
    if 'action_distribution' in patterns:
        for action, count in patterns['action_distribution'].items():
            percentage = (count / patterns['total_activities']) * 100
            print(f"  {action}: {count} trades ({percentage:.1f}%)")
    
    wait_for_user()
    
    print_step(4, "Making Decision Predictions")
    
    # Make prediction
    prediction = analyzer.predict_next_decision()
    
    if 'error' not in prediction:
        print("🔮 Next Action Prediction:")
        print(f"   Predicted Action: {prediction['predicted_action']}")
        print(f"   Confidence: {prediction['confidence']:.1%}")
        
        print(f"\n📊 Action Probabilities:")
        for action, prob in prediction['action_probabilities'].items():
            # Create visual probability bar
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {action:4s}: {bar} {prob:.1%}")
        
        print(f"\n💡 State Value Estimate: {prediction['estimated_value']:.3f}")
    else:
        print(f"❌ Prediction Error: {prediction['error']}")
    
    wait_for_user()
    
    print_step(5, "Generating Analysis Report")
    
    # Generate comprehensive report
    report = analyzer.generate_decision_report()
    
    print("📋 Comprehensive Analysis Report Generated:")
    print("\n" + "─" * 50)
    # Show first part of report
    report_lines = report.split('\n')
    for line in report_lines[:25]:  # Show first 25 lines
        print(line)
    
    if len(report_lines) > 25:
        print("...")
        print(f"({len(report_lines) - 25} more lines)")
    
    print("─" * 50)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"demo_report_{timestamp}.txt"
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"\n📁 Full report saved as: {filename}")
    
    wait_for_user()
    
    print_step(6, "Creating Visualizations")
    
    try:
        print("📊 Generating decision pattern visualizations...")
        analyzer.visualize_decision_patterns(save_plots=True)
        print("✅ Visualizations created and saved!")
    except Exception as e:
        print(f"⚠️  Visualization error (this is normal in some environments): {str(e)}")
    
    wait_for_user()
    
    print_header("DEMO COMPLETED SUCCESSFULLY!")
    
    print("🎉 You have successfully seen the Polymarket Decision Analysis System in action!")
    print("\nKey Capabilities Demonstrated:")
    print("  ✅ Deep learning model training")
    print("  ✅ Decision pattern recognition")
    print("  ✅ Next action prediction with confidence scores")
    print("  ✅ Comprehensive analysis reporting")
    print("  ✅ Data visualization")
    
    print(f"\nNext Steps:")
    print("  1. Try with real user data using: python decision_analyzer.py --address YOUR_ADDRESS")
    print("  2. Experiment with different model types (--model-type transformer)")
    print("  3. Analyze specific markets (--analyze-market 'market_name')")
    print("  4. Use real-time monitoring (--monitor)")

def demo_real_user_analysis():
    """Demo with real user data"""
    print_header("DEMO: REAL USER ANALYSIS")
    
    # Use the default test address
    user_address = "0xfb1c3c1ab4fb2d0cbcbb9538c8d4d357dd95963e"
    
    print(f"📊 Analyzing real trading data for user: {user_address}")
    print("   This will fetch actual data from Polymarket's API")
    
    wait_for_user()
    
    analyzer = DecisionAnalyzer(user_address, model_type='lstm')
    
    print_step(1, "Fetching Real Trading Data")
    
    try:
        activities = analyzer.collect_training_data(num_activities=200)
        
        if not activities:
            print("❌ No activities found for this user")
            print("   This could mean:")
            print("   - The user hasn't made any trades recently")
            print("   - The address is incorrect")
            print("   - API connectivity issues")
            return False
        
        print(f"✅ Successfully collected {len(activities)} real trading activities")
        
        # Show sample real activity
        sample = activities[0]
        print(f"\nSample Real Activity:")
        print(f"  Time: {sample.get('timestamp', 'N/A')}")
        print(f"  Market: {sample.get('title', 'N/A')}")
        print(f"  Action: {sample.get('side', 'N/A')}")
        print(f"  Outcome: {sample.get('outcome', 'N/A')}")
        print(f"  Price: ${float(sample.get('price', 0)):.3f}")
        
        wait_for_user()
        
        print_step(2, "Training on Real Data")
        
        patterns = analyzer.train_decision_model(epochs=30)
        
        print("✅ Model trained on real trading behavior!")
        print(f"Total activities analyzed: {patterns['total_activities']}")
        
        wait_for_user()
        
        print_step(3, "Real Decision Prediction")
        
        prediction = analyzer.predict_next_decision()
        
        if 'error' not in prediction:
            print("🔮 Real User Decision Prediction:")
            print(f"   Next Action: {prediction['predicted_action']}")
            print(f"   Confidence: {prediction['confidence']:.1%}")
            
            context = prediction.get('context', {})
            print(f"   Latest Market: {context.get('latest_market', 'Unknown')}")
        else:
            print(f"❌ Prediction Error: {prediction['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during real data analysis: {str(e)}")
        return False

def interactive_demo():
    """Interactive demo with user choices"""
    print_header("POLYMARKET DECISION ANALYSIS SYSTEM DEMO")
    
    print("🧠 Welcome to the Polymarket Decision Analysis System!")
    print("   This system uses deep learning to reverse engineer trading decisions")
    print("   and predict future actions based on historical patterns.")
    
    print("\nDemo Options:")
    print("  1. Quick Demo with Synthetic Data (recommended for first time)")
    print("  2. Real User Analysis (requires active trading history)")
    print("  3. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                demo_with_synthetic_data()
                break
            elif choice == '2':
                success = demo_real_user_analysis()
                if not success:
                    print("\n🔄 Falling back to synthetic data demo...")
                    time.sleep(2)
                    demo_with_synthetic_data()
                break
            elif choice == '3':
                print("👋 Goodbye!")
                sys.exit(0)
            else:
                print("❌ Please enter 1, 2, or 3")
                
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """Main demo function"""
    try:
        interactive_demo()
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        print("\nThis might be due to:")
        print("- Missing dependencies (run: pip install -r requirements.txt)")
        print("- Network connectivity issues")
        print("- System compatibility issues")

if __name__ == "__main__":
    main()
