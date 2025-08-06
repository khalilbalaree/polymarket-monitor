# Polymarket Decision Analysis System

A comprehensive Python system that combines real-time activity monitoring with deep learning-based decision pattern analysis for Polymarket users. This system can reverse engineer trading decisions using Markov Decision Process modeling and neural networks.

## 🚀 Features

### Core Monitoring (Original)
- Real-time monitoring of user trading activities
- Color-coded output for better visualization
- Market filtering capabilities
- JSON logging support
- P&L calculation and display
- Activity statistics and analysis

### 🧠 Deep Learning Decision Analysis (New)
- **Markov Decision Process Modeling**: Reverse engineer user decision patterns
- **Neural Network Architectures**: LSTM and Transformer models for sequence prediction
- **State Representation**: Advanced feature extraction from market conditions, positions, and prices
- **Action Prediction**: Predict next trading actions (Buy/Sell/Hold) with confidence scores
- **Pattern Recognition**: Identify decision-making patterns and trading strategies
- **Real-time Inference**: Live prediction of user decisions based on current market state
- **Comprehensive Reporting**: Detailed analysis reports with visualizations
- **Model Comparison**: Compare different neural network architectures

## 📦 Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**System Requirements:**
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## 🎯 Usage

### 1. Basic Activity Monitoring

Monitor activities for a specific user:
```bash
python polymarket_monitor.py --address 0xUSER_WALLET_ADDRESS
```

### 2. Decision Pattern Analysis

Analyze and predict trading decisions using deep learning:

```bash
# Train model and generate comprehensive report
python decision_analyzer.py --address 0xUSER_ADDRESS --train --report --epochs 50

# Make predictions for next actions
python decision_analyzer.py --address 0xUSER_ADDRESS --predict

# Analyze specific market patterns
python decision_analyzer.py --address 0xUSER_ADDRESS --analyze-market "election" --visualize

# Real-time decision monitoring
python decision_analyzer.py --address 0xUSER_ADDRESS --monitor
```

### 3. Run Examples

Explore the system capabilities:
```bash
python example_usage.py
```

## 🧠 Deep Learning Architecture

### State Representation
The system extracts comprehensive state features:
- **Price Features**: Current price, recent price trends, volatility
- **Volume Features**: Trade volume, recent volume patterns
- **Position Features**: Current positions, P&L, portfolio value
- **Time Features**: Hour of day, day of week (cyclical encoding)
- **Market Features**: Market type, outcome preferences
- **Context Features**: Recent trading activity, market momentum

### Model Architectures

#### LSTM Model
```python
# Sequential decision modeling with attention
- LSTM layers for sequence processing
- Multi-head attention mechanism
- Dual heads: action prediction + value estimation
- Dropout regularization
```

#### Transformer Model
```python
# Advanced attention-based modeling
- Multi-head self-attention
- Positional encoding
- Layer normalization
- Parallel processing capability
```

### Training Process
1. **Data Collection**: Fetch historical trading activities
2. **Feature Engineering**: Extract state-action sequences
3. **Model Training**: Train neural networks with validation
4. **Evaluation**: Assess prediction accuracy and model performance
5. **Inference**: Real-time decision prediction

## 📊 Analysis Capabilities

### Decision Pattern Recognition
- **Action Distribution**: Buy/Sell/Hold preferences
- **Price Sensitivity**: Correlation between prices and actions
- **Time Patterns**: Trading activity by time of day/week
- **Market Preferences**: Most traded markets and outcomes
- **Risk Patterns**: Position sizing and risk management behavior

### Prediction Features
- **Next Action Prediction**: Predict Buy/Sell/Hold with confidence
- **Action Probabilities**: Probability distribution over actions
- **State Value Estimation**: Estimated value of current market state
- **Decision Confidence**: Model confidence in predictions

### Visualization Tools
- Action distribution pie charts
- Price distribution histograms by action type
- Trading activity heatmaps over time
- Outcome preference analysis
- Model training progress plots

## 🔧 Command Line Options

### Basic Monitoring (`polymarket_monitor.py`)
- `--address`: User wallet address (required)
- `--interval`: Check interval in seconds (default: 10)
- `--market`: Filter activities by market name
- `--no-colors`: Disable color highlighting
- `--num-activities`: Number of activities to fetch (default: 200)

### Decision Analysis (`decision_analyzer.py`)
- `--address`: User wallet address (required)
- `--model-type`: Model architecture (lstm/transformer)
- `--train`: Train the decision model
- `--predict`: Make next action prediction
- `--analyze-market`: Analyze specific market patterns
- `--report`: Generate comprehensive analysis report
- `--visualize`: Create decision pattern visualizations
- `--monitor`: Start real-time decision monitoring
- `--epochs`: Training epochs (default: 50)
- `--num-activities`: Training data size (default: 1000)

## 📈 Example Workflows

### 1. Complete Analysis Workflow
```bash
# Step 1: Collect data and train model
python decision_analyzer.py --address 0xUSER_ADDRESS --train --epochs 100 --num-activities 2000

# Step 2: Generate comprehensive report
python decision_analyzer.py --address 0xUSER_ADDRESS --report --visualize

# Step 3: Make predictions
python decision_analyzer.py --address 0xUSER_ADDRESS --predict

# Step 4: Start real-time monitoring
python decision_analyzer.py --address 0xUSER_ADDRESS --monitor
```

### 2. Market-Specific Analysis
```bash
# Analyze election market patterns
python decision_analyzer.py --address 0xUSER_ADDRESS --analyze-market "election" --train --report

# Compare different markets
python decision_analyzer.py --address 0xUSER_ADDRESS --analyze-market "crypto" --visualize
```

### 3. Model Comparison
```bash
# Train LSTM model
python decision_analyzer.py --address 0xUSER_ADDRESS --model-type lstm --train --epochs 50

# Train Transformer model
python decision_analyzer.py --address 0xUSER_ADDRESS --model-type transformer --train --epochs 50
```

## 🎨 Color Coding

- 🟢 **Green**: UP/YES outcomes, BUY actions, positive P&L
- 🔴 **Red**: DOWN/NO outcomes, SELL actions, negative P&L
- 🔵 **Blue**: BUY actions
- 🟡 **Yellow**: Other actions and neutral outcomes

## 📊 Output Examples

### Decision Prediction Output
```
🔮 Predicted Next Action: BUY
📊 Confidence: 87.3%

Action Probabilities:
  Hold: ░░░░░░░░░░░░░░░░░░░░ 12.1%
  Buy:  ████████████████████ 87.3%
  Sell: ░░░░░░░░░░░░░░░░░░░░ 0.6%

📈 Latest Market: Will Bitcoin hit $100k in 2024?
⏱️  Activities Analyzed: 45
```

### Analysis Report Sample
```
==============================================================
POLYMARKET DECISION ANALYSIS REPORT
==============================================================
User: 0x...dd95963e
Generated: 2024-01-15 14:30:22

OVERALL TRADING PATTERNS:
------------------------------
Total Activities: 1,247
Action Distribution:
  BUY: 623 (50.0%)
  SELL: 624 (50.0%)
Average Buy Price: $0.456
Average Sell Price: $0.544

NEXT ACTION PREDICTION:
------------------------------
Predicted Action: BUY
Confidence: 87.3%
Action Probabilities:
  Hold: 12.1%
  Buy: 87.3%
  Sell: 0.6%
Estimated State Value: 0.234

MODEL PERFORMANCE:
------------------------------
Final Training Accuracy: 78.5%
Final Validation Accuracy: 76.2%
Model Type: LSTM
==============================================================
```

## 🔬 Technical Details

### Feature Engineering
- **Cyclical Time Encoding**: Sin/cos encoding for temporal features
- **Price Normalization**: StandardScaler for numerical features
- **Categorical Encoding**: Label encoding for outcomes and markets
- **Sequence Padding**: Handle variable-length sequences
- **Attention Weights**: Interpretable decision factors

### Model Training
- **Loss Function**: Combined cross-entropy (actions) + MSE (values)
- **Optimizer**: Adam with learning rate scheduling
- **Regularization**: Dropout, early stopping
- **Validation**: Time-based train/validation split
- **Metrics**: Accuracy, precision, recall, F1-score

### Performance Optimization
- **GPU Support**: Automatic CUDA detection and usage
- **Batch Processing**: Efficient batch training
- **Memory Management**: Gradient checkpointing for large models
- **Caching**: Activity data caching for faster iterations

## 🔍 API Information

**Polymarket Data API:**
- Base URL: `https://data-api.polymarket.com`
- No API key required for public data
- Rate limits may apply
- Endpoints used: `/activity`, `/price`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA Issues**: Install PyTorch with CUDA support if using GPU
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Memory Issues**: Reduce batch size or sequence length for large datasets

4. **No Training Data**: Ensure the user address has sufficient trading history

### Getting Help

- Check the example usage script: `python example_usage.py`
- Review command line options: `python decision_analyzer.py --help`
- Examine the generated reports and visualizations for insights

## 🔮 Future Enhancements

- **Multi-user Analysis**: Compare decision patterns across users
- **Market Prediction**: Predict market outcomes based on user behavior
- **Strategy Optimization**: Recommend optimal trading strategies
- **Risk Assessment**: Quantify trading risk and portfolio optimization
- **Real-time Alerts**: Notify when unusual decision patterns are detected
