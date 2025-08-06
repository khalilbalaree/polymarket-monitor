# Polymarket Activity Monitor

A real-time monitoring tool for tracking user activities on Polymarket prediction markets. This tool fetches and displays trading activities, calculates profit/loss, and provides market statistics with a beautifully formatted terminal interface.

## Features

- 🔄 **Real-time Activity Monitoring**: Continuously monitors user activities with customizable refresh intervals
- 📊 **Market Statistics**: Comprehensive trading statistics including UP/DOWN ratios and volume analysis
- 💰 **Profit & Loss Tracking**: Real-time P&L calculations with current market prices
- 🎯 **Market Filtering**: Filter activities by specific markets (partial or exact matching)
- 🌈 **Color-coded Display**: Beautiful terminal output with color-coded outcomes and trade types
- ⚡ **Optimized Alignment**: Perfectly aligned activity feed with proper column spacing
- 🔍 **Debug Mode**: Optional debug information for troubleshooting
- 📝 **Flexible Display**: Configurable number of activities to display

## Installation

1. **Clone or download the project:**
   ```bash
   git clone <repository-url>
   cd poly-monitor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the monitor:**
   ```bash
   python polymarket_monitor.py --address YOUR_WALLET_ADDRESS
   ```

## Requirements

- Python 3.7+
- `requests` library for API calls
- Terminal with ANSI color support (most modern terminals)

## Usage

### Basic Usage

Monitor all activities for a wallet address:
```bash
python polymarket_monitor.py --address 0x...
```

### Advanced Options

```bash
python polymarket_monitor.py \
  --address 0x... \
  --interval 30 \
  --market "trump" \
  --max-activity-lines 50 \
  --num-activities 500
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--address` | string | Required | User wallet address to monitor |
| `--interval` | int | 10 | Check interval in seconds |
| `--market` | string | None | Filter activities by market name (partial match) |
| `--max-activity-lines` | int | 20 | Maximum activities to show in feed |
| `--num-activities` | int | 200 | Number of activities to fetch from API |
| `--no-colors` | flag | False | Disable color highlighting |
| `--no-debug` | flag | False | Disable debug information |

## Examples

### 1. Monitor Specific Market
```bash
python polymarket_monitor.py --address 0x... --market "2024 Presidential Election"
```

### 2. High-Frequency Monitoring
```bash
python polymarket_monitor.py --address 0x... --interval 5 --max-activity-lines 100
```

### 3. No Colors (for logging/scripts)
```bash
python polymarket_monitor.py --address 0x... --no-colors > trading_log.txt
```

## Display Format

The activity feed shows:
- **Timestamp**: Time of the activity (HH:MM:SS)
- **Type**: Activity type (TRADE, MERGE, REDEEM, etc.)
- **Side**: BUY/SELL for trades
- **Outcome**: UP/DOWN/YES/NO for prediction outcomes
- **Shares**: Number of shares traded
- **Price**: Price per share
- **Total**: Total USD amount

Example output:
```
🔄 ACTIVITY FEED - Last updated: 2024-01-15 12:42:39
======================================================================
  • [12:42:39] TRADE BUY  Down │      95 shares @ $0.780 │      $74.10
  • [12:37:55] TRADE BUY  Up   │     199 shares @ $0.540 │     $107.46
  • [12:34:21] MERGE           │     508 shares @ $0.000 │     $508.00
  • [12:33:25] TRADE BUY  Down │     100 shares @ $0.470 │      $47.00
```

## Market Statistics

When filtering by market, the tool displays comprehensive statistics:

- **Total Trades & Volume**: Overall activity summary
- **Outcome Breakdown**: Trades and amounts by outcome (UP/DOWN/YES/NO)
- **Trading Ratios**: Percentage breakdown of trading activity
- **Share Analysis**: Share distribution across outcomes

## Profit & Loss Tracking

The P&L section shows:
- **Current Positions**: Shares held by outcome
- **Cost Basis**: Total amount invested
- **Current Value**: Real-time market valuation
- **Unrealized P&L**: Profit/loss with percentage

Example P&L display:
```
💰 PROFIT & LOSS
======================================================================
UP              Shares:        1,234 | Cost:   $617.00
                Price:         $0.520 | Value:  $641.68
                P&L:          +$24.68 (+4.0%)
----------------------------------------------------------------------
TOTAL COST:      $617.00
CURRENT VALUE:   $641.68
TOTAL P&L:       +$24.68 (+4.0%)
```

## Market Selection

When multiple markets match your filter, the tool will prompt you to select:
```
🎯 Found 3 different markets:
============================================================
 1. 2024 Presidential Election (45 activities)
 2. 2024 Presidential Election - Popular Vote (12 activities)
 3. 2024 Presidential Election Winner (23 activities)
============================================================
Select market to monitor (1-3): 1
```

## API Integration

The monitor uses the official Polymarket Data API:
- **Base URL**: `https://data-api.polymarket.com`
- **Price API**: `https://clob.polymarket.com/price`
- **Rate Limiting**: Respects API limits with configurable intervals
- **Error Handling**: Graceful handling of API failures

## Color Coding

- 🟢 **Green**: UP/YES outcomes, positive P&L
- 🔴 **Red**: DOWN/NO outcomes, negative P&L  
- 🔵 **Blue**: BUY transactions
- 🟡 **Yellow**: SELL transactions, other activities

## Troubleshooting

### Common Issues

1. **"No activities found"**: Check that the wallet address is correct and has recent activity
2. **API errors**: Verify internet connection and try increasing the interval
3. **Alignment issues**: Ensure terminal supports ANSI colors or use `--no-colors`

### Debug Mode

Enable debug mode to see additional information:
```bash
python polymarket_monitor.py --address 0x... --market "trump"
# Debug mode is enabled by default, use --no-debug to disable
```

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the monitor.

## License

This project is open source. Please check the repository for license details.

## Disclaimer

This tool is for informational purposes only. Always verify trading information through official Polymarket channels. The authors are not responsible for any trading decisions made using this tool.
