# Stock Ichimoku Cloud Analysis Tool

A Python script for downloading stock price data and generating Ichimoku Cloud charts with automated trading signal analysis.

## Features

- Downloads historical stock data using yfinance
- Generates professional candlestick charts with Ichimoku Cloud indicators
- Analyzes trading signals based on Ichimoku methodology
- Processes multiple stocks from a JSON configuration file
- Generates comprehensive trading reports with BUY/SELL/HOLD recommendations
- Supports both US and Copenhagen Stock Exchange listings

## Ichimoku Cloud Components

The tool calculates and visualizes all five Ichimoku components:

1. **Tenkan-sen (Conversion Line)**: 9-period moving average
2. **Kijun-sen (Base Line)**: 26-period moving average
3. **Senkou Span A (Leading Span A)**: Average of Tenkan and Kijun, shifted 26 periods forward
4. **Senkou Span B (Leading Span B)**: 52-period moving average, shifted 26 periods forward
5. **Chikou Span (Lagging Span)**: Closing price shifted 26 periods backward

## Installation

1. Clone this repository
2. Install required dependencies:

```bash
pip install yfinance pandas matplotlib
```

## Usage

1. Configure stocks in `stocks_config.json` (25 stocks included by default)
2. Run the script:

```bash
python download_novo_stock.py
```

## Output

The script generates:

- **CSV files**: Historical price data for each stock
- **PNG charts**: High-resolution Ichimoku Cloud candlestick charts
- **Trading report**: Detailed analysis with trading recommendations

## Configuration

Edit `stocks_config.json` to customize:

- Stock tickers to analyze
- Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
- Interval (1d, 1wk, 1mo, etc.)
- Output preferences (CSV/Chart generation)

## Trading Signals

The tool analyzes multiple Ichimoku signals:

### Bullish Signals
- Price above the cloud
- Tenkan-sen crosses above Kijun-sen
- Chikou Span above historical price
- Green cloud (Senkou Span A > Senkou Span B)

### Bearish Signals
- Price below the cloud
- Tenkan-sen crosses below Kijun-sen
- Chikou Span below historical price
- Red cloud (Senkou Span A < Senkou Span B)

## Disclaimer

This tool is for educational purposes only and should not be considered financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.

## License

MIT License
