#!/usr/bin/env python3
"""
Script to download stock prices and create Ichimoku Cloud charts
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import json
import os

def download_stock(ticker, stock_name, period="1y", interval="1d", save_to_csv=True):
    """
    Download stock prices for a given ticker

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    stock_name : str
        Full name of the stock
    period : str
        Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    interval : str
        Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    save_to_csv : bool
        If True, saves the data to a CSV file

    Returns:
    --------
    pandas.DataFrame : Stock price data
    """

    print(f"Downloading {ticker} stock data...")
    print(f"Period: {period}, Interval: {interval}")

    # Create ticker object
    stock = yf.Ticker(ticker)

    # Download historical data
    df = stock.history(period=period, interval=interval)

    if df.empty:
        print("No data retrieved. Please check your parameters.")
        return None

    print(f"\nSuccessfully downloaded {len(df)} records")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Display basic statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Latest Close Price: ${df['Close'].iloc[-1]:.2f}")
    print(f"Highest Price: ${df['High'].max():.2f}")
    print(f"Lowest Price: ${df['Low'].min():.2f}")
    print(f"Average Volume: {df['Volume'].mean():,.0f}")

    # Save to CSV if requested
    if save_to_csv:
        filename = f"{ticker.lower()}_stock_{period}_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename)
        print(f"\nData saved to: {filename}")

    return df

def calculate_ichimoku(df):
    """
    Calculate Ichimoku Cloud components

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC data

    Returns:
    --------
    pandas.DataFrame : DataFrame with Ichimoku indicators added
    """

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df['High'].rolling(window=9).max()
    period9_low = df['Low'].rolling(window=9).min()
    df['tenkan_sen'] = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df['High'].rolling(window=26).max()
    period26_low = df['Low'].rolling(window=26).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = df['High'].rolling(window=52).max()
    period52_low = df['Low'].rolling(window=52).min()
    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

    # Chikou Span (Lagging Span): Close plotted 26 periods in the past
    df['chikou_span'] = df['Close'].shift(-26)

    return df

def plot_ichimoku(df, ticker, stock_name, filename=None):
    """
    Create Ichimoku Cloud chart and save as PNG

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data and Ichimoku indicators
    ticker : str
        Stock ticker symbol
    stock_name : str
        Full name of the stock
    filename : str
        Output filename for the PNG (default: auto-generated)
    """

    if filename is None:
        filename = f"{ticker.lower()}_ichimoku_{datetime.now().strftime('%Y%m%d')}.png"

    print(f"\nCreating Ichimoku Cloud chart...")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot candlesticks
    # Calculate bar width based on the date range
    total_days = (df.index[-1] - df.index[0]).days
    width = total_days / len(df) * 0.6  # 60% of the spacing between bars

    for idx in range(len(df)):
        date = df.index[idx]
        open_price = df['Open'].iloc[idx]
        high_price = df['High'].iloc[idx]
        low_price = df['Low'].iloc[idx]
        close_price = df['Close'].iloc[idx]

        # Determine color: green if close > open, red otherwise
        if close_price >= open_price:
            color = 'green'
            body_bottom = open_price
            body_height = close_price - open_price
        else:
            color = 'red'
            body_bottom = close_price
            body_height = open_price - close_price

        # Draw the high-low line (wick)
        ax.plot([date, date], [low_price, high_price], color=color, linewidth=0.8, zorder=1)

        # Draw the body (rectangle)
        if body_height > 0:
            rect = Rectangle((mdates.date2num(date) - width/2, body_bottom),
                           width, body_height,
                           facecolor=color, edgecolor=color,
                           linewidth=0.5, zorder=2)
            ax.add_patch(rect)
        else:
            # Doji candle (open == close)
            ax.plot([mdates.date2num(date) - width/2, mdates.date2num(date) + width/2],
                   [close_price, close_price], color=color, linewidth=1, zorder=2)

    # Plot Tenkan-sen (Conversion Line) - dark red
    ax.plot(df.index, df['tenkan_sen'], label='Tenkan-sen (Conversion)',
            color='darkred', linewidth=1, alpha=0.8, zorder=3)

    # Plot Kijun-sen (Base Line) - blue
    ax.plot(df.index, df['kijun_sen'], label='Kijun-sen (Base)',
            color='blue', linewidth=1, alpha=0.8, zorder=3)

    # Plot Chikou Span (Lagging Span) - light green
    ax.plot(df.index, df['chikou_span'], label='Chikou Span (Lagging)',
            color='lightgreen', linewidth=1, alpha=0.8, zorder=3)

    # Plot Senkou Span A - orange
    ax.plot(df.index, df['senkou_span_a'], label='Senkou Span A',
            color='orange', linewidth=1, alpha=0.5, zorder=3)

    # Plot Senkou Span B - purple
    ax.plot(df.index, df['senkou_span_b'], label='Senkou Span B',
            color='purple', linewidth=1, alpha=0.5, zorder=3)

    # Fill the cloud (Kumo) - behind candlesticks
    ax.fill_between(df.index, df['senkou_span_a'], df['senkou_span_b'],
                     where=df['senkou_span_a'] >= df['senkou_span_b'],
                     facecolor='palegreen', alpha=0.25, interpolate=True,
                     label='Bullish Cloud', zorder=0)

    ax.fill_between(df.index, df['senkou_span_a'], df['senkou_span_b'],
                     where=df['senkou_span_a'] < df['senkou_span_b'],
                     facecolor='lightcoral', alpha=0.25, interpolate=True,
                     label='Bearish Cloud', zorder=0)

    # Formatting
    ax.set_title(f'{stock_name} ({ticker}) - Ichimoku Cloud Chart', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, ha='right')

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Ichimoku chart saved to: {filename}")

    # Close the plot to free memory
    plt.close()

def load_stocks_config(config_file="stocks_config.json"):
    """
    Load stocks configuration from JSON file

    Parameters:
    -----------
    config_file : str
        Path to the JSON configuration file

    Returns:
    --------
    dict : Configuration dictionary
    """
    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} not found!")
        return None

    with open(config_file, 'r') as f:
        config = json.load(f)

    return config

def analyze_ichimoku_signals(df, ticker, stock_name):
    """
    Analyze Ichimoku Cloud signals for trading decisions

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with Ichimoku indicators
    ticker : str
        Stock ticker symbol
    stock_name : str
        Full name of the stock

    Returns:
    --------
    dict : Analysis results
    """
    # Get the most recent data point (excluding NaN values for chikou span)
    latest = df.iloc[-1]
    latest_valid_chikou_idx = -27  # chikou is shifted -26, so we need -27 for comparison

    # Ensure we have valid data
    if latest_valid_chikou_idx < -len(df):
        latest_valid_chikou_idx = -len(df)

    analysis = {
        'ticker': ticker,
        'name': stock_name,
        'date': latest.name.strftime('%Y-%m-%d'),
        'close_price': latest['Close'],
        'signals': [],
        'trend': 'NEUTRAL',
        'strength': 'WEAK',
        'recommendation': 'HOLD'
    }

    # Signal 1: Price vs Cloud
    if pd.notna(latest['senkou_span_a']) and pd.notna(latest['senkou_span_b']):
        cloud_top = max(latest['senkou_span_a'], latest['senkou_span_b'])
        cloud_bottom = min(latest['senkou_span_a'], latest['senkou_span_b'])

        if latest['Close'] > cloud_top:
            analysis['signals'].append("Price above cloud (BULLISH)")
            analysis['trend'] = 'BULLISH'
        elif latest['Close'] < cloud_bottom:
            analysis['signals'].append("Price below cloud (BEARISH)")
            analysis['trend'] = 'BEARISH'
        else:
            analysis['signals'].append("Price inside cloud (NEUTRAL - consolidation)")

    # Signal 2: Tenkan-sen vs Kijun-sen (TK Cross)
    if pd.notna(latest['tenkan_sen']) and pd.notna(latest['kijun_sen']):
        if latest['tenkan_sen'] > latest['kijun_sen']:
            analysis['signals'].append("Tenkan-sen above Kijun-sen (BULLISH)")
            if analysis['trend'] == 'BULLISH':
                analysis['strength'] = 'MODERATE'
        elif latest['tenkan_sen'] < latest['kijun_sen']:
            analysis['signals'].append("Tenkan-sen below Kijun-sen (BEARISH)")
            if analysis['trend'] == 'BEARISH':
                analysis['strength'] = 'MODERATE'

    # Signal 3: Cloud Color (Future Cloud)
    if pd.notna(latest['senkou_span_a']) and pd.notna(latest['senkou_span_b']):
        if latest['senkou_span_a'] > latest['senkou_span_b']:
            analysis['signals'].append("Cloud is green/bullish (future support)")
        else:
            analysis['signals'].append("Cloud is red/bearish (future resistance)")

    # Signal 4: Chikou Span vs Price
    if len(df) >= 27:
        chikou_comparison_price = df.iloc[latest_valid_chikou_idx]['Close']
        chikou_span_value = df.iloc[latest_valid_chikou_idx]['chikou_span']

        if pd.notna(chikou_span_value):
            if chikou_span_value > chikou_comparison_price:
                analysis['signals'].append("Chikou Span above price (BULLISH)")
                if analysis['trend'] == 'BULLISH' and analysis['strength'] == 'MODERATE':
                    analysis['strength'] = 'STRONG'
            elif chikou_span_value < chikou_comparison_price:
                analysis['signals'].append("Chikou Span below price (BEARISH)")
                if analysis['trend'] == 'BEARISH' and analysis['strength'] == 'MODERATE':
                    analysis['strength'] = 'STRONG'

    # Signal 5: Price vs Kijun-sen
    if pd.notna(latest['kijun_sen']):
        if latest['Close'] > latest['kijun_sen']:
            analysis['signals'].append("Price above Kijun-sen (support)")
        else:
            analysis['signals'].append("Price below Kijun-sen (resistance)")

    # Generate recommendation
    if analysis['trend'] == 'BULLISH' and analysis['strength'] in ['MODERATE', 'STRONG']:
        analysis['recommendation'] = 'BUY'
    elif analysis['trend'] == 'BEARISH' and analysis['strength'] in ['MODERATE', 'STRONG']:
        analysis['recommendation'] = 'SELL'
    else:
        analysis['recommendation'] = 'HOLD'

    return analysis

def generate_report(analyses, filename="ichimoku_trading_report.txt"):
    """
    Generate a trading report based on Ichimoku analysis

    Parameters:
    -----------
    analyses : list
        List of analysis dictionaries
    filename : str
        Output filename for the report
    """
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("ICHIMOKU CLOUD TRADING ANALYSIS REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*80)
    report_lines.append("")

    report_lines.append("ICHIMOKU TRADING RULES SUMMARY:")
    report_lines.append("-" * 80)
    report_lines.append("1. BULLISH SIGNALS:")
    report_lines.append("   - Price above the cloud")
    report_lines.append("   - Tenkan-sen crosses above Kijun-sen")
    report_lines.append("   - Chikou Span above price from 26 periods ago")
    report_lines.append("   - Cloud is green (Senkou Span A > Senkou Span B)")
    report_lines.append("")
    report_lines.append("2. BEARISH SIGNALS:")
    report_lines.append("   - Price below the cloud")
    report_lines.append("   - Tenkan-sen crosses below Kijun-sen")
    report_lines.append("   - Chikou Span below price from 26 periods ago")
    report_lines.append("   - Cloud is red (Senkou Span A < Senkou Span B)")
    report_lines.append("")
    report_lines.append("3. NEUTRAL/CONSOLIDATION:")
    report_lines.append("   - Price inside the cloud")
    report_lines.append("   - Mixed signals from different indicators")
    report_lines.append("="*80)
    report_lines.append("")

    # Summary table
    report_lines.append("SUMMARY TABLE:")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Ticker':<10} {'Trend':<12} {'Strength':<12} {'Recommendation':<15} {'Price':<12}")
    report_lines.append("-" * 80)

    for analysis in analyses:
        report_lines.append(
            f"{analysis['ticker']:<10} "
            f"{analysis['trend']:<12} "
            f"{analysis['strength']:<12} "
            f"{analysis['recommendation']:<15} "
            f"${analysis['close_price']:<11.2f}"
        )

    report_lines.append("="*80)
    report_lines.append("")

    # Detailed analysis for each stock
    report_lines.append("DETAILED ANALYSIS:")
    report_lines.append("="*80)

    for analysis in analyses:
        report_lines.append("")
        report_lines.append(f"Stock: {analysis['name']} ({analysis['ticker']})")
        report_lines.append(f"Date: {analysis['date']}")
        report_lines.append(f"Close Price: ${analysis['close_price']:.2f}")
        report_lines.append(f"Trend: {analysis['trend']}")
        report_lines.append(f"Signal Strength: {analysis['strength']}")
        report_lines.append(f"Trading Recommendation: {analysis['recommendation']}")
        report_lines.append("")
        report_lines.append("Ichimoku Signals:")
        for i, signal in enumerate(analysis['signals'], 1):
            report_lines.append(f"  {i}. {signal}")
        report_lines.append("-" * 80)

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("DISCLAIMER:")
    report_lines.append("This report is for educational purposes only and should not be considered")
    report_lines.append("as financial advice. Always conduct your own research and consult with a")
    report_lines.append("qualified financial advisor before making investment decisions.")
    report_lines.append("="*80)

    # Write to file
    with open(filename, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nTrading report saved to: {filename}")

    # Also print summary to console
    print("\n" + "="*80)
    print("TRADING RECOMMENDATIONS SUMMARY:")
    print("-" * 80)
    print(f"{'Ticker':<10} {'Trend':<12} {'Strength':<12} {'Recommendation':<15} {'Price':<12}")
    print("-" * 80)
    for analysis in analyses:
        print(
            f"{analysis['ticker']:<10} "
            f"{analysis['trend']:<12} "
            f"{analysis['strength']:<12} "
            f"{analysis['recommendation']:<15} "
            f"${analysis['close_price']:<11.2f}"
        )
    print("="*80)

def process_stock(stock_info, period, interval, save_csv, save_chart):
    """
    Process a single stock: download data and create chart

    Parameters:
    -----------
    stock_info : dict
        Dictionary containing ticker, name, and exchange
    period : str
        Time period for data download
    interval : str
        Data interval
    save_csv : bool
        Whether to save CSV file
    save_chart : bool
        Whether to save chart

    Returns:
    --------
    dict : Analysis results or None if failed
    """
    ticker = stock_info['ticker']
    name = stock_info['name']

    print("\n" + "="*70)
    print(f"Processing: {name} ({ticker})")
    print("="*70)

    # Download stock data
    df = download_stock(ticker, name, period=period, interval=interval, save_to_csv=save_csv)

    if df is not None:
        print("\n" + "="*50)
        print("FIRST 5 ROWS")
        print("="*50)
        print(df.head())

        print("\n" + "="*50)
        print("LAST 5 ROWS")
        print("="*50)
        print(df.tail())

        # Calculate Ichimoku indicators
        df = calculate_ichimoku(df)

        # Create and save Ichimoku chart
        if save_chart:
            plot_ichimoku(df, ticker, name)

        # Analyze Ichimoku signals
        analysis = analyze_ichimoku_signals(df, ticker, name)

        print(f"\n✓ Successfully processed {ticker}")
        return analysis
    else:
        print(f"\n✗ Failed to process {ticker}")
        return None

def main():
    """Main function to run the script"""

    # Load configuration
    config = load_stocks_config()

    if config is None:
        print("Error: Could not load configuration. Exiting.")
        return

    # Extract settings
    stocks = config.get('stocks', [])
    settings = config.get('settings', {})
    period = settings.get('period', '1y')
    interval = settings.get('interval', '1d')
    save_csv = settings.get('save_csv', True)
    save_chart = settings.get('save_chart', True)

    print("="*70)
    print(f"STOCK DATA DOWNLOAD AND CHART GENERATION")
    print("="*70)
    print(f"Total stocks to process: {len(stocks)}")
    print(f"Period: {period}, Interval: {interval}")
    print(f"Save CSV: {save_csv}, Save Chart: {save_chart}")
    print("="*70)

    # Process each stock and collect analyses
    analyses = []
    for stock_info in stocks:
        try:
            analysis = process_stock(stock_info, period, interval, save_csv, save_chart)
            if analysis is not None:
                analyses.append(analysis)
        except Exception as e:
            print(f"\n✗ Error processing {stock_info['ticker']}: {str(e)}")
            continue

    print("\n" + "="*70)
    print("ALL STOCKS PROCESSED!")
    print("="*70)

    # Generate trading report
    if analyses:
        generate_report(analyses)

if __name__ == "__main__":
    main()
