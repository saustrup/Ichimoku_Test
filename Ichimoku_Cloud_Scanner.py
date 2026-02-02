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
import shutil

# PDF generation imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.platypus.flowables import AnchorFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

def download_stock(ticker, stock_name, period="1y", interval="1d", save_to_csv=True, output_folder=None):
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
    output_folder : str
        Folder to save output files (default: current directory)

    Returns:
    --------
    pandas.DataFrame : Stock price data
    """

    # Create ticker object
    stock = yf.Ticker(ticker)

    # Download historical data
    df = stock.history(period=period, interval=interval)

    if df.empty:
        print(f"    No data retrieved for {ticker}. Skipping.")
        return None

    # Save to CSV if requested
    if save_to_csv:
        filename = f"{ticker.lower()}_stock_{period}_{datetime.now().strftime('%Y%m%d')}.csv"
        if output_folder:
            filename = os.path.join(output_folder, filename)
        df.to_csv(filename)

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

def plot_ichimoku(df, ticker, stock_name, filename=None, output_folder=None):
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
    output_folder : str
        Folder to save output files (default: current directory)
    """

    if filename is None:
        filename = f"{ticker.lower()}_ichimoku_{datetime.now().strftime('%Y%m%d')}.png"
    if output_folder:
        filename = os.path.join(output_folder, filename)

    # Create Ichimoku Cloud chart

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

def analyze_single_day(df, day_idx, chikou_offset):
    """
    Analyze Ichimoku signals for a single day

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with Ichimoku indicators
    day_idx : int
        Index of the day to analyze (negative for from end)
    chikou_offset : int
        Offset for chikou span comparison

    Returns:
    --------
    dict : Component contributions for that day
    """
    if abs(day_idx) > len(df):
        return None

    day_data = df.iloc[day_idx]
    chikou_idx = day_idx - 26  # chikou comparison point

    if abs(chikou_idx) > len(df):
        chikou_idx = -len(df)

    components = {}

    # Component 1: Price vs Cloud (Kumo)
    if pd.notna(day_data['senkou_span_a']) and pd.notna(day_data['senkou_span_b']):
        cloud_top = max(day_data['senkou_span_a'], day_data['senkou_span_b'])
        cloud_bottom = min(day_data['senkou_span_a'], day_data['senkou_span_b'])

        if day_data['Close'] > cloud_top:
            components['kumo'] = 2
        elif day_data['Close'] < cloud_bottom:
            components['kumo'] = -2
        else:
            components['kumo'] = 0
    else:
        components['kumo'] = 0

    # Component 2: Tenkan-sen vs Kijun-sen
    if pd.notna(day_data['tenkan_sen']) and pd.notna(day_data['kijun_sen']):
        if day_data['tenkan_sen'] > day_data['kijun_sen']:
            components['tk_cross'] = 1
        elif day_data['tenkan_sen'] < day_data['kijun_sen']:
            components['tk_cross'] = -1
        else:
            components['tk_cross'] = 0
    else:
        components['tk_cross'] = 0

    # Component 3: Cloud Color
    if pd.notna(day_data['senkou_span_a']) and pd.notna(day_data['senkou_span_b']):
        if day_data['senkou_span_a'] > day_data['senkou_span_b']:
            components['cloud_color'] = 1
        else:
            components['cloud_color'] = -1
    else:
        components['cloud_color'] = 0

    # Component 4: Chikou Span
    if len(df) >= 27 and abs(chikou_idx) <= len(df):
        chikou_comparison_price = df.iloc[chikou_idx]['Close']
        chikou_span_value = df.iloc[chikou_idx]['chikou_span']
        chikou_senkou_a = df.iloc[chikou_idx]['senkou_span_a']
        chikou_senkou_b = df.iloc[chikou_idx]['senkou_span_b']

        if pd.notna(chikou_span_value):
            if pd.notna(chikou_senkou_a) and pd.notna(chikou_senkou_b):
                chikou_cloud_top = max(chikou_senkou_a, chikou_senkou_b)
                chikou_cloud_bottom = min(chikou_senkou_a, chikou_senkou_b)
            else:
                chikou_cloud_top = None
                chikou_cloud_bottom = None

            if chikou_span_value > chikou_comparison_price:
                if chikou_cloud_top is not None and chikou_span_value > chikou_cloud_top:
                    components['chikou'] = 2
                elif chikou_cloud_bottom is not None and chikou_span_value < chikou_cloud_bottom:
                    components['chikou'] = 0.5
                else:
                    components['chikou'] = 1
            elif chikou_span_value < chikou_comparison_price:
                if chikou_cloud_bottom is not None and chikou_span_value < chikou_cloud_bottom:
                    components['chikou'] = -2
                elif chikou_cloud_top is not None and chikou_span_value > chikou_cloud_top:
                    components['chikou'] = -0.5
                else:
                    components['chikou'] = -1
            else:
                components['chikou'] = 0
        else:
            components['chikou'] = 0
    else:
        components['chikou'] = 0

    # Component 5: Price vs Kijun-sen
    if pd.notna(day_data['kijun_sen']):
        if day_data['Close'] > day_data['kijun_sen']:
            components['kijun'] = 1
        else:
            components['kijun'] = -1
    else:
        components['kijun'] = 0

    return components


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

    # Analyze previous day for comparison
    prev_day_components = None
    if len(df) >= 2:
        prev_day_components = analyze_single_day(df, -2, -28)

    analysis = {
        'ticker': ticker,
        'name': stock_name,
        'date': latest.name.strftime('%Y-%m-%d'),
        'close_price': latest['Close'],
        'signals': [],
        'components': {},  # Store each Ichimoku component's contribution
        'prev_components': prev_day_components,  # Store previous day for comparison
        'changes': {},  # Track which components changed
        'trend': 'NEUTRAL',
        'strength': 'WEAK',
        'recommendation': 'HOLD'
    }

    # Component 1: Price vs Cloud (Kumo)
    component_kumo = {'name': 'Price vs Cloud (Kumo)', 'signal': 'NEUTRAL', 'description': '', 'contribution': 0}
    if pd.notna(latest['senkou_span_a']) and pd.notna(latest['senkou_span_b']):
        cloud_top = max(latest['senkou_span_a'], latest['senkou_span_b'])
        cloud_bottom = min(latest['senkou_span_a'], latest['senkou_span_b'])

        if latest['Close'] > cloud_top:
            analysis['signals'].append("Price above cloud (BULLISH)")
            analysis['trend'] = 'BULLISH'
            component_kumo['signal'] = 'BULLISH'
            component_kumo['description'] = f"Price ${latest['Close']:.2f} is above cloud top ${cloud_top:.2f}"
            component_kumo['contribution'] = 2
        elif latest['Close'] < cloud_bottom:
            analysis['signals'].append("Price below cloud (BEARISH)")
            analysis['trend'] = 'BEARISH'
            component_kumo['signal'] = 'BEARISH'
            component_kumo['description'] = f"Price ${latest['Close']:.2f} is below cloud bottom ${cloud_bottom:.2f}"
            component_kumo['contribution'] = -2
        else:
            analysis['signals'].append("Price inside cloud (NEUTRAL - consolidation)")
            component_kumo['signal'] = 'NEUTRAL'
            component_kumo['description'] = f"Price ${latest['Close']:.2f} is inside cloud (${cloud_bottom:.2f} - ${cloud_top:.2f})"
            component_kumo['contribution'] = 0
    analysis['components']['kumo'] = component_kumo

    # Component 2: Tenkan-sen vs Kijun-sen (TK Cross)
    component_tk = {'name': 'Tenkan-sen vs Kijun-sen', 'signal': 'NEUTRAL', 'description': '', 'contribution': 0}
    if pd.notna(latest['tenkan_sen']) and pd.notna(latest['kijun_sen']):
        if latest['tenkan_sen'] > latest['kijun_sen']:
            analysis['signals'].append("Tenkan-sen above Kijun-sen (BULLISH)")
            if analysis['trend'] == 'BULLISH':
                analysis['strength'] = 'MODERATE'
            component_tk['signal'] = 'BULLISH'
            component_tk['description'] = f"Tenkan ${latest['tenkan_sen']:.2f} > Kijun ${latest['kijun_sen']:.2f}"
            component_tk['contribution'] = 1
        elif latest['tenkan_sen'] < latest['kijun_sen']:
            analysis['signals'].append("Tenkan-sen below Kijun-sen (BEARISH)")
            if analysis['trend'] == 'BEARISH':
                analysis['strength'] = 'MODERATE'
            component_tk['signal'] = 'BEARISH'
            component_tk['description'] = f"Tenkan ${latest['tenkan_sen']:.2f} < Kijun ${latest['kijun_sen']:.2f}"
            component_tk['contribution'] = -1
        else:
            component_tk['signal'] = 'NEUTRAL'
            component_tk['description'] = f"Tenkan ${latest['tenkan_sen']:.2f} = Kijun ${latest['kijun_sen']:.2f}"
            component_tk['contribution'] = 0
    analysis['components']['tk_cross'] = component_tk

    # Component 3: Cloud Color (Future Cloud / Senkou Spans)
    component_cloud = {'name': 'Future Cloud Color', 'signal': 'NEUTRAL', 'description': '', 'contribution': 0}
    if pd.notna(latest['senkou_span_a']) and pd.notna(latest['senkou_span_b']):
        if latest['senkou_span_a'] > latest['senkou_span_b']:
            analysis['signals'].append("Cloud is green/bullish (future support)")
            component_cloud['signal'] = 'BULLISH'
            component_cloud['description'] = f"Senkou A ${latest['senkou_span_a']:.2f} > Senkou B ${latest['senkou_span_b']:.2f} (green cloud)"
            component_cloud['contribution'] = 1
        else:
            analysis['signals'].append("Cloud is red/bearish (future resistance)")
            component_cloud['signal'] = 'BEARISH'
            component_cloud['description'] = f"Senkou A ${latest['senkou_span_a']:.2f} < Senkou B ${latest['senkou_span_b']:.2f} (red cloud)"
            component_cloud['contribution'] = -1
    analysis['components']['cloud_color'] = component_cloud

    # Component 4: Chikou Span vs Price and Cloud
    component_chikou = {'name': 'Chikou Span (Lagging)', 'signal': 'NEUTRAL', 'description': '', 'contribution': 0}
    if len(df) >= 27:
        chikou_comparison_price = df.iloc[latest_valid_chikou_idx]['Close']
        chikou_span_value = df.iloc[latest_valid_chikou_idx]['chikou_span']
        # Get the cloud values at the chikou comparison point
        chikou_senkou_a = df.iloc[latest_valid_chikou_idx]['senkou_span_a']
        chikou_senkou_b = df.iloc[latest_valid_chikou_idx]['senkou_span_b']

        if pd.notna(chikou_span_value):
            # Determine cloud boundaries at chikou position
            if pd.notna(chikou_senkou_a) and pd.notna(chikou_senkou_b):
                chikou_cloud_top = max(chikou_senkou_a, chikou_senkou_b)
                chikou_cloud_bottom = min(chikou_senkou_a, chikou_senkou_b)
            else:
                chikou_cloud_top = None
                chikou_cloud_bottom = None

            if chikou_span_value > chikou_comparison_price:
                # Chikou above price - bullish, but check cloud position for strength
                if chikou_cloud_top is not None and chikou_span_value > chikou_cloud_top:
                    analysis['signals'].append("Chikou Span above price and above cloud (STRONG BULLISH)")
                    if analysis['trend'] == 'BULLISH' and analysis['strength'] == 'MODERATE':
                        analysis['strength'] = 'STRONG'
                    component_chikou['signal'] = 'STRONG BULLISH'
                    component_chikou['description'] = f"Chikou ${chikou_span_value:.2f} above price ${chikou_comparison_price:.2f} and above cloud"
                    component_chikou['contribution'] = 2
                elif chikou_cloud_bottom is not None and chikou_span_value < chikou_cloud_bottom:
                    analysis['signals'].append("Chikou Span above price but below cloud (WEAK BULLISH)")
                    component_chikou['signal'] = 'WEAK BULLISH'
                    component_chikou['description'] = f"Chikou ${chikou_span_value:.2f} above price ${chikou_comparison_price:.2f} but below cloud"
                    component_chikou['contribution'] = 0.5
                else:
                    analysis['signals'].append("Chikou Span above price but inside cloud (GOOD BULLISH)")
                    if analysis['trend'] == 'BULLISH' and analysis['strength'] == 'WEAK':
                        analysis['strength'] = 'MODERATE'
                    component_chikou['signal'] = 'GOOD BULLISH'
                    component_chikou['description'] = f"Chikou ${chikou_span_value:.2f} above price ${chikou_comparison_price:.2f} but inside cloud"
                    component_chikou['contribution'] = 1
            elif chikou_span_value < chikou_comparison_price:
                # Chikou below price - bearish, but check cloud position for strength
                if chikou_cloud_bottom is not None and chikou_span_value < chikou_cloud_bottom:
                    analysis['signals'].append("Chikou Span below price and below cloud (STRONG BEARISH)")
                    if analysis['trend'] == 'BEARISH' and analysis['strength'] == 'MODERATE':
                        analysis['strength'] = 'STRONG'
                    component_chikou['signal'] = 'STRONG BEARISH'
                    component_chikou['description'] = f"Chikou ${chikou_span_value:.2f} below price ${chikou_comparison_price:.2f} and below cloud"
                    component_chikou['contribution'] = -2
                elif chikou_cloud_top is not None and chikou_span_value > chikou_cloud_top:
                    analysis['signals'].append("Chikou Span below price but above cloud (WEAK BEARISH)")
                    component_chikou['signal'] = 'WEAK BEARISH'
                    component_chikou['description'] = f"Chikou ${chikou_span_value:.2f} below price ${chikou_comparison_price:.2f} but above cloud"
                    component_chikou['contribution'] = -0.5
                else:
                    analysis['signals'].append("Chikou Span below price but inside cloud (GOOD BEARISH)")
                    if analysis['trend'] == 'BEARISH' and analysis['strength'] == 'WEAK':
                        analysis['strength'] = 'MODERATE'
                    component_chikou['signal'] = 'GOOD BEARISH'
                    component_chikou['description'] = f"Chikou ${chikou_span_value:.2f} below price ${chikou_comparison_price:.2f} but inside cloud"
                    component_chikou['contribution'] = -1
    analysis['components']['chikou'] = component_chikou

    # Component 5: Price vs Kijun-sen
    component_kijun = {'name': 'Price vs Kijun-sen', 'signal': 'NEUTRAL', 'description': '', 'contribution': 0}
    if pd.notna(latest['kijun_sen']):
        if latest['Close'] > latest['kijun_sen']:
            analysis['signals'].append("Price above Kijun-sen (support)")
            component_kijun['signal'] = 'BULLISH'
            component_kijun['description'] = f"Price ${latest['Close']:.2f} above Kijun ${latest['kijun_sen']:.2f} (support level)"
            component_kijun['contribution'] = 1
        else:
            analysis['signals'].append("Price below Kijun-sen (resistance)")
            component_kijun['signal'] = 'BEARISH'
            component_kijun['description'] = f"Price ${latest['Close']:.2f} below Kijun ${latest['kijun_sen']:.2f} (resistance level)"
            component_kijun['contribution'] = -1
    analysis['components']['kijun'] = component_kijun

    # Calculate total score
    total_score = sum(c['contribution'] for c in analysis['components'].values())
    analysis['total_score'] = total_score

    # Compare with previous day and mark changes
    if prev_day_components:
        for comp_key in ['kumo', 'tk_cross', 'cloud_color', 'chikou', 'kijun']:
            current_val = analysis['components'].get(comp_key, {}).get('contribution', 0)
            prev_val = prev_day_components.get(comp_key, 0)
            if current_val != prev_val:
                analysis['changes'][comp_key] = {
                    'prev': prev_val,
                    'current': current_val,
                    'direction': 'up' if current_val > prev_val else 'down'
                }

    # Calculate previous total score for comparison
    if prev_day_components:
        prev_total = sum(prev_day_components.values())
        analysis['prev_total_score'] = prev_total
        if total_score != prev_total:
            analysis['changes']['total'] = {
                'prev': prev_total,
                'current': total_score,
                'direction': 'up' if total_score > prev_total else 'down'
            }

    # Generate recommendation (long-only perspective)
    if analysis['trend'] == 'BULLISH' and analysis['strength'] == 'STRONG':
        analysis['recommendation'] = 'BUY'
    elif analysis['trend'] == 'BULLISH' and analysis['strength'] == 'MODERATE':
        analysis['recommendation'] = 'BUY (MODERATE)'
    elif analysis['trend'] == 'BULLISH' and analysis['strength'] == 'WEAK':
        analysis['recommendation'] = 'WAIT'
    elif analysis['trend'] == 'NEUTRAL':
        analysis['recommendation'] = 'WAIT'
    elif analysis['trend'] == 'BEARISH':
        analysis['recommendation'] = 'AVOID'

    return analysis

def get_recommendation_priority(recommendation):
    """Return sort priority for recommendations (lower = higher priority)."""
    priority_map = {
        'BUY': 1,
        'BUY (MODERATE)': 2,
        'WAIT': 3,
        'AVOID': 4
    }
    return priority_map.get(recommendation, 999)  # Unknown recommendations go last

def generate_report(analyses, filename="ichimoku_trading_report.txt", output_folder=None, market_name=None, currency="USD"):
    """
    Generate a trading report based on Ichimoku analysis

    Parameters:
    -----------
    analyses : list
        List of analysis dictionaries
    filename : str
        Output filename for the report
    output_folder : str
        Folder to save output files (default: current directory)
    market_name : str
        Name of the market for the report header
    currency : str
        Currency symbol for prices
    """
    if output_folder:
        filename = os.path.join(output_folder, filename)
    report_lines = []
    report_lines.append("="*80)
    title = "ICHIMOKU CLOUD LONG-ONLY ANALYSIS REPORT"
    if market_name:
        title += f" - {market_name}"
    report_lines.append(title)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*80)
    report_lines.append("")

    report_lines.append("LONG-ONLY ICHIMOKU TRADING RULES:")
    report_lines.append("-" * 80)
    report_lines.append("BULLISH SIGNALS (Favorable for Long Entry):")
    report_lines.append("   - Price above the cloud")
    report_lines.append("   - Tenkan-sen above Kijun-sen")
    report_lines.append("   - Chikou Span above price and cloud from 26 periods ago")
    report_lines.append("   - Cloud is green (Senkou Span A > Senkou Span B)")
    report_lines.append("")
    report_lines.append("WARNING SIGNALS (Avoid Long Entry):")
    report_lines.append("   - Price below the cloud")
    report_lines.append("   - Tenkan-sen below Kijun-sen")
    report_lines.append("   - Chikou Span below price from 26 periods ago")
    report_lines.append("   - Cloud is red (Senkou Span A < Senkou Span B)")
    report_lines.append("")
    report_lines.append("RECOMMENDATIONS:")
    report_lines.append("   - BUY: Strong bullish signals - favorable for long entry")
    report_lines.append("   - BUY (MODERATE): Good bullish signals - consider long entry")
    report_lines.append("   - WAIT: Weak or neutral signals - wait for better setup")
    report_lines.append("   - AVOID: Bearish signals - not suitable for long entry")
    report_lines.append("="*80)
    report_lines.append("")

    # Summary table with component breakdown
    report_lines.append("SUMMARY TABLE (* = changed from previous day):")
    report_lines.append("-" * 130)
    report_lines.append(
        f"{'Ticker':<12} {'Kumo':>6} {'TK':>6} {'Cloud':>6} {'Chik':>6} {'Kij':>6} "
        f"{'Score':>7} {'Long Entry':<15} {'Price':<10}"
    )
    report_lines.append("-" * 130)

    for analysis in analyses:
        components = analysis.get('components', {})
        changes = analysis.get('changes', {})

        kumo = components.get('kumo', {}).get('contribution', 0)
        tk = components.get('tk_cross', {}).get('contribution', 0)
        cloud = components.get('cloud_color', {}).get('contribution', 0)
        chikou = components.get('chikou', {}).get('contribution', 0)
        kijun = components.get('kijun', {}).get('contribution', 0)
        total = analysis.get('total_score', 0)

        # Format values with asterisk if changed
        kumo_str = f"*{kumo:+.1f}" if 'kumo' in changes else f"{kumo:+.1f}"
        tk_str = f"*{tk:+.1f}" if 'tk_cross' in changes else f"{tk:+.1f}"
        cloud_str = f"*{cloud:+.1f}" if 'cloud_color' in changes else f"{cloud:+.1f}"
        chikou_str = f"*{chikou:+.1f}" if 'chikou' in changes else f"{chikou:+.1f}"
        kijun_str = f"*{kijun:+.1f}" if 'kijun' in changes else f"{kijun:+.1f}"
        total_str = f"*{total:+.1f}" if 'total' in changes else f"{total:+.1f}"

        report_lines.append(
            f"{analysis['ticker']:<12} "
            f"{kumo_str:>6} "
            f"{tk_str:>6} "
            f"{cloud_str:>6} "
            f"{chikou_str:>6} "
            f"{kijun_str:>6} "
            f"{total_str:>7} "
            f"{analysis['recommendation']:<15} "
            f"${analysis['close_price']:<9.2f}"
        )

    report_lines.append("-" * 130)
    report_lines.append("Legend: Kumo=Price vs Cloud | TK=Tenkan vs Kijun | Cloud=Future Cloud Color | Chik=Chikou Span | Kij=Price vs Kijun")
    report_lines.append("        * = value changed from previous day (ALERT)")
    report_lines.append("="*130)
    report_lines.append("")

    # Detailed analysis for each stock
    report_lines.append("DETAILED COMPONENT ANALYSIS:")
    report_lines.append("="*80)

    for analysis in analyses:
        changes = analysis.get('changes', {})
        report_lines.append("")
        report_lines.append(f"{'='*80}")
        report_lines.append(f"Stock: {analysis['name']} ({analysis['ticker']})")
        report_lines.append(f"{'='*80}")
        report_lines.append(f"Date: {analysis['date']}")
        report_lines.append(f"Close Price: ${analysis['close_price']:.2f}")
        report_lines.append("")

        # Show if there are any changes
        if changes:
            report_lines.append("*** ALERT: Signal changes detected from previous day ***")

        total_str = f"{analysis.get('total_score', 0):+.1f}"
        if 'total' in changes:
            prev_total = changes['total']['prev']
            total_str += f" (was {prev_total:+.1f})"

        report_lines.append(f"OVERALL: {analysis['trend']} | Strength: {analysis['strength']} | Score: {total_str}")
        report_lines.append(f"LONG ENTRY RECOMMENDATION: {analysis['recommendation']}")
        report_lines.append("")
        report_lines.append("COMPONENT BREAKDOWN:")
        report_lines.append("-" * 80)

        # Display each component
        component_order = ['kumo', 'tk_cross', 'cloud_color', 'chikou', 'kijun']
        for comp_key in component_order:
            if comp_key in analysis.get('components', {}):
                comp = analysis['components'][comp_key]
                # Determine indicator symbol based on contribution
                if comp['contribution'] > 0:
                    indicator = "+"
                    for_long = "Positive"
                elif comp['contribution'] < 0:
                    indicator = "-"
                    for_long = "Warning"
                else:
                    indicator = "~"
                    for_long = "Neutral"

                # Check if this component changed
                changed_marker = ""
                change_info = ""
                if comp_key in changes:
                    changed_marker = " *CHANGED*"
                    prev_val = changes[comp_key]['prev']
                    direction = changes[comp_key]['direction']
                    arrow = "↑" if direction == 'up' else "↓"
                    change_info = f" (was {prev_val:+.1f} {arrow})"

                report_lines.append(f"  {comp['name']}{changed_marker}")
                report_lines.append(f"    Signal: {comp['signal']} ({for_long} for Long)")
                report_lines.append(f"    {comp['description']}")
                report_lines.append(f"    Contribution: {indicator}{abs(comp['contribution']):.1f}{change_info}")
                report_lines.append("")

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



def generate_pdf_report(analyses, charts_folder, output_folder, filename="ichimoku_report.pdf", market_name=None, currency="USD"):
    """
    Generate a PDF report with analysis and charts

    Parameters:
    -----------
    analyses : list
        List of analysis dictionaries
    charts_folder : str
        Folder containing chart PNG files
    output_folder : str
        Folder to save the PDF report
    filename : str
        Output PDF filename
    market_name : str
        Name of the market for the report header
    currency : str
        Currency symbol for prices
    """
    pdf_path = os.path.join(output_folder, filename)

    # Create document with landscape orientation for better chart display
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=landscape(A4),
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=20
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=10
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=10,
        spaceBefore=15
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=5
    )
    alert_style = ParagraphStyle(
        'AlertStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.red,
        spaceAfter=5
    )

    # Build content
    content = []

    # Title page
    content.append(Spacer(1, 1*inch))
    title_text = "ICHIMOKU CLOUD ANALYSIS REPORT"
    if market_name:
        title_text += f"<br/>{market_name}"
    content.append(Paragraph(title_text, title_style))
    content.append(Paragraph("Long-Only Trading Evaluation", subtitle_style))
    content.append(Spacer(1, 0.3*inch))
    content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    content.append(Spacer(1, 0.5*inch))

    # Summary table
    content.append(AnchorFlowable('summary_table'))
    content.append(Paragraph("SUMMARY TABLE (* = changed from previous day)", heading_style))

    # Build table data
    table_data = [['Ticker', 'Kumo', 'TK', 'Cloud', 'Chik', 'Kij', 'Score', 'Recommendation', 'Price']]

    for analysis in analyses:
        components = analysis.get('components', {})
        changes = analysis.get('changes', {})

        kumo = components.get('kumo', {}).get('contribution', 0)
        tk = components.get('tk_cross', {}).get('contribution', 0)
        cloud = components.get('cloud_color', {}).get('contribution', 0)
        chikou = components.get('chikou', {}).get('contribution', 0)
        kijun = components.get('kijun', {}).get('contribution', 0)
        total = analysis.get('total_score', 0)

        # Format values with asterisk if changed
        kumo_str = f"{kumo:+.1f}*" if 'kumo' in changes else f"{kumo:+.1f}"
        tk_str = f"{tk:+.1f}*" if 'tk_cross' in changes else f"{tk:+.1f}"
        cloud_str = f"{cloud:+.1f}*" if 'cloud_color' in changes else f"{cloud:+.1f}"
        chikou_str = f"{chikou:+.1f}*" if 'chikou' in changes else f"{chikou:+.1f}"
        kijun_str = f"{kijun:+.1f}*" if 'kijun' in changes else f"{kijun:+.1f}"
        total_str = f"{total:+.1f}*" if 'total' in changes else f"{total:+.1f}"

        # Create clickable ticker link to detail section
        dest_name = f'stock_{analysis["ticker"]}'
        ticker_link = Paragraph(
            f'<a href="#{dest_name}" color="blue"><u>{analysis["ticker"]}</u></a>',
            normal_style
        )

        table_data.append([
            ticker_link,
            kumo_str,
            tk_str,
            cloud_str,
            chikou_str,
            kijun_str,
            total_str,
            analysis['recommendation'],
            f"${analysis['close_price']:.2f}"
        ])

    # Create table with styling
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))

    # Color code recommendations
    for i, analysis in enumerate(analyses, 1):
        rec = analysis['recommendation']
        if rec == 'BUY':
            table.setStyle(TableStyle([('BACKGROUND', (7, i), (7, i), colors.lightgreen)]))
        elif rec == 'BUY (MODERATE)':
            table.setStyle(TableStyle([('BACKGROUND', (7, i), (7, i), colors.palegreen)]))
        elif rec == 'AVOID':
            table.setStyle(TableStyle([('BACKGROUND', (7, i), (7, i), colors.lightcoral)]))
        elif rec == 'WAIT':
            table.setStyle(TableStyle([('BACKGROUND', (7, i), (7, i), colors.lightyellow)]))

        # Highlight rows with changes
        if analysis.get('changes'):
            table.setStyle(TableStyle([('BACKGROUND', (0, i), (0, i), colors.yellow)]))

    content.append(table)
    content.append(Spacer(1, 0.2*inch))

    # Legend
    legend_text = "Legend: Kumo=Price vs Cloud | TK=Tenkan vs Kijun | Cloud=Future Cloud Color | Chik=Chikou Span | Kij=Price vs Kijun"
    content.append(Paragraph(legend_text, normal_style))
    content.append(Paragraph("* = value changed from previous day (ALERT) | Yellow ticker = signals changed today", normal_style))

    content.append(PageBreak())

    # Individual stock pages with charts
    for analysis in analyses:
        ticker = analysis['ticker']
        changes = analysis.get('changes', {})

        # Add anchor for linking from summary table
        dest_name = f'stock_{ticker}'
        content.append(AnchorFlowable(dest_name))

        # Back to summary link
        content.append(Paragraph(
            '<a href="#summary_table" color="blue"><u>&lt;&lt; Back to Summary Table</u></a>',
            normal_style
        ))
        content.append(Spacer(1, 0.1*inch))

        # Stock header
        content.append(Paragraph(f"{analysis['name']} ({ticker})", heading_style))
        content.append(Paragraph(f"Date: {analysis['date']} | Close: ${analysis['close_price']:.2f}", normal_style))

        # Alert if changes detected
        if changes:
            content.append(Paragraph("⚠ ALERT: Signal changes detected from previous day!", alert_style))

        # Overall assessment
        total_str = f"{analysis.get('total_score', 0):+.1f}"
        if 'total' in changes:
            prev_total = changes['total']['prev']
            total_str += f" (was {prev_total:+.1f})"

        content.append(Paragraph(
            f"<b>Trend:</b> {analysis['trend']} | <b>Strength:</b> {analysis['strength']} | "
            f"<b>Score:</b> {total_str} | <b>Recommendation:</b> {analysis['recommendation']}",
            normal_style
        ))

        # Component breakdown table
        comp_data = [['Component', 'Signal', 'Contribution', 'Changed?']]
        component_names = {
            'kumo': 'Price vs Cloud (Kumo)',
            'tk_cross': 'Tenkan vs Kijun',
            'cloud_color': 'Future Cloud Color',
            'chikou': 'Chikou Span',
            'kijun': 'Price vs Kijun'
        }

        for comp_key in ['kumo', 'tk_cross', 'cloud_color', 'chikou', 'kijun']:
            comp = analysis['components'].get(comp_key, {})
            contrib = comp.get('contribution', 0)

            changed = 'YES' if comp_key in changes else ''
            if comp_key in changes:
                prev_val = changes[comp_key]['prev']
                changed = f"YES ({prev_val:+.1f} → {contrib:+.1f})"

            comp_data.append([
                component_names.get(comp_key, comp_key),
                comp.get('signal', 'N/A'),
                f"{contrib:+.1f}",
                changed
            ])

        comp_table = Table(comp_data, colWidths=[2.5*inch, 1.5*inch, 1*inch, 1.5*inch])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))

        # Highlight changed components
        for i, comp_key in enumerate(['kumo', 'tk_cross', 'cloud_color', 'chikou', 'kijun'], 1):
            if comp_key in changes:
                comp_table.setStyle(TableStyle([('BACKGROUND', (-1, i), (-1, i), colors.yellow)]))

        content.append(Spacer(1, 0.1*inch))
        content.append(comp_table)
        content.append(Spacer(1, 0.2*inch))

        # Add chart image
        chart_filename = f"{ticker.lower()}_ichimoku_{datetime.now().strftime('%Y%m%d')}.png"
        chart_path = os.path.join(charts_folder, chart_filename)

        if os.path.exists(chart_path):
            # Scale image to fit page width while maintaining aspect ratio
            img = Image(chart_path, width=9*inch, height=5*inch)
            content.append(img)
        else:
            content.append(Paragraph(f"Chart not found: {chart_filename}", normal_style))

        content.append(PageBreak())

    # Disclaimer page
    content.append(Spacer(1, 2*inch))
    content.append(Paragraph("DISCLAIMER", heading_style))
    content.append(Paragraph(
        "This report is for educational purposes only and should not be considered as financial advice. "
        "Always conduct your own research and consult with a qualified financial advisor before making "
        "investment decisions. Past performance is not indicative of future results.",
        normal_style
    ))

    # Build PDF
    doc.build(content)

    return pdf_path


def process_stock(stock_info, period, interval, save_csv, save_chart, charts_folder=None, data_folder=None, stock_index=0, stock_total=0):
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
    charts_folder : str
        Folder to save chart PNG files (default: current directory)
    data_folder : str
        Folder to save CSV data files (default: current directory)
    stock_index : int
        Current stock number (1-based) for progress display
    stock_total : int
        Total number of stocks in this market for progress display

    Returns:
    --------
    dict : Analysis results or None if failed
    """
    ticker = stock_info['ticker']
    name = stock_info['name']
    progress = f"[{stock_index}/{stock_total}]" if stock_total > 0 else ""

    print(f"  {progress} {name} ({ticker})...", end=" ", flush=True)

    # Download stock data
    df = download_stock(ticker, name, period=period, interval=interval, save_to_csv=save_csv, output_folder=data_folder)

    if df is not None:
        # Calculate Ichimoku indicators
        df = calculate_ichimoku(df)

        # Create and save Ichimoku chart
        if save_chart:
            plot_ichimoku(df, ticker, name, output_folder=charts_folder)

        # Analyze Ichimoku signals
        analysis = analyze_ichimoku_signals(df, ticker, name)

        print(f"OK ({analysis['recommendation']})")
        return analysis
    else:
        print("FAILED")
        return None

def archive_previous_output(base_output_folder, archive_folder):
    """
    Move previous output files to archive folder with timestamp

    Parameters:
    -----------
    base_output_folder : str
        The main output folder to archive
    archive_folder : str
        The archive folder to move files to
    """
    if not os.path.exists(base_output_folder):
        print("No previous output to archive.")
        return

    # Check if there are any files/folders to archive
    contents = os.listdir(base_output_folder)
    if not contents:
        print("Output folder is empty, nothing to archive.")
        return

    # Create archive folder if it doesn't exist
    os.makedirs(archive_folder, exist_ok=True)

    # Create timestamped subfolder in archive
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_destination = os.path.join(archive_folder, f"run_{timestamp}")

    print(f"Archiving previous output to: {archive_destination}")

    # Move all contents from output to archive
    os.makedirs(archive_destination, exist_ok=True)

    for item in contents:
        source_path = os.path.join(base_output_folder, item)
        dest_path = os.path.join(archive_destination, item)
        shutil.move(source_path, dest_path)


def main():
    """Main function to run the script"""

    # Define folders
    base_output_folder = "Output"
    archive_folder = "Archive"

    # Archive previous output before starting new run
    archive_previous_output(base_output_folder, archive_folder)

    # Create base Output folder
    os.makedirs(base_output_folder, exist_ok=True)

    # Load configuration
    config = load_stocks_config()

    if config is None:
        print("Error: Could not load configuration. Exiting.")
        return

    # Extract settings
    settings = config.get('settings', {})
    period = settings.get('period', '1y')
    interval = settings.get('interval', '1d')
    save_csv = settings.get('save_csv', True)
    save_chart = settings.get('save_chart', True)

    # Get markets from config
    markets = config.get('markets', {})

    if not markets:
        print("Error: No markets found in configuration. Exiting.")
        return

    # Count total stocks across all markets
    total_stocks = sum(len(market_data.get('stocks', [])) for market_data in markets.values())

    print(f"Ichimoku Cloud Scanner — {total_stocks} stocks across {len(markets)} market(s) ({period}, {interval})")

    # Process each market
    for market_key, market_data in markets.items():
        market_name = market_data.get('name', market_key)
        currency = market_data.get('currency', 'USD')
        stocks = market_data.get('stocks', [])

        if not stocks:
            print(f"\nSkipping {market_name} - no stocks configured")
            continue

        print(f"\n--- {market_name} ({len(stocks)} stocks, {currency}) ---")

        # Create market-specific output folders
        market_output_folder = os.path.join(base_output_folder, market_key)
        charts_folder = os.path.join(market_output_folder, "charts")
        data_folder = os.path.join(market_output_folder, "data")
        os.makedirs(market_output_folder, exist_ok=True)
        os.makedirs(charts_folder, exist_ok=True)
        os.makedirs(data_folder, exist_ok=True)

        # Process each stock in this market and collect analyses
        analyses = []
        num_stocks = len(stocks)
        for i, stock_info in enumerate(stocks, start=1):
            try:
                analysis = process_stock(stock_info, period, interval, save_csv, save_chart,
                                         charts_folder=charts_folder, data_folder=data_folder,
                                         stock_index=i, stock_total=num_stocks)
                if analysis is not None:
                    analyses.append(analysis)
            except Exception as e:
                print(f"  [{i}/{num_stocks}] {stock_info['ticker']}... ERROR: {str(e)}")
                continue

        # Sort analyses by recommendation (descending priority: BUY first, then AVOID last)
        analyses.sort(key=lambda x: get_recommendation_priority(x['recommendation']))

        succeeded = len(analyses)
        failed = num_stocks - succeeded
        print(f"  Done: {succeeded} succeeded, {failed} failed")

        # Generate trading report for this market
        if analyses:
            print(f"  Generating reports for {market_name}...", end=" ", flush=True)
            report_filename = f"ichimoku_trading_report_{market_key}.txt"
            pdf_filename = f"ichimoku_report_{market_key}.pdf"
            generate_report(analyses, filename=report_filename, output_folder=market_output_folder,
                          market_name=market_name, currency=currency)
            # Generate PDF report with charts
            generate_pdf_report(analyses, charts_folder, market_output_folder,
                              filename=pdf_filename, market_name=market_name, currency=currency)
            print("OK")

    print(f"\nAll markets processed. Output in: {base_output_folder}/")

if __name__ == "__main__":
    main()
