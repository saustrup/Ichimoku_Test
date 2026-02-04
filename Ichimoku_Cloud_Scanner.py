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

    # --- Enhancement indicator columns ---

    # Volume analysis: 20-day SMA and ratio
    df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

    # Cloud thickness (absolute and as percentage of price)
    df['cloud_thickness'] = abs(df['senkou_span_a'] - df['senkou_span_b'])
    df['cloud_thickness_pct'] = (df['cloud_thickness'] / df['Close']) * 100

    # Price distance from Kijun-sen as percentage
    df['kijun_distance_pct'] = ((df['Close'] - df['kijun_sen']) / df['kijun_sen']) * 100

    # Flat line detection: 5-period rolling std dev
    df['kijun_flat'] = df['kijun_sen'].rolling(window=5).std()
    df['tenkan_flat'] = df['tenkan_sen'].rolling(window=5).std()

    # Unshifted (future) Senkou values for Kumo twist detection
    df['future_senkou_a'] = (df['tenkan_sen'] + df['kijun_sen']) / 2
    df['future_senkou_b'] = (df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2

    return df

def plot_ichimoku(df, ticker, stock_name, filename=None, output_folder=None, analysis=None):
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
    analysis : dict
        Analysis results with trade targets (optional)
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

    # Draw trade target lines (if analysis provided)
    if analysis and analysis.get('trade_targets'):
        targets = analysis['trade_targets']
        last_date = df.index[-1]
        date_range_start = df.index[int(len(df) * 0.8)]

        if targets.get('stop_loss_primary'):
            ax.hlines(y=targets['stop_loss_primary'], xmin=date_range_start, xmax=last_date,
                      colors='red', linestyles='dashed', linewidth=1.2, alpha=0.8, zorder=4)
            ax.annotate('SL', xy=(last_date, targets['stop_loss_primary']),
                        fontsize=8, color='red', fontweight='bold',
                        xytext=(5, 0), textcoords='offset points')

        if targets.get('take_profit_1'):
            ax.hlines(y=targets['take_profit_1'], xmin=date_range_start, xmax=last_date,
                      colors='green', linestyles='dashed', linewidth=1.2, alpha=0.8, zorder=4)
            ax.annotate('TP1', xy=(last_date, targets['take_profit_1']),
                        fontsize=8, color='green', fontweight='bold',
                        xytext=(5, 0), textcoords='offset points')

        if targets.get('take_profit_2'):
            ax.hlines(y=targets['take_profit_2'], xmin=date_range_start, xmax=last_date,
                      colors='darkgreen', linestyles='dotted', linewidth=1, alpha=0.6, zorder=4)
            ax.annotate('TP2', xy=(last_date, targets['take_profit_2']),
                        fontsize=8, color='darkgreen',
                        xytext=(5, 0), textcoords='offset points')

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

    # --- Enhancement signals ---

    # Enhancement 1: Volume confirmation
    if 'volume_ratio' in day_data.index and pd.notna(day_data.get('volume_ratio')):
        vol_ratio = day_data['volume_ratio']
        trend_score = components.get('kumo', 0) + components.get('tk_cross', 0)
        if vol_ratio >= 1.5:
            components['volume_confirm'] = 1 if trend_score > 0 else (-1 if trend_score < 0 else 0)
        elif vol_ratio <= 0.5:
            components['volume_confirm'] = -0.5
        else:
            components['volume_confirm'] = 0
    else:
        components['volume_confirm'] = 0

    # Enhancement 2: Kumo twist detection
    components['kumo_twist'] = 0
    if ('future_senkou_a' in day_data.index and 'future_senkou_b' in day_data.index
            and pd.notna(day_data.get('future_senkou_a')) and pd.notna(day_data.get('future_senkou_b'))):
        actual_idx = len(df) + day_idx if day_idx < 0 else day_idx
        if actual_idx > 0:
            prev_data = df.iloc[actual_idx - 1]
            if (pd.notna(prev_data.get('future_senkou_a')) and pd.notna(prev_data.get('future_senkou_b'))):
                prev_diff = prev_data['future_senkou_a'] - prev_data['future_senkou_b']
                curr_diff = day_data['future_senkou_a'] - day_data['future_senkou_b']
                # Only trigger if sign change AND meaningful difference (>0.1% of price)
                if prev_diff * curr_diff < 0 and abs(curr_diff) > day_data['Close'] * 0.001:
                    components['kumo_twist'] = 1 if curr_diff > 0 else -1

    # Enhancement 3: Cloud thickness
    if 'cloud_thickness_pct' in day_data.index and pd.notna(day_data.get('cloud_thickness_pct')):
        thickness = day_data['cloud_thickness_pct']
        kumo_direction = components.get('kumo', 0)
        if thickness < 1.0:
            components['cloud_thickness'] = 0
        elif thickness > 4.0:
            components['cloud_thickness'] = 1 if kumo_direction > 0 else (-1 if kumo_direction < 0 else 0)
        else:
            components['cloud_thickness'] = 0.5 if kumo_direction > 0 else (-0.5 if kumo_direction < 0 else 0)
    else:
        components['cloud_thickness'] = 0

    # Enhancement 4: TK cross location relative to cloud
    if (pd.notna(day_data.get('tenkan_sen')) and pd.notna(day_data.get('kijun_sen'))
            and pd.notna(day_data.get('senkou_span_a')) and pd.notna(day_data.get('senkou_span_b'))):
        tk_midpoint = (day_data['tenkan_sen'] + day_data['kijun_sen']) / 2
        cloud_top = max(day_data['senkou_span_a'], day_data['senkou_span_b'])
        cloud_bottom = min(day_data['senkou_span_a'], day_data['senkou_span_b'])
        tk_dir = components.get('tk_cross', 0)
        if tk_dir > 0 and tk_midpoint > cloud_top:
            components['tk_location'] = 1
        elif tk_dir < 0 and tk_midpoint < cloud_bottom:
            components['tk_location'] = -1
        else:
            components['tk_location'] = 0
    else:
        components['tk_location'] = 0

    # Enhancement 5: Kijun distance (overextension)
    if 'kijun_distance_pct' in day_data.index and pd.notna(day_data.get('kijun_distance_pct')):
        dist = day_data['kijun_distance_pct']
        if abs(dist) > 8.0:
            components['kijun_distance'] = -1
        elif abs(dist) > 5.0:
            components['kijun_distance'] = -0.5
        else:
            components['kijun_distance'] = 0
    else:
        components['kijun_distance'] = 0

    # Enhancement 6: Flat Kijun/Tenkan detection
    components['flat_lines'] = 0
    if 'kijun_flat' in day_data.index and pd.notna(day_data.get('kijun_flat')):
        threshold = day_data['Close'] * 0.001
        kijun_is_flat = day_data['kijun_flat'] < threshold
        tenkan_is_flat = ('tenkan_flat' in day_data.index
                          and pd.notna(day_data.get('tenkan_flat'))
                          and day_data['tenkan_flat'] < threshold)
        if kijun_is_flat and tenkan_is_flat:
            components['flat_lines'] = -0.5
        elif kijun_is_flat:
            components['flat_lines'] = -0.25

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
        'enhancements': {},  # Enhancement signal layer
        'prev_components': prev_day_components,  # Store previous day for comparison
        'changes': {},  # Track which components changed
        'trend': 'NEUTRAL',
        'strength': 'WEAK',
        'recommendation': 'HOLD',
        'confidence_score': 0,
        'confidence_label': '',
        'trade_targets': {},
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

    # --- Enhancement signals ---

    # Enhancement 1: Volume Confirmation
    enh_volume = {'name': 'Volume Confirmation', 'signal': 'NEUTRAL', 'description': '', 'contribution': 0}
    if 'volume_ratio' in latest.index and pd.notna(latest.get('volume_ratio')) and pd.notna(latest.get('volume_sma_20')):
        vol_ratio = latest['volume_ratio']
        vol_sma = latest['volume_sma_20']
        trend_direction = 1 if analysis['trend'] == 'BULLISH' else (-1 if analysis['trend'] == 'BEARISH' else 0)

        if vol_ratio >= 1.5:
            if trend_direction > 0:
                enh_volume['signal'] = 'CONFIRMS BULLISH'
                enh_volume['description'] = f"Volume {latest['Volume']:.0f} is {vol_ratio:.1f}x above 20-day avg {vol_sma:.0f} — confirms bullish move"
                enh_volume['contribution'] = 1
            elif trend_direction < 0:
                enh_volume['signal'] = 'CONFIRMS BEARISH'
                enh_volume['description'] = f"Volume {latest['Volume']:.0f} is {vol_ratio:.1f}x above 20-day avg {vol_sma:.0f} — confirms bearish pressure"
                enh_volume['contribution'] = -1
            else:
                enh_volume['signal'] = 'HIGH VOLUME'
                enh_volume['description'] = f"Volume {latest['Volume']:.0f} is {vol_ratio:.1f}x above 20-day avg {vol_sma:.0f} — no clear trend"
                enh_volume['contribution'] = 0
        elif vol_ratio <= 0.5:
            enh_volume['signal'] = 'LOW CONVICTION'
            enh_volume['description'] = f"Volume {latest['Volume']:.0f} is only {vol_ratio:.1f}x of 20-day avg {vol_sma:.0f} — weak conviction"
            enh_volume['contribution'] = -0.5
        else:
            enh_volume['signal'] = 'NORMAL'
            enh_volume['description'] = f"Volume {latest['Volume']:.0f} is {vol_ratio:.1f}x of 20-day avg {vol_sma:.0f}"
            enh_volume['contribution'] = 0
    else:
        enh_volume['description'] = "Volume data unavailable"
    analysis['enhancements']['volume_confirm'] = enh_volume

    # Enhancement 2: Kumo Twist Detection
    enh_twist = {'name': 'Kumo Twist', 'signal': 'NONE', 'description': 'No Kumo twist detected', 'contribution': 0}
    if ('future_senkou_a' in latest.index and 'future_senkou_b' in latest.index
            and pd.notna(latest.get('future_senkou_a')) and pd.notna(latest.get('future_senkou_b')) and len(df) >= 2):
        prev = df.iloc[-2]
        if pd.notna(prev.get('future_senkou_a')) and pd.notna(prev.get('future_senkou_b')):
            prev_diff = prev['future_senkou_a'] - prev['future_senkou_b']
            curr_diff = latest['future_senkou_a'] - latest['future_senkou_b']
            if prev_diff * curr_diff < 0 and abs(curr_diff) > latest['Close'] * 0.001:
                if curr_diff > 0:
                    enh_twist['signal'] = 'BULLISH TWIST'
                    enh_twist['description'] = "Senkou A crossing above Senkou B — future cloud turning bullish"
                    enh_twist['contribution'] = 1
                else:
                    enh_twist['signal'] = 'BEARISH TWIST'
                    enh_twist['description'] = "Senkou A crossing below Senkou B — future cloud turning bearish"
                    enh_twist['contribution'] = -1
    analysis['enhancements']['kumo_twist'] = enh_twist

    # Enhancement 3: Cloud Thickness
    enh_thickness = {'name': 'Cloud Thickness', 'signal': 'NEUTRAL', 'description': '', 'contribution': 0}
    if ('cloud_thickness_pct' in latest.index and pd.notna(latest.get('cloud_thickness_pct'))
            and pd.notna(latest.get('cloud_thickness'))):
        thickness_pct = latest['cloud_thickness_pct']
        thickness_abs = latest['cloud_thickness']
        kumo_contribution = analysis['components']['kumo']['contribution']

        if thickness_pct < 1.0:
            enh_thickness['signal'] = 'THIN CLOUD'
            enh_thickness['description'] = f"Cloud thickness {thickness_pct:.1f}% of price ({thickness_abs:.2f}) — weak support/resistance"
            enh_thickness['contribution'] = 0
        elif thickness_pct > 4.0:
            if kumo_contribution > 0:
                enh_thickness['signal'] = 'STRONG SUPPORT'
                enh_thickness['description'] = f"Cloud thickness {thickness_pct:.1f}% of price ({thickness_abs:.2f}) — strong support below"
                enh_thickness['contribution'] = 1
            elif kumo_contribution < 0:
                enh_thickness['signal'] = 'STRONG RESISTANCE'
                enh_thickness['description'] = f"Cloud thickness {thickness_pct:.1f}% of price ({thickness_abs:.2f}) — strong resistance above"
                enh_thickness['contribution'] = -1
            else:
                enh_thickness['signal'] = 'THICK CLOUD'
                enh_thickness['description'] = f"Cloud thickness {thickness_pct:.1f}% of price ({thickness_abs:.2f}) — price inside thick cloud"
                enh_thickness['contribution'] = 0
        else:
            enh_thickness['signal'] = 'MODERATE'
            enh_thickness['description'] = f"Cloud thickness {thickness_pct:.1f}% of price ({thickness_abs:.2f})"
            enh_thickness['contribution'] = 0.5 if kumo_contribution > 0 else (-0.5 if kumo_contribution < 0 else 0)
    else:
        enh_thickness['description'] = "Cloud thickness data unavailable"
    analysis['enhancements']['cloud_thickness'] = enh_thickness

    # Enhancement 4: TK Cross Location
    enh_tk_loc = {'name': 'TK Cross Location', 'signal': 'NEUTRAL', 'description': '', 'contribution': 0}
    if (pd.notna(latest['tenkan_sen']) and pd.notna(latest['kijun_sen'])
            and pd.notna(latest['senkou_span_a']) and pd.notna(latest['senkou_span_b'])):
        tk_midpoint = (latest['tenkan_sen'] + latest['kijun_sen']) / 2
        cloud_top = max(latest['senkou_span_a'], latest['senkou_span_b'])
        cloud_bottom = min(latest['senkou_span_a'], latest['senkou_span_b'])
        tk_direction = analysis['components']['tk_cross']['contribution']

        if tk_direction > 0 and tk_midpoint > cloud_top:
            enh_tk_loc['signal'] = 'ABOVE CLOUD'
            enh_tk_loc['description'] = f"Bullish TK cross above cloud — strong bullish signal"
            enh_tk_loc['contribution'] = 1
        elif tk_direction < 0 and tk_midpoint < cloud_bottom:
            enh_tk_loc['signal'] = 'BELOW CLOUD'
            enh_tk_loc['description'] = f"Bearish TK cross below cloud — strong bearish signal"
            enh_tk_loc['contribution'] = -1
        elif cloud_bottom <= tk_midpoint <= cloud_top:
            enh_tk_loc['signal'] = 'INSIDE CLOUD'
            enh_tk_loc['description'] = f"TK cross inside cloud — weakened signal"
            enh_tk_loc['contribution'] = 0
        else:
            enh_tk_loc['description'] = f"TK cross outside cloud, direction mixed"
            enh_tk_loc['contribution'] = 0
    else:
        enh_tk_loc['description'] = "TK/Cloud data unavailable"
    analysis['enhancements']['tk_location'] = enh_tk_loc

    # Enhancement 5: Price-to-Kijun Distance (Overextension)
    enh_dist = {'name': 'Kijun Distance', 'signal': 'NEUTRAL', 'description': '', 'contribution': 0}
    if 'kijun_distance_pct' in latest.index and pd.notna(latest.get('kijun_distance_pct')):
        dist = latest['kijun_distance_pct']
        if dist > 8.0:
            enh_dist['signal'] = 'OVEREXTENDED LONG'
            enh_dist['description'] = f"Price is {dist:.1f}% above Kijun — high risk of pullback"
            enh_dist['contribution'] = -1
        elif dist > 5.0:
            enh_dist['signal'] = 'STRETCHED LONG'
            enh_dist['description'] = f"Price is {dist:.1f}% above Kijun — mildly overextended"
            enh_dist['contribution'] = -0.5
        elif dist < -8.0:
            enh_dist['signal'] = 'OVEREXTENDED SHORT'
            enh_dist['description'] = f"Price is {abs(dist):.1f}% below Kijun — may bounce"
            enh_dist['contribution'] = 0.5
        elif dist < -5.0:
            enh_dist['signal'] = 'STRETCHED SHORT'
            enh_dist['description'] = f"Price is {abs(dist):.1f}% below Kijun — mildly oversold"
            enh_dist['contribution'] = 0.25
        else:
            enh_dist['signal'] = 'NORMAL RANGE'
            enh_dist['description'] = f"Price is {dist:+.1f}% from Kijun — normal range"
            enh_dist['contribution'] = 0
    else:
        enh_dist['description'] = "Kijun distance data unavailable"
    analysis['enhancements']['kijun_distance'] = enh_dist

    # Enhancement 6: Flat Line Detection
    enh_flat = {'name': 'Flat Lines', 'signal': 'NONE', 'description': 'No flat lines — normal momentum', 'contribution': 0}
    if 'kijun_flat' in latest.index and pd.notna(latest.get('kijun_flat')):
        threshold = latest['Close'] * 0.001
        kijun_is_flat = latest['kijun_flat'] < threshold
        tenkan_is_flat = ('tenkan_flat' in latest.index
                          and pd.notna(latest.get('tenkan_flat'))
                          and latest['tenkan_flat'] < threshold)
        if kijun_is_flat and tenkan_is_flat:
            enh_flat['signal'] = 'CONSOLIDATION'
            enh_flat['description'] = "Both Kijun and Tenkan flat — strong consolidation, expect breakout"
            enh_flat['contribution'] = -0.5
        elif kijun_is_flat:
            enh_flat['signal'] = 'KIJUN FLAT'
            enh_flat['description'] = "Kijun-sen flat — price consolidating around base line"
            enh_flat['contribution'] = -0.25
        elif tenkan_is_flat:
            enh_flat['signal'] = 'TENKAN FLAT'
            enh_flat['description'] = "Tenkan-sen flat — short-term momentum stalling"
            enh_flat['contribution'] = 0
    analysis['enhancements']['flat_lines'] = enh_flat

    # --- Confidence Score ---
    all_contributions = [c['contribution'] for c in analysis['components'].values()]
    all_contributions += [e['contribution'] for e in analysis['enhancements'].values()]
    bullish_count = sum(1 for c in all_contributions if c > 0)
    bearish_count = sum(1 for c in all_contributions if c < 0)
    non_neutral_count = bullish_count + bearish_count
    if non_neutral_count > 0:
        analysis['confidence_score'] = round(max(bullish_count, bearish_count) / non_neutral_count * 100)
    else:
        analysis['confidence_score'] = 0
    if analysis['confidence_score'] >= 80:
        analysis['confidence_label'] = 'HIGH'
    elif analysis['confidence_score'] >= 50:
        analysis['confidence_label'] = 'MODERATE'
    else:
        analysis['confidence_label'] = 'LOW'

    # --- Trade Targets ---
    trade_targets = {}
    if pd.notna(latest['kijun_sen']):
        trade_targets['stop_loss_kijun'] = latest['kijun_sen']
    if pd.notna(latest['senkou_span_a']) and pd.notna(latest['senkou_span_b']):
        cloud_bottom = min(latest['senkou_span_a'], latest['senkou_span_b'])
        cloud_top = max(latest['senkou_span_a'], latest['senkou_span_b'])
        trade_targets['stop_loss_cloud'] = cloud_bottom
        trade_targets['resistance_cloud'] = cloud_top

    # Primary stop-loss: Kijun if price above it, else cloud bottom
    if latest['Close'] > latest.get('kijun_sen', 0):
        trade_targets['stop_loss_primary'] = trade_targets.get('stop_loss_kijun', trade_targets.get('stop_loss_cloud'))
    else:
        trade_targets['stop_loss_primary'] = trade_targets.get('stop_loss_cloud', trade_targets.get('stop_loss_kijun'))

    # Take-profit targets (only meaningful for longs where price > stop-loss)
    if trade_targets.get('stop_loss_primary'):
        risk = latest['Close'] - trade_targets['stop_loss_primary']
        if risk > 0:
            trade_targets['take_profit_1'] = latest['Close'] + risk        # 1:1 R:R
            trade_targets['take_profit_2'] = latest['Close'] + (risk * 2)  # 1:2 R:R

    # Overextension flag
    if 'kijun_distance_pct' in latest.index and pd.notna(latest.get('kijun_distance_pct')):
        trade_targets['kijun_distance_pct'] = latest['kijun_distance_pct']
        trade_targets['overextended'] = abs(latest['kijun_distance_pct']) > 8.0
    analysis['trade_targets'] = trade_targets

    # Calculate total score
    total_score = sum(c['contribution'] for c in analysis['components'].values())
    analysis['total_score'] = total_score

    # Compare with previous day and mark changes
    if prev_day_components:
        # Core components
        for comp_key in ['kumo', 'tk_cross', 'cloud_color', 'chikou', 'kijun']:
            current_val = analysis['components'].get(comp_key, {}).get('contribution', 0)
            prev_val = prev_day_components.get(comp_key, 0)
            if current_val != prev_val:
                analysis['changes'][comp_key] = {
                    'prev': prev_val,
                    'current': current_val,
                    'direction': 'up' if current_val > prev_val else 'down'
                }
        # Enhancement signals
        for enh_key in ['volume_confirm', 'kumo_twist', 'cloud_thickness', 'tk_location', 'kijun_distance', 'flat_lines']:
            current_val = analysis['enhancements'].get(enh_key, {}).get('contribution', 0)
            prev_val = prev_day_components.get(enh_key, 0)
            if current_val != prev_val:
                analysis['changes'][enh_key] = {
                    'prev': prev_val,
                    'current': current_val,
                    'direction': 'up' if current_val > prev_val else 'down'
                }

    # Calculate previous total score for comparison (core 5 components only)
    if prev_day_components:
        core_keys = ['kumo', 'tk_cross', 'cloud_color', 'chikou', 'kijun']
        prev_total = sum(prev_day_components.get(k, 0) for k in core_keys)
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

    # Confidence-based recommendation adjustment
    if analysis['recommendation'] == 'BUY (MODERATE)' and analysis['confidence_score'] >= 85:
        analysis['recommendation'] = 'BUY'
        analysis['recommendation_note'] = 'Upgraded from BUY (MODERATE) — high signal confidence'
    elif analysis['recommendation'] == 'BUY' and analysis['confidence_score'] < 40:
        analysis['recommendation'] = 'BUY (MODERATE)'
        analysis['recommendation_note'] = 'Downgraded from BUY — low signal confidence'

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
    report_lines.append("")
    report_lines.append("ENHANCEMENT SIGNALS:")
    report_lines.append("   - Volume Confirmation: confirms/weakens trend based on 20-day volume average")
    report_lines.append("   - Kumo Twist: Senkou A/B crossover signals major trend change")
    report_lines.append("   - Cloud Thickness: thin cloud = weak S/R, thick = strong S/R")
    report_lines.append("   - TK Cross Location: TK cross above cloud = strong, inside = weak")
    report_lines.append("   - Kijun Distance: overextension risk when price deviates >8% from Kijun")
    report_lines.append("   - Flat Lines: flat Kijun/Tenkan indicates consolidation phase")
    report_lines.append("")
    report_lines.append("CONFIDENCE SCORE:")
    report_lines.append("   Percentage of all signals (11 total) agreeing on direction")
    report_lines.append("   HIGH (>=80%) | MODERATE (50-79%) | LOW (<50%)")
    report_lines.append("="*80)
    report_lines.append("")

    # Summary table with component breakdown
    report_lines.append("SUMMARY TABLE (* = changed from previous day):")
    report_lines.append("-" * 155)
    report_lines.append(
        f"{'Ticker':<12} {'Kumo':>6} {'TK':>6} {'Cloud':>6} {'Chik':>6} {'Kij':>6} "
        f"{'Score':>7} {'Conf':>5} {'Long Entry':<15} {'Price':>10} {'SL':>10} {'TP1':>10}"
    )
    report_lines.append("-" * 155)

    for analysis in analyses:
        components = analysis.get('components', {})
        changes = analysis.get('changes', {})
        targets = analysis.get('trade_targets', {})

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
        conf_str = f"{analysis.get('confidence_score', 0)}%"
        sl_str = f"{targets['stop_loss_primary']:.2f}" if targets.get('stop_loss_primary') else "N/A"
        tp1_str = f"{targets['take_profit_1']:.2f}" if targets.get('take_profit_1') else "N/A"

        report_lines.append(
            f"{analysis['ticker']:<12} "
            f"{kumo_str:>6} "
            f"{tk_str:>6} "
            f"{cloud_str:>6} "
            f"{chikou_str:>6} "
            f"{kijun_str:>6} "
            f"{total_str:>7} "
            f"{conf_str:>5} "
            f"{analysis['recommendation']:<15} "
            f"{analysis['close_price']:>10.2f} "
            f"{sl_str:>10} "
            f"{tp1_str:>10}"
        )

    report_lines.append("-" * 155)
    report_lines.append("Legend: Kumo=Price vs Cloud | TK=Tenkan vs Kijun | Cloud=Future Cloud Color | Chik=Chikou Span | Kij=Price vs Kijun")
    report_lines.append("        Conf=Signal Confidence | SL=Stop-Loss | TP1=Take-Profit Target 1 | * = changed from previous day")
    report_lines.append("="*155)
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

        # Enhancement signals
        report_lines.append("ENHANCEMENT SIGNALS:")
        report_lines.append("-" * 80)
        enhancement_order = ['volume_confirm', 'kumo_twist', 'cloud_thickness',
                             'tk_location', 'kijun_distance', 'flat_lines']
        for enh_key in enhancement_order:
            if enh_key in analysis.get('enhancements', {}):
                enh = analysis['enhancements'][enh_key]
                changed_marker = ""
                change_info = ""
                if enh_key in changes:
                    changed_marker = " *CHANGED*"
                    prev_val = changes[enh_key]['prev']
                    direction = changes[enh_key]['direction']
                    arrow = "↑" if direction == 'up' else "↓"
                    change_info = f" (was {prev_val:+.1f} {arrow})"

                report_lines.append(f"  {enh['name']}{changed_marker}")
                report_lines.append(f"    Signal: {enh['signal']}")
                report_lines.append(f"    {enh['description']}")
                report_lines.append(f"    Enhancement: {enh['contribution']:+.1f}{change_info}")
                report_lines.append("")

        # Confidence score
        report_lines.append(f"SIGNAL CONFIDENCE: {analysis.get('confidence_score', 0)}% ({analysis.get('confidence_label', 'N/A')})")
        if analysis.get('recommendation_note'):
            report_lines.append(f"  Note: {analysis['recommendation_note']}")
        report_lines.append("")

        # Trade targets
        targets = analysis.get('trade_targets', {})
        report_lines.append("TRADE TARGETS:")
        report_lines.append("-" * 80)
        if targets.get('stop_loss_primary'):
            report_lines.append(f"  Stop-Loss (Primary):  {targets['stop_loss_primary']:.2f}")
        if targets.get('stop_loss_kijun'):
            report_lines.append(f"  Stop-Loss (Kijun):    {targets['stop_loss_kijun']:.2f}")
        if targets.get('stop_loss_cloud'):
            report_lines.append(f"  Stop-Loss (Cloud):    {targets['stop_loss_cloud']:.2f}")
        if targets.get('take_profit_1'):
            report_lines.append(f"  Take-Profit 1 (1:1):  {targets['take_profit_1']:.2f}")
        if targets.get('take_profit_2'):
            report_lines.append(f"  Take-Profit 2 (1:2):  {targets['take_profit_2']:.2f}")
        if not any(targets.get(k) for k in ['stop_loss_primary', 'take_profit_1']):
            report_lines.append("  N/A (insufficient data or AVOID recommendation)")
        if targets.get('overextended'):
            report_lines.append(f"  *** WARNING: Price overextended ({targets['kijun_distance_pct']:+.1f}% from Kijun) ***")
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
    table_data = [['Ticker', 'Kumo', 'TK', 'Cloud', 'Chik', 'Kij', 'Score', 'Conf%', 'Recommendation', 'Price', 'SL', 'TP1']]

    for analysis in analyses:
        components = analysis.get('components', {})
        changes = analysis.get('changes', {})
        targets = analysis.get('trade_targets', {})

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
        conf_str = f"{analysis.get('confidence_score', 0)}%"
        sl_str = f"{targets['stop_loss_primary']:.2f}" if targets.get('stop_loss_primary') else "N/A"
        tp1_str = f"{targets['take_profit_1']:.2f}" if targets.get('take_profit_1') else "N/A"

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
            conf_str,
            analysis['recommendation'],
            f"{analysis['close_price']:.2f}",
            sl_str,
            tp1_str,
        ])

    # Create table with styling
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))

    # Color code recommendations and confidence
    for i, analysis in enumerate(analyses, 1):
        rec = analysis['recommendation']
        rec_col = 8  # Recommendation column index
        conf_col = 7  # Confidence column index
        if rec == 'BUY':
            table.setStyle(TableStyle([('BACKGROUND', (rec_col, i), (rec_col, i), colors.lightgreen)]))
        elif rec == 'BUY (MODERATE)':
            table.setStyle(TableStyle([('BACKGROUND', (rec_col, i), (rec_col, i), colors.palegreen)]))
        elif rec == 'AVOID':
            table.setStyle(TableStyle([('BACKGROUND', (rec_col, i), (rec_col, i), colors.lightcoral)]))
        elif rec == 'WAIT':
            table.setStyle(TableStyle([('BACKGROUND', (rec_col, i), (rec_col, i), colors.lightyellow)]))

        # Color code confidence
        conf = analysis.get('confidence_score', 0)
        if conf >= 80:
            table.setStyle(TableStyle([('BACKGROUND', (conf_col, i), (conf_col, i), colors.lightgreen)]))
        elif conf >= 50:
            table.setStyle(TableStyle([('BACKGROUND', (conf_col, i), (conf_col, i), colors.lightyellow)]))
        else:
            table.setStyle(TableStyle([('BACKGROUND', (conf_col, i), (conf_col, i), colors.lightcoral)]))

        # Highlight rows with changes
        if analysis.get('changes'):
            table.setStyle(TableStyle([('BACKGROUND', (0, i), (0, i), colors.yellow)]))

    content.append(table)
    content.append(Spacer(1, 0.2*inch))

    # Legend
    legend_text = "Legend: Kumo=Price vs Cloud | TK=Tenkan vs Kijun | Cloud=Future Cloud Color | Chik=Chikou Span | Kij=Price vs Kijun"
    content.append(Paragraph(legend_text, normal_style))
    content.append(Paragraph("Conf%=Signal Confidence | SL=Stop-Loss | TP1=Take-Profit Target 1 | * = changed from previous day", normal_style))

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
        content.append(Spacer(1, 0.15*inch))

        # Enhancement signals table
        enh_data = [['Enhancement', 'Signal', 'Value', 'Changed?']]
        enhancement_names = {
            'volume_confirm': 'Volume Confirmation',
            'kumo_twist': 'Kumo Twist',
            'cloud_thickness': 'Cloud Thickness',
            'tk_location': 'TK Cross Location',
            'kijun_distance': 'Kijun Distance',
            'flat_lines': 'Flat Lines',
        }
        for enh_key in ['volume_confirm', 'kumo_twist', 'cloud_thickness',
                        'tk_location', 'kijun_distance', 'flat_lines']:
            enh = analysis.get('enhancements', {}).get(enh_key, {})
            contrib = enh.get('contribution', 0)
            changed = ''
            if enh_key in changes:
                prev_val = changes[enh_key]['prev']
                changed = f"YES ({prev_val:+.1f} → {contrib:+.1f})"
            enh_data.append([
                enhancement_names.get(enh_key, enh_key),
                enh.get('signal', 'N/A'),
                f"{contrib:+.1f}",
                changed,
            ])

        enh_table = Table(enh_data, colWidths=[2.5*inch, 1.5*inch, 1*inch, 1.5*inch])
        enh_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkslategray),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))

        # Highlight changed enhancements
        for i, enh_key in enumerate(['volume_confirm', 'kumo_twist', 'cloud_thickness',
                                      'tk_location', 'kijun_distance', 'flat_lines'], 1):
            if enh_key in changes:
                enh_table.setStyle(TableStyle([('BACKGROUND', (-1, i), (-1, i), colors.yellow)]))

        content.append(enh_table)
        content.append(Spacer(1, 0.1*inch))

        # Confidence score
        conf = analysis.get('confidence_score', 0)
        conf_label = analysis.get('confidence_label', 'N/A')
        conf_color = 'green' if conf >= 80 else ('orange' if conf >= 50 else 'red')
        conf_text = f"<b>Signal Confidence:</b> <font color='{conf_color}'>{conf}% ({conf_label})</font>"
        if analysis.get('recommendation_note'):
            conf_text += f" — {analysis['recommendation_note']}"
        content.append(Paragraph(conf_text, normal_style))

        # Trade targets
        targets = analysis.get('trade_targets', {})
        if targets:
            targets_parts = []
            if targets.get('stop_loss_primary'):
                targets_parts.append(f"<b>Stop-Loss:</b> {targets['stop_loss_primary']:.2f}")
            if targets.get('take_profit_1'):
                targets_parts.append(f"<b>TP1 (1:1):</b> {targets['take_profit_1']:.2f}")
            if targets.get('take_profit_2'):
                targets_parts.append(f"<b>TP2 (1:2):</b> {targets['take_profit_2']:.2f}")
            if targets_parts:
                content.append(Paragraph(" | ".join(targets_parts), normal_style))
            if targets.get('overextended'):
                content.append(Paragraph(
                    f"<font color='red'><b>WARNING:</b> Price overextended — {targets['kijun_distance_pct']:+.1f}% from Kijun</font>",
                    normal_style
                ))

        content.append(Spacer(1, 0.15*inch))

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


def generate_html_dashboard(analyses, charts_folder, output_folder, filename="ichimoku_dashboard.html", market_name=None, currency="USD"):
    """
    Generate a self-contained HTML dashboard with grid view and interactive chart modal.

    Parameters:
    -----------
    analyses : list
        List of analysis dictionaries from analyze_ichimoku_signals()
    charts_folder : str
        Folder containing chart PNG files
    output_folder : str
        Folder to save the HTML file
    filename : str
        Output filename
    market_name : str
        Display name for the market
    currency : str
        Currency symbol for price display
    """
    if not analyses:
        return None

    market_name = market_name or "Stock Market"
    scan_date = analyses[0].get('date', datetime.now().strftime('%Y-%m-%d'))

    # Count recommendations
    rec_counts = {}
    for a in analyses:
        rec = a['recommendation']
        rec_counts[rec] = rec_counts.get(rec, 0) + 1

    # Prepare stock data for JSON embedding
    stocks_json = []
    for a in analyses:
        ticker = a['ticker']
        chart_filename = f"{ticker.lower()}_ichimoku_{datetime.now().strftime('%Y%m%d')}.png"
        chart_path = os.path.join(charts_folder, chart_filename)
        chart_rel = f"charts/{chart_filename}" if os.path.exists(chart_path) else ""

        # Serialize components
        components = {}
        for key, comp in a.get('components', {}).items():
            components[key] = {
                'name': comp.get('name', key),
                'signal': str(comp.get('signal', 'N/A')),
                'description': str(comp.get('description', '')),
                'contribution': float(comp.get('contribution', 0))
            }

        # Serialize enhancements
        enhancements = {}
        for key, enh in a.get('enhancements', {}).items():
            enhancements[key] = {
                'signal': str(enh.get('signal', 'N/A')),
                'description': str(enh.get('description', '')),
                'contribution': float(enh.get('contribution', 0))
            }

        # Serialize trade targets
        targets = a.get('trade_targets', {})
        trade_targets = {
            'stop_loss_primary': float(targets.get('stop_loss_primary', 0)),
            'stop_loss_kijun': float(targets.get('stop_loss_kijun', 0)),
            'stop_loss_cloud': float(targets.get('stop_loss_cloud', 0)),
            'take_profit_1': float(targets.get('take_profit_1', 0)),
            'take_profit_2': float(targets.get('take_profit_2', 0)),
            'kijun_distance_pct': float(targets.get('kijun_distance_pct', 0)),
            'overextended': bool(targets.get('overextended', False))
        }

        # Serialize changes
        changes = {}
        raw_changes = a.get('changes', {})
        for key, ch in raw_changes.items():
            if isinstance(ch, dict):
                changes[key] = {
                    'prev': float(ch.get('prev', 0)),
                    'current': float(ch.get('current', 0)),
                    'direction': str(ch.get('direction', ''))
                }

        stocks_json.append({
            'ticker': ticker,
            'name': a.get('name', ticker),
            'date': a.get('date', scan_date),
            'close_price': round(float(a.get('close_price', 0)), 2),
            'currency': currency,
            'recommendation': a.get('recommendation', 'N/A'),
            'confidence_score': int(a.get('confidence_score', 0)),
            'confidence_label': a.get('confidence_label', 'N/A'),
            'trend': a.get('trend', 'N/A'),
            'strength': a.get('strength', 'N/A'),
            'total_score': float(a.get('total_score', 0)),
            'components': components,
            'enhancements': enhancements,
            'trade_targets': trade_targets,
            'changes': changes,
            'chart_file': chart_rel,
            'has_changes': len(changes) > 0,
            'recommendation_note': a.get('recommendation_note', ''),
            'prev_total_score': float(a.get('prev_total_score', a.get('total_score', 0)))
        })

    stocks_data = json.dumps(stocks_json, indent=2)

    # Build summary stats for header
    buy_count = rec_counts.get('BUY', 0)
    moderate_count = rec_counts.get('BUY (MODERATE)', 0)
    wait_count = rec_counts.get('WAIT', 0)
    avoid_count = rec_counts.get('AVOID', 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ichimoku Dashboard — {market_name}</title>
<style>
:root {{
  --bg-primary: #0d1117;
  --bg-secondary: #161b22;
  --bg-tertiary: #1c2333;
  --bg-card: #161b22;
  --bg-card-hover: #1c2333;
  --bg-modal: #0d1117ee;
  --border: #30363d;
  --border-light: #3d444d;
  --text-primary: #e6edf3;
  --text-secondary: #8b949e;
  --text-muted: #6e7681;
  --green: #00ff88;
  --green-dim: #00cc6a;
  --green-bg: #00ff8818;
  --red: #ff4757;
  --red-dim: #cc3945;
  --red-bg: #ff475718;
  --amber: #ffaa00;
  --amber-dim: #cc8800;
  --amber-bg: #ffaa0018;
  --yellow: #f0e040;
  --yellow-bg: #f0e04018;
  --blue: #58a6ff;
  --blue-dim: #388bfd;
  --blue-bg: #58a6ff18;
  --purple: #bc8cff;
  --radius: 10px;
  --radius-sm: 6px;
  --shadow: 0 4px 24px rgba(0,0,0,.4);
  --shadow-lg: 0 8px 48px rgba(0,0,0,.6);
  --transition: .2s ease;
  --font: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'SF Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
}}

[data-theme="light"] {{
  --bg-primary: #ffffff;
  --bg-secondary: #f6f8fa;
  --bg-tertiary: #eef1f5;
  --bg-card: #ffffff;
  --bg-card-hover: #f6f8fa;
  --bg-modal: rgba(255,255,255,0.92);
  --border: #d0d7de;
  --border-light: #bbc0c7;
  --text-primary: #1f2328;
  --text-secondary: #57606a;
  --text-muted: #8b949e;
  --green: #1a7f37;
  --green-dim: #2da44e;
  --green-bg: rgba(26,127,55,0.12);
  --red: #cf222e;
  --red-dim: #d1242f;
  --red-bg: rgba(207,34,46,0.12);
  --amber: #bf8700;
  --amber-dim: #9a6700;
  --amber-bg: rgba(191,135,0,0.12);
  --yellow: #7d6e00;
  --yellow-bg: rgba(125,110,0,0.12);
  --blue: #0969da;
  --blue-dim: #0550ae;
  --blue-bg: rgba(9,105,218,0.12);
  --purple: #8250df;
  --shadow: 0 4px 24px rgba(0,0,0,.08);
  --shadow-lg: 0 8px 48px rgba(0,0,0,.12);
}}

* {{ margin:0; padding:0; box-sizing:border-box; }}

body {{
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: var(--font);
  min-height: 100vh;
  overflow-x: hidden;
}}

/* ─── Header ─── */
.header {{
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 24px 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
  position: sticky;
  top: 0;
  z-index: 100;
  backdrop-filter: blur(12px);
}}
.header-left h1 {{
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -.3px;
}}
.header-left h1 span {{
  color: var(--blue);
}}
.header-left .subtitle {{
  color: var(--text-secondary);
  font-size: 13px;
  margin-top: 2px;
}}
.header-stats {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}}
.stat-pill {{
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 600;
  border: 1px solid transparent;
}}
.stat-pill .dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
}}
.stat-buy {{ background: var(--green-bg); color: var(--green); border-color: var(--green); }}
.stat-buy .dot {{ background: var(--green); }}
.stat-moderate {{ background: var(--amber-bg); color: var(--amber); border-color: var(--amber); }}
.stat-moderate .dot {{ background: var(--amber); }}
.stat-wait {{ background: var(--yellow-bg); color: var(--yellow); border-color: var(--yellow); }}
.stat-wait .dot {{ background: var(--yellow); }}
.stat-avoid {{ background: var(--red-bg); color: var(--red); border-color: var(--red); }}
.stat-avoid .dot {{ background: var(--red); }}

/* Theme toggle */
.theme-toggle {{
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  transition: all var(--transition);
  flex-shrink: 0;
}}
.theme-toggle:hover {{
  border-color: var(--text-secondary);
  color: var(--text-primary);
  background: var(--bg-card-hover, var(--bg-tertiary));
}}

/* ─── Toolbar ─── */
.toolbar {{
  padding: 16px 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-primary);
}}
.filter-group {{
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}}
.filter-btn {{
  padding: 7px 16px;
  border-radius: 20px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  transition: all var(--transition);
  font-family: var(--font);
}}
.filter-btn:hover {{
  border-color: var(--text-secondary);
  color: var(--text-primary);
}}
.filter-btn.active {{
  background: var(--blue);
  border-color: var(--blue);
  color: #fff;
}}
.sort-select {{
  padding: 7px 14px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: var(--bg-secondary);
  color: var(--text-primary);
  font-size: 13px;
  cursor: pointer;
  font-family: var(--font);
}}
.sort-select:focus {{ outline: 1px solid var(--blue); }}

/* ─── Grid ─── */
.grid-container {{
  padding: 24px 32px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
  gap: 16px;
}}
.card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  cursor: pointer;
  transition: all var(--transition);
  position: relative;
  overflow: hidden;
}}
.card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  border-radius: var(--radius) var(--radius) 0 0;
}}
.card.rec-buy::before {{ background: var(--green); }}
.card.rec-moderate::before {{ background: var(--amber); }}
.card.rec-wait::before {{ background: var(--yellow); }}
.card.rec-avoid::before {{ background: var(--red); }}
.card:hover {{
  border-color: var(--border-light);
  background: var(--bg-card-hover);
  transform: translateY(-2px);
  box-shadow: var(--shadow);
}}
.card-header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 14px;
}}
.card-ticker {{
  font-size: 18px;
  font-weight: 700;
  font-family: var(--font-mono);
  letter-spacing: .5px;
}}
.card-name {{
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 2px;
  max-width: 180px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.badge {{
  display: inline-flex;
  align-items: center;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .5px;
  text-transform: uppercase;
  white-space: nowrap;
}}
.badge-buy {{ background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }}
.badge-moderate {{ background: var(--amber-bg); color: var(--amber); border: 1px solid var(--amber); }}
.badge-wait {{ background: var(--yellow-bg); color: var(--yellow); border: 1px solid var(--yellow); }}
.badge-avoid {{ background: var(--red-bg); color: var(--red); border: 1px solid var(--red); }}

.card-body {{
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.card-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.card-label {{
  font-size: 12px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: .5px;
}}
.card-value {{
  font-size: 15px;
  font-weight: 600;
  font-family: var(--font-mono);
}}
.confidence-bar {{
  width: 100%;
  height: 4px;
  background: var(--bg-tertiary);
  border-radius: 2px;
  margin-top: 4px;
  overflow: hidden;
}}
.confidence-fill {{
  height: 100%;
  border-radius: 2px;
  transition: width .4s ease;
}}
.score-indicator {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
}}
.change-dot {{
  position: absolute;
  top: 12px;
  right: 12px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--amber);
  box-shadow: 0 0 6px var(--amber);
}}

/* Card trend & strength indicators */
.card-indicators {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}}
.card-trend {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .3px;
}}
.trend-arrow {{
  font-size: 14px;
  line-height: 1;
}}
.trend-bullish {{ color: var(--green); }}
.trend-bearish {{ color: var(--red); }}
.trend-neutral {{ color: var(--text-muted); }}
.strength-tag {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .5px;
}}
.str-strong {{ background: var(--green-bg); color: var(--green); }}
.str-moderate {{ background: var(--amber-bg); color: var(--amber); }}
.str-weak {{ background: var(--red-bg); color: var(--text-muted); }}
.score-delta {{
  font-size: 11px;
  font-family: var(--font-mono);
  margin-left: 6px;
}}
.score-delta-up {{ color: var(--green); }}
.score-delta-down {{ color: var(--red); }}

/* ─── Modal ─── */
.modal-backdrop {{
  display: none;
  position: fixed;
  inset: 0;
  background: var(--bg-modal);
  z-index: 1000;
  overflow-y: auto;
  padding: 40px 20px;
  backdrop-filter: blur(4px);
}}
.modal-backdrop.open {{ display: flex; justify-content: center; align-items: flex-start; }}
.modal {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  max-width: 1100px;
  width: 100%;
  box-shadow: var(--shadow-lg);
  animation: modalIn .25s ease;
}}
@keyframes modalIn {{
  from {{ opacity: 0; transform: translateY(20px) scale(.98); }}
  to {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
.modal-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 28px;
  border-bottom: 1px solid var(--border);
}}
.modal-header h2 {{
  font-size: 20px;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 12px;
}}
.modal-header h2 .ticker {{
  font-family: var(--font-mono);
  color: var(--blue);
}}
.modal-close {{
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text-secondary);
  font-size: 18px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition);
}}
.modal-close:hover {{ background: var(--bg-tertiary); color: var(--text-primary); }}
.modal-body {{ padding: 24px 28px; }}

/* Modal overview strip */
.overview-strip {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}}
.overview-item {{
  background: var(--bg-tertiary);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  text-align: center;
}}
.overview-item .ov-label {{
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: .5px;
  margin-bottom: 4px;
}}
.overview-item .ov-value {{
  font-size: 18px;
  font-weight: 700;
  font-family: var(--font-mono);
}}

/* Chart area */
.chart-container {{
  margin-bottom: 24px;
  background: var(--bg-primary);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  overflow: hidden;
  position: relative;
}}
.chart-container img {{
  width: 100%;
  height: auto;
  display: block;
}}
.chart-placeholder {{
  padding: 60px;
  text-align: center;
  color: var(--text-muted);
}}

/* Signals section */
.section-title {{
  font-size: 14px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .5px;
  color: var(--text-secondary);
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}}
.signals-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 24px;
}}
@media (max-width: 700px) {{
  .signals-grid {{ grid-template-columns: 1fr; }}
}}
.signal-table {{
  width: 100%;
  border-collapse: collapse;
}}
.signal-table th {{
  text-align: left;
  padding: 8px 12px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .5px;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
}}
.signal-table td {{
  padding: 10px 12px;
  font-size: 13px;
  border-bottom: 1px solid var(--border);
}}
.signal-table tr:last-child td {{ border-bottom: none; }}
.signal-name {{
  font-weight: 600;
  white-space: nowrap;
}}
.signal-badge {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  font-family: var(--font-mono);
}}
.sig-bullish {{ background: var(--green-bg); color: var(--green); }}
.sig-bearish {{ background: var(--red-bg); color: var(--red); }}
.sig-neutral {{ background: var(--blue-bg); color: var(--blue); }}
.contrib-pos {{ color: var(--green); font-family: var(--font-mono); font-weight: 600; }}
.contrib-neg {{ color: var(--red); font-family: var(--font-mono); font-weight: 600; }}
.contrib-zero {{ color: var(--text-muted); font-family: var(--font-mono); }}

/* Tooltip styles */
.has-tooltip {{
  position: relative;
  cursor: help;
  border-bottom: 1px dotted var(--text-muted);
}}
.has-tooltip:hover::after {{
  content: attr(data-tooltip);
  position: absolute;
  left: 0;
  top: 100%;
  margin-top: 6px;
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  font-size: 12px;
  font-weight: 400;
  line-height: 1.5;
  white-space: normal;
  width: 300px;
  z-index: 1000;
  box-shadow: 0 4px 16px rgba(0,0,0,0.3);
  pointer-events: none;
}}
.signal-badge.has-tooltip {{
  border-bottom: none;
}}
.signal-badge.has-tooltip:hover::after {{
  left: auto;
  right: 0;
  width: 260px;
}}

/* Trade targets */
.targets-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}}
.target-box {{
  background: var(--bg-tertiary);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  border-left: 3px solid var(--border);
}}
.target-box.sl {{ border-left-color: var(--red); }}
.target-box.tp1 {{ border-left-color: var(--green); }}
.target-box.tp2 {{ border-left-color: var(--green-dim); }}
.target-box.info {{ border-left-color: var(--blue); }}
.target-label {{
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: .5px;
}}
.target-value {{
  font-size: 17px;
  font-weight: 700;
  font-family: var(--font-mono);
  margin-top: 4px;
}}

/* Changes section */
.changes-list {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 24px;
}}
.change-tag {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-family: var(--font-mono);
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
}}
.change-up {{ color: var(--green); }}
.change-down {{ color: var(--red); }}

/* Recommendation note */
.rec-note {{
  background: var(--blue-bg);
  border: 1px solid var(--blue-dim);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  font-size: 13px;
  color: var(--blue);
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.rec-note .note-icon {{
  font-size: 16px;
  flex-shrink: 0;
}}

/* No results */
.no-results {{
  text-align: center;
  padding: 80px 20px;
  color: var(--text-muted);
  font-size: 16px;
}}

/* Responsive */
@media (max-width: 640px) {{
  .header {{ padding: 16px; }}
  .toolbar {{ padding: 12px 16px; }}
  .grid-container {{ padding: 16px; }}
  .grid {{ grid-template-columns: 1fr; }}
  .modal-body {{ padding: 16px; }}
  .overview-strip {{ grid-template-columns: repeat(2, 1fr); }}
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--border-light); }}

/* Light mode overrides */
[data-theme="light"] ::-webkit-scrollbar-track {{ background: var(--bg-secondary); }}
[data-theme="light"] ::-webkit-scrollbar-thumb {{ background: var(--border); }}
[data-theme="light"] .chart-container {{ background: #1a1a2e; border-radius: var(--radius-sm); padding: 4px; }}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1><span>Ichimoku</span> Dashboard — {market_name}</h1>
    <div class="subtitle">Scan date: {scan_date} &middot; {len(analyses)} stocks analyzed &middot; {currency}</div>
  </div>
  <div class="header-stats">
    <div class="stat-pill stat-buy"><span class="dot"></span> {buy_count} BUY</div>
    <div class="stat-pill stat-moderate"><span class="dot"></span> {moderate_count} MODERATE</div>
    <div class="stat-pill stat-wait"><span class="dot"></span> {wait_count} WAIT</div>
    <div class="stat-pill stat-avoid"><span class="dot"></span> {avoid_count} AVOID</div>
    <button class="theme-toggle" id="themeToggle" title="Toggle light/dark mode" aria-label="Toggle theme">&#9790;</button>
  </div>
</div>

<div class="toolbar">
  <div class="filter-group">
    <button class="filter-btn active" data-filter="ALL">All</button>
    <button class="filter-btn" data-filter="BUY">Buy</button>
    <button class="filter-btn" data-filter="BUY (MODERATE)">Moderate</button>
    <button class="filter-btn" data-filter="WAIT">Wait</button>
    <button class="filter-btn" data-filter="AVOID">Avoid</button>
  </div>
  <select class="sort-select" id="sortSelect">
    <option value="recommendation">Sort: Recommendation</option>
    <option value="score-desc">Sort: Score (High to Low)</option>
    <option value="score-asc">Sort: Score (Low to High)</option>
    <option value="confidence-desc">Sort: Confidence (High to Low)</option>
    <option value="name-asc">Sort: Name (A-Z)</option>
    <option value="price-desc">Sort: Price (High to Low)</option>
  </select>
</div>

<div class="grid-container">
  <div class="grid" id="stockGrid"></div>
  <div class="no-results" id="noResults" style="display:none;">No stocks match the current filter.</div>
</div>

<div class="modal-backdrop" id="modalBackdrop">
  <div class="modal" id="modal">
    <div class="modal-header">
      <h2 id="modalTitle"></h2>
      <button class="modal-close" id="modalClose">&times;</button>
    </div>
    <div class="modal-body" id="modalBody"></div>
  </div>
</div>

<script>
const STOCKS = {stocks_data};

const REC_PRIORITY = {{'BUY': 1, 'BUY (MODERATE)': 2, 'WAIT': 3, 'AVOID': 4}};

let currentFilter = 'ALL';
let currentSort = 'recommendation';

function recClass(rec) {{
  if (rec === 'BUY') return 'buy';
  if (rec === 'BUY (MODERATE)') return 'moderate';
  if (rec === 'WAIT') return 'wait';
  return 'avoid';
}}

function confidenceColor(score) {{
  if (score >= 80) return 'var(--green)';
  if (score >= 50) return 'var(--amber)';
  return 'var(--red)';
}}

function contribHtml(val) {{
  if (val > 0) return '<span class="contrib-pos">+' + val.toFixed(1) + '</span>';
  if (val < 0) return '<span class="contrib-neg">' + val.toFixed(1) + '</span>';
  return '<span class="contrib-zero">0.0</span>';
}}

function signalClass(signal) {{
  const s = signal.toUpperCase();
  if (s.includes('BULLISH') || s.includes('CONFIRMS') || s.includes('STRONG SUPPORT') || s.includes('ABOVE')) return 'sig-bullish';
  if (s.includes('BEARISH') || s.includes('OVEREXTENDED') || s.includes('BELOW') || s.includes('WEAK')) return 'sig-bearish';
  return 'sig-neutral';
}}

function trendHtml(trend) {{
  if (trend === 'BULLISH') return '<span class="card-trend trend-bullish"><span class="trend-arrow">&#9650;</span> Bullish</span>';
  if (trend === 'BEARISH') return '<span class="card-trend trend-bearish"><span class="trend-arrow">&#9660;</span> Bearish</span>';
  return '<span class="card-trend trend-neutral"><span class="trend-arrow">&#9654;</span> Neutral</span>';
}}

function strengthTag(strength) {{
  if (strength === 'STRONG') return '<span class="strength-tag str-strong">Strong</span>';
  if (strength === 'MODERATE') return '<span class="strength-tag str-moderate">Moderate</span>';
  return '<span class="strength-tag str-weak">Weak</span>';
}}

function scoreDeltaHtml(s) {{
  if (s.has_changes && s.changes.total) {{
    const dir = s.changes.total.direction;
    const prev = s.changes.total.prev;
    const arrow = dir === 'up' ? '&#9650;' : '&#9660;';
    const cls = dir === 'up' ? 'score-delta-up' : 'score-delta-down';
    return `<span class="score-delta ${{cls}}">${{arrow}} from ${{prev >= 0 ? '+' : ''}}${{prev.toFixed(1)}}</span>`;
  }}
  return '';
}}

function formatPrice(price, currency) {{
  if (currency === 'DKK') return price.toLocaleString('da-DK', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + ' DKK';
  return '$' + price.toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
}}

function getFiltered() {{
  let list = STOCKS;
  if (currentFilter !== 'ALL') list = list.filter(s => s.recommendation === currentFilter);

  list = [...list];
  switch (currentSort) {{
    case 'score-desc': list.sort((a, b) => b.total_score - a.total_score); break;
    case 'score-asc': list.sort((a, b) => a.total_score - b.total_score); break;
    case 'confidence-desc': list.sort((a, b) => b.confidence_score - a.confidence_score); break;
    case 'name-asc': list.sort((a, b) => a.name.localeCompare(b.name)); break;
    case 'price-desc': list.sort((a, b) => b.close_price - a.close_price); break;
    default: list.sort((a, b) => (REC_PRIORITY[a.recommendation] || 9) - (REC_PRIORITY[b.recommendation] || 9));
  }}
  return list;
}}

function renderGrid() {{
  const grid = document.getElementById('stockGrid');
  const noResults = document.getElementById('noResults');
  const filtered = getFiltered();

  if (filtered.length === 0) {{
    grid.innerHTML = '';
    noResults.style.display = 'block';
    return;
  }}
  noResults.style.display = 'none';

  grid.innerHTML = filtered.map(s => `
    <div class="card rec-${{recClass(s.recommendation)}}" onclick="openModal('${{s.ticker}}')">
      ${{s.has_changes ? '<div class="change-dot" title="Signals changed from previous day"></div>' : ''}}
      <div class="card-header">
        <div>
          <div class="card-ticker">${{s.ticker}}</div>
          <div class="card-name" title="${{s.name}}">${{s.name}}</div>
        </div>
        <div class="badge badge-${{recClass(s.recommendation)}}">${{s.recommendation}}</div>
      </div>
      <div class="card-indicators">
        ${{trendHtml(s.trend)}}
        ${{strengthTag(s.strength)}}
      </div>
      <div class="card-body">
        <div class="card-row">
          <span class="card-label">Price</span>
          <span class="card-value">${{formatPrice(s.close_price, s.currency)}}</span>
        </div>
        <div class="card-row">
          <span class="card-label">Score</span>
          <span class="card-value" style="color:${{s.total_score >= 0 ? 'var(--green)' : 'var(--red)'}}">${{s.total_score >= 0 ? '+' : ''}}${{s.total_score.toFixed(1)}}${{scoreDeltaHtml(s)}}</span>
        </div>
        <div>
          <div class="card-row">
            <span class="card-label">Confidence</span>
            <span class="card-value" style="color:${{confidenceColor(s.confidence_score)}}">${{s.confidence_score}}% <small style="font-weight:400;font-size:11px">${{s.confidence_label}}</small></span>
          </div>
          <div class="confidence-bar"><div class="confidence-fill" style="width:${{s.confidence_score}}%;background:${{confidenceColor(s.confidence_score)}}"></div></div>
        </div>
      </div>
    </div>
  `).join('');
}}

function openModal(ticker) {{
  const s = STOCKS.find(x => x.ticker === ticker);
  if (!s) return;

  const rc = recClass(s.recommendation);
  document.getElementById('modalTitle').innerHTML = `<span class="ticker">${{s.ticker}}</span> ${{s.name}} <span class="badge badge-${{rc}}">${{s.recommendation}}</span>`;

  // Tooltip descriptions for component names
  const compTooltips = {{
    'kumo': 'Price vs Cloud (Kumo): Where price sits relative to the Ichimoku cloud. Above cloud = bullish (+2), below = bearish (-2), inside = neutral.',
    'tk_cross': 'Tenkan-Kijun Cross: The Tenkan-sen (9-period) crossing above/below the Kijun-sen (26-period). Bullish cross = +1, bearish = -1.',
    'cloud_color': 'Future Cloud Color: Whether Senkou Span A is above or below Senkou Span B projected 26 periods ahead. Green cloud = +1, red = -1.',
    'chikou': 'Chikou Span (Lagging Line): Current price plotted 26 periods back vs past price action. Above = bullish (up to +2), below = bearish (down to -2).',
    'kijun': 'Kijun-sen Support: Whether price is holding above or below the Kijun-sen (26-period baseline). Above = support (+1), below = resistance (-1).'
  }};
  const enhTooltips = {{
    'volume_confirm': 'Volume Confirmation: Compares current volume to 20-day average. 1.5x+ confirms trend direction (\\u00b11), 0.5x or less = weak conviction (-0.5).',
    'kumo_twist': 'Kumo Twist: Detects when Senkou Span A and B cross each other, signaling a potential major trend reversal. Bullish twist = +1, bearish = -1.',
    'cloud_thickness': 'Cloud Thickness: Measures cloud width as % of price. Thin (<1%) = weak support/resistance, thick (>4%) = strong. Scores \\u00b11.',
    'tk_location': 'TK Cross Location: Where the Tenkan/Kijun cross occurs relative to the cloud. Above cloud = strong (+1), inside = moderate (0), below = weak.',
    'kijun_distance': 'Kijun Distance: How far price has extended from Kijun-sen. >8% = overextended risk (-1), >5% = mild caution (-0.5).',
    'flat_lines': 'Flat Kijun/Tenkan Lines: Detects consolidation periods when both lines are flat. Both flat = consolidation warning (-0.5).'
  }};

  // Build component rows
  const compNames = {{'kumo': 'Kumo (Cloud)', 'tk_cross': 'TK Cross', 'cloud_color': 'Cloud Color', 'chikou': 'Chikou Span', 'kijun': 'Kijun Support'}};
  let compRows = '';
  for (const [key, label] of Object.entries(compNames)) {{
    const c = s.components[key];
    if (!c) continue;
    const compTip = (compTooltips[key] || '').replace(/"/g, '&quot;');
    const sigTip = (c.description || '').replace(/"/g, '&quot;');
    compRows += `<tr>
      <td class="signal-name"><span class="has-tooltip" data-tooltip="${{compTip}}">${{label}}</span></td>
      <td><span class="signal-badge ${{signalClass(c.signal)}} has-tooltip" data-tooltip="${{sigTip}}">${{c.signal}}</span></td>
      <td>${{contribHtml(c.contribution)}}</td>
    </tr>`;
  }}

  // Build enhancement rows
  const enhNames = {{'volume_confirm': 'Volume', 'kumo_twist': 'Kumo Twist', 'cloud_thickness': 'Cloud Thickness', 'tk_location': 'TK Location', 'kijun_distance': 'Kijun Distance', 'flat_lines': 'Flat Lines'}};
  let enhRows = '';
  for (const [key, label] of Object.entries(enhNames)) {{
    const e = s.enhancements[key];
    if (!e) continue;
    const enhTip = (enhTooltips[key] || '').replace(/"/g, '&quot;');
    const sigTip = (e.description || '').replace(/"/g, '&quot;');
    enhRows += `<tr>
      <td class="signal-name"><span class="has-tooltip" data-tooltip="${{enhTip}}">${{label}}</span></td>
      <td><span class="signal-badge ${{signalClass(e.signal)}} has-tooltip" data-tooltip="${{sigTip}}">${{e.signal}}</span></td>
      <td>${{contribHtml(e.contribution)}}</td>
    </tr>`;
  }}

  // Build trade targets
  const t = s.trade_targets;
  const targetsHtml = `
    <div class="targets-grid">
      <div class="target-box sl">
        <div class="target-label">Stop Loss</div>
        <div class="target-value" style="color:var(--red)">${{formatPrice(t.stop_loss_primary, s.currency)}}</div>
      </div>
      <div class="target-box tp1">
        <div class="target-label">Take Profit 1 (1:1)</div>
        <div class="target-value" style="color:var(--green)">${{formatPrice(t.take_profit_1, s.currency)}}</div>
      </div>
      <div class="target-box tp2">
        <div class="target-label">Take Profit 2 (1:2)</div>
        <div class="target-value" style="color:var(--green-dim)">${{formatPrice(t.take_profit_2, s.currency)}}</div>
      </div>
      <div class="target-box info">
        <div class="target-label">Kijun Distance</div>
        <div class="target-value" style="color:${{Math.abs(t.kijun_distance_pct) > 5 ? 'var(--amber)' : 'var(--text-primary)'}}">${{t.kijun_distance_pct.toFixed(1)}}%${{t.overextended ? ' ⚠' : ''}}</div>
      </div>
    </div>`;

  // Build changes
  let changesHtml = '';
  if (s.has_changes) {{
    changesHtml = '<div class="section-title">Day-over-Day Changes</div><div class="changes-list">';
    for (const [key, ch] of Object.entries(s.changes)) {{
      if (key === 'total') continue;
      const arrow = ch.direction === 'up' ? '&#9650;' : '&#9660;';
      const cls = ch.direction === 'up' ? 'change-up' : 'change-down';
      changesHtml += `<span class="change-tag ${{cls}}">${{arrow}} ${{key.replace('_', ' ')}} ${{ch.prev.toFixed(1)}} &rarr; ${{ch.current.toFixed(1)}}</span>`;
    }}
    changesHtml += '</div>';
  }}

  const chartHtml = s.chart_file
    ? `<div class="chart-container"><img src="${{s.chart_file}}" alt="${{s.ticker}} Ichimoku Chart" loading="lazy"></div>`
    : '<div class="chart-container"><div class="chart-placeholder">Chart not available</div></div>';

  document.getElementById('modalBody').innerHTML = `
    <div class="overview-strip">
      <div class="overview-item">
        <div class="ov-label">Price</div>
        <div class="ov-value">${{formatPrice(s.close_price, s.currency)}}</div>
      </div>
      <div class="overview-item">
        <div class="ov-label">Score</div>
        <div class="ov-value" style="color:${{s.total_score >= 0 ? 'var(--green)' : 'var(--red)'}}">${{s.total_score >= 0 ? '+' : ''}}${{s.total_score.toFixed(1)}}</div>
      </div>
      <div class="overview-item">
        <div class="ov-label">Confidence</div>
        <div class="ov-value" style="color:${{confidenceColor(s.confidence_score)}}">${{s.confidence_score}}%</div>
      </div>
      <div class="overview-item">
        <div class="ov-label">Trend</div>
        <div class="ov-value">${{s.trend}}</div>
      </div>
      <div class="overview-item">
        <div class="ov-label">Strength</div>
        <div class="ov-value">${{s.strength}}</div>
      </div>
    </div>

    ${{s.recommendation_note ? `<div class="rec-note"><span class="note-icon">&#9432;</span> ${{s.recommendation_note}}</div>` : ''}}

    ${{chartHtml}}

    <div class="section-title">Trade Targets</div>
    ${{targetsHtml}}

    <div class="signals-grid">
      <div>
        <div class="section-title">Core Components</div>
        <table class="signal-table">
          <tr><th>Component</th><th>Signal</th><th>Score</th></tr>
          ${{compRows}}
        </table>
      </div>
      <div>
        <div class="section-title">Enhancement Signals</div>
        <table class="signal-table">
          <tr><th>Signal</th><th>Status</th><th>Score</th></tr>
          ${{enhRows}}
        </table>
      </div>
    </div>

    ${{changesHtml}}
  `;

  document.getElementById('modalBackdrop').classList.add('open');
  document.body.style.overflow = 'hidden';
}}

function closeModal() {{
  document.getElementById('modalBackdrop').classList.remove('open');
  document.body.style.overflow = '';
}}

// Event listeners
document.getElementById('modalClose').addEventListener('click', closeModal);
document.getElementById('modalBackdrop').addEventListener('click', e => {{
  if (e.target === document.getElementById('modalBackdrop')) closeModal();
}});
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});

document.querySelectorAll('.filter-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentFilter = btn.dataset.filter;
    renderGrid();
  }});
}});

document.getElementById('sortSelect').addEventListener('change', e => {{
  currentSort = e.target.value;
  renderGrid();
}});

// Initial render
renderGrid();

// Theme toggle
(function() {{
  const toggle = document.getElementById('themeToggle');
  const KEY = 'ichimoku-theme';
  function applyTheme(theme) {{
    document.documentElement.setAttribute('data-theme', theme);
    toggle.innerHTML = theme === 'light' ? '&#9728;' : '&#9790;';
    toggle.title = theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode';
  }}
  const saved = localStorage.getItem(KEY) || 'dark';
  if (saved === 'light') applyTheme('light');
  toggle.addEventListener('click', function() {{
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'light' ? 'dark' : 'light';
    applyTheme(next);
    localStorage.setItem(KEY, next);
  }});
}})();
</script>
</body>
</html>"""

    filepath = os.path.join(output_folder, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)

    return filepath


def generate_index_page(market_summaries, output_folder):
    """
    Generate an index.html page linking to all market dashboards.

    Parameters:
    -----------
    market_summaries : list
        List of dicts with keys: market_key, market_name, total, buy, moderate, wait, avoid, dashboard_file
    output_folder : str
        Base output folder
    """
    scan_date = datetime.now().strftime('%Y-%m-%d')
    total_stocks = sum(m['total'] for m in market_summaries)

    cards_html = ''
    for m in market_summaries:
        cards_html += f"""
        <a href="{m['market_key']}/{m['dashboard_file']}" class="market-card">
          <h2>{m['market_name']}</h2>
          <div class="market-count">{m['total']} stocks</div>
          <div class="market-pills">
            <span class="mp mp-buy">{m['buy']} BUY</span>
            <span class="mp mp-mod">{m['moderate']} MOD</span>
            <span class="mp mp-wait">{m['wait']} WAIT</span>
            <span class="mp mp-avoid">{m['avoid']} AVOID</span>
          </div>
        </a>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ichimoku Cloud Scanner</title>
<style>
:root {{
  --bg: #0d1117; --bg2: #161b22; --bg3: #1c2333;
  --border: #30363d; --text: #e6edf3; --text2: #8b949e; --text3: #6e7681;
  --blue: #58a6ff; --green: #00ff88; --amber: #ffaa00; --yellow: #f0e040; --red: #ff4757;
  --font: 'Segoe UI', -apple-system, sans-serif; --mono: 'SF Mono', 'Consolas', monospace;
}}
[data-theme="light"] {{
  --bg: #ffffff; --bg2: #f6f8fa; --bg3: #eef1f5;
  --border: #d0d7de; --text: #1f2328; --text2: #57606a; --text3: #8b949e;
  --blue: #0969da; --green: #1a7f37; --amber: #bf8700; --yellow: #7d6e00; --red: #cf222e;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:var(--font); min-height:100vh; display:flex; flex-direction:column; align-items:center; padding:60px 24px; }}
h1 {{ font-size:32px; font-weight:700; margin-bottom:6px; }}
h1 span {{ color:var(--blue); }}
.subtitle {{ color:var(--text2); font-size:14px; margin-bottom:48px; }}
.markets {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:20px; width:100%; max-width:900px; }}
.market-card {{
  background:var(--bg2); border:1px solid var(--border); border-radius:12px; padding:28px;
  text-decoration:none; color:var(--text); transition:all .2s ease;
}}
.market-card:hover {{ border-color:var(--blue); transform:translateY(-3px); box-shadow:0 8px 32px rgba(0,0,0,.4); }}
.market-card h2 {{ font-size:20px; margin-bottom:4px; }}
.market-count {{ color:var(--text2); font-size:14px; margin-bottom:16px; }}
.market-pills {{ display:flex; gap:8px; flex-wrap:wrap; }}
.mp {{ padding:4px 10px; border-radius:12px; font-size:12px; font-weight:600; font-family:var(--mono); }}
.mp-buy {{ background:#00ff8818; color:var(--green); }}
.mp-mod {{ background:#ffaa0018; color:var(--amber); }}
.mp-wait {{ background:#f0e04018; color:var(--yellow); }}
.mp-avoid {{ background:#ff475718; color:var(--red); }}
.footer {{ margin-top:60px; color:var(--text3); font-size:12px; }}
.theme-toggle {{
  position:fixed; top:20px; right:20px;
  width:40px; height:40px; border-radius:50%;
  border:1px solid var(--border); background:var(--bg2); color:var(--text2);
  cursor:pointer; font-size:18px; display:flex; align-items:center; justify-content:center;
  transition:all .2s ease; z-index:100;
}}
.theme-toggle:hover {{ border-color:var(--text2); color:var(--text); }}
[data-theme="light"] .market-card:hover {{ box-shadow:0 8px 32px rgba(0,0,0,.12); }}
</style>
</head>
<body>
  <button class="theme-toggle" id="themeToggle" title="Toggle light/dark mode" aria-label="Toggle theme">&#9790;</button>
  <h1><span>Ichimoku</span> Cloud Scanner</h1>
  <div class="subtitle">Scan date: {scan_date} &middot; {total_stocks} stocks across {len(market_summaries)} markets</div>
  <div class="markets">{cards_html}</div>
  <div class="footer">Generated by Ichimoku Cloud Scanner</div>
<script>
(function() {{
  var toggle = document.getElementById('themeToggle');
  var KEY = 'ichimoku-theme';
  function applyTheme(theme) {{
    document.documentElement.setAttribute('data-theme', theme);
    toggle.innerHTML = theme === 'light' ? '&#9728;' : '&#9790;';
    toggle.title = theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode';
  }}
  var saved = localStorage.getItem(KEY) || 'dark';
  if (saved === 'light') applyTheme('light');
  toggle.addEventListener('click', function() {{
    var current = document.documentElement.getAttribute('data-theme');
    var next = current === 'light' ? 'dark' : 'light';
    applyTheme(next);
    localStorage.setItem(KEY, next);
  }});
}})();
</script>
</body>
</html>"""

    filepath = os.path.join(output_folder, 'index.html')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)

    return filepath


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

        # Analyze Ichimoku signals (must run before chart to provide trade targets)
        analysis = analyze_ichimoku_signals(df, ticker, name)

        # Create and save Ichimoku chart (with trade target lines)
        if save_chart:
            plot_ichimoku(df, ticker, name, output_folder=charts_folder, analysis=analysis)

        print(f"OK ({analysis['recommendation']}, {analysis['confidence_score']}% conf)")
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
    market_summaries = []

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
            dashboard_filename = f"ichimoku_dashboard_{market_key}.html"
            generate_report(analyses, filename=report_filename, output_folder=market_output_folder,
                          market_name=market_name, currency=currency)
            # Generate PDF report with charts
            generate_pdf_report(analyses, charts_folder, market_output_folder,
                              filename=pdf_filename, market_name=market_name, currency=currency)
            # Generate HTML dashboard
            generate_html_dashboard(analyses, charts_folder, market_output_folder,
                                   filename=dashboard_filename, market_name=market_name, currency=currency)
            print("OK")

            # Collect summary for index page
            rec_counts = {}
            for a in analyses:
                rec = a['recommendation']
                rec_counts[rec] = rec_counts.get(rec, 0) + 1
            market_summaries.append({
                'market_key': market_key,
                'market_name': market_name,
                'total': len(analyses),
                'buy': rec_counts.get('BUY', 0),
                'moderate': rec_counts.get('BUY (MODERATE)', 0),
                'wait': rec_counts.get('WAIT', 0),
                'avoid': rec_counts.get('AVOID', 0),
                'dashboard_file': dashboard_filename
            })

    # Generate index page linking all market dashboards
    if market_summaries:
        generate_index_page(market_summaries, base_output_folder)

    print(f"\nAll markets processed. Output in: {base_output_folder}/")

if __name__ == "__main__":
    main()
