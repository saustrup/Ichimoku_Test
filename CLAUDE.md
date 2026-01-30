# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ichimoku Cloud Stock Scanner — a Python tool that downloads historical stock data (via yfinance), calculates all five Ichimoku Cloud indicators, generates candlestick charts, and produces text + PDF trading reports with BUY/WAIT/AVOID recommendations. Follows a long-only trading strategy.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the scanner (processes all configured markets)
python Ichimoku_Cloud_Scanner.py
```

There is no test suite, linter, or build system configured.

## Architecture

**Single-script application** (`Ichimoku_Cloud_Scanner.py`, ~1300 lines) with a functional pipeline:

1. **Config loading** — reads `stocks_config.json` for market definitions and settings
2. **Archive** — moves previous `Output/` to `Archive/run_<timestamp>/` before each run
3. **Per-market processing loop**:
   - Download stock data → Calculate Ichimoku indicators → Generate PNG chart → Analyze signals
4. **Report generation** — text report + PDF report per market, sorted by recommendation strength

### Signal Scoring System

Each stock gets a composite score (-8 to +8) from five components:
- **Kumo** (price vs cloud): ±2 pts
- **TK Cross** (Tenkan vs Kijun): ±1 pt
- **Cloud Color** (Senkou A vs B): ±1 pt
- **Chikou Span** (lagging): ±2/±1/±0.5 pts
- **Kijun Support**: ±1 pt

Score maps to: BUY (strong), BUY MODERATE, WAIT, or AVOID.

Reports track day-over-day signal changes and mark shifted components with asterisks.

### Key Configuration: `stocks_config.json`

Defines markets (Copenhagen, NASDAQ), each with a Yahoo Finance suffix, currency, and stock list. Settings control period, interval, and output toggles. Adding a new market or stock only requires editing this JSON file.

### Output Structure

```
Output/<MarketName>/
  ├── charts/    (PNG candlestick + Ichimoku overlay)
  ├── data/      (CSV price data)
  ├── ichimoku_trading_report_<Market>.txt
  └── ichimoku_report_<Market>.pdf
```

### Key Dependencies

- **yfinance** — stock data download from Yahoo Finance
- **pandas** — data manipulation and indicator calculation
- **matplotlib** — candlestick chart rendering
- **reportlab** — PDF report generation (imported but not in requirements.txt)
