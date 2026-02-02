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

**Single-script application** (`Ichimoku_Cloud_Scanner.py`, ~1700 lines) with a functional pipeline:

1. **Config loading** — reads `stocks_config.json` for market definitions and settings
2. **Archive** — moves previous `Output/` to `Archive/run_<YYYYMMDD_HHMMSS>/` before each run
3. **Per-market processing loop**:
   - Download stock data → Calculate Ichimoku indicators → Generate PNG chart → Analyze signals
4. **Report generation** — text report + PDF report per market, sorted by recommendation strength

Individual stock processing is wrapped in try/except so a single ticker failure does not abort the entire run.

### Functions

| Function | Purpose |
|---|---|
| `download_stock()` | Downloads historical data via yfinance, optionally saves CSV |
| `calculate_ichimoku()` | Computes 5 Ichimoku indicators + enhancement columns (volume ratio, cloud thickness, Kijun distance, flat line detection, future Senkou values) |
| `plot_ichimoku()` | Generates 16x9 PNG candlestick charts with Ichimoku overlay + SL/TP target lines (300 DPI) |
| `load_stocks_config()` | Loads and returns `stocks_config.json` |
| `analyze_single_day()` | Analyzes Ichimoku signals for a specific day — returns 5 core + 6 enhancement component values |
| `analyze_ichimoku_signals()` | Main signal analysis: composite score, enhancements, confidence score, trade targets, recommendation |
| `get_recommendation_priority()` | Sorting helper — BUY=1, BUY MODERATE=2, WAIT=3, AVOID=4 |
| `generate_report()` | Text report with summary table and per-stock component breakdown |
| `generate_pdf_report()` | PDF with clickable TOC, color-coded recommendations, embedded charts |
| `process_stock()` | Orchestrates single stock: download → calculate → analyze → chart (analysis before chart for trade target lines) |
| `archive_previous_output()` | Moves `Output/` to `Archive/run_<timestamp>/` |
| `main()` | Entry point: archive → load config → process markets → generate reports |

### Signal Scoring System

**Two-tier architecture**: a core scoring layer (5 components) and an enhancement layer (6 signals).

#### Core Components (composite score -8 to +7)
- **Kumo** (price vs cloud): ±2 pts
- **TK Cross** (Tenkan vs Kijun): ±1 pt
- **Cloud Color** (Senkou A vs B): ±1 pt
- **Chikou Span** (lagging): ±2/±1/±0.5 pts
- **Kijun Support**: ±1 pt

Core score maps to: BUY (strong), BUY MODERATE, WAIT, or AVOID via trend + strength heuristic.

#### Enhancement Signals (separate layer, does not change core score)
- **Volume Confirmation**: ≥1.5x 20-day avg confirms trend (±1), ≤0.5x = weak conviction (-0.5)
- **Kumo Twist**: Senkou A/B crossover signals major trend change (±1)
- **Cloud Thickness**: thin <1% = weak S/R, thick >4% = strong S/R (±1)
- **TK Cross Location**: bullish cross above cloud = strong (+1), inside cloud = weak (0)
- **Kijun Distance**: overextension >8% = risk (-1), >5% = mild (-0.5)
- **Flat Lines**: both Kijun+Tenkan flat = consolidation (-0.5)

#### Confidence Score
Percentage of all 11 signals (5 core + 6 enhancements) agreeing on direction. Labels: HIGH (≥80%), MODERATE (50-79%), LOW (<50%). High confidence can upgrade BUY MODERATE → BUY; low confidence can downgrade BUY → BUY MODERATE.

#### Trade Targets
- **Stop-loss primary**: Kijun-sen (if price above), else cloud bottom
- **Take-profit 1**: 1:1 risk/reward from stop-loss distance
- **Take-profit 2**: 1:2 risk/reward from stop-loss distance
- **Overextension warning**: flagged when price >8% from Kijun

Day-over-day signal change tracking compares all 11 signals against the previous trading day, recording change direction (up/down arrows), previous values, and marking shifted components with asterisks. Changed stocks get yellow highlighting in the PDF.

### Key Configuration: `stocks_config.json`

Defines markets and global settings. Currently configured: **Copenhagen** (40 stocks, `.CO` suffix, DKK) and **NASDAQ** (15 stocks, no suffix, USD).

```json
{
  "markets": {
    "<MarketKey>": {
      "name": "Display Name",
      "suffix": ".CO",
      "currency": "DKK",
      "stocks": [["TICKER", "Company Name"], ...]
    }
  },
  "settings": {
    "period": "1y",
    "interval": "1d",
    "save_csv": true,
    "save_chart": true
  }
}
```

Adding a new market or stock only requires editing this JSON file.

### Output Structure

```
Output/<MarketName>/
  ├── charts/    (PNG candlestick + Ichimoku overlay)
  ├── data/      (CSV price data)
  ├── ichimoku_trading_report_<Market>.txt
  └── ichimoku_report_<Market>.pdf
```

Previous runs are preserved in `Archive/run_<YYYYMMDD_HHMMSS>/` with the same structure.

### Key Dependencies

- **yfinance** — stock data download from Yahoo Finance
- **pandas** — data manipulation and indicator calculation
- **matplotlib** — candlestick chart rendering
- **reportlab** — PDF report generation (imported but **not in requirements.txt** — must be installed separately: `pip install reportlab`)

### Known Issues

- `reportlab` is missing from `requirements.txt` — users must install it manually
- `README.md` references the old script name `download_novo_stock.py` instead of `Ichimoku_Cloud_Scanner.py`
