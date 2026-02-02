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
2. **Archive** — moves previous `Output/` to `Archive/run_<YYYYMMDD_HHMMSS>/` before each run
3. **Per-market processing loop**:
   - Download stock data → Calculate Ichimoku indicators → Generate PNG chart → Analyze signals
4. **Report generation** — text report + PDF report per market, sorted by recommendation strength

Individual stock processing is wrapped in try/except so a single ticker failure does not abort the entire run.

### Functions

| Function | Purpose |
|---|---|
| `download_stock()` | Downloads historical data via yfinance, optionally saves CSV |
| `calculate_ichimoku()` | Computes 5 Ichimoku indicators (Tenkan, Kijun, Senkou A/B, Chikou) |
| `plot_ichimoku()` | Generates 16x9 PNG candlestick charts with Ichimoku overlay (300 DPI) |
| `load_stocks_config()` | Loads and returns `stocks_config.json` |
| `analyze_single_day()` | Analyzes Ichimoku signals for a specific day (used for day-over-day comparison) |
| `analyze_ichimoku_signals()` | Main signal analysis: composite score, component change tracking, recommendation |
| `get_recommendation_priority()` | Sorting helper — BUY=1, BUY MODERATE=2, WAIT=3, AVOID=4 |
| `generate_report()` | Text report with summary table and per-stock component breakdown |
| `generate_pdf_report()` | PDF with clickable TOC, color-coded recommendations, embedded charts |
| `process_stock()` | Orchestrates single stock: download → calculate → chart → analyze |
| `archive_previous_output()` | Moves `Output/` to `Archive/run_<timestamp>/` |
| `main()` | Entry point: archive → load config → process markets → generate reports |

### Signal Scoring System

Each stock gets a composite score (-8 to +8) from five components:
- **Kumo** (price vs cloud): +2 pts
- **TK Cross** (Tenkan vs Kijun): +1 pt
- **Cloud Color** (Senkou A vs B): +1 pt
- **Chikou Span** (lagging): +2/+1/+0.5 pts
- **Kijun Support**: +1 pt

Score maps to: BUY (strong), BUY MODERATE, WAIT, or AVOID. Results are sorted by recommendation priority before reporting.

Day-over-day signal change tracking compares each component against the previous trading day, recording change direction (up/down arrows), previous values, and marking shifted components with asterisks. Changed stocks get yellow highlighting in the PDF.

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
