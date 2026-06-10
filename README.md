# swarm-alpha-notebooks

Colab-first strategy research notebooks for the Swarm Alpha Factory pipeline.

## Architecture

```
Local (Alpha Factory)              Colab (this repo)
┌──────────────────┐              ┌──────────────────┐
│ S1: Ingest       │              │ Strategy notebook │
│ S2: Decompose    │──── spec ───▶│ (vectorbt + ta)  │
│ S3: Fabricate    │              │                  │
│                  │              │ QC gate check    │
│ S5: Vault Writer │◀── JSON ────│ Export results   │
│ S6: Digest       │              └──────────────────┘
└──────────────────┘
```

**Local pipeline** handles paper ingestion, decomposition, and vault writing (lightweight deps).
**Colab notebooks** handle backtesting with heavy quant libraries (vectorbt, pandas-ta, scipy).

Results flow back as JSON files that Station 5 (obsidian_writer) ingests into the Obsidian vault.

## Quick Start

1. Open any notebook in `strategies/` via Colab badge
2. Run the setup cell (installs deps)
3. Execute the strategy
4. Download the QC results JSON
5. Drop into `12-Alpha-Factory/data/results/` for Station 5 pickup

## Repo Structure

```
swarm-alpha-notebooks/
├── README.md
├── requirements.txt
├── lib/
│   ├── qc_gate.py          # QC thresholds mirrored from Alpha Factory
│   ├── backtest_utils.py    # Common backtest helpers
│   └── ehlers_dsp.py        # Custom Ehlers/MESA indicators
├── strategies/
│   ├── vwap_mean_reversion.ipynb
│   └── mesa_adaptive.ipynb  (coming soon)
└── results/                 # Exported QC JSONs (gitignored)
```

## QC Gate Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Sharpe IS | > 0.8 | In-sample Sharpe ratio |
| Sharpe OOS | > 0.6 | Out-of-sample Sharpe ratio |
| OOS Degradation | < 35% | Max Sharpe drop IS→OOS |
| Max Drawdown | < 25% | Maximum drawdown |
| Calmar Ratio | > 0.5 | Return / max drawdown |
| Min Trades | > 100 | Minimum trade count |
| Correlation | < 0.4 | Max correlation to existing book |

## Compliance

- All notebooks use **public data only** (yfinance, FRED)
- No proprietary strategy code, positions, P&L, or AUM figures
- Zone-1 compliant: safe to share externally
