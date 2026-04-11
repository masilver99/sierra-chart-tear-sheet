# Sierra Chart Tearsheet

A comprehensive, self-contained HTML trade analysis report generated from a Sierra Chart Trade Activity Log export. The tearsheet turns raw fill data into an interactive, single-file report packed with charts and statistics — no server, no database, just open the HTML file in your browser.

Inspired by the [QuantStats](https://github.com/ranaroussi/quantstats) tearsheet for portfolio analysis.

Donations are not wanted for this project.  Send your money to a great charity, instead.

## Usage

### Python

```
python -m tearsheet --input YourExportFileName.txt --output report.html
```

Open `report.html` in any modern browser to explore your results.

### Windows Executable

A standalone EXE for Windows (no Python required) is available in [Releases](https://github.com/masilver99/sierra-chart-tear-sheet/releases).

### Sample Report

A live demo is available here: [sample report](https://masilver99.github.io/sierra-chart-tear-sheet/report.html)

---

## What's in the Tearsheet

### 📊 Charts

| Chart | Description |
|-------|-------------|
| **Equity Curve** | Cumulative gross and net P&L over time with cash-flow adjustments |
| **Fee Drag** | Gross vs. net equity side-by-side to visualise commission impact |
| **Daily Returns** | Bar chart of daily P&L |
| **Drawdown** | Underwater equity curve showing peak-to-trough drawdowns |
| **Rolling Analytics** | 20-trade rolling window: expectancy, win rate, profit factor, Sharpe |
| **Win Rate Over Time** | Rolling win rate plotted trade-by-trade |
| **Daily P&L** | Bar chart coloured green/red by profitable day |
| **Daily P&L Distribution** | Histogram of daily profit/loss values |
| **Trade P&L Waterfall** | Cumulative waterfall chart of every trade outcome |
| **Trade P&L Distribution** | Histogram of individual trade P&L |
| **Winners vs Losers Distribution** | Overlaid histograms comparing winner and loser sizes |
| **MFE vs MAE** | Scatter plot of Maximum Favourable vs Maximum Adverse Excursion |
| **Time in Trade vs P&L** | Duration scatter coloured by outcome (std-dev bands) |
| **Timing Heatmap** | P&L heat map by day-of-week × entry hour |
| **Trade Mix** | Pie charts: direction mix (long/short), session mix, outcome mix |
| **R-Multiple Distribution** | Histogram of R-multiples with 1R and 2R markers |
| **Exit Type Analysis** | P&L breakdown by exit type (target, stop, manual) |
| **P&L Calendar** | Interactive calendar — click any day to drill into individual trades |
| **Monte Carlo Simulation** | Bootstrap fan chart (p5–p95) with ruin-probability estimate |
| **Benchmark Comparison** | Equity curve vs S&P 500 (SPY) total return |
| **Streak Distribution** | Bar charts of consecutive-winner and consecutive-loser run lengths |

### 📋 Statistics & Metrics

#### Performance
| Metric | Description |
|--------|-------------|
| Win Rate | % of trades with positive gross P&L |
| Profit Factor | Gross winners ÷ gross losers |
| Expectancy | Average gross P&L per trade |
| SQN | System Quality Number (Van Tharp) |
| Payoff Ratio | Average win ÷ average loss |
| Gain-to-Pain Ratio | Total gross profit ÷ sum of losses |
| Biggest Winner / Loser | Largest single-trade P&L in each direction |

#### Risk & Drawdown
| Metric | Description |
|--------|-------------|
| Max Drawdown ($) | Largest peak-to-trough dollar drop |
| Max Drawdown (%) | Same, expressed as % of peak equity |
| Ulcer Index | RMS of drawdown depth (penalises prolonged drawdowns) |
| Calmar Ratio | Annualised net return ÷ max drawdown |
| Sterling Ratio | Annualised return ÷ (max drawdown % + 10% buffer) |
| Recovery Factor | Total net P&L ÷ max drawdown |
| V2 Ratio | Annualised return ÷ (Ulcer Index + 1) |
| CVaR 95% | Average loss in the worst 5% of days |

#### Risk-Adjusted Returns
| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Annualised mean daily return ÷ daily return std dev |
| Sortino Ratio | Sharpe using only downside deviation |
| Omega Ratio | Probability-weighted gains ÷ losses |
| Upside Potential Ratio | Mean upside ÷ downside deviation |

#### Trade Dynamics
| Metric | Description |
|--------|-------------|
| Avg / Max Trade Duration | Hold time statistics, split for winners vs losers |
| MFE Capture % | How much of the maximum run-up was captured |
| Avg MFE / MAE | Average maximum favourable/adverse excursion |
| MFE/MAE Quality Ratio | > 1 means average run-up exceeds average drawdown |
| Avg MAE (Winners) | Adversity that winning trades survived before closing |
| Long / Short Win Rate | Win rates broken out by direction |
| Max Consecutive Wins/Losses | Longest winning and losing streaks |

#### Edge Quality & Position Sizing
| Metric | Description |
|--------|-------------|
| Kelly Criterion | Optimal fraction of capital to risk per trade |
| Breakeven Win Rate | Win rate needed for zero expectancy at current payoff ratio |
| R-Multiple Stats | Avg R, median R, % ≥ 1R, % ≥ 2R, total R |
| Concentration Ratio | % of gross profit from the top-5 winning trades |

#### Calendar Stats
| Metric | Description |
|--------|-------------|
| Trading Days | Total number of days with at least one trade |
| % Profitable Days / Weeks / Months | Fraction of periods with positive P&L |
| Avg Trades per Day | Average daily trade frequency |
| Avg Daily Net P&L | Average net profit per trading day |
| Max Winning / Losing Day | Best and worst single-day P&L |

#### Execution Quality
| Metric | Description |
|--------|-------------|
| Entry / Exit Chase Points | Average and max limit-order fill slippage |
| Fill Rate / Cancel Rate | Order outcome statistics |
| Modify Rate | % of orders that were modified before fill |
| Avg Time to Fill | Entry and exit order fill speed (seconds) |
| Target / Stop / Manual Exit % | Breakdown of how trades were closed |

### 📅 Segmentation Breakdowns

Performance statistics are broken out across multiple dimensions:

- **Direction** — long vs. short
- **Instrument** — per traded symbol
- **Session** — open (before 10:30), midday (10:30–14:00), close (after 14:00)
- **Config Tag** — Sierra Chart trade note / strategy tag
- **Exit Type** — target, stop, manual
- **Day of Week** — Monday through Friday
- **Entry Hour** — by hour of day
- **Weekly / Monthly P&L** — period summaries

### 🗓 Period Summary

A collapsible hierarchy table (Year → Quarter → Month → Week → Day) shows net P&L, trade count, win rate, profit factor, fees, and an estimated tax provision for every period.

### 📑 SC Trade Statistics

A replica of Sierra Chart's built-in Trade Statistics window, so you can cross-check figures directly against the platform output.

### 📜 Trade Log

A full, searchable trade-by-trade log with entry/exit times, P&L, MFE, MAE, fees, exit type, and R-multiple. Linked to the P&L Calendar for drill-down by day.

