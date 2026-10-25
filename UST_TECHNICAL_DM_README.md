# Technical analysis and DM government bonds

`python ust_technical_monitor.py` now generates the technical and DM analyses
together in `ust_technical_output/UST_Technical_Monitor_<latest-data-date>.html`.
The combined Treasury report includes this file in **Technical & DM gov bonds**.
`python ust_combined_report.py` refreshes all analyses and packages the result.

Country views cover US, France, Italy, Germany, Japan, Spain and UK at
2Y/5Y/7Y/10Y/20Y/30Y, plus FN/G2 current-coupon OAS. Yield data is downloaded
once and reused by both analyses. The additional funding query uses the four
existing DM tickers for USD/EUR/JPY/GBP.

Shared yield levels, 1/5/20/60-observation changes, RSI, Bollinger indicators,
realized volatility, SMAs and Fibonacci charts remain in the technical views.
The **DM comparison** button adds the complementary views:

- 21/63/126/252-observation yield changes, 5/21/63 momentum votes, funding and carry proxies.
- UST 10Y minus Bund/OAT/Bono spread summaries and histories.
- Latest yield curves across all seven countries.
- Last-60-observation yield quantiles with a country selector.
- A 21-observation yield-change heatmap.

DM horizon definitions and calculations are retained, with these conventions
made explicit in the HTML:

- The technical 1d change ends at the latest observation. Legacy DM spread 1d
  changes end at the previous observation. The other DM horizons end at the latest.
- 1m/3m in DM mean 21/63 observations; technical changes use 20/60.
- Each series uses its available observations; no yields are filled. Spread
  calculations use the paired dates on which both yields were observed.
- Spread levels, bands and quantiles are displayed in bp. The empirical spread
  percentile uses 250 observations; spread charts use 252.
- Yield quantile range position means `(current-min)/(max-min)`, not percentile.
- Funding updates are forward-aligned causally to each yield date. The annualized
  yield-minus-funding proxy divided by the 50-observation daily yield-change std
  retains the legacy DM ratio; it is not a return Sharpe or realized bond carry.

The standalone DM generators remain available for compatibility. They are not
called by the combined-report refresh: the technical monitor builds DM content
directly, avoiding another yield download or a duplicate technical summary.
