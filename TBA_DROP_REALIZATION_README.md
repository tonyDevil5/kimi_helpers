# TBA same-contract holding and dollar-roll monitor

Run `python tba_drop_realization_monitor.py` in mercury_env.
Interactive: tba_drop_realization_output/tba_drop_realization_monitor.html.
Defaults: hold FN/G2 5.5/6/6.5 from 2025-01-01, $10mm face per coupon.
Override --start, --notional, --workbook and --calendar.

User-authorized mapping: the workbook Price is the front delivery contract
through settlement day inclusive, then switches to the next delivery contract.
Drop is front minus back for the same pair. Use this assumption explicitly;
provider timing has not been independently established from a price jump.

Start with the first available front-contract close. On each notification day,
sell that same held contract at the front quote, realizing sale minus its purchase
cost, then buy the next delivery contract at front - current drop/32. The new
position is marked at that same back price, so rolling creates no immediate P&L.
Before the outgoing front settles, mark the held back at current front minus
current drop/32. After settlement, mark it directly at the front quote.

Completed cycle = next notification front sale minus prior notification inferred
back purchase. Its decomposition is front-to-front price change plus the ENTRY
DROP. The exit-date drop belongs to the next purchase. Keep the initial front
holding stub separate, with zero entry drop. The latest cycle remains open and
uses the held-contract mark, including a back-month adjustment where needed.

Total holding P&L = cumulative closed-cycle P&L + open marked P&L. Independent
check: generic front change + sum of all roll-date drops - current front-to-held
mark adjustment. The sum of drops is quote reference, not realized income.
Yearly total = closed-cycle P&L booked that year + change in open marked P&L.
Yearly, daily and cycle totals reconcile. JPM hedged performance remains separate,
with no drops added and no inclusion of the initial date's preceding daily move.

ClassA (FN) and ClassC (G2) come from tba_settle_dates.xlsx. Notification is two
US bank business days before settlement with USFederalHolidayCalendar. All 24
2026 Class A/C dates match the published SIFMA notification calendar. Missing
exact roll quotes are not filled or moved to another date; unknown entry/exit
prices propagate to the cumulative holding result. Calendar must cover held
contracts. Unrelated workbook fields do not remove valid quotes.

Gross modeled quote execution only: no funding, coupon income, accrued interest,
paydowns/amortization, reinvestment, bid/ask or fees. Constant face notional.
No Citi filters apply. Prices are points; Drop/Perf are already ticks. The monitor
reads only the user workbooks and never changes them.

Outputs: six-coupon summary, yearly results, same-contract cycle ledger in
roll_events.csv, daily states/marks/P&L, per-coupon roll_trades.csv with entry
front/back prices, held-contract versus generic-front charts, drop quotes, and
separate JPM charts. This replaces the old incorrectly timed monthly drop table.

Tests: python tests/test_tba_drop_realization.py
Dashboard: node tests/check_tba_drop_monitor.cjs
Sources:
https://www.sifma.org/resources/general/mbs-notification-and-settlement-dates
https://www.newyorkfed.org/markets/ambs-treasury-faq-05012024
