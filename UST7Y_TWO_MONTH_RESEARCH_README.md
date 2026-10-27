# UST 7Y two-month yield-change research

The final two-page research note is
`output/pdf/UST7Y_TwoMonth_Rises_Over60bp.pdf`. It includes the non-overlapping
calendar-period distribution and the five periods with strictly greater than
60 bp yield rises, starting/ending yields, subsequent two-month moves, and
linked primary sources for the market backdrop.

## Definitions and source

- Source: Bloomberg `GT07 Govt`, `PX_LAST`, requested from 2009-01-01.
  The downloaded history starts 2009-02-26; the supplied PDF uses quotes through
  2026-10-26. The final daily quote in an intraday download may not be a final close.
- Initial analysis: daily overlapping 42-observation yield changes, plus an
  exact two-calendar-month check and latest-anchored non-overlapping 42-observation
  periods. This is separate from the final calendar schedule.
- Final schedule: first available quoted weekday of Jan to Mar, Mar to May,
  May to Jul, Jul to Sep, Sep to Nov, Nov to Jan. Consecutive periods share only
  their boundary quote. Yield change is `(end yield % - start yield %) * 100` bp.
- The February-start schedule is a separate sensitivity analysis. The schedules
  overlap with each other and must not be pooled into a non-overlapping sample.
- No yield filling or interpolation. A boundary first quoted after day 7 is
  excluded as incomplete; this excludes February 2009. Missing boundary months
  are not bridged. Incomplete final intervals are excluded.
- The main schedule has 105 completed intervals from March 2009 to September
  2026. The strict `change_bp > 60` filter selects five episodes in the supplied
  PDF. Subsequent changes start at each episode's endpoint and end at the next
  scheduled two-month boundary.
- Market context is historical interpretation from the linked Fed sources,
  not an estimated causal attribution. Removing overlap does not establish
  statistical independence or a predictive trading rule.

## Reproduce

Run from this repository in `mercury_env`, with Bloomberg available:

```powershell
python adhoc_ust7y_2m_change.py
python adhoc_ust7y_calendar_2m.py
```

The first command queries Bloomberg; use `--from-cache` to reuse its existing
download. Both commands write charts, HTML and CSV data under
`adhoc_output/ust7y_2m_change/` (ignored by Git).

To build the PDF in one environment containing NumPy, pandas, Matplotlib and
ReportLab:

```powershell
python adhoc_ust7y_tail_pdf.py
```

On the current desktop, Matplotlib is in `mercury_env` and ReportLab is in the
bundled Python runtime. Build the chart first, then the PDF:

```powershell
python adhoc_ust7y_tail_pdf.py --chart-only
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" adhoc_ust7y_tail_pdf.py --pdf-only
```

The chart intermediate is `tmp/pdfs/ust7y_over60/distribution.png`. The PDF
builder uses Windows Arial fonts. It writes the filtered cases CSV to the
ignored analysis folder and the final PDF to `output/pdf/`.

The PDF builder deliberately asserts the five-case count for this dated note;
review that assertion and update the narrative if reproducing with later data.
The raw Bloomberg CSVs and temporary rendered pages are not included in Git.
