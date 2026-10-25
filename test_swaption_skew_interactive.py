from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

import swaption_skew_report as skew
from swaption_skew_interactive import build_change_summary, build_interactive_report, summary_table


class SwaptionMonitorTests(unittest.TestCase):
    def fixture(self, n=40):
        index=pd.bdate_range('2026-01-01',periods=n)
        t=np.arange(n,dtype=float)
        df=pd.DataFrame({'10y':4+t*.01},index=index)
        for tenor in skew.OPTION_TENORS:
            df[f'{tenor}_atm']=100+t*2
            df[f'{tenor}_p50']=110+t*3
            df[f'{tenor}_r50']=100+t
        return df

    def test_absolute_normal_vol_and_payer_minus_receiver_changes(self):
        df=self.fixture()
        result=build_change_summary(df,skew.compute_risk_reversals(df))
        self.assertEqual(result['ATM level'].tolist(),[178]*3)
        self.assertEqual(result['Skew level'].tolist(),[88]*3)
        for n in [1,5,20,30]:
            self.assertEqual(result[f'ATM Δ{n}d'].tolist(),[2*n]*3)
            self.assertEqual(result[f'Skew Δ{n}d'].tolist(),[2*n]*3)

    def test_missing_quotes_do_not_shorten_horizon_or_mask_other_series(self):
        df=self.fixture()
        df.loc[df.index[-6],'3m10y_atm']=np.nan
        df.loc[df.index[-1],'1y10y_p50']=np.nan
        df.loc[df.index[-1],'10y']=np.nan
        result=build_change_summary(df,skew.compute_risk_reversals(df)).set_index('Tenor')
        self.assertTrue(pd.isna(result.loc['3m10y','ATM Δ5d']))
        self.assertTrue(pd.isna(result.loc['1y10y','Skew level']))
        self.assertEqual(result.loc['3y10y','ATM Δ5d'],10)
        self.assertIn('Unavailable',summary_table(result.reset_index()))

    def test_short_history_is_unavailable(self):
        df=self.fixture(n=6)
        result=build_change_summary(df,skew.compute_risk_reversals(df))
        self.assertTrue(result['ATM Δ30d'].isna().all())
        self.assertEqual(result['ATM Δ5d'].tolist(),[10]*3)

    def test_loader_uses_required_columns_preserves_date_calendar_rejects_duplicates(self):
        raw=self.fixture().reset_index().rename(columns={'index':'Unnamed: 0','10y':'Unnamed: 1'})
        raw['unrelated missing column']=np.nan
        with patch.object(skew.pd,'read_excel',return_value=raw):
            result=skew.load_data(__file__)
        self.assertEqual(len(result),40)
        self.assertEqual(result['10y'].iloc[-1],4.39)
        duplicate=pd.concat([raw,raw.tail(1)])
        with patch.object(skew.pd,'read_excel',return_value=duplicate):
            with self.assertRaisesRegex(ValueError,'Duplicate dates'):
                skew.load_data(__file__)

    def test_html_preserves_interactive_charts_grouped_changes_and_csv(self):
        df=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output=build_interactive_report(df,skew.compute_risk_reversals(df),Path('source.xlsx'),Path(tmp)/'report.html')
            page=output.read_text(encoding='utf-8')
            self.assertEqual(page.count('Plotly.newPlot('),6)
            self.assertEqual(page.count('plotly.js v'),1)
            self.assertIn('Δ30d',page)
            self.assertIn('ATM normal volatility',page)
            self.assertIn('Skew: payer +50bp',page)
            self.assertIn('Download table CSV',page)
            result=pd.read_csv(output.parent/'swaption_atm_skew_changes.csv')
            self.assertEqual(result['Skew Δ30d'].tolist(),[60]*3)


if __name__=='__main__':unittest.main()
