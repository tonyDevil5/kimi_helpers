import unittest
import numpy as np
import pandas as pd

import dm_govy_report as dg
import ust_technical_monitor as tech
from ust_technical_dm import build_payload, section_html


class TechnicalDMTests(unittest.TestCase):
    def fixture(self):
        index=pd.bdate_range('2024-01-01',periods=320)
        base=pd.Series(4+np.arange(320)*.001+np.sin(np.arange(320)/7)*.04,index=index)
        data={country:{tenor:base+ci*.1+ti*.04 for ti,tenor in enumerate(tech.TENORS)}
            for ci,country in enumerate(['ust','frf','btp','bund','jgb','esp','uk'])}
        funding=pd.DataFrame({ccy:2.5+np.arange(320)*.0005 for ccy in ['usd','eur','jpy','gbp']},index=index)
        return data,funding

    def test_existing_dm_performance_carry_and_quantiles_match(self):
        data,funding=self.fixture()
        payload=build_payload(data,funding,tech.COUNTRY_NAMES,tech.TENORS)
        expected=dg.build_summary_table(data['ust']['10Y'].to_frame('us_10y'),funding).loc['us_10y']
        row=payload['performance'].query("Country == 'UST (US)' and Tenor == '10Y'").iloc[0]
        for horizon in ['1m','3m','6m','1y']:
            self.assertAlmostEqual(row[f'Chg {horizon} (bp)'],expected[f'chg_{horizon}'])
        self.assertAlmostEqual(row['Yield − funding (bp/yr)'],expected.carry_bps)
        self.assertAlmostEqual(row['Carry / daily vol'],expected.carry_ratio)
        self.assertEqual(row['Momentum score'],expected.mom_score)
        quantile=payload['quantiles']['ust'].query("Tenor == '10Y'").iloc[0]
        self.assertAlmostEqual(quantile['Median (%)'],data['ust']['10Y'].tail(60).median())
        self.assertEqual(len(payload['performance']),42)

    def test_spread_units_and_legacy_change_timing(self):
        data,funding=self.fixture()
        # Nonconstant spread gives useful volatility and band checks.
        data['bund']['10Y']=data['bund']['10Y']+np.sin(np.arange(320)/11)*.1
        payload=build_payload(data,funding,tech.COUNTRY_NAMES,tech.TENORS)
        df=pd.DataFrame({'us_10y':data['ust']['10Y'],'dem_10y':data['bund']['10Y'],
            'frf_10y':data['frf']['10Y'],'esp_10y':data['esp']['10Y']})
        expected=dg.build_spread_summary_table(df).loc['GT10 - Bund10']
        row=payload['spreads'].query("Spread == 'UST − Bund 10Y'").iloc[0]
        self.assertAlmostEqual(row['Level (bp)'],expected['level']*100)
        self.assertAlmostEqual(row['BB high (bp)'],expected.bb_high*100)
        self.assertAlmostEqual(row['Chg 1d (bp)'],expected.chg_1d)
        self.assertAlmostEqual(row['Vol20 (bp/yr)'],expected.rv20)
        self.assertAlmostEqual(row['Percentile (%)'],expected.pct_1y*100)

    def test_missing_yields_do_not_fill_or_remove_other_country_rows_and_funding_is_causal(self):
        data,funding=self.fixture()
        data['ust']['2Y']=data['ust']['2Y'].iloc[:-1]
        yield_end=data['ust']['2Y'].index[-1]
        funding.loc[funding.index[-1],'usd']=99 # after the last available 2Y observation
        data['uk']['7Y']=None
        payload=build_payload(data,funding,tech.COUNTRY_NAMES,tech.TENORS)
        row=payload['performance'].query("Country == 'UST (US)' and Tenor == '2Y'").iloc[0]
        self.assertAlmostEqual(row['Funding (%)'],funding.loc[yield_end,'usd'])
        self.assertEqual(len(payload['performance']),41)
        self.assertEqual(len(payload['quantiles']['uk']),5)

    def test_merged_html_has_unique_dm_views_and_one_set_of_technical_charts(self):
        data,funding=self.fixture()
        payload=build_payload(data,funding,tech.COUNTRY_NAMES,tech.TENORS)
        body=section_html(tech._ReportBuilder(),payload)
        for key in ['performance','spreads','curves','quantiles','heatmap']:
            self.assertEqual(body.count(f'id="dm-{key}"'),1)
        self.assertNotIn('<th>RSI(14)</th>',body.split('id="dm-spreads"')[0])
        self.assertNotIn('plotly.js v',body)
        self.assertEqual(body.count('class="plotly-fig"'),3)
        self.assertEqual(len(tech.TENORS),6)
        self.assertEqual(tech.COUNTRY_TICKERS['uk']['7Y'],'GUKG7 Index')


if __name__=='__main__':unittest.main()
