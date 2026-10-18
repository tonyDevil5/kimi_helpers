from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from tba_drop_realization_monitor import BANK_DAY,build_coupon

class DropMonitorTests(unittest.TestCase):
    def fixture(self):
        idx=pd.bdate_range('2025-01-02','2025-03-03')
        q=pd.DataFrame({'FN 6 Price':100.,'FN 6 Drop':4.,'FN 6 Perf':1.},index=idx)
        cal=pd.DataFrame({'agency':['FN']*3,'settlement_month':['2025-01','2025-02','2025-03'],
            'front_settlement':pd.to_datetime(['2025-01-14','2025-02-13','2025-03-13']),
            'back_settlement':pd.to_datetime(['2025-02-13','2025-03-13','2025-04-14']),
            'notification_date':pd.to_datetime(['2025-01-10','2025-02-11','2025-03-11'])})
        return q,cal

    def test_roll_creates_no_immediate_profit_and_back_mark(self):
        q,cal=self.fixture();d,e,s,y=build_coupon(q,cal,'FN 6','2025-01-01',10_000_000)
        self.assertEqual(d.loc['2025-01-10','open_unrealized_pnl_ticks'],0)
        self.assertEqual(d.loc['2025-01-10','total_holding_pnl_ticks'],0)
        self.assertEqual(d.loc['2025-01-14','held_contract_mark_points'],99.875)
        self.assertEqual(d.loc['2025-01-14','held_delivery_month'],'2025-02')
        self.assertEqual(d.loc['2025-01-15','held_contract_mark_points'],100.)
        self.assertEqual(d.loc['2025-01-15','quoted_front_delivery_month'],'2025-02')
        self.assertEqual(s['total_holding_pnl_usd'],25000)
        self.assertEqual(s['closed_holding_pnl_usd'],12500)
        self.assertEqual(s['open_holding_pnl_usd'],12500)
        self.assertEqual(s['jpm_hedged_perf_usd'],(len(q)-1)*3125)

    def test_closed_cycle_uses_entry_drop_not_exit_drop(self):
        q,cal=self.fixture();q.loc[q.index>='2025-02-01','FN 6 Drop']=8.
        q=q.loc[:'2025-02-11']
        d,e,s,y=build_coupon(q,cal,'FN 6','2025-01-01',10_000_000)
        feb=e[(e.delivery_month=='2025-02')&e.status.str.startswith('Closed')].iloc[0]
        self.assertEqual(feb.entry_drop_ticks,4)
        self.assertEqual(feb.purchase_price_points,99.875)
        self.assertEqual(feb.sale_or_mark_price_points,100)
        self.assertEqual(feb.holding_pnl_ticks,4)
        self.assertEqual(s['total_holding_pnl_usd'],12500)
        self.assertEqual(s['quoted_drop_total_usd'],37500)
        self.assertEqual(s['open_holding_pnl_usd'],0)

    def test_pure_contract_switch_does_not_create_pnl(self):
        q,cal=self.fixture()
        q.loc[q.index>'2025-01-14','FN 6 Price']=99.875
        q.loc[q.index>'2025-02-13','FN 6 Price']=99.75
        d,e,s,y=build_coupon(q,cal,'FN 6','2025-01-01',10_000_000)
        np.testing.assert_allclose(d.total_holding_pnl_ticks,0)
        self.assertEqual(s['quoted_drop_total_ticks'],8)
        self.assertEqual(s['total_holding_pnl_usd'],0)

    def test_missing_entry_quote_not_filled(self):
        q,cal=self.fixture();q.loc['2025-01-10','FN 6 Drop']=np.nan
        d,e,s,y=build_coupon(q,cal,'FN 6','2025-01-01',10_000_000)
        self.assertEqual(s['missing_roll_quotes'],1)
        self.assertTrue(pd.isna(s['total_holding_pnl_usd']))
        self.assertTrue(pd.isna(s['closed_holding_pnl_usd']))

    def test_notional_scaling_no_future_and_cycle_reconciliation(self):
        q,cal=self.fixture();d,e,s,y=build_coupon(q,cal,'FN 6','2025-01-01',10_000_000)
        other,_,double,_=build_coupon(q,cal,'FN 6','2025-01-01',20_000_000)
        self.assertEqual(double['total_holding_pnl_usd'],s['total_holding_pnl_usd']*2)
        self.assertEqual(s['next_notification'],'2025-03-11')
        self.assertAlmostEqual(e.holding_pnl_usd.sum(),s['total_holding_pnl_usd'])
        changed=q.copy();changed.iloc[-1,0]=101.
        newer,_,_,_=build_coupon(changed,cal,'FN 6','2025-01-01',10_000_000)
        np.testing.assert_allclose(d.total_holding_pnl_ticks.iloc[:-1],newer.total_holding_pnl_ticks.iloc[:-1])

    def test_zero_roll_and_yearly_reconciliation(self):
        q,cal=self.fixture();d,e,s,y=build_coupon(q,cal,'FN 6','2025-03-03',10_000_000)
        self.assertEqual(s['rolls'],0);self.assertEqual(s['total_holding_pnl_usd'],0)
        dates=[BANK_DAY.rollforward(m+pd.Timedelta(days=13)) for m in pd.date_range('2025-01-01',periods=14,freq='MS')]
        cal=pd.DataFrame([dict(agency='FN',settlement_month=str(t.to_period('M')),front_settlement=t,
            back_settlement=dates[i+1],notification_date=t-2*BANK_DAY) for i,t in enumerate(dates[:-1])])
        q=pd.DataFrame({'FN 6 Price':100.,'FN 6 Drop':4.,'FN 6 Perf':1.},index=pd.bdate_range('2025-01-02','2026-01-02'))
        d,e,s,y=build_coupon(q,cal,'FN 6','2025-01-01',10_000_000)
        self.assertAlmostEqual(y.total_holding_pnl_usd.sum(),s['total_holding_pnl_usd'])
        self.assertAlmostEqual(y.closed_holding_pnl_usd.sum(),s['closed_holding_pnl_usd'])
        self.assertAlmostEqual(y.change_in_open_pnl_usd.sum(),s['open_holding_pnl_usd'])

    def test_holidays_match_published_2026_class_a_c(self):
        front_a=['01-14','02-12','03-12','04-13','05-13','06-11','07-13','08-13','09-14','10-13','11-12','12-10']
        notice_a=['01-12','02-10','03-10','04-09','05-11','06-09','07-09','08-11','09-10','10-08','11-09','12-08']
        front_c=['01-22','02-19','03-19','04-21','05-20','06-22','07-20','08-20','09-21','10-20','11-19','12-17']
        notice_c=['01-20','02-17','03-17','04-17','05-18','06-17','07-16','08-18','09-17','10-16','11-17','12-15']
        for front,notice in zip(front_a+front_c,notice_a+notice_c):
            self.assertEqual(pd.Timestamp('2026-'+front)-2*BANK_DAY,pd.Timestamp('2026-'+notice))

if __name__=='__main__':unittest.main()
