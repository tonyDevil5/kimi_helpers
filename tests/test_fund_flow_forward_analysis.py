import unittest
import numpy as np
import pandas as pd
from fund_flow_forward_analysis import FACTORS,CONTROLS,FLOW,forward_targets,available_cohort_flows,walkforward,score


class ForwardFlowTests(unittest.TestCase):
    def test_targets_start_after_monday_not_previous_friday(self):
        dates=pd.to_datetime(['2026-09-25','2026-09-28','2026-10-05'])
        levels=pd.DataFrame({f:[0,100,103] for f in FACTORS},index=dates)
        y=forward_targets(levels,pd.to_datetime(['2026-09-25']))
        self.assertEqual(y.CT10.iloc[0],3)
        self.assertEqual(y.actual_start.iloc[0],pd.Timestamp('2026-09-28'))

    def test_holiday_rolls_forward_never_back_to_friday(self):
        dates=pd.to_datetime(['2026-09-25','2026-09-29','2026-10-05'])
        levels=pd.DataFrame({f:[0,100,103] for f in FACTORS},index=dates)
        y=forward_targets(levels,pd.to_datetime(['2026-09-25']))
        self.assertEqual(y.actual_start.iloc[0],pd.Timestamp('2026-09-29'))
        self.assertEqual(y.CT10.iloc[0],3)

    def test_reporting_screen_ignores_future_coverage(self):
        weeks=pd.date_range('2024-01-05',periods=30,freq='W-FRI')
        weekly=pd.DataFrame({'A':2.,'P':3.},index=weeks)
        cohort=pd.DataFrame({'ticker':['A','P'],'group':['Active','Passive'],'included':[True,True]})
        a,counts,_=available_cohort_flows(cohort,weekly,weeks[15])
        changed=weekly.copy();changed.loc[weeks[16]:,'A']=np.nan
        b,other,_=available_cohort_flows(cohort,changed,weeks[15])
        self.assertEqual(counts,other)
        pd.testing.assert_frame_equal(a.loc[:weeks[15]],b.loc[:weeks[15]])

    def test_training_timing_and_future_target_invariance(self):
        rng=np.random.default_rng(10)
        weeks=pd.date_range('2024-01-05',periods=82,freq='W-FRI')
        flow=pd.DataFrame(rng.normal(size=(82,2)),index=weeks,columns=FLOW)
        x=pd.DataFrame(rng.normal(size=(82,5)),index=weeks,columns=CONTROLS)
        y=pd.DataFrame({f:rng.normal(size=82) for f in FACTORS},index=weeks)
        y['scheduled_start']=weeks+pd.Timedelta(days=3);y['scheduled_end']=weeks+pd.Timedelta(days=10)
        y['actual_start']=y.scheduled_start;y['actual_end']=y.scheduled_end
        logs,_=walkforward(flow,x,y)
        self.assertTrue((logs.training_last_target_end<logs.forecast_start).all())
        self.assertTrue((logs.training_obs>=52).all())
        changed=y.copy();changed.loc[weeks[-1],FACTORS]=99999
        again,_=walkforward(flow,x,changed)
        np.testing.assert_allclose(logs.predicted_bps,again.predicted_bps)
        np.testing.assert_allclose(logs.baseline_bps,again.baseline_bps)

    def test_incremental_error_metric(self):
        d=pd.DataFrame({'factor':['CT10']*3,'extra_flow_lag_weeks':[0]*3,
                        'actual_bps':[1.,-2.,3.],'predicted_bps':[1.,-2.,3.],
                        'baseline_bps':[0.,0.,0.]})
        m=score(d).iloc[0]
        self.assertEqual(m.improvement_vs_baseline_pct,100)
        self.assertEqual(m.directional_hit_pct,100)


if __name__=='__main__':unittest.main()
