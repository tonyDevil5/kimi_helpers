import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock
import pandas as pd
import mbs_rv_report as m


OAS='RATES.MBS.MBS_FIXED_RATE.FNMA.MBS_CPN_2_5.30Y.T_OAS - CLOSE'
PERF='RATES.MBS.MBS_PERF.FNMA.30Y.MBS_CPN_2_5.VS_7S_UST.DESK - CLOSE'


def sheet(column, dates, values, header=1):
    return pd.DataFrame([[None,None]]*header+[['Date',column]]+[[pd.Timestamp(d),v] for d,v in zip(dates,values)])


class CitiSheetDetectionTests(unittest.TestCase):
    def fixtures(self):
        return {'Unrelated':sheet('Other',['2026-10-16'],[9]),
                'OAS renamed':sheet(OAS,['2026-10-15','2026-10-16'],[20,21],header=3),
                'Daily renamed':sheet(PERF,['2026-10-15','2026-10-16'],[.1,.2]),
                'Hourly renamed':sheet(PERF,['2026-10-17 04:00','2026-10-17 05:00','2026-10-17 06:00'],[.3,.4,.5])}

    def mocked(self,frames):
        book=MagicMock();book.sheet_names=list(frames);book.__enter__.return_value=book
        def read(_book,sheet_name,header=None,nrows=None):
            raw=frames[sheet_name].copy()
            if header is None:return raw.iloc[:nrows] if nrows else raw
            result=raw.iloc[header+1:].copy();result.columns=raw.iloc[header].tolist()
            return result.reset_index(drop=True)
        return patch.object(m.pd,'ExcelFile',return_value=book),patch.object(m.pd,'read_excel',side_effect=read)

    def test_renamed_reordered_sheets_and_shifted_header(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'book.xlsx';path.touch()
            a,b=self.mocked(self.fixtures())
            with a,b:
                oas=m.load_citi_oas_data(path);daily=m.load_citi_perf_data(path);hourly=m.load_citi_perf_intraday(path)
            self.assertEqual(oas['FN 2.5'].tolist(),[20,21])
            self.assertEqual(daily['FN 2.5'].tolist(),[.1,.2])
            self.assertAlmostEqual(hourly['FN 2.5'].iloc[0],.4)
            self.assertEqual(hourly.attrs['source_sheet'],'Hourly renamed')

    def test_optional_intraday_absent(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'book.xlsx';path.touch()
            frames=self.fixtures();del frames['Hourly renamed']
            a,b=self.mocked(frames)
            with a,b:self.assertTrue(m.load_citi_perf_intraday(path).empty)

    def test_ambiguous_daily_tabs_are_not_silently_selected(self):
        frames=self.fixtures();frames['Daily copy']=frames['Daily renamed'].copy()
        a,b=self.mocked(frames)
        with a,b:
            with self.assertRaisesRegex(ValueError,'Ambiguous Citi performance'):
                m._read_citi_sheet('unused','performance')

    def test_old_name_with_wrong_content_does_not_override_detection(self):
        frames=self.fixtures();frames['Sheet6']=frames.pop('Unrelated')
        a,b=self.mocked(frames)
        with a,b:
            result=m._read_citi_sheet('unused','intraday',preferred='Sheet6')
        self.assertEqual(result.attrs['source_sheet'],'Hourly renamed')


if __name__=='__main__':unittest.main()
