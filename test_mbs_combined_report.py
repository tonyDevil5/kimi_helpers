import base64
import gzip
import json
import re
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import mbs_combined_report as mbs
import ust_combined_report as shared


def decode(value):
    return gzip.decompress(base64.b64decode(value)).decode('utf-8')


class CombinedMBSReportTests(unittest.TestCase):
    def test_selects_mbs_rv_by_date_across_both_filename_conventions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for name in ['MBS_RV_Report_Interactive_20261004.html','MBS_RV_Report_20261018.html']:
                (folder/name).touch()
            self.assertEqual(mbs.latest_mbs_rv(folder).name,'MBS_RV_Report_20261018.html')
            (folder/'MBS_RV_Report_Interactive_20261018.html').touch()
            self.assertEqual(mbs.latest_mbs_rv(folder).name,'MBS_RV_Report_Interactive_20261018.html')

    def test_credit_cdn_is_replaced_with_local_library_without_changing_chart_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'credit.html'
            original='<html><head><script src="https://cdn.plot.ly/plotly-3.1.1.min.js"></script></head><body><script>Plotly.newPlot("chart",[1]);</script></body></html>'
            path.write_text(original,encoding='utf-8')
            data=shared.prepare_documents([('credit','MBS vs credit',path)],inline_plotly_js='/* plotly.js vlocal */')
            doc=data['documents'][0]
            bundled=decode(doc['data']).replace(shared.PLOTLY_SLOT,decode(data['libraries'][doc['library']]))
            self.assertNotIn('cdn.plot.ly',bundled)
            self.assertIn('Plotly.newPlot("chart",[1]);',bundled)
            self.assertTrue(doc['plotly_inlined'])
            self.assertEqual(path.read_text(encoding='utf-8'),original)

    def test_refresh_five_jobs_updates_caches_and_uses_current_monitor_imports(self):
        with patch.object(mbs.subprocess,'run') as run:
            mbs.refresh_reports(Path('BackTestingTool'))
        calls=run.call_args_list
        self.assertEqual(len(calls),5)
        self.assertEqual(calls[0].args[0][-2:],['--format','interactive'])
        self.assertIn('--update',calls[1].args[0])
        self.assertIn('jpm_coupon_mean_reversion.py',calls[2].args[0][-1])
        self.assertIn('--update',calls[3].args[0])
        self.assertEqual(calls[4].args[0][-2:],['strategy_example/fair_value_120_backtest.py','--update'])
        for call in calls:
            self.assertTrue(call.kwargs['check'])
            self.assertTrue(call.kwargs['env']['PYTHONPATH'].startswith(str(mbs.ROOT)))

    def test_refresh_stops_on_failed_report(self):
        with patch.object(mbs.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'report')) as run:
            with self.assertRaises(subprocess.CalledProcessError):mbs.refresh_reports(Path('BackTestingTool'))
            self.assertEqual(run.call_count,1)

    def test_custom_title_and_all_five_navigation_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=[]
            for key in ['mbs-rv','tba-strategies','jpm-mean-reversion','mbs-credit','fair-value-120']:
                path=Path(tmp)/(key+'.html')
                path.write_text('<html><head><script>/* plotly.js vtest */</script></head><body>Interactive content</body></html>',encoding='utf-8')
                sources.append((key,key,path))
            out=shared.build_report(sources,Path(tmp)/'combined.html',title='MBS Research & Strategies')
            page=out.read_text(encoding='utf-8')
            self.assertIn('<title>MBS Research &amp; Strategies</title>',page)
            self.assertNotIn('Treasury Research Monitor',page)
            for key,_,_ in sources:self.assertIn(f'href="#{key}"',page)
            node=shutil.which('node')
            if not node:
                bundled=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
                node=str(bundled) if bundled.exists() else None
            if node:
                script=re.search(r'<script>\s*(\x27use strict\x27;.*?)</script>',page,re.S).group(1)
                script_path=Path(tmp)/'navigation.js';script_path.write_text(script,encoding='utf-8')
                subprocess.run([node,'--check',str(script_path)],check=True,capture_output=True)

    def test_source_manifest_includes_fair_value_120(self):
        with patch.object(mbs,'latest_mbs_rv',return_value=Path('mbs.html')):
            sources=mbs.report_sources(Path('BackTestingTool'))
        self.assertEqual(len(sources),5)
        self.assertEqual(sources[-1][0],'fair-value-120')
        self.assertEqual(sources[-1][2],Path('BackTestingTool/strategy_example/fair_value_120_output/fair_value_120_backtest.html'))


if __name__=='__main__':unittest.main()
