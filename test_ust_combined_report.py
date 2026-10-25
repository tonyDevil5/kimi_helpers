import base64
import gzip
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import ust_combined_report as combined


def unpack(value):
    return gzip.decompress(base64.b64decode(value)).decode('utf-8')


class CombinedReportTests(unittest.TestCase):
    def test_reports_keep_scripts_styles_unicode_and_distinct_library_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources = []
            for i, version in enumerate(['same', 'same', 'different']):
                path = Path(tmp)/f'report_{i}_20261022.html'
                source = f'<html><head><script>/* plotly.js v{version} */</script><style>body{{color:red}}</style></head><body>收益<script>const x="$&";</script></body></html>'
                path.write_text(source, encoding='utf-8')
                sources.append((str(i), f'Analysis {i}', path))
            payload = combined.prepare_documents(sources)
            self.assertEqual(len(payload['libraries']), 2)
            for doc, (_, _, path) in zip(payload['documents'], sources):
                restored = unpack(doc['data']).replace(combined.PLOTLY_SLOT, unpack(payload['libraries'][doc['library']]))
                restored = re.sub(r'<base href="[^"]*">', '', restored)
                self.assertEqual(restored, path.read_text(encoding='utf-8'))
                self.assertEqual(doc['source_label'], 'Report dated 2026-10-22')

    def test_embedded_only_navigation_and_missing_library_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'input.html'
            source.write_text('<html><head><script>/* plotly.js vtest */</script></head><body>Chart</body></html>', encoding='utf-8')
            out = combined.build_report([('fair-value', 'Fair value', source)], Path(tmp)/'report.html')
            page = out.read_text(encoding='utf-8')
            self.assertIn('href="#fair-value"', page)
            self.assertNotIn('iframe src=', page)
            payload = json.loads(re.search(r'id="report-data">(.*?)</script>', page, re.S).group(1))
            self.assertIn('Chart', unpack(payload['documents'][0]['data']))
            self.assertIn('frame.srcdoc=content', page)
            source.write_text('<html><body>No library</body></html>', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Expected one embedded Plotly'):
                combined.prepare_documents([('fair-value', 'Fair value', source)])

    def test_refresh_stops_on_source_failure(self):
        with patch.object(combined.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'report')) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                combined.refresh_reports(Path('BackTestingTool'))
            self.assertEqual(run.call_count, 1)

    def test_refresh_includes_swaption_interactive(self):
        with patch.object(combined.subprocess,'run') as run:
            combined.refresh_reports(Path('BackTestingTool'))
        commands=[call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands),4)
        self.assertIn([combined.sys.executable,'-u','swaption_skew_report.py','--format','interactive'],commands)


if __name__ == '__main__':
    unittest.main()
