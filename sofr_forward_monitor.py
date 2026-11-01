"""Standalone offline HTML report for the SOFR forward curve calculator."""
import html
import json
from pathlib import Path


def write_monitor(path, forwards, quotes, calibration, metadata):
    """Embed a complete snapshot; no server, CDN or additional packages required."""
    payload = json.dumps(dict(forwards=forwards, quotes=quotes,
                              calibration=calibration, metadata=metadata),
                         allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("sofr_forward_monitor.html").read_text(encoding="utf-8")
    document = template.replace("__SNAPSHOT_JSON__", payload).replace(
        "__ASOF__", html.escape(metadata["asof"]))
    Path(path).write_text(document, encoding="utf-8")
