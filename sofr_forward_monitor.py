"""Standalone offline HTML report for the SOFR forward curve calculator."""
import html
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs


def basis_zscores(forwards, history, asof, window=250):
    """Latest basis vs trailing valid observations, including the snapshot date."""
    result = {}
    for row in forwards:
        dated = {h['date']: float(h['cme_minus_lch_bps']) for h in history
                 if h['forward'] == row['forward'] and h['date'] <= asof
                 and np.isfinite(float(h['cme_minus_lch_bps']))}
        dates = sorted(dated)[-window:]
        values = [dated[d] for d in dates]
        mean = float(np.mean(values)) if values else None
        std = float(np.std(values, ddof=1)) if len(values) >= 2 else None
        valid = len(values) == window and dates[-1] == asof and std is not None and std > 1e-12
        result[row['forward']] = dict(z=(row['cme_minus_lch_bps']-mean)/std if valid else None,
            count=len(values), mean=mean, std=std, start=dates[0] if dates else None,
            end=dates[-1] if dates else None)
    return result


def write_monitor(path, forwards, quotes, calibration, metadata, history=None):
    """Embed a complete snapshot; no server, CDN or additional packages required."""
    payload = json.dumps(dict(forwards=forwards, quotes=quotes,
                              calibration=calibration, metadata=metadata, history=history or [],
                              basis_zscores=basis_zscores(forwards, history or [], metadata['asof'])),
                         allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("sofr_forward_monitor.html").read_text(encoding="utf-8")
    document = template.replace("__SNAPSHOT_JSON__", payload).replace(
        "__ASOF__", html.escape(metadata["asof"]))
    document = document.replace('__PLOTLY_JS__', get_plotlyjs())
    Path(path).write_text(document, encoding="utf-8")
