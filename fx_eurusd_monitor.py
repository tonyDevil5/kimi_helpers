r"""
EUR/USD Forward Monitor - thin wrapper around fx_forward_monitor.py.

EUR is quoted as EURUSD per market convention (the pair is EUR/USD).

Run:
    python fx_eurusd_monitor.py

Output:
    ./fx_eurusd_output/EURUSD_Forward_Monitor_<latest data date>.html
"""

from fx_forward_monitor import PAIRS, run_pair

if __name__ == "__main__":
    run_pair(PAIRS["EURUSD"])
