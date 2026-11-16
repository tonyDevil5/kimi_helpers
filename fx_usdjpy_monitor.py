r"""
USD/JPY Forward Monitor - thin wrapper around fx_forward_monitor.py.

Run:
    python fx_usdjpy_monitor.py

Output:
    ./fx_usdjpy_output/USDJPY_Forward_Monitor_<latest data date>.html
"""

from fx_forward_monitor import PAIRS, run_pair

if __name__ == "__main__":
    run_pair(PAIRS["USDJPY"])
