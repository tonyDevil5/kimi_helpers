r"""
USD/SGD Forward Monitor - thin wrapper around fx_forward_monitor.py.

Run:
    python fx_usdsgd_monitor.py

Output:
    ./fx_usdsgd_output/USDSGD_Forward_Monitor_<latest data date>.html
"""

from fx_forward_monitor import PAIRS, run_pair

if __name__ == "__main__":
    run_pair(PAIRS["USDSGD"])
