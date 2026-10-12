"""Standalone Bloomberg indicator; not a valuation-model predictor."""
from pathlib import Path
from datetime import date
import time
import pandas as pd

TICKER = "SD0490FS 1Y1Y BLC Curncy"
START = "2026-08-01"
CACHE = Path(__file__).parent / "adhoc_output" / "cmo_1y1y_indicator.csv"


def refresh():
    import blpapi
    options = blpapi.SessionOptions()
    options.setServerHost("127.0.0.1")
    options.setServerPort(8194)
    options.setConnectTimeout(3000)
    options.setNumStartAttempts(1)
    session = blpapi.Session(options)
    rows = []
    try:
        if not session.start() or not session.openService("//blp/refdata"):
            raise RuntimeError("Bloomberg Desktop API unavailable")
        request = session.getService("//blp/refdata").createRequest("HistoricalDataRequest")
        request.getElement("securities").appendValue(TICKER)
        request.getElement("fields").appendValue("PX_LAST")
        request.set("startDate", START.replace("-", ""))
        request.set("endDate", date.today().strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")
        session.sendRequest(request)
        deadline = time.monotonic()+25
        complete = False
        while time.monotonic() < deadline:
            event = session.nextEvent(500)
            for message in event:
                if message.hasElement("responseError"):
                    raise RuntimeError(str(message.getElement("responseError")))
                if not message.hasElement("securityData"):
                    continue
                security = message.getElement("securityData")
                if security.hasElement("securityError"):
                    raise RuntimeError(str(security.getElement("securityError")))
                if security.hasElement("fieldExceptions") and security.getElement("fieldExceptions").numValues():
                    raise RuntimeError(str(security.getElement("fieldExceptions")))
                fields = security.getElement("fieldData")
                for i in range(fields.numValues()):
                    row = fields.getValueAsElement(i)
                    if row.hasElement("PX_LAST") and not row.getElement("PX_LAST").isNull():
                        rows.append((str(row.getElementAsDatetime("date")), row.getElementAsFloat("PX_LAST")))
            if event.eventType() == blpapi.Event.RESPONSE:
                complete = True
                break
        if not complete:
            raise RuntimeError("Bloomberg history request timed out")
        if not rows:
            raise RuntimeError("Bloomberg returned no PX_LAST observations")
        df = pd.DataFrame(rows, columns=["date", "px_last"])
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.drop_duplicates("date", keep="last").sort_values("date")
        CACHE.parent.mkdir(exist_ok=True)
        df.to_csv(CACHE, index=False)
        print(f"{TICKER}: {len(df)} observations, {df.date.iloc[0]} to {df.date.iloc[-1]}; latest {df.px_last.iloc[-1]}")
    finally:
        session.stop()


def payload():
    out = {"ticker": TICKER, "start": START, "dates": [], "values": []}
    if CACHE.exists():
        df = pd.read_csv(CACHE).dropna(subset=["date", "px_last"])
        df = df.loc[df.date >= START].sort_values("date")
        out.update(dates=df.date.tolist(), values=df.px_last.astype(float).tolist())
    return out


if __name__ == "__main__":
    refresh()
