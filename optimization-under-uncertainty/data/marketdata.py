"""Price and factor data from sources that need no API key.

Every source here is chosen for one reason: it can be retrieved by anyone,
without an account, so a reader can re-run the fetch and get the same file.
A dataset behind a login is not reproducible evidence.

Stooq  - daily OHLCV per ticker, CSV, no key, no rate limit published
French - the Fama-French research factors, the canonical benchmark series
CoinGecko - free tier market chart, no key (rate limited to a few per minute)
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import date, datetime

from .datakit import Source

STOOQ = "https://stooq.com/q/d/l/?s={sym}&d1={start}&d2={end}&i=d"
FRENCH = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_daily_CSV.zip")
FRENCH_INDUSTRIES = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
                     "ftp/10_Industry_Portfolios_daily_CSV.zip")
COINGECKO = ("https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
             "?vs_currency=usd&days={days}&interval=daily")
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def stooq_source(symbol: str, start: str, end: str, note: str = "") -> Source:
    """Stooq symbols carry a market suffix: spy.us, ^spx, btcusd."""
    return Source(
        name=f"Stooq {symbol}",
        url=STOOQ.format(sym=symbol, start=start.replace("-", ""),
                         end=end.replace("-", "")),
        dest=f"stooq/{symbol.replace('^', '_')}.csv",
        publisher="Stooq", terms="free for personal/research use",
        note=note or f"daily OHLCV {start}..{end}",
    )


def french_source() -> Source:
    return Source(
        name="Fama-French daily research factors", url=FRENCH,
        dest="french/ff_daily.zip", publisher="Kenneth R. French Data Library",
        terms="free for research use, attribution requested",
        note="Mkt-RF, SMB, HML and the daily risk-free rate",
    )


def french_industries_source() -> Source:
    """The 10 Fama-French industry portfolios, daily.

    Ten value-weighted return series -- consumer non-durables, durables,
    manufacturing, energy, tech, telecom, shops, health, utilities and a
    residual -- that differ in their tails. Breadth of that kind is exactly what
    the deterministic/SAA/CVaR/robust comparison needs: with ten sectors that
    behave differently in a crash, a risk-aware allocation has something to earn
    that equal weighting does not.

    Unlike a per-ticker price feed, these are the canonical academic series,
    retrievable by anyone with no account, so the fetch is reproducible.
    """
    return Source(
        name="Fama-French 10 industry portfolios, daily",
        url=FRENCH_INDUSTRIES,
        dest="french/10_industry_daily.zip",
        publisher="Kenneth R. French Data Library",
        terms="free for research use, attribution requested",
        note="daily value-weighted returns for 10 US industry portfolios",
    )


def coingecko_source(coin: str, days: int = 365) -> Source:
    return Source(
        name=f"CoinGecko {coin} daily", url=COINGECKO.format(coin=coin, days=days),
        dest=f"coingecko/{coin}.json", publisher="CoinGecko",
        terms="free API tier, attribution requested",
        note=f"{days} days of daily close, volume and market cap",
    )


def fred_source(series: str, note: str = "") -> Source:
    return Source(
        name=f"FRED {series}", url=FRED.format(series=series),
        dest=f"fred/{series}.csv", publisher="Federal Reserve Bank of St. Louis",
        terms="free redistribution permitted for most series",
        note=note,
    )


# --- parsers ---------------------------------------------------------------

def parse_stooq(raw: bytes) -> tuple:
    """Return (dates, closes) from a Stooq daily CSV, oldest first.

    Stooq answers a bad symbol with the body 'No data' and HTTP 200, so an
    unrecognised ticker looks exactly like a successful request until you read
    the bytes. That check has to happen here.
    """
    text = raw.decode("utf-8", errors="replace").strip()
    if not text or text.lower().startswith("no data") or "<html" in text[:200].lower():
        raise ValueError("Stooq returned no data for this symbol "
                         "(check the market suffix, e.g. 'spy.us' not 'SPY')")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or "Close" not in rows[0]:
        raise ValueError(f"unexpected Stooq columns: {list(rows[0]) if rows else []}")
    dates, closes = [], []
    for r in rows:
        if not r.get("Close") or r["Close"] in ("", "N/A"):
            continue          # halted sessions appear as empty cells
        dates.append(date.fromisoformat(r["Date"]))
        closes.append(float(r["Close"]))
    if len(closes) < 2:
        raise ValueError("Stooq returned fewer than two usable closes")
    return dates, closes


def parse_french(raw: bytes) -> tuple:
    """Return (dates, {factor: [values]}) from the Fama-French daily zip.

    The file has a multi-line preamble, then the daily block, then a second
    annual block after a blank line. Reading past the blank line silently mixes
    annual returns into a daily series, which is the classic way to get a
    Sharpe ratio that is wrong by a factor of sixteen.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        text = z.read(name).decode("utf-8", errors="replace")

    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("mkt-rf") or (
                "Mkt-RF" in ln and "SMB" in ln):
            start = i
            break
    if start is None:
        raise ValueError("could not find the factor header row in the French CSV")

    header = [h.strip() for h in lines[start].split(",")]
    cols = header[1:] if header[0] == "" else header
    dates, data = [], {c: [] for c in cols}
    for ln in lines[start + 1:]:
        s = ln.strip()
        if not s:
            break                      # end of the daily block
        parts = [p.strip() for p in s.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            break                      # the annual block uses 4-digit years
        dates.append(datetime.strptime(parts[0], "%Y%m%d").date())
        for c, v in zip(cols, parts[1:]):
            data[c].append(float(v) / 100.0)   # the file is in percent
    if not dates:
        raise ValueError("no daily rows parsed from the French CSV")
    return dates, data


def parse_french_industries(raw: bytes) -> tuple:
    """Return (dates, {industry: [returns]}) from the 10-industry daily zip.

    The file stacks several blocks: value-weighted daily returns first, then
    equal-weighted daily, then monthly and annual blocks, each behind a blank
    line with its own header. Only the FIRST block -- value-weighted daily -- is
    read; walking past the blank line would splice equal-weighted or annual
    returns onto the same series and quietly corrupt every downstream number.

    Returns are already returns, not prices, and the file quotes them in
    percent, so each value is divided by 100. Missing observations are written
    as -99.99 or -999; a day carrying either in any column is dropped whole,
    because aligning ten industries requires one common calendar.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        text = z.read(name).decode("utf-8", errors="replace")

    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        parts = [p.strip() for p in ln.split(",")]
        # The header row has an empty first cell (the date column) followed by
        # the industry names, and the row beneath it is an 8-digit date.
        if parts and parts[0] == "" and len(parts) >= 4 and all(parts[1:]):
            nxt = lines[i + 1].split(",")[0].strip() if i + 1 < len(lines) else ""
            if nxt.isdigit() and len(nxt) == 8:
                start = i
                break
    if start is None:
        raise ValueError("could not find the industry header row in the "
                         "French CSV")

    cols = [p.strip() for p in lines[start].split(",")][1:]
    dates, data = [], {c: [] for c in cols}
    for ln in lines[start + 1:]:
        s = ln.strip()
        if not s:
            break                      # end of the value-weighted daily block
        parts = [p.strip() for p in s.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            break                      # monthly/annual blocks use short years
        vals = [float(v) for v in parts[1:1 + len(cols)]]
        if any(v <= -99.98 for v in vals):
            continue                   # -99.99 / -999 mark a missing day
        dates.append(datetime.strptime(parts[0], "%Y%m%d").date())
        for c, v in zip(cols, vals):
            data[c].append(v / 100.0)  # the file is in percent
    if not dates:
        raise ValueError("no daily industry rows parsed from the French CSV")
    return dates, data


def parse_coingecko(raw: bytes) -> tuple:
    """Return (dates, closes) from a CoinGecko market_chart response."""
    d = json.loads(raw)
    if "prices" not in d:
        raise ValueError(f"CoinGecko response has no 'prices' key: {list(d)[:5]}")
    dates, closes = [], []
    for ms, px in d["prices"]:
        dates.append(datetime.utcfromtimestamp(ms / 1000).date())
        closes.append(float(px))
    if len(closes) < 2:
        raise ValueError("CoinGecko returned fewer than two prices")
    return dates, closes


def parse_fred(raw: bytes) -> tuple:
    """Return (dates, values) from a FRED CSV, skipping '.' for missing days."""
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8", errors="replace"))))
    if not rows:
        raise ValueError("empty FRED response")
    dates, vals = [], []
    for r in rows[1:]:
        if len(r) < 2 or r[1].strip() in (".", ""):
            continue      # FRED writes '.' on holidays; float('.') raises
        dates.append(date.fromisoformat(r[0].strip()))
        vals.append(float(r[1]))
    if not vals:
        raise ValueError("FRED returned no usable observations")
    return dates, vals


def align(series: dict) -> tuple:
    """Intersect several {name: (dates, values)} series onto common dates.

    Assets trade on different calendars -- crypto every day, equities not --
    and correlating two series of different lengths by position rather than by
    date is a mistake that produces plausible, wrong numbers.
    """
    if not series:
        raise ValueError("nothing to align")
    common = None
    for dates, _ in series.values():
        s = set(dates)
        common = s if common is None else (common & s)
    common = sorted(common)
    if len(common) < 2:
        raise ValueError(
            f"only {len(common)} overlapping dates across "
            f"{len(series)} series -- check the requested date ranges")
    out = {}
    for name, (dates, vals) in series.items():
        idx = {d: v for d, v in zip(dates, vals)}
        out[name] = [idx[d] for d in common]
    return common, out


def to_returns(prices) -> list:
    return [prices[i] / prices[i - 1] - 1.0 for i in range(1, len(prices))]
