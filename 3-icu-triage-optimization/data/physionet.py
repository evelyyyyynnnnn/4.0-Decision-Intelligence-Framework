"""PhysioNet's open-access demo databases.

Every dataset referenced here is OPEN ACCESS: no credentialing, no data use
agreement, no login. That is a deliberate constraint. The full MIMIC-IV and
eICU databases require a credentialed PhysioNet account and a signed DUA, and
a pipeline that only runs for people who hold one is not reproducible by a
reader. The demo subsets are the same schema at a hundredth of the size.

The size is the honest limitation, and it is not small: the MIMIC-IV demo holds
roughly 100 patients. That is enough to prove a pipeline parses real clinical
records, computes real features and produces calibrated output. It is not
enough to estimate an AUROC anyone should rely on, and any result from it must
say so rather than quoting a confidence interval built on a hundred people.
"""
from __future__ import annotations

import csv
import gzip
import io

from .datakit import Source

BASE = "https://physionet.org/files/{project}/{version}/{path}"

MIMIC = ("mimic-iv-demo", "2.2")
EICU = ("eicu-crd-demo", "2.0.1")
BIDMC = ("bidmc", "1.0.0")

TERMS = ("PhysioNet open-access; see the project's own licence page. "
         "Open access means no credentialing is required -- it does not mean "
         "the data may be redistributed, so the cache is gitignored.")


def source(project: str, version: str, path: str, note: str = "") -> Source:
    return Source(
        name=f"{project} {path}",
        url=BASE.format(project=project, version=version, path=path),
        dest=f"{project}/{path}",
        publisher=f"PhysioNet ({project} v{version})",
        terms=TERMS, note=note,
    )


def mimic(path: str, note: str = "") -> Source:
    return source(*MIMIC, path, note)


def eicu(path: str, note: str = "") -> Source:
    return source(*EICU, path, note)


def bidmc(path: str, note: str = "") -> Source:
    return source(*BIDMC, path, note)


def read_csv(path_or_bytes) -> list:
    """Read a PhysioNet CSV, transparently handling gzip.

    Nearly every MIMIC file ships gzipped and a few do not, so sniffing the
    magic number is more reliable than trusting the extension.
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        raw = bytes(path_or_bytes)
    else:
        raw = open(path_or_bytes, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def to_float(value, default=None):
    """MIMIC writes empty strings, and occasionally text, into numeric columns."""
    if value is None:
        return default
    s = str(value).strip()
    if not s or s.upper() in ("NULL", "NA", "___"):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def parse_ts(value):
    """MIMIC timestamps are 'YYYY-MM-DD HH:MM:SS', sometimes without seconds."""
    from datetime import datetime
    if not value:
        return None
    s = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None
