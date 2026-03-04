#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from obspy import read  # type: ignore
    from obspy.core.utcdatetime import UTCDateTime  # type: ignore
except ImportError as e:
    raise SystemExit(
        "ObsPy is required to read MiniSEED times.\n"
        "Install it with: pip install obspy\n"
    ) from e


# ----------------------------
# Parsing the input pick file
# ----------------------------

PICK_LINE_RE = re.compile(
    r"""
    ^
    (?P<station>\S+)          # station code (e.g. TROJ)
    \s+
    (?:\?\s+){3}              # three '?' fields
    (?P<phase>[PS])           # P or S
    \s+
    \?                        # another '?'
    \s+
    (?P<date>\d{8})           # YYYYMMDD
    \s+
    (?P<hhmm>\d{4})           # HHMM
    \s+
    (?P<sec>[0-9]+(?:\.[0-9]+)?)  # seconds as float
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Pick:
    station: str
    phase: str  # "P" or "S"
    time: UTCDateTime  # absolute pick time


@dataclass
class Event:
    header_line: str
    picks: List[Pick]


def parse_pick_time(date_yyyymmdd: str, hhmm: str, sec_str: str) -> UTCDateTime:
    year = int(date_yyyymmdd[0:4])
    month = int(date_yyyymmdd[4:6])
    day = int(date_yyyymmdd[6:8])

    hour = int(hhmm[0:2])
    minute = int(hhmm[2:4])

    sec_float = float(sec_str)
    whole_sec = int(sec_float)
    frac = sec_float - whole_sec
    microsecond = int(round(frac * 1_000_000))

    # normalize microsecond rounding overflow (e.g., 59.9999996)
    base = UTCDateTime(year, month, day, hour, minute, whole_sec)
    return base + (microsecond / 1_000_000.0)


def parse_events(pick_file: Path) -> List[Event]:
    events: List[Event] = []
    current: Optional[Event] = None

    for raw in pick_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("#EQVENT"):
            if current is not None and current.picks:
                events.append(current)
            current = Event(header_line=line, picks=[])
            continue

        if current is None:
            # ignore lines before first EQVENT
            continue

        m = PICK_LINE_RE.match(line)
        if not m:
            # line doesn't match expected pick shape; ignore safely
            continue

        station = m.group("station")
        phase = m.group("phase")
        t = parse_pick_time(m.group("date"), m.group("hhmm"), m.group("sec"))
        current.picks.append(Pick(station=station, phase=phase, time=t))

    if current is not None and current.picks:
        events.append(current)

    return events


# ----------------------------
# Finding the matching mseed folder
# ----------------------------

def month_folder_name(year: int, month: int) -> str:
    # e.g. month=11 -> "11_96", month=3 -> "03_96"
    return f"{month:02d}_{year % 100:02d}"


def read_any_mseed_timerange(folder: Path) -> Optional[Tuple[UTCDateTime, UTCDateTime]]:
    # pick any *.mseed in this folder (non-recursive)
    mseed_files = sorted(folder.glob("*.mseed"))
    if not mseed_files:
        return None

    try:
        st = read(str(mseed_files[0]), headonly=True)
        start = min(tr.stats.starttime for tr in st)
        end = max(tr.stats.endtime for tr in st)
        return start, end
    except Exception:
        return None


def all_picks_in_range(picks: List[Pick], start: UTCDateTime, end: UTCDateTime) -> bool:
    # inclusive check with a tiny tolerance
    tol = 1e-6
    for p in picks:
        if p.time < (start - tol) or p.time > (end + tol):
            return False
    return True


def infer_type_suffix(folder: Path) -> str:
    """
    Determine event 'type' from filenames like XXXX_T.mseed, XXXX_E.mseed, XXXX_M.mseed.
    Returns "T" / "E" / "M" or "unknown".
    """
    suffixes = set()
    for f in folder.glob("*.mseed"):
        # match "..._<TYPE>.mseed"
        m = re.search(r"_([A-Za-z])\.mseed$", f.name)
        if m:
            suffixes.add(m.group(1).upper())

    for candidate in ("E", "M", "T"):
        if candidate in suffixes:
            return candidate
    return "unknown"


def clean_event_name(folder_name: str) -> str:
    # remove leading 'P' if present and trailing '_decoded' if present
    name = folder_name
    if name.startswith("P"):
        name = name[1:]
    if name.endswith("_decoded"):
        name = name[: -len("_decoded")]
    return name


def find_event_folder(mseed_root: Path, picks: List[Pick]) -> Optional[Tuple[Path, UTCDateTime, UTCDateTime]]:
    if not picks:
        return None

    # Use the event date from the first pick (assumes all picks are same day/time neighborhood)
    t0 = picks[0].time
    year = t0.year
    month = t0.month

    base = mseed_root / str(year) / month_folder_name(year, month)
    if not base.is_dir():
        return None

    # Candidate folders: immediate subfolders (each contains ~5 mseed files)
    for sub in sorted([p for p in base.iterdir() if p.is_dir()]):
        trange = read_any_mseed_timerange(sub)
        if trange is None:
            continue
        start, end = trange
        if all_picks_in_range(picks, start, end):
            return sub, start, end

    return None


# ----------------------------
# CSV writing
# ----------------------------

def build_station_phase_columns(events: List[Event]) -> List[str]:
    stations = sorted({p.station for ev in events for p in ev.picks})
    cols: List[str] = []
    for st in stations:
        cols.append(f"{st}_P")
        cols.append(f"{st}_S")
    return cols


def event_row(
    ev: Event,
    folder: Optional[Path],
    start: Optional[UTCDateTime],
    type_suffix: str,
    station_phase_cols: List[str],
) -> Dict[str, str]:
    row: Dict[str, str] = {}
    row["name"] = clean_event_name(folder.name) if folder else ""
    row["type"] = type_suffix if folder else "unknown"
    row["path"] = str(folder) if folder else ""

    # initialize blanks
    for c in station_phase_cols:
        row[c] = ""

    if folder and start is not None:
        # Fill picks: seconds since starttime
        for p in ev.picks:
            key = f"{p.station}_{p.phase}"
            if key in row:
                dt = float(p.time - start)  # seconds
                row[key] = f"{dt:.6f}"

    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse seismological pick file and match to mseed event folders.")
    ap.add_argument("pick_file", type=Path, help="Input text file with #EQVENT sections and pick lines.")
    ap.add_argument("mseed_root", type=Path, help="Root folder, e.g. mseed/")
    ap.add_argument("out_csv", type=Path, help="Output CSV path.")
    args = ap.parse_args()

    events = parse_events(args.pick_file)
    if not events:
        raise SystemExit("No events found in input file.")

    station_phase_cols = build_station_phase_columns(events)
    fieldnames = ["name", "type"] + station_phase_cols + ["path"]

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for ev in events:
            found = find_event_folder(args.mseed_root, ev.picks)
            if found is None:
                folder = None
                start = None
                type_suffix = "unknown"
            else:
                folder, start, end = found
                type_suffix = infer_type_suffix(folder)

            row = event_row(ev, folder, start, type_suffix, station_phase_cols)
            w.writerow(row)

    print(f"Wrote {len(events)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
