#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

from obspy import read
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


STATIONS = ["CEL", "PAL", "PST", "TROJ", "VYS"]
COMPS = ["Z", "N", "E"]
A4_PORTRAIT_IN = (8.27, 11.69)  # inches


def load_name_list(path: str | None) -> set[str]:
    """
    Text file: one event name per line. Lines starting with # are comments.
    """
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s)
    return out


def station_colors():
    cyc = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cyc:
        cyc = ["C0", "C1", "C2", "C3", "C4", "C5"]
    return {st: cyc[i % len(cyc)] for i, st in enumerate(STATIONS)}


def comp_from_channel(ch: str) -> str:
    ch = (ch or "").strip().upper()
    if ch and ch[-1] in ("Z", "N", "E"):
        return ch[-1]
    return "?"


def list_mseed_files(event_dir: Path) -> list[Path]:
    return sorted(set(list(event_dir.glob("*.mseed")) + list(event_dir.glob("*.MSEED"))))


def station_from_filename(path: Path) -> str:
    # Accept STATION_X.mseed or STATION.mseed -> station is prefix before first "_"
    return path.stem.split("_")[0].strip().upper()


def load_traces_by_station_comp(event_dir: Path) -> dict[tuple[str, str], object]:
    """
    Return (STATION, Z/N/E) -> trace (first match found).
    Reads all mseeds in directory; ignores the filename letter (M/T/E).
    """
    out: dict[tuple[str, str], object] = {}
    for f in list_mseed_files(event_dir):
        stname = station_from_filename(f)
        try:
            st = read(str(f))
        except Exception:
            continue
        for tr in st:
            comp = comp_from_channel(getattr(tr.stats, "channel", ""))
            if comp not in ("Z", "N", "E"):
                continue
            key = (stname, comp)
            if key not in out:
                out[key] = tr
    return out


def event_type_letter(event_dir: Path) -> str:
    """
    Extract directory-level letter M/T/E from any filename like VYS_M.mseed
    (assumed constant within directory). If not found, '?'.
    """
    pat = re.compile(r"^[A-Za-z0-9]+_([A-Za-z])\.(mseed|MSEED)$")
    for f in list_mseed_files(event_dir):
        m = pat.match(f.name)
        if m:
            return m.group(1).upper()
    return "?"


def get_pick(row: pd.Series, station: str, phase: str) -> float | None:
    """
    Return numeric pick time (seconds) or None.
    Columns are like CEL_P, CEL_S, ...
    """
    col = f"{station}_{phase}"
    if col not in row:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    try:
        return float(v)
    except Exception:
        return None

def plot_event_page(pdf: PdfPages, title: str, traces_map: dict[tuple[str, str], object], row: pd.Series):
    colors = station_colors()

    nrows = len(STATIONS) * len(COMPS)  # 15
    fig, axes = plt.subplots(nrows, 1, figsize=A4_PORTRAIT_IN, sharex=False)
    if nrows == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=11)

    idx = 0
    for st in STATIONS:
        p_time = get_pick(row, st, "P")
        s_time = get_pick(row, st, "S")

        for comp in COMPS:
            ax = axes[idx]
            idx += 1

            ax.grid(True, linewidth=0.3)
            ax.tick_params(labelsize=6)
            ax.set_ylabel(f"{st} {comp}", fontsize=7)

            tr = traces_map.get((st, comp))
            if tr is None:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7)
                if ax is not axes[-1]:
                    ax.set_xticklabels([])
                continue

            data = tr.data
            sr = float(tr.stats.sampling_rate)
            npts = int(tr.stats.npts)
            t = np.arange(npts) / sr

            ax.plot(t, data, linewidth=0.5, color=colors.get(st))

            # Draw arrivals ONLY if available, but on ALL 3 comps for that station
            if p_time is not None:
                ax.axvline(p_time, linewidth=1.0, linestyle="--")
                ax.text(p_time, 0.92, "P", transform=ax.get_xaxis_transform(),
                        fontsize=7, va="top")
            if s_time is not None:
                ax.axvline(s_time, linewidth=1.0, linestyle="--")
                ax.text(s_time, 0.92, "S", transform=ax.get_xaxis_transform(),
                        fontsize=7, va="top")

            if ax is not axes[-1]:
                ax.set_xticklabels([])

    axes[-1].set_xlabel("Time from trace start (s)", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Plot events from CSV into one PDF (A4 portrait, 15 stacked traces), with whitelist/blacklist filtering."
    )
    ap.add_argument("--csv", required=True, help="Input CSV with columns: name,type,...,path")
    ap.add_argument("--root", required=True, help="Root folder. Event folder is root/path from CSV.")
    ap.add_argument("--outpdf", default="all_events.pdf", help="Output PDF (default: all_events.pdf)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")



    df = pd.read_csv(args.csv, sep=None, engine="python")

    # Make sure required columns exist
    required_cols = ["name", "type", "path"] + [f"{st}_{ph}" for st in STATIONS for ph in ("P", "S")]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Normalize
    df["name"] = df["name"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.upper()
    df["path"] = df["path"].astype(str).str.strip()

    # Convert pick columns to numeric where possible
    for st in STATIONS:
        for ph in ("P", "S"):
            col = f"{st}_{ph}"
            df[col] = pd.to_numeric(df[col], errors="coerce")

    outpdf = Path(args.outpdf)

    missing_dir = 0

    with PdfPages(outpdf) as pdf:
        for _, row in df.iterrows():
            name = row["name"]
            rel = Path(row["path"])
            event_dir = (root / rel)

            if not event_dir.exists() or not event_dir.is_dir():
                print(f"[SKIP] {name}: folder not found: {event_dir}")
                missing_dir += 1
                continue

            traces_map = load_traces_by_station_comp(event_dir)
            etype_letter = event_type_letter(event_dir)

            title = f"{name}  [{etype_letter}]  ({event_dir})"
            print(f"   {name}, (traces: {len(traces_map)})")
            plot_event_page(pdf, title, traces_map, row)

    print(f"\nDone. Wrote: {outpdf}")
    print(f"Skipped (missing dir): {missing_dir}")


if __name__ == "__main__":
    main()

