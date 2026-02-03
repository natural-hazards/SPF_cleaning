#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from obspy import read


DEFAULT_COMPONENTS = ("Z", "N", "E")


def is_truthy_error_value(x) -> bool:
    """
    Return True if value should count as an 'error present' for selection.

    Selection rules (matches your description):
    - empty / NaN => NOT selected
    - the word 'false' (any case, with surrounding whitespace) => NOT selected
    - anything else => selected (e.g. True, numbers, 'N', 'EZ', etc.)
    """
    if x is None:
        return False

    # pandas NaN handling
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass

    s = str(x).strip()
    if s == "":
        return False
    if s.lower() == "false":
        return False
    return True


def select_rows(df: pd.DataFrame, event_col: str) -> pd.DataFrame:
    """
    Select rows where:
    - event_col is non-empty
    - AND at least one other column has a truthy error value
    """
    if event_col not in df.columns:
        raise ValueError(f"CSV must contain column '{event_col}'")

    other_cols = [c for c in df.columns if c != event_col]
    if not other_cols:
        return df.iloc[0:0].copy()

    def row_selected(row) -> bool:
        ev = row.get(event_col, None)
        if not is_truthy_error_value(ev):
            return False
        for c in other_cols:
            if is_truthy_error_value(row.get(c, None)):
                return True
        return False

    mask = df.apply(row_selected, axis=1)
    return df[mask].copy()


def pick_component_traces(st, components=DEFAULT_COMPONENTS):
    """
    Build a dict {component_letter: Trace or None}.

    Identify component by the *last character* of channel code:
      e.g. BHZ -> 'Z', HHN -> 'N', BHE -> 'E'

    If multiple traces match the same component, pick the longest one.
    """
    comp_map = {c: [] for c in components}
    for tr in st:
        ch = (tr.stats.channel or "").strip()
        if not ch:
            continue
        last = ch[-1].upper()
        if last in comp_map:
            comp_map[last].append(tr)

    picked = {}
    for c in components:
        if not comp_map[c]:
            picked[c] = None
        else:
            picked[c] = max(comp_map[c], key=lambda t: int(getattr(t.stats, "npts", 0)))
    return picked


def row_error_summary(row: pd.Series, event_col: str):
    """
    Return list of (col, value_str) for columns that are truthy error values.
    """
    items = []
    for col, val in row.items():
        if col == event_col:
            continue
        if is_truthy_error_value(val):
            items.append((col, str(val).strip()))
    return items


def plot_one_page(pdf: PdfPages, mseed_path: Path, row: pd.Series, event_col: str):
    """
    Create one PDF page:
    - path at top
    - selected error columns + values at top
    - 3 component plots (Z, N, E) when present
    """
    fig = plt.figure(figsize=(11.69, 8.27))  # A4-ish landscape
    gs = fig.add_gridspec(
        nrows=4, ncols=1,
        height_ratios=[0.9, 1.0, 1.0, 1.0],
        hspace=0.35
    )

    ax_text = fig.add_subplot(gs[0, 0])
    ax_text.axis("off")

    errors = row_error_summary(row, event_col)
    error_lines = []
    if errors:
        error_lines.append("Selected error fields (column = value):")
        for c, v in errors:
            error_lines.append(f"  - {c} = {v}")
    else:
        error_lines.append("Selected (no error fields found?)")

    title = f"MiniSEED: {mseed_path}"
    ax_text.text(0.0, 0.9, title, fontsize=12, fontweight="bold",
                 va="top", ha="left", wrap=True)
    ax_text.text(0.0, 0.55, "\n".join(error_lines), fontsize=10,
                 va="top", ha="left", family="monospace")

    try:
        st = read(str(mseed_path))
    except Exception as e:
        ax_text.text(0.0, 0.15, f"ERROR reading file: {e}",
                     fontsize=10, va="top", ha="left", color="red")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    picked = pick_component_traces(st, components=DEFAULT_COMPONENTS)

    for i, comp in enumerate(DEFAULT_COMPONENTS, start=1):
        ax = fig.add_subplot(gs[i, 0])
        tr = picked.get(comp)

        if tr is None:
            ax.text(0.5, 0.5, f"{comp}: missing", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            continue

        data = tr.data
        npts = int(tr.stats.npts)
        sr = float(tr.stats.sampling_rate)
        t = (pd.RangeIndex(npts) / sr)

        ax.plot(t, data, linewidth=0.6)
        ax.set_title(f"Component {comp}  |  {tr.id}", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Counts")
        ax.grid(True, alpha=0.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def pdf_path_with_suffix(base_path: Path, index: int) -> Path:
    """
    index = 0  -> base_path (e.g., output.pdf)
    index > 0  -> output_XXX.pdf (e.g., output_001.pdf)
    """
    if index == 0:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{index:03d}{base_path.suffix}")


def main():
    ap = argparse.ArgumentParser(
        description="Visualise 3-component MiniSEED files listed in a CSV into multi-page PDF(s)."
    )
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    ap.add_argument("--out", required=True, help="Output PDF path (e.g. output.pdf)")
    ap.add_argument("--event-col", default="event",
                    help="CSV column containing the MiniSEED path (default: event)")
    ap.add_argument("--base-dir", default=".",
                    help="Optional base directory prepended to event paths (default: current directory)")
    ap.add_argument("--max-pages", type=int, default=0,
                    help="Maximum pages per PDF (0 = unlimited)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_pdf = Path(args.out)
    base_dir = Path(args.base_dir)

    df = pd.read_csv(csv_path)

    selected = select_rows(df, event_col=args.event_col)
    if selected.empty:
        print("No rows selected (no truthy error values found). Nothing to draw.")
        return

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    max_pages = int(args.max_pages)
    pdf_index = 0
    page_count = 0

    current_pdf_path = pdf_path_with_suffix(out_pdf, pdf_index)
    pdf = PdfPages(current_pdf_path)

    total_pages = 0
    try:
        for _, row in selected.iterrows():

            # rotate to a new file if page limit reached
            if max_pages > 0 and page_count >= max_pages:
                pdf.close()
                pdf_index += 1
                page_count = 0
                current_pdf_path = pdf_path_with_suffix(out_pdf, pdf_index)
                pdf = PdfPages(current_pdf_path)

            raw_path = str(row[args.event_col]).strip()
            mseed_path = (base_dir / raw_path).resolve()

            plot_one_page(pdf, mseed_path, row, event_col=args.event_col)

            page_count += 1
            total_pages += 1

    finally:
        pdf.close()

    print(f"Selected rows: {len(selected)}")
    print(f"Total pages written: {total_pages}")
    if max_pages > 0:
        print(f"Max pages per PDF: {max_pages}")
        print(f"Created {pdf_index + 1} PDF file(s) starting at: {out_pdf}")


if __name__ == "__main__":
    main()
