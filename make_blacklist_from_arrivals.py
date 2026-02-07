import sys
import argparse
import pandas as pd

STATION_NAMES = {
    ('CEL'),
    ('PAL'),
    ('PST'),
    ('TROJ'),
    ('VYS')
}

#Parse arguments
parser = argparse.ArgumentParser()

# ============================
# Input / Output
# ============================
parser.add_argument(
    '--input',
    dest='input_csv',
    type=str,
    help='Input CSV'
)

STATIONS = set()
for stat in STATION_NAMES:
    STATIONS.add((stat, stat + '_P', stat + '_S'))

args = parser.parse_args()

df = pd.read_csv(args.input_csv)

blacklist = []

for row in df.itertuples(index=False):
    time_found = False
    to_report = []

    for stat in STATIONS:
        val = getattr(row, stat[1], None)
        # treat NaN / empty as "not found"
        if pd.notna(val) and val != "":
            time_found = True
        else:
            to_report.append(f"{row.path}/{stat[0]}_{row.type}.mseed")
        
        #CHECK S
        val = getattr(row, stat[2], None)
        if pd.notna(val) and val != "":
            time_found = True

    if time_found:
        blacklist.extend(to_report)

for ms in blacklist:
    print(ms)