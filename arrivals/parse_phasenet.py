import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    ap = argparse.ArgumentParser(description="TODO")
    ap.add_argument("--input_csv", required=True,type=Path, help="Input csv with picks done by PhaseNet.")
    ap.add_argument("--output_csv", required=True,type=Path, help="Output CSV file.")
    ap.add_argument("--sampling_rate", required=True,type=int, default=125, help="Sampling rate of the input picks. Default 125 Hz.") 
    args = ap.parse_args()

    #Final columns
    COLUMNS = ["name", "type", "CEL_P", "CEL_S", "PAL_P", "PAL_S", "PST_P", "PST_S", "TROJ_P", "TROJ_S", "VYS_P", "VYS_S", "path"]


    # Load CSV
    df = pd.read_csv(args.input_csv)

    # Step 1: find duplicate combinations of phase_type + file_name
    dup_mask = df.duplicated(subset=["phase_type", "file_name"], keep=False)

    # Step 2: get all file_names that are duplicated
    dup_file_names = df.loc[dup_mask, "file_name"].unique()

    # Step 3: remove all rows with those file_names
    events = df[~df["file_name"].isin(dup_file_names)]



    #output 
    db = {}
    for i, r in events.iterrows():
        pnet_filename = r["file_name"]

        # Split path and filename
        path, filename = pnet_filename.rsplit("/", 1)

        # Extract station and type
        station, rest = filename.split("_", 1)
        event_type = rest.rsplit(".mseed", 1)[0]

        #extract name
        name = path.split("/P", 1)[1].split("_decoded", 1)[0]

        if not name in db.keys():
            #Initialize
            db[name] = {}
            for c in COLUMNS:
                db[name][c] = ""
            db[name]["name"] = name
            db[name]["type"] = event_type
            db[name]["path"] = f'mseed/{path}'
        station = station + "_" + r["phase_type"]
        db[name][station] = r["phase_index"] / args.sampling_rate

    csv_list = []

    for e in db:
        #check arrivals time
        OK = True
        for P, S in zip(COLUMNS[2:-1][::2], COLUMNS[2:-1][1::2]):
            if db[e][P] != "" and db[e][S] != "" and float(db[e][P]) >= float(db[e][S]):
                OK = False
                break
        if OK:
            csv_list.append(",".join([str(db[e][c]) for c in COLUMNS]))

        csv_list.sort()
    print(",".join(COLUMNS))
    for r in csv_list:
        print(r)
        

if __name__ == "__main__":
    main()
