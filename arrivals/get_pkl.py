import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from obspy import read, UTCDateTime
import pickle
import math

class Station_arrival:
    def __init__(self):

        self._P_highest_prio = -1
        self._P_time = None

        self._S_highest_prio = -1
        self._S_time = -1

    def update_P_time(self, time, prio):
        if prio >= self._P_highest_prio:
            self._P_highest_prio = prio
            self._P_time = time
            
    def update_S_time(self, time, prio):
        if prio >= self._S_highest_prio:
            self._S_highest_prio = prio
            self._S_time = time

    def get_P_time(self):
        return self._P_time
    
    def get_S_time(self):
        return self._S_time

class event_metadata:
    def __init__(self, e_type):
        self.e_type = e_type
        self.arrivals = {}

    def update(self, st_name, phase_type, time, prio):
        if phase_type not in ['P', 'S']:
            return
        if not st_name in self.arrivals:
            self.arrivals[st_name] = Station_arrival()
        if phase_type == 'P':
            self.arrivals[st_name].update_P_time(time, prio)
        else:
            self.arrivals[st_name].update_S_time(time, prio)

    def get_arrival_time(self, st_name, phase_type):
        if not st_name in self.arrivals:
            return None
        if phase_type == 'P':
            return self.arrivals[st_name].get_P_time()
        if phase_type == 'S':
             return self.arrivals[st_name].get_S_time()
        
    def get_type(self):
        return self.e_type


def parse_source(value: str) -> tuple[str, int]:
    """
    Parse --source argument in the form:
        path/to/file.csv:10
    """
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid --source '{value}'. Expected format: file.csv:priority"
        )

    path, priority_str = value.rsplit(":", 1)

    if not path:
        raise argparse.ArgumentTypeError("CSV path cannot be empty")

    try:
        priority = int(priority_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid priority in '{value}'. Priority must be an integer"
        ) from e

    return path, priority

def main():
    ap = argparse.ArgumentParser(
        description="Prepare "
    )
    ap.add_argument(
        "--years",
        type=int,
        nargs='+',
        default=list(range(1992, 2003)),
        help="Years from which the events are processed. Default 1992 - 2002."
    )
    ap.add_argument(
        "--stations",
        type=str,
        nargs='+',
        default=['CEL', 'PAL', 'PST', 'TROJ', 'VYS'],
        help="List of stations whose data are used, default all."
    )
    ap.add_argument(
        "--types",
        type=str,
        nargs='+',
        default=['M', 'T', 'E'],
        help="List of event types, default M, T, E."
    )

    ap.add_argument(
        "--source",
        action="append",
        required=True,
        type=parse_source,
        help="CSV source in format file.csv:priority. Can be repeated.",
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Root folder. Event folder is root/path from CSV."
    )
    ap.add_argument(
        "--out",
        default="all_events.pkl",
        help="Output pkl (default: all_events.pkl)"
    )

    ap.add_argument(
            "--p_prefix",
            type=int,
            default=1,
            help="Prefix before P arrival in seconds"
        )

    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")
    
    id_col = 'path'
    event_type_col = 'type'
    st_cols = [x for s in args.stations for x in (s + '_P', s + '_S')]
    print(st_cols)
    events = {}

    years_str = [str(y) for y in args.years]
    statistics = {}
    for event_type in args.types:
        statistics[event_type] = 0


    for csv_path, priority in args.source:
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=[id_col])
            print(f"CSV file {csv_path} read successfully:")
            #print(df.head())  # Display the first few rows
        except FileNotFoundError:
            print(f"Error: File '{args.csv}' not found.")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
        for row in df[[id_col, event_type_col] + st_cols].itertuples(index=False):

            obj_id = getattr(row, id_col)
            #print(obj_id)
            skip = True
            for y in years_str:
                if y in obj_id:
                    skip = False
                    break
            if skip:
                continue

            event_type = getattr(row, event_type_col)
            if event_type not in args.types:
                continue

            # create if not exists
            if obj_id not in events:
                events[obj_id] = event_metadata(event_type)

            obj = events[obj_id]

            # update attributes dynamically
            for col in st_cols:
                time = getattr(row, col)
                if not math.isnan(time):
                    obj.update(col[:-2], col[-1], time, priority)
                
            
        #print(df)
        #print(priority)
    #prepare data structures for the output
    dataframes = []
    labels = []
    filenames = []

    for event_name in events.keys():
        event = events[event_name]
        event_type = event.get_type()
        for st in args.stations:
            P_arrival = event.get_arrival_time(st, 'P')
            if P_arrival is None:
                continue
            mseed_path = f'{root}/{event_name}/{st}_{event_type}.mseed'

            error = False
            try:
                st = read(mseed_path)

            except FileNotFoundError:
                print(f"[ERROR] File '{mseed_path}' not found.")
                error = True
                continue
                #TODO report
            except Exception as e:
                print(f"[ERROR] reading Excel file {mseed_path}: {e}")
                continue

            event_data = {}
            t1 = st[0].stats.starttime + max(0, P_arrival - args.p_prefix)
            t2 = st[0].stats.endtime

            st.trim(t1, t2, pad=True, fill_value=0)
            l1 = len(st[0].data)
            l2 = len(st[1].data)
            l3 = len(st[2].data)
            if not (l1 == l2 and l2 == l3):
                print(f"[ERROR] a problem with channel lengths {l1} {l2} {l3}")
            for tr in st:
                event_data[tr.stats.channel[-1]] = tr.data
                

            # Create DataFrame with correct column names
            time_series = pd.DataFrame(event_data)

            dataframes.append(time_series)
            labels.append(event_type)
            #filenames.append(mseed_filename)
            statistics[event_type] += 1
            #print(f'{mseed_path} : {P_arrival}')

            #print(statistics)
        #print(f"{ev} CEL P: {events[ev].get_arrival_time('CEL', 'P')}")
    print(statistics)

    # Final structure
    #final_data = [dataframes, np.array(labels), np.array(filenames)]
    final_data = [dataframes, np.array(labels)]

        # Save to one pickle
    with open(args.out, "wb") as out_f:
        pickle.dump(final_data, out_f)
        print(f'{args.out} saved')

if __name__ == "__main__":
    main()