import sys
import argparse
import pandas as pd
import numpy as np

from obspy import read
from sklearn.cluster import KMeans

from signal_proc import zscore, fft_features, event_features_from_mseed, video_features_from_mseed
from deterministic import Deterministic_tester
from autoencoder import AutoencoderCleaner, AEConfig
from autoencoder_video import AutoencoderCleaner, AEConfig

#Parse arguments
parser = argparse.ArgumentParser()

# ============================
# Input / Output
# ============================
parser.add_argument(
    '--mseed_list',
    dest='mseed_list_txt',
    type=str,
    help='List of paths to mseeds'
)

parser.add_argument(
    '--ok_mseed_list',
    dest='ok_mseed_list_txt',
    type=str,
    default=None,
    help='Optional: list of paths to OK mseeds for AE training'
)

parser.add_argument(
    '--output',
    dest='output',
    type=str,
    default='output.csv',
    help ='Output csv filename. Default: output.csv'
)

# ============================
# Signal processing
# ============================

parser.add_argument(
    '--n_fft',
    dest='n_fft', 
    type=int,
    default=4096,
    help='Fixed FFT length in samples (trim/pad). Default: 4096'
)

# ============================
# K-means
# ============================

parser.add_argument(
    '--seed',
    dest='seed',
    type=int,
    default=0,
    help='Random seed for KMeans. Default: 0'
)

# ============================
# Autoencoder
# ============================

parser.add_argument(
    '--ae_model_out',
    dest='ae_model_out',
    type=str,
    default=None,
    help='Optional: path to save trained AE model (torch .pt)'
)

parser.add_argument(
    '--ae_percentile',
    dest='ae_percentile',
    type=float,
    default=99.0,
    help='AE threshold percentile on OK validation. Default: 99'
)

parser.add_argument(
    '--ae_epochs',
    dest='ae_epochs',
    type=int,
    default=60,
    help='AE training epochs. Default: 60'
)

parser.add_argument(
    '--ae_device',
    dest='ae_device',
    type=str,
    default='cpu',
    help='AE device: cpu or cuda. Default: cpu'
)

parser.add_argument('--ae_working_length', type=int, default=8192,
                    help='Pad/crop length (samples) before windowing. Default: 8192')
parser.add_argument('--ae_win_len', type=int, default=256,
                    help='Sliding window length (samples). Default: 256')
parser.add_argument('--ae_hop', type=int, default=64,
                    help='Window hop (samples). Default: 64')
parser.add_argument('--ae_agg', type=str, default="max",
                    help='Aggregate window scores -> event score: max|p95|mean. Default: max')
parser.add_argument('--ae_p', type=float, default=95.0,
                    help='Percentile if ae_agg=p95. Default: 95')

# ============================
# Deterministic cleaning parameters (Michael)
# ============================

parser.add_argument(
    '--amp_threshold',
    dest='amp_threshold',
    type = float,
    default = 1.0,
    help='Max amplitude threshold.'
)

parser.add_argument(
    '--amp_threshold_multiplier',
    dest='amp_threshold_multiplier',
    type = float,
    default = .8,
    help='Max amplitude threshold coeff.'
)

parser.add_argument(
    '--amp_threshold_ratio',
    dest='amp_threshold_ratio',
    type = float,
    default = .35,
    help='Max allowed ratio of values greater than max coff * amplitude threshold.'
)

parser.add_argument(
    '--mean_threshold',
    dest='mean_threshold',
    type = float,
    default = .95,
    help='Max allowd ratio of values less than mean.'
)

parser.add_argument(
    '--zero_threshold',
    dest='zero_threshold',
    type = float,
    default = .4,
    help='Max allowed ratio of values equal to zero.'
)

# ============================
# Deterministic cleaning parameters (Slavek)
# ============================

parser.add_argument(
    '--mean_multiplier',
    dest='mean_multiplier',
    type = float,
    default = 1.e5,
    help='Max allowed ratio of max value and mean'
)

parser.add_argument(
    '--trace_ratio',
    dest='trace_ratio',
    type = float,
    default = 1.e1,
    help='Max allowed ratio of max values of channels'
)

parser.add_argument(
    '--correlation_level',
    dest='correlation_level',
    type = float,
    default = 0.6,
    help='Minimum correlation level of spectra'
)

args = parser.parse_args()

#Read OK mseeds and train autoencoder
"""cleaner = None
if args.ok_mseed_list_txt is not None:
    with open(args.ok_mseed_list_txt) as f:
        ok_files = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    print(f"[INFO] Training autoencoder on {len(ok_files)} OK signals...")

     X_ok = []
    for fn in ok_files:
        mseed_ok = read(fn)
        mseed_ok.sort(keys=["channel"])
        feats_ok = event_features_from_mseed(mseed_ok, n_fft=args.n_fft)
        if np.isnan(feats_ok).any():
            continue
        X_ok.append(feats_ok)

    if len(X_ok) < 10:
        print("[WARNING] Too few OK samples for AE; skipping AE training.")
    else:
        X_ok = np.vstack(X_ok).astype(np.float64)

        cfg = AEConfig(
            epochs=args.ae_epochs,
            threshold_percentile=args.ae_percentile,
            seed=args.seed,
            device=args.ae_device,
        )
        cleaner = AutoencoderCleaner(cfg).fit(X_ok)
        print(f"AE threshold set to {cleaner.threshold_:.6f}")

        if args.ae_model_out:
            cleaner.save(args.ae_model_out)
            print(f"Saved AE model to {args.ae_model_out}")
 """
# Read OK mseeds and train autoencoder (WINDOW-based)
cleaner = None
if args.ok_mseed_list_txt is not None:
    with open(args.ok_mseed_list_txt) as f:
        ok_files = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    print(f"[INFO] Training autoencoder on {len(ok_files)} OK signals (sliding windows)...")

    X_ok_all = []
    for fn in ok_files:
        mseed_ok = read(fn)
        mseed_ok.sort(keys=["channel"])

        vid_ok = video_features_from_mseed(
            mseed_ok,
            working_length=args.ae_working_length,
            win_len=args.ae_win_len,
            hop=args.ae_hop,
        )  # (N, win_len, 3)

        if np.isnan(vid_ok).any():
            continue

        X_ok_all.append(vid_ok)

    if len(X_ok_all) == 0:
        print("[WARNING] No usable OK windows for AE; skipping AE training.")
    else:
        X_ok = np.concatenate(X_ok_all, axis=0).astype(np.float32)  # (M, win_len, 3)

        cfg = AEConfig(
            epochs=args.ae_epochs,
            threshold_percentile=args.ae_percentile,
            seed=args.seed,
            device=args.ae_device,
        )
        cleaner = AutoencoderCleaner(cfg).fit(X_ok)
        print(f"AE threshold set to {cleaner.threshold_:.6f}")

        if args.ae_model_out:
            cleaner.save(args.ae_model_out)
            print(f"Saved AE model to {args.ae_model_out}")
# Collect KMeans features for ALL events
X = []
valid_files = []

#Dataframe for the CSV with the results
corrupted = pd.DataFrame(
    columns=[
        "amp_threshold",
        "amp_threshold_mean",
        "mean_problem",
        "zeros_problem",
        "mean_vs_max",
        "traces_amp_ratio",
        "traces_spectra",
        "kmeans_corrupted",
        "ae_score",
        "NaN"])
corrupted.index.name = "event"

with open(args.mseed_list_txt) as f:
    filenames = f.read().splitlines()

d_test = Deterministic_tester(
    amp_threshold = args.amp_threshold,
    amp_threshold_multiplier = args.amp_threshold_multiplier,
    amp_threshold_ratio = args.amp_threshold_ratio,
    mean_threshold = args.mean_threshold,
    zero_threshold = args.zero_threshold,
    mean_multiplier = args.mean_multiplier, #Slavek
    trace_ratio = args.trace_ratio, #Slavek
    correlation_level = args.correlation_level #Slavek
)

for filename in filenames:
    #read mseed
    print(f"[INFO] reading {filename}.")
    mseed = read(filename)
    mseed.sort(keys=["channel"])
    #Deterministic tests

    #Single component
    checks = {
        "amp_threshold": d_test.check_amp_threshold,
        "amp_threshold_mean": d_test.check_amp_threshold_ratio,
        "mean_problem": d_test.check_mean,
        "zeros_problem": d_test.check_zeros,
        "mean_vs_max": d_test.check_mean_vs_max_amp,
    }
    problems = {key: "" for key in checks}

    for ch in mseed:
        channel_id = ch.id[-1]

        for key, func in checks.items():
            result = func(ch.data)
            if result:
                problems[key] += f"{channel_id}: {result} "

    for key, value in problems.items():
        if value:
            corrupted.loc[filename, key] = value.rstrip()

    #All three components
    checks = {
        "traces_amp_ratio": d_test.check_traces_amp_ratio,
        "traces_spectra": d_test.check_trace_spectra,
    }

    data = [ch.data for ch in mseed[:3]]

    for key, func in checks.items():
        result = func(*data)
        if result:
            corrupted.loc[filename, key] = result

    #K-means preparation
    """ transforemd_data = event_features_from_mseed(mseed, n_fft=args.n_fft)
    #Autoencoder tests
    if np.isnan(transforemd_data).any():
        corrupted.loc[filename, "NaN"] = str(True)
    else:
         # AE scoring (if trained)
        if cleaner is not None:
            ae_score = float(cleaner.score(transforemd_data)[0])
            if ae_score > cleaner.threshold_:
                corrupted.loc[filename, "ae_score"] = ae_score

        X.append(transforemd_data)
        valid_files.append(filename) """
    transforemd_data = event_features_from_mseed(mseed, n_fft=args.n_fft)

    # AE (window-based) scoring
    vid = video_features_from_mseed(
        mseed,
        working_length=args.ae_working_length,
        win_len=args.ae_win_len,
        hop=args.ae_hop,
    )

    if np.isnan(transforemd_data).any() or np.isnan(vid).any():
        corrupted.loc[filename, "NaN"] = str(True)
    else:
        if cleaner is not None:
            win_scores = cleaner.score(vid)  # (N_windows,)
            if args.ae_agg == "max":
                ae_score = float(np.max(win_scores))
            elif args.ae_agg == "mean":
                ae_score = float(np.mean(win_scores))
            elif args.ae_agg == "p95":
                ae_score = float(np.percentile(win_scores, args.ae_p))
            else:
                raise ValueError(f"Unknown --ae_agg {args.ae_agg}")

            if ae_score > cleaner.threshold_:
                corrupted.loc[filename, "ae_score"] = ae_score

        # KMeans data (unchanged)
        X.append(transforemd_data)
        valid_files.append(filename)

print("[INFO] Starting k-means")

# Fit KMeans on valid feature rows
if len(X) > 1:
    X = np.vstack(X).astype(np.float64)

    kmeans = KMeans(n_clusters=2, n_init=20, random_state=args.seed)
    labels = kmeans.fit_predict(X)

    # Minority cluster = corrupted
    counts = np.bincount(labels)
    minority_cluster = int(np.argmin(counts))

    for fn, lab in zip(valid_files, labels):
        if int(lab) != minority_cluster:
            continue
        corrupted.loc[fn, "kmeans_corrupted"] = (int(lab) == minority_cluster)

print(f"[INFO] Print output to {args.output}.")
corrupted.to_csv(args.output)
