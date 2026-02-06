import sys
import argparse
import pandas as pd
import numpy as np

from obspy import read
from sklearn.cluster import KMeans

from signal_proc import zscore, fft_features, event_features_from_mseed
from deterministic import Deterministic_tester
from autoencoder import AutoencoderCleaner, AEConfig

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
    help='Max allowd ratio of values equal to zero.'
)

args = parser.parse_args()

#Read OK mseeds and train autoencoder
cleaner = None
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
    mean_multiplier = 1.e5, #Slavek
    trace_ratio = 1.e1, #Slavek
    correlation_level = 0.6 #Slavek
)

for filename in filenames:
    #read mseed
    print(f"[INFO] reading {filename}.")
    mseed = read(filename)
    mseed.sort(keys=["channel"])
    #Deterministic tests
    problems = {"amp_threshold" : "", "amp_threshold_mean" : "", "mean_problem" : "", "zeros_problem" : "", "mean_vs_max" : ""}
    for ch in mseed:
        if d_test.check_amp_threshold(ch.data):
            problems["amp_threshold"] += ch.id[-1]
        if d_test.check_amp_threshold_ratio(ch.data):
            problems["amp_threshold_mean"] += ch.id[-1]
        if d_test.check_mean(ch.data):
            problems["mean_problem"] += ch.id[-1]
        if d_test.check_zeros(ch.data):
            problems["zeros_problem"] += ch.id[-1]
        if d_test.check_mean_vs_max_amp(ch.data):
            problems["mean_vs_max"] += ch.id[-1]
        for key, value in problems.items():
            if value != "":
                corrupted.loc[filename, key] = value
        if d_test.check_traces_amp_ratio(mseed[0].data, mseed[1].data, mseed[2].data):
            corrupted.loc[filename, "traces_amp_ratio"] = str(True)
        if d_test.check_trace_spectra(mseed[0].data, mseed[1].data, mseed[2].data):
            corrupted.loc[filename, "traces_spectra"] = str(True)

    #K-means preparation
    transforemd_data = event_features_from_mseed(mseed, n_fft=args.n_fft)
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
