import sys
import argparse

from obspy import read

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

#Deterministic staff
def check_amp_treshold(signal):
    return np.any(np.abs(signal) > amp_treshold)

def check_amp_treshold_ratio(signal):
    return np.mean(np.abs(signal) >= amp_treshold_multiplier * amp_treshold) > amp_treshold_ratio

def check_mean(signal):
    abs_arr = np.abs(signal)
    mean_abs = abs_arr.mean()
    return np.mean(abs_arr <= mean_abs) >= mean_treshold

def check_zeros(signal):
    return np.mean(signal == 0.0) >= zero_treshold

#K-means staff
def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    mu = x.mean()
    sd = x.std()
    if sd < eps:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd

def fft_features(x: np.ndarray, n_fft: int, use_log: bool = True) -> np.ndarray:
    X = np.fft.rfft(x, n=n_fft)  # pad or trim here
    mag = np.abs(X)
    if use_log:
        mag = np.log1p(mag)
    return mag

def event_features_from_mseed(mseed, n_fft: int) -> np.ndarray:
    """
    z-score -> fixed length -> FFT mag -> concat.
    """
    transformed_data = []
    for i in range(3):
        sig = mseed[i].data
        sig = zscore(sig)
        transformed_data.append(fft_features(sig, n_fft=n_fft, use_log=True))
    return np.concatenate(transformed_data, axis=0)

#Dataframe for the CSV with the results
corrupted = pd.DataFrame(columns=["amp_treshold", "amp_treshold_mean", "mean_problem", "zeros_problem", "kmeans_corrupted"])
corrupted.index.name = "event"

#Parse arguments
parser = argparse.ArgumentParser()
#txt file with the list of mseed files
parser.add_argument('--mseed_list', dest='mseed_list_txt', type=str, help='List of paths to mseeds')

# Add FFT length arg (samples)
parser.add_argument('--n_fft', dest='n_fft', type=int, default=4096,
                    help='Fixed FFT length in samples (trim/pad). Default: 4096')
#Random seed
parser.add_argument('--seed', dest='seed', type=int, default=0,
                    help='Random seed for KMeans. Default: 0')

args = parser.parse_args()

#parameters
amp_treshold = 1
amp_treshold_multiplier = .8
amp_treshold_ratio = .35

mean_treshold = .95

zero_treshold = .4


with open(args.mseed_list_txt) as f:
    filenames = f.read().splitlines()
max_value = 0.0


# Collect KMeans features for ALL events
X = []
valid_files = []

for filename in filenames:
    mseed = read(filename)
    mseed.sort(keys=["channel"])
    
    problems = {"amp_treshold" : "", "amp_treshold_mean" : "", "mean_problem" : "", "zeros_problem" : ""}

    for ch in mseed:
        if check_amp_treshold(ch.data):
            problems["amp_treshold"] += ch.id[-1]
            #corrupted.loc[filename, "amp_treshold"] = i

        if check_amp_treshold_ratio(ch.data):
            problems["amp_treshold_mean"] += ch.id[-1]
            #corrupted.loc[filename, "amp_treshold_mean"] = i

        if check_mean(ch.data):
            problems["mean_problem"] += ch.id[-1]
            #corrupted.loc[filename, "mean_problem"] = i

        if check_zeros(ch.data):
            problems["zeros_problem"] += ch.id[-1]
            #corrupted.loc[filename, "zeros_problem"] = i

        for key, value in problems.items():
            if value != "":
                corrupted.loc[filename, key] = value
    #K-means preparation
    transforemd_data = event_features_from_mseed(mseed, n_fft=args.n_fft)
    X.append(transforemd_data)

# Fit KMeans on valid feature rows
if len(X) > 1:
    X = np.vstack(X).astype(np.float64)

    kmeans = KMeans(n_clusters=2, n_init=20, random_state=args.seed)
    labels = kmeans.fit_predict(X)

    # Minority cluster = corrupted
    counts = np.bincount(labels)
    minority_cluster = int(np.argmin(counts))

    for fn, lab in zip(filenames, labels):
        if int(lab) != minority_cluster:
            continue
        #corrupted.loc[fn, "kmeans_cluster"] = int(lab)
        corrupted.loc[fn, "kmeans_corrupted"] = (int(lab) == minority_cluster)

corrupted.to_csv("corrupted.csv")
