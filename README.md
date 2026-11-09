# SPF_cleaning

This repository implements a two-stage pipeline for preparing ML seismic datasets from MiniSEED waveform files and arrival-time annotations.

---

# Overview

The pipeline consists of two main steps:

1. **Cleaning (blacklist generation)**  
2. **Arrival-based dataset assembly**

These steps are designed to work together:

- The cleaning stage identifies problematic waveform files
- The arrivals stage uses this blacklist to build a clean, labeled dataset

---

## Data

The SPF (Seismic Polygon Frenštát, 1992–2002) event data sets
used in the IJRMMS manuscript are available on Zenodo (CC BY 4.0):
https://doi.org/10.5281/zenodo.24010919 (`SPF.tar.gz`).

---

# Pipeline Structure

Raw inputs:

- MiniSEED waveform files
- Arrival CSV files (possibly multiple sources)
- Optional clean dataset for autoencoder training

Processing:

1. Cleaning script scans waveform files
2. Generates blacklist of corrupted/suspicious files
3. Arrivals script loads arrival annotations
4. Resolves conflicts using source priorities
5. Filters out blacklisted waveform files
6. Trims waveforms relative to P arrival
7. Exports dataset as a pickle file

---

# Stage 1: Cleaning Script

## Purpose

Detect low-quality or corrupted waveform files.

## Methods

- Deterministic signal checks:
  - amplitude thresholds
  - zero-value ratios
  - mean/max inconsistencies
  - cross-channel consistency
  - spectral correlation

- Optional autoencoder:
  - trained on known-good signals
  - detects subtle anomalies

- K-means clustering:
  - separates outliers in feature space

## Output

CSV file (blacklist) containing waveform paths flagged as problematic.

This file is used by the arrivals stage to exclude bad data.

---

# Stage 2: Arrivals Script

## Purpose

Build the final dataset using:

- arrival-time annotations
- waveform files
- blacklist from cleaning stage

## Key Features

- Supports multiple arrival sources with priorities
- Keeps best arrival per station/phase
- Filters by year, station, and event type
- Skips blacklisted waveform files
- Trims waveforms relative to P arrival
- Outputs ML-ready dataset

---

# Arrival Source Priority

Arrival sources are passed as:

--source file.csv:priority

Example:

--source pamela_T.csv:1  
--source DB_arrivals.csv:5  

Higher priority overrides lower priority for the same station and phase.

---

# Input Requirements

## Arrival CSV

Must contain:

- path (event folder)
- type (event type)
- station-phase columns:

Example:

CEL_P, CEL_S, PAL_P, PAL_S, ...

## Waveform Structure

Expected path:

root / event_path / STATION_TYPE.mseed

Example:

root/mseed/1997/08_97/P7807032.250_decoded/TROJ_T.mseed

---

# Output Format

The arrivals script produces:

all_events.pkl

Structure:

[dataframes, labels]

- dataframes: list of pandas DataFrames (waveforms)
- labels: NumPy array of event types

Each DataFrame contains three channels (e.g. Z, N, E).

---

# Usage

## Step 1: Cleaning

python find_errors.py \
  --mseed_list mseeds.txt \
  --output blacklist.csv

Optional (with autoencoder):

python cleaning.py \
  --mseed_list mseeds.txt \
  --ok_mseed_list ok_mseeds.txt \
  --output blacklist.csv

Vizualize (optional)
python visualize.py \
  --csv blacklist.csv \
  --out output.pdf \
  --max-pages 1000 #max pages for one pdf
  
## Step 2: Dataset Assembly

cd arrivals

python get_pkl.py \
  --root ../ \
  --source DB_arrivals.csv:10 \
  --source arrivals_auto.csv:5 \
  --source phasenet_arrivals.csv:2 \
  --source pamela_E.csv:1 \
  --source pamela_M.csv:1 \
  --source pamela_T.csv:1 \
  --blacklist blacklist.csv \
  --out all_events.pkl

---

# Key Parameters (Arrivals)

--years        Years to include (default 1992 -- 2002)  
--stations     Stations to include (default CEL, PAL, PRAZ, PST, TROJ)  
--types        Event types (default E, M, T)  
--p_prefix     Seconds before P arrival (default 1 s)  

---

# Internal Logic

## Arrival Handling

- Only P and S phases are used
- Highest-priority arrival is retained

## Filtering

An event is skipped if:

- missing P arrival
- blacklisted
- waveform file missing
- P arrival < prefix
- channel lengths mismatch

## Trimming

Waveforms are trimmed:

start = P_arrival - prefix  
end = trace end  

---

# Statistics

The script prints:

- number of accepted samples per event type
- number of skipped samples per event type

---

# Notes & Caveats

- Assumes 3-component waveforms
- Blacklist matching is string-based
- NaN or corrupted signals are skipped
- K-means assumes anomalies are minority
- Autoencoder requires sufficient clean data

---

# Repository Structure

SPF_cleaning/
├── check_event.py
├── signal_proc.py
├── deterministic.py
├── autoencoder_video.py
├── autoencoder.py
├── README.md
└── arrivals/
    └── get_pkl.py


---

# Summary

This repository provides a preprocessing pipeline:

- Cleaning stage removes problematic waveform files
- Arrivals stage builds a clean, labeled dataset ready for analysis or machine learning

