import numpy as np

#Deterministic staff
def check_amp_threshold(signal, amp_threshold = 1):
    return np.any(np.abs(signal) > amp_threshold)

def check_amp_threshold_ratio(signal, amp_threshold = 1, amp_threshold_multiplier = .8, amp_threshold_ratio = .95):
    return np.mean(np.abs(signal) >= amp_threshold_multiplier * amp_threshold) > amp_threshold_ratio

def check_mean(signal, mean_threshold = .95):
    abs_arr = np.abs(signal)
    mean_abs = abs_arr.mean()
    return np.mean(abs_arr <= mean_abs) >= mean_threshold

def check_zeros(signal, zero_threshold = .4):
    return np.mean(signal == 0.0) >= zero_threshold
