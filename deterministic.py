import numpy as np

class Deterministic_tester:
    def __init__(
        self,
        amp_threshold=1.0,
        amp_threshold_multiplier=.8,
        amp_threshold_ratio=.35,
        mean_threshold=.95,
        zero_threshold=.4,
        mean_multiplier=1.e5,
        trace_ratio=1.e1,
        correlation_level=0.6
    ):
        self.amp_threshold = amp_threshold
        self.amp_threshold_multiplier = amp_threshold_multiplier
        self.amp_threshold_ratio = amp_threshold_ratio
        self.mean_threshold = mean_threshold
        self.zero_threshold = zero_threshold
        self.mean_multiplier = mean_multiplier
        self.trace_ratio = trace_ratio
        self.correlation_level = correlation_level     

        print(self.amp_threshold)
        print(self.amp_threshold_multiplier)   

    # ============================
    # Michael
    # ============================
    def check_amp_threshold(self,signal):
        return np.any(np.abs(signal) > self.amp_threshold)

    def check_amp_threshold_ratio(self, signal):
        return np.mean(np.abs(signal) >= self.amp_threshold_multiplier * self.amp_threshold) > self.amp_threshold_ratio

    def check_mean(self, signal):
        abs_signal = np.abs(signal)
        mean_abs = abs_signal.mean()
        return np.mean(abs_signal <= mean_abs) >= self.mean_threshold

    def check_zeros(self, signal):
        return np.mean(signal == 0.0) >= self.zero_threshold

    # ============================
    # Slavek
    # ============================

    def check_mean_vs_max_amp(self, signal):
        abs_signal = np.abs(signal)
        max_abs = np.max(abs_signal)
        mean_abs = np.mean(abs_signal)
        return max_abs > self.mean_multiplier * mean_abs

    def check_traces_amp_ratio(self, signal_Z, signal_N, signal_E):
        zmax = np.max(np.abs(signal_Z))
        nmax = np.max(np.abs(signal_N))
        emax = np.max(np.abs(signal_E))

        z2n = zmax/nmax
        z2e = zmax/emax
        n2e = nmax/emax

        r = False
        if (z2n > self.trace_ratio) or (z2n < 1 / self.trace_ratio):
            r = True
        if (z2e > self.trace_ratio) or (z2e < 1 / self.trace_ratio):
            r = True
        if (n2e > self.trace_ratio) or (n2e < 1 / self.trace_ratio):
            r = True
        return r

    def check_trace_spectra(self, signal_Z, signal_N, signal_E):
        z = signal_Z - np.median(signal_Z)
        n = signal_N - np.median(signal_N)
        e = signal_E - np.median(signal_E)
        Z = np.abs(np.fft.fft(z))
        N = np.abs(np.fft.fft(n))
        E = np.abs(np.fft.fft(e))
        Z = Z/np.linalg.norm(Z)
        N = N/np.linalg.norm(N)
        E = E/np.linalg.norm(E)
        Z2N = np.correlate(Z, N)
        Z2E = np.correlate(Z, E)
        N2E = np.correlate(N, E)

        #print("%5s corr_coeff Z-N %5.3f Z-E %5.3f N-E %5.3f" % (stat, Z2N, Z2E, N2E))

        #if (Z2N < corr_lev) or (Z2E < corr_lev) or (N2E < corr_lev):
            #self.status[ii] += 1

        return (Z2N < self.correlation_level) or (Z2E < self.correlation_level) or (N2E < self.correlation_level)