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
        max_amp = np.max(np.abs(signal))
        if max_amp >= self.amp_threshold:
            return f"{max_amp:.3f}>={self.amp_threshold}"
        return None

    def check_amp_threshold_ratio(self, signal):
        amp_ratio = np.mean(np.abs(signal) >= self.amp_threshold_multiplier * self.amp_threshold)
        if amp_ratio >= self.amp_threshold_ratio:
            return f"{amp_ratio:.3f}>={self.amp_threshold_ratio}"
        return None

    def check_mean(self, signal):
        abs_signal = np.abs(signal)
        mean_abs = abs_signal.mean()
        mean_ratio = np.mean(abs_signal <= mean_abs)
        if mean_ratio >= self.mean_threshold:
            return f"{mean_ratio:.3f}>={self.mean_threshold}"
        return  None

    def check_zeros(self, signal):
        zeros_ratio = np.mean(signal == 0.0)
        if zeros_ratio >= self.zero_threshold:
            return f"f{zeros_ratio:.3f}>={self.zero_threshold}"
        return None

    # ============================
    # Slavek
    # ============================

    def check_mean_vs_max_amp(self, signal):
        abs_signal = np.abs(signal)
        max_abs = np.max(abs_signal)
        mean_abs = np.mean(abs_signal)

        threshold = self.mean_multiplier * mean_abs
        if max_abs > self.mean_multiplier * mean_abs:
            return f"{max_abs/mean_abs:.3f}>{self.mean_multiplier}"
        return None

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
        if r:
            return f"z/n {z2n:.3e} z/e {z2e:.3e} n/e {n2e:.3e}"              
        return None

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
        #Z2N = np.correlate(Z, N)
        #Z2E = np.correlate(Z, E)
        #N2E = np.correlate(N, E)
        Z2N = float(np.dot(Z, N))
        Z2E = float(np.dot(Z, E))
        N2E = float(np.dot(N, E))

        #print("%5s corr_coeff Z-N %5.3f Z-E %5.3f N-E %5.3f" % (stat, Z2N, Z2E, N2E))

        #if (Z2N < corr_lev) or (Z2E < corr_lev) or (N2E < corr_lev):
            #self.status[ii] += 1

        if (Z2N < self.correlation_level) or (Z2E < self.correlation_level) or (N2E < self.correlation_level):
            return f"Z-N {Z2N:.3f} Z-E {Z2E:.3f} N-E {N2E:.3f}"
