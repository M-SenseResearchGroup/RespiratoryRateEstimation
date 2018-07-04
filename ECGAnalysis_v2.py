"""
V0.2.0
January 19, 2018

Lukas Adamowicz

Python 3.6.3 on Windows 10 with 64-bit Anaconda
"""
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
from scipy import signal, interpolate
from pickle import dump as pickle_dump, load as pickle_load


class EcgData:
    def __init__(self):
        self.t = dict()
        self.v = dict()


class OrderError(Exception):
    pass


class TSN:
    """
    Class for storing Threshold, Signal, Noise values for R-peak detection
    """
    def __init__(self):
        self.t = 0.0  # threshold
        self.s = 0.0  # signal peak moving average
        self.n = 0.0  # noise peak moving average


class ECGAnalysis(object):
    def __init__(self, master, moving_filter_len=150, delay=175):
        """
        Class defining steps for analyzing ECG data to obtain an estimation
        of Respiratory Rate.

        Parameters
        ---------
        master : EcgData
            EcgData class (DataStructures) with voltage and timing dictionaries for
            desired events to be analyzed
        moving_filter_len : float, int, optional
            Trailing length of the moving filter in ms.  Defaults to 150ms
        delay : float, int, optional
            Delay in ms to wait before declaring descending half-peak.
            Defaults to 175ms
        """
        self.v = master.v  # should reference the same data so no duplicates
        self.t = master.t

        self.mfl = moving_filter_len
        self.delay = delay

        self.vd = list(master.v.keys())
        self.td = list(master.t.keys())  # these 2 should be the same

        # calculate sampling frequency
        self.fs = round(1/(self.t[self.td[0]][1] - self.t[self.td[0]][0]))
        master.fs = self.fs  # assign this to master for possible future use

        self.nyq = 0.5*self.fs  # Nyquist frequency is half sampling frequency

    def TestFunctions(self, count='adv', fuse=True):
        self.ElimLowFreq()
        self.ElimVeryHighFreq()
        self.ElimMainsFreq()
        self.CheckLeadInversion()
        self.ElimSubCardiacFreq()
        self.DerivativeFilter()
        self.MovingAverageFilter()
        self.DetectRPeaks(debug=False)
        self.PlotHeartRate()
        self.RespRateExtraction()
        if count == 'orig':
            self.CountOriginal(debug=False)
        if count == 'adv':
            self.CountAdv(debug=True)
        if fuse:
            self.SmartModFusion(use_given_time=True)

    def ElimLowFreq(self, cutoff=3, N=1, debug=False):
        """
        Step 1: Eliminate low frequencies for ECG data using a high-pass filter.

        Parameters
        ---------
        cutoff : float, int, optional
            -3dB high-pass filter cutoff frequency in Hz.  Defaults to 3Hz
        N : int, optional
            Filter order for a double (forwards-backwards) linear filter.
            Defaults to 1
        """
        if debug:
            self.v_old = self.v.copy()  # store previous step's voltage for reference

        w_cut = cutoff/self.nyq  # cutoff frequency as percentage of nyquist frequency

        b, a = signal.butter(N, w_cut, 'highpass')  # setup high pass filter

        # perform filtering on the data
        for key in self.vd:
            self.v[key] = signal.filtfilt(b, a, self.v[key])

        if debug:
            n = len(self.vd)
            f, ax = pl.subplots(n, figsize=(16, 8))
            for i, key in zip(range(n), self.vd):
                ax[i].plot(self.t[key], self.v_old[key], label='initial')
                ax[i].plot(self.t[key], self.v[key], label='Step 1')
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel('Voltage [mV]')
                ax[i].legend(title=f'{key}')
            ax[0].set_title('1 - Eliminate Low Frequencies')
            pl.tight_layout()

    def ElimVeryHighFreq(self,cutoff=20, N=1, detrend=True, debug=False):
        """
        Step 2: Eliminate very high frequencies for ECG data using a low-pass filter
        Should be run after "ElimLowFreq" function is run

        Parameters
        ---------
        cutoff : float, int, optional
            -3dB cutoff for low-pass filter.  Defaults to 20Hz.
        N : int, optional
            Filter order for a double (forwards-backwards) linear filter.
            Defaults to 1
        detrend : bool, optional
            Detrend data.  Defaults to True
        """
        if debug:
            # copy old voltages for reference/debugging
            self.v_old = self.v.copy()

        # cutoff frequency as percentage of nyquist frequency
        w_cut = cutoff/self.nyq

        # setup filter parameters
        b, a = signal.butter(N, w_cut, 'lowpass')

        for key in self.vd:
            # detrend - remove mean, linear associations
            if detrend:
                data_det = signal.detrend(self.v[key])
                self.v[key] = signal.filtfilt(b, a, data_det)
            else:
                # filter input data
                self.v[key] = signal.filtfilt(b, a, self.v[key])

        if debug:
            n = len(self.vd)
            f, ax = pl.subplots(n, figsize=(16, 8))
            for i, key in zip(range(n), self.vd):
                ax[i].plot(self.t[key], self.v_old[key], label='Step 1')
                ax[i].plot(self.t[key], self.v[key], label='Step 2')
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel('Voltage [mV]')
                ax[i].legend(title=f'{key}')
            ax[0].set_title('2 - Eliminate Very High Frequencies')
            pl.tight_layout()

    def ElimMainsFreq(self, cutoff=60, Q=10, detrend=True, debug=False):
        """
        Step 3: Eliminate Mains frequency caused by electronics.
        60Hz in Americas.  50Hz in Europe.
        Run after ElimVeryHighFreq

        Parameters
        ---------
        cutoff : float, int, optional
        -3dB cutoff for notch filter.  Defaults to 60Hz (US).
        Q : int, optional
        Filter quality for double (forwards-backwards) linear filter.
        Defaults to 10.
        detrend : bool, optional
        Detrend data.  Defaults to True
        """
        if debug:
            self.v_old = self.v.copy()

        w0 = cutoff/self.nyq
        b, a = signal.iirnotch(w0, Q)

        for key in self.vd:
            if detrend:
                data_det = signal.detrend(self.v[key])
                self.v[key] = signal.filtfilt(b, a, data_det)
            else:
                self.v[key] = signal.filtfilt(b, a, self.v[key])

        if debug:
            n = len(self.vd)
            f, ax = pl.subplots(n, figsize=(16, 8))
            for i, key in zip(range(n), self.vd):
                ax[i].plot(self.t[key], self.v_old[key], label='Step 2')
                ax[i].plot(self.t[key], self.v[key], label='Step 3')
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel('Voltage [mV]')
                ax[i].legend(title=f'{key}')
                ax[0].set_title('3 - Eliminate Mains Frequencies')
                pl.tight_layout()

    def CheckLeadInversion(self, debug=False):
        """
        Step 4: Check for lead inversion by examining percentage of local
        extrema that are negative. R-peaks (maximum local extrema) should be positive.
        Run after ElimMainsFreq
        """

        d2 = self.v[self.vd[0]]**2  # square data for all positive and prominent peaks
        d2_max = max(d2)  # maximum of squared data
        d2_max_pks = np.zeros_like(d2)  # allocate array for peaks

        # indices where squared data is greater than 20% of squared max
        inds = np.argwhere(d2>0.2*d2_max)
        d2_max_pks[inds] = d2[inds]  # squared values greater than 20% max, 0s elsewhere
        d2_pks = signal.argrelmax(d2_max_pks)[0]  # locations of maximum peaks

        # data values at local extrema (found from squard values)
        d_pks = self.v[self.vd[0]][d2_pks]

        # percentage of peaks with values that are negative
        p_neg_pks = len(np.where(d_pks<0)[0])/len(d_pks)

        if debug:
            x = np.array([i for i in range(len(self.v[self.vd[0]]))])
            f, ax = pl.subplots(figsize=(9,5))
            ax.plot(x, self.v[self.vd[0]])
            ax.plot(x[d2_pks], self.v[self.vd[0]][d2_pks], '+')
            ax.set_xlabel('Sample No.')
            ax.set_ylabel('Voltage [mV]')
            ax.set_title('4 - Check Lead Inversion')
            pl.tight_layout()

        if p_neg_pks>=0.5:
            for key in self.vd:
                self.v[key] *= -1
                self.lead_inv = True
        elif p_neg_pks<0.5:
            self.lead_inv = False

    def ElimSubCardiacFreq(self, cutoff=0.5, N=4, debug=False):
        """
        Step 5: Eliminate sub-cardiac frequencies (30 BPM - 0.5Hz)

        Parameters
        ----------
        cutoff : float, int, optional
        -3dB cutoff for high-pass filter.  Defaults to 0.5Hz
        N : int, optional
        Filter order for a double (forwards-backwards) linear filter.
        Defaults to 4
        """
        if debug:
            self.v_old = self.v.copy()

        w_cut = cutoff/self.nyq  # define cutoff frequency as % of nyq. freq.
        b, a = signal.butter(N, w_cut, 'highpass')  # setup high pass filter

        for key in self.vd:
            self.v[key] = signal.filtfilt(b, a, self.v[key])

        if debug:
            n = len(self.vd)
            f, ax = pl.subplots(n, figsize=(16,8))
            for i, key in zip(range(n), self.vd):
                ax[i].plot(self.t[key], self.v_old[key], label='Step 4')
                ax[i].plot(self.t[key], self.v[key], label='Step 5')
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel('Voltage [mV]')
                ax[i].legend(title=f'{key}')
                ax[0].set_title('5 - Eliminate Sub-Cardiac Frequencies')
                pl.tight_layout()

        try:
            self.v_old = None  # remove from memory
        except:
            pass

    def DerivativeFilter(self, debug=False):
        """
        Calculated derivative of filtered data,
        as well as squaring data before next step

        Class Results
        -------------
        v_der : dict
        Derivative of filtered voltage signal data for each activity
        t_der : dict
        Timestamps for derivative voltage data for each activity
        """
        self.v_der = dict()
        self.t_der = dict()
        self.v_sq = dict()
        self.dt = 1/self.fs  # timestep

        for key in self.vd:
            self.v_der[key] = (-self.v[key][:-4]-2*self.v[key][1:-3]+2*self.v[key][3:-1]+self.v[key][4:])/(8*self.dt)
            self.t_der[key] = self.t[key][2:-2]*1
            self.t_der[key][:2] = 0  # timesteps are cut by 2 on either end

        self.v_der[key] = np.insert(self.v_der[key],0,[0]*2)

        if debug:
            f, ax = pl.subplots(2, figsize=(9, 5), sharex=True)
            ax[0].plot(self.t_der[key][2:], self.v_der[key][:-2], label='Derivative')
            ax[1].plot(self.t_der[key][2:], self.v_der[key][:-2]**2, label='Squared')
            ax[0].legend()
            ax[1].legend()
            f.tight_layout()
            f.subplots_adjust(hspace=0)
            ax[0].set_title('Derivative and squared filter example')

    def MovingAverageFilter(self, debug=False):
        """
        Moving average of squared data.

        Class Returns
        -------------
        v_int : ndarray
        Moving average resulting signal
        t_int : ndarray
        Timestamps for resulting average signal
        """
        # number of samples to use for moving average window
        # window width in seconds/timestep
        self.mfl_n = int(round((self.mfl/1000)/self.dt))

        self.v_ma = dict()
        self.t_ma = dict()
        for key in self.vd:
            # take preceding cumulative sum [1,2,3] -> [1,3,6]
            cs = np.cumsum(self.v_der[key][1:]**2)
            self.v_ma[key] = (cs[self.mfl_n:]-cs[:-self.mfl_n])/self.mfl_n
            # Array is shortened by N-1 entries by above operation
            self.t_ma[key] = self.t_der[key][self.mfl_n-1:]

            # add zeros back in to shortened arrays
            # adding n-1(moving average) + 2 (derivative) = n+1
            self.v_ma[key] = np.insert(self.v_ma[key], 0, [0]*(self.mfl_n+1))
            self.t_ma[key] = np.insert(self.t_ma[key], 0, [0]*(self.mfl_n-1))

        if debug:
            f, ax = pl.subplots(figsize=(16, 5))
            ax.plot(self.t_ma[key][self.mfl_n+1:], self.v_ma[key][self.mfl_n+1:], label='Moving Average')
            ax.legend()
            ax.set_xlabel('Time [s]')
            ax.set_title('Moving average filter example')

    def DetectRPeaks(self, debug=False):
        """
        Detect R-peaks in the filtered ECG signal
        """

        # TODO something wrong in checking back or checking for r-peaks
        # TODO minimum time is 0.12s -> HR that is way too high

        # number of samples in delay time
        ndly = int(round(self.delay/1000*self.fs))
        # number of samples in 25,125,225ms
        n25 = int(round(0.025*self.fs))
        n125 = int(round(0.125*self.fs))
        n225 = int(round(0.225*self.fs))

        self.r_pks = dict()  # allocate dictionary for R-peaks
        self.q_trs = dict()  # allocate dictionary for Q-troughs (for resp rate paramters)

        for key in self.vd:
            # peak indices in region with width equal to moving average width
            va_pts = signal.argrelmax(self.v_ma[key], order=int(round(0.5*self.mfl_n)))[0]
            # remove any points that are less than 0.1% of the mean of the peaks
            va_pts = va_pts[np.where(self.v_ma[key][va_pts] > 0.001*np.mean(self.v_ma[key][va_pts]))[0]]
            # descending half-peak value allocation
            hpt = np.zeros_like(va_pts)
            # maximum values and indices for filtered data
            vf_pks = np.zeros((len(va_pts),2))

            # maximum slopes and indices for each peak
            m_pos = np.zeros((len(va_pts),2))

            for i in range(len(va_pts)):
                # finding maximum slope in the +- width surrounding peaks
                i1 = va_pts[i]-self.mfl_n if va_pts[i]-self.mfl_n > 0 else 0
                i2 = va_pts[i]+self.mfl_n if va_pts[i]+self.mfl_n<=len(self.v_der[key]) else len(self.v_der[key])
                m_pos[i] = [max(self.v_der[key][i1:i2])+va_pts[i]-self.mfl_n, np.argmax(self.v_der[key][i1:i2])+i1]

                # find descending half-peak value in moving average data
                try:
                    hpt[i] = np.where(self.v_ma[key][va_pts[i]:va_pts[i]+ndly]<0.5*self.v_ma[key][va_pts[i]])[0][0] + va_pts[i]
                    longwave = False  # half peak found => not a long QRS complex
                except:
                    hpt[i] = m_pos[i,1] + ndly if m_pos[i,1] + ndly < len(self.v_ma[key]) else len(self.v_ma[key])-1
                    longwave = True  # half peak not found => long QRS complex

                # find maximum values in filtered data preceding descending half-peak in
                # moving average data
                if not longwave:  # if not a long wave search preceding 225 to 125 ms
                    i1 = hpt[i]-n225 if hpt[i]-n225>0 else 0
                    i2 = hpt[i]-n125
                    vf_pks[i] = [max(self.v[key][i1:i2]),np.argmax(self.v[key][i1:i2])+i1]
                elif longwave:  # if long wave, search preceding 250 to 150 ms
                    i1 = hpt[i]-n225-n25 if hpt[i]-n225-n25>0 else 0
                    i2 = hpt[i]-n125-n25
                    vf_pks[i] = [max(self.v[key][i1:i2]), np.argmax(self.v[key][i1:i2])+i1]

            sntf = TSN()  # signal/Noise thresholds for filtered data
            snta = TSN()  # signal/Noise thresholds for moving average data
            sntf.t = 0.125*np.median(vf_pks[:,0])  # initialize thresholds with 1/8 of median
            snta.t = 0.125*np.median(self.v_ma[key][va_pts])

            r_pk_b = np.full_like(hpt, False, dtype=bool)  # store if peak is R-peak (T) or not (F)
            # Determine the type of peak (R,T,etc) for learning phase (9 peaks->8 RR times)
            k = 1  # loop counter
            j = 0  # last detected R-peak counter
            nrpk = 0  # keep count for first 8 R-peaks (threshold learning)
            while nrpk < 9:
                # Are the moving average and filtered peaks above the thresholds
                if vf_pks[k,0] > sntf.t and self.v_ma[key][va_pts[k]] > snta.t:
                    # is this peak more than 0.36s away from the last R-peak?
                    if (vf_pks[k,1]-vf_pks[j,1])*self.dt > 0.36:
                        r_pk_b[k] = True  # r-peak was detected
                        nrpk += 1  # update number found for learning
                        j = k  # update last found r-peak index
                        if nrpk == 1:  # change the thresholds to values based on the R-peak
                            sntf.t = 0.125*vf_pks[k,0]
                            snta.t = 0.125*self.v_ma[key][va_pts[k]]
                        # Update all the signal/noise/threshold values
                        self._UpdateThresholds(self.v_ma[key][va_pts[k]], vf_pks[k,0], snta, sntf, sig=True)
                    # is this peak more than 0.2s but less than 0.36s away from last R-peak?
                    elif (vf_pks[k,1]-vf_pks[j,1])*self.dt > 0.2:
                        # is the slope of this peak greater than half the slope of the last R-peak?
                        if m_pos[k,0] > 0.5*m_pos[j,0]:
                            r_pk_b[k] = True
                            nrpk += 1
                            j = k
                            if nrpk == 1:
                                sntf.t = 0.125*vf_pks[k,0]
                                snta.t = 0.125*self.v_ma[key][va_pts[k]]
                            self._UpdateThresholds(self.v_ma[key][va_pts[k]], vf_pks[k,0], snta, sntf, sig=True)
                    # check for duplicate points, don't want to double count for updating thresholds
                    elif (vf_pks[k,1]-vf_pks[j,1])*self.dt ==0:
                        pass
                    # if the peak was less than 0.2s away from the last R-peak, it is noise
                    else:
                        self._UpdateThresholds(self.v_ma[key][va_pts[k]], vf_pks[k,0], snta, sntf, sig=False)
                #if the peak was not above the thresholds, it is noise
                else:
                    self._UpdateThresholds(self.v_ma[key][va_pts[k]], vf_pks[k,0], snta, sntf, sig=False)
                k += 1

            # calculate initial learning RR times
            rr8 = self.t[key][vf_pks[r_pk_b,1][1:].astype(int)] - self.t[key][vf_pks[r_pk_b,1][:-1].astype(int)]
            rr8_lim = rr8.copy()  # initialize limited R-R array with normal R-R array

            # determe if the peaks are R-peaks or other
            for i in range(k, len(va_pts)):
                # Is the point greater than moving thresholds?
                if vf_pks[i,0] > sntf.t and self.v_ma[key][va_pts[i]] > snta.t:
                    # is the time difference between consecutive peaks greater than 0.36s
                    if (vf_pks[i,1]-vf_pks[j,1])*self.dt > 0.36:
                        r_pk_b[i] = True
                        # if the time interval to last R-peak is greater than 166% of the limited average
                        if (vf_pks[i,1]-vf_pks[j,1])*self.dt > 1.66*np.mean(rr8_lim):
                            index, snta, sntf = self._SearchForPeak(self.v_ma[key][va_pts[j:i]], vf_pks[j:i,0],\
                                                                    m_pos[j:i,0], self.t[key][vf_pks[j:i,1].astype(int)],\
                                                                    snta, sntf)
                            if index != []:
                                index = np.array(index) + j
                                rr8, rr8_lim = self._UpdateAvgRR(self.t[key][int(vf_pks[i,1])]-\
                                                                 self.t[key][int(vf_pks[index[-1],1])], rr8, rr8_lim)
                                r_pk_b[index] = True
                        else:
                            rr8, rr8_lim = self._UpdateAvgRR(self.t[key][int(vf_pks[i,1])]-self.t[key][int(vf_pks[j,1])],\
                                                             rr8, rr8_lim)
                        j = i
                        self._UpdateThresholds(self.v_ma[key][va_pts[i]],vf_pks[i,0], snta, sntf, sig=True)

                    # is the time difference between 0.2 and 0.36s?
                    # TODO i+1 here?
                    elif (vf_pks[i,1]-vf_pks[j,1])*self.dt > 0.2:
                        # is the slope greater than 50% of previous R-peak slope?
                        if m_pos[i,0] > 0.5*m_pos[j,0]:
                            r_pk_b[i] = True
                            rr8, rr8_lim = self._UpdateAvgRR(self.t[key][int(vf_pks[i,1])]-self.t[key][int(vf_pks[j,1])],\
                                                             rr8, rr8_lim)
                            self._UpdateThresholds(self.v_ma[key][va_pts[i]], vf_pks[i,0], snta, sntf, sig=True)
                            j = i
                    # check for duplicate points, don't want to double count
                    # TODO i+1 here?
                    elif (vf_pks[i,1]-vf_pks[j,1])*self.dt == 0:
                        pass
                    # if the peak is less than 0.2s away from last R-peak, it is noise
                    else:
                        self._UpdateThresholds(self.v_ma[key][va_pts[i]],vf_pks[i,0], snta, sntf, sig=False)
                #if the peak is not above the thresholds, it is noise
                else:
                    self._UpdateThresholds(self.v_ma[key][va_pts[i]],vf_pks[i,0], snta, sntf, sig=False)

            indices = np.unique(vf_pks[r_pk_b,1].astype(int))
            self.r_pks[key] = np.zeros((len(indices),2))  # allocate R-peak array
            self.r_pks[key][:,0] = self.t[key][indices]  # set timings
            self.r_pks[key][:,1] = self.v[key][indices]  # set peak values

            self.q_trs[key] = np.zeros_like(self.r_pks[key])  # allocate Q-trough array

            for i in range(len(self.r_pks[key][:,0])):
                # determining index for R-peak
                i_rpk = np.where(self.t[key]==self.r_pks[key][i,0])[0][0]
                # search in preceding 0.1s for minimum value
                i_qtr = np.argmin(self.v[key][i_rpk-int(round(0.1*self.fs)):i_rpk]) + i_rpk-int(round(0.1*self.fs))
                # set time of minimum value, as well as value itself for Q-trough
                self.q_trs[key][i] = [self.t[key][i_qtr],self.v[key][i_qtr]]


            if debug:
                f, ax = pl.subplots(2,figsize=(14,5), sharex=True)
                ax[0].plot(self.t[key], self.v[key])
                ax[1].plot(self.t_ma[key][self.mfl_n:], self.v_ma[key][self.mfl_n:])

                ax[0].plot(self.t[key][vf_pks[r_pk_b,1].astype(int)], vf_pks[r_pk_b,0], 'bo')
                ax[0].plot(self.t[key][vf_pks[:,1].astype(int)], vf_pks[:,0], 'r+')
                ax[1].plot(self.t_ma[key][hpt], self.v_ma[key][hpt], 'ro')
                ax[1].plot(self.t_ma[key][va_pts], self.v_ma[key][va_pts], 'ko')

                ax[0].set_title(f'{key}')
                ax[0].set_xlabel('Time [s]')
                ax[0].set_ylabel('Voltage [mv]')
                f.tight_layout()
                f.subplots_adjust(hspace=0)

    def _SearchForPeak(self, a_pks, f_pks, slp, t, snta, sntf):
        """
        Search in abnormally large R-peak to R-peak interval for missed R-peaks

        Parameters
        ----------
        a_pks : array
        Moving average value local maxima between previously detected R-peaks
        f_pks : array
        Filtered value peaks between previously detected R-peaks
        slp : array
        Slope values between previously detected R-peaks
        t : array
        Time values between previously detected R-peaks
        snta : TSN
        Moving average signal/noise/threshold values
        sntf : TSN
        Filtered signal/noise/threshold values

        Returns
        ------
        k : array, None
        Indices of detected R-peaks or None
        snta : TSN
        Updated moving average signal/noise/threshold values
        sntf : TSN
        Updated filtered signal/noise/threshold values
        """
        k = []
        j = 0
        for i in range(1, len(a_pks)):
            if a_pks[i] > 0.5*snta.t and f_pks[i] > 0.5*sntf.t:
                if t[i] - t[j] > 0.36:
                    k.append(i)
                    j = i
                    self._UpdateThresholds(a_pks[i], f_pks[i], snta, sntf, sig=True,searchback=True)
                elif t[i] - t[j] > 0.2:
                    if slp[i] > 0.5*slp[j]:
                        k.append(i)
                        j = i
                        self._UpdateThresholds(a_pks[i], f_pks[i], snta, sntf, sig=True, searchback=True)
        return k, snta, sntf

    @staticmethod
    def _UpdateAvgRR(rr_t, rr8, rr8_lim):
        """
        Update R-R 8 value storage arrays for R-peak detection

        Parameters
        ---------
        rr_t : float
        R-R time in seconds
        rr8 : list of floats
        List/array of last 8 R-R times in seconds
        rr8_lim : list of floats
        List/array of last 8 R-R times in seconds that were between 92-116% of avg(rr8)

        Returns
        -------
        rr8 : list of floats
        Updated list/array of last 8 R-R times in seconds
        rr8_lim : list of floats
        Updated list/array of last 8 R-R times that were between 92-116% of avg(rr8)
        """
        if rr_t>0.92*np.mean(rr8) and rr_t<1.16*np.mean(rr8):
            rr8_lim[:-1] = rr8_lim[1:]
            rr8_lim[-1] = rr_t

        rr8[:-1] = rr8[1:]  # move values down 1 (ie remove oldest)
        rr8[-1] = rr_t  # insert newest value in last spot in list/array

        return rr8, rr8_lim

    @staticmethod
    def _UpdateThresholds(ma_peak, f_peak, tsn_ma, tsn_f, sig=True, searchback=False):
        """
        Update thresholds for R-peak detection

        Parameters
        ----------
        ma_peak : float
        Moving average signal peak
        f_peak : float
        Filtered signal peak
        tsn_ma : TSN
        Threshold, signal and noise moving averages for moving average data
        tsn_f : TSN
        Threshold, signal and noise moving averages for filtered data
        sig : bool, optional
        Is the peak data provided a signal(True) or noise(False) peak?  Defaults to True
        searchback : bool, optional
        Is the peak data from searching a large R-peak interval?  Defaults to False

        Returns
        -------
        tsn_ma : TSN
        Updated threshold, signal and noise moving averages for moving average data
        tsn_f : TSN
        Updated threshold, signal and noise moving averages for filtered data
        """

        if sig:
            if not searchback:
                tsn_ma.s = 0.125*ma_peak + 0.875*tsn_ma.s  # update moving average signal peak value
                tsn_f.s = 0.125*f_peak + 0.875*tsn_f.s  # update filtered signal peak value
            else:
                tsn_ma.s = 0.25*ma_peak + 0.75*tsn_ma.s
                tsn_f.s = 0.125*f_peak + 0.75*tsn_f.s
        elif not sig:
            tsn_ma.n = 0.125*ma_peak+ 0.875*tsn_ma.n  # update moving average noise peak value
            tsn_f.n = 0.125*f_peak + 0.875*tsn_f.n  # update filtered noise peak value

            tsn_ma.t = tsn_ma.n + 0.25*(tsn_ma.s-tsn_ma.n)  # update moving average signal threshold
            tsn_f.t = tsn_f.n + 0.25*(tsn_f.s-tsn_f.n)  # update filtered signal threshold

        return tsn_ma, tsn_f

    @staticmethod
    def _SplineCreation(data, dt=0.2):
        """
        Create spline data

        Parameters
        ----------
        data : array
        N by 2 array of time and parameters
        dt : float, optional
        Timestep in seconds for spline data points.  Defaults to 0.2s

        Returns
        -------
        spl : array
        M by 2 array of time and parameters, with timestep = dt
        """

        cs1 = interpolate.CubicSpline(data[:, 0], data[:, 1])  # setup spline function
        cs2 = interpolate.CubicSpline(data[:, 0], data[:, 2])  # setup 2nd spline function
        cs3 = interpolate.CubicSpline(data[:, 0], data[:, 3])  # setup 3rd spline function

        x = np.arange(data[0, 0], data[-1, 0], dt)  # setup x values.  dt second intervals
        # if x-values don't include end time, add it to the array
        if data[-1, 0] not in x:
            x = np.append(x, data[-1, 0])

        spl = np.zeros((len(x), 4))
        spl[:, 0] = x
        spl[:, 1] = cs1(x)
        spl[:, 2] = cs2(x)
        spl[:, 3] = cs3(x)

        return spl

    def PlotHeartRate(self):
        """
        Plot the results of R-peak detection by plotting Heart Rate (Hz) vs time
        """
        if 'r_pks' not in self.__dict__.keys():
            raise OrderError("Plotting Heart Rate must be done after detecting R-peaks.")
        else:
            f, ax = pl.subplots(figsize=(16, 6))

            for i in self.vd:
                ax.plot(self.r_pks[i][1:, 0]-self.r_pks[i][0, 0], 60/(self.r_pks[i][1:, 0]-self.r_pks[i][:-1, 0]),
                        label=f'{i}')

                ax.legend(title='Activity')
                ax.set_ylabel('Heart Rate [BPM]')
                ax.set_xlabel('Time [s]')

                f.tight_layout()

    def RespRateExtraction(self):
        """
        Compute parameters for respiratory rate estimation.  Compute spline for parameters
        """
        self.rr_p = dict()  # dictionary for all 3 respiratory rate parameters
        # time, FM, AM, BW

        self.rr_spl = dict()  # dictionary for all 3 respiratory rate parameter splines
        # spline time, FM spline, AM spline, BW spline

        for key in self.vd:
            # allocate array.  FM is cut 1 short due to difference
            # so exclude first of other parameters as well
            self.rr_p[key] = np.zeros((len(self.r_pks[key])-1, 4))
            self.rr_p[key][:, 0] = self.r_pks[key][1:, 0]  # set timings
            # timings assumed to be at the R-peak (second R-peak for difference)

            # FM: Charleton X_b3 - dt between successive R-peaks
            self.rr_p[key][:, 1] = self.r_pks[key][1:, 0]-self.r_pks[key][:-1, 0]

            # AM: Charleton X_b2 - amplitude difference between Q-trough and R-peak
            self.rr_p[key][:, 2] = self.r_pks[key][1:, 1]-self.q_trs[key][1:, 1]

            # BW: Charleton X_b1 - mean of R-peak and Q-trough values
            self.rr_p[key][:, 3] = np.mean([self.r_pks[key][1:, 1], self.q_trs[key][1:, 1]], axis=0)

            # create spline data points
            self.rr_spl[key] = self._SplineCreation(self.rr_p[key], dt=0.2)

    def CountOriginal(self, debug=False):
        """
        Implementation of Original Count Method from
        Axel Schafer, Karl Kratky.  "Estimation of Breathing Rate from Respiratory
        Sinus Arrhythmia: Comparison of Various Methods."  Ann. of Biomed. Engr.
        Vol 36 No 3, 2008.
        """
        self.rr = dict()
        for key in self.vd:
            self.rr[key] = dict()  # another dictionary to store obtained RR times since they will be different lengths
            # for different parameters

            rr_splf = np.zeros((len(self.rr_spl[key][:, 1]), 3))  # spline filtered allocation
            # step 1 - bandpass filter with pass region between 0.1-0.5Hz
            fs = 1/(self.rr_spl[key][1, 0]-self.rr_spl[key][0, 0])  # spline data frequency
            wl = 0.1/(0.5*fs)  # low cutoff frequency as % of nyquist frequency
            wh = 0.5/(0.5*fs)  # high cutoff freq as % of nyquist freq

            b, a = signal.butter(5, [wl, wh], 'bandpass')  # filtfilt, -> order is 2x given

            # remove mean from data and filter
            mn = np.mean(self.rr_spl[key][:, 1:], axis=0)
            rr_splf[:,0] = signal.filtfilt(b, a, self.rr_spl[key][:, 1]-mn[0])
            rr_splf[:,1] = signal.filtfilt(b, a, self.rr_spl[key][:, 2]-mn[1])
            rr_splf[:,2] = signal.filtfilt(b, a, self.rr_spl[key][:, 3]-mn[2])

            if debug:
                f, ax = pl.subplots(3,figsize=(16, 5), sharex=True)
                ax[0].plot(self.rr_spl[key][:, 0], self.rr_spl[key][:, 1], 'k--', alpha=0.5, label='initial')
                ax[1].plot(self.rr_spl[key][:, 0], self.rr_spl[key][:, 2], 'k--', alpha=0.5, label='initial')
                ax[2].plot(self.rr_spl[key][:, 0], self.rr_spl[key][:, 3], 'k--', alpha=0.5, label='initial')

                ax[0].plot(self.rr_spl[key][:, 0], rr_splf[:, 0], 'b', label='filtered')
                ax[1].plot(self.rr_spl[key][:, 0], rr_splf[:, 1], 'b', label='filtered')
                ax[2].plot(self.rr_spl[key][:, 0], rr_splf[:, 2], 'b', label='filtered')

            for i, param in zip(range(3), ['fm', 'am', 'bw']):
                # step 2 - find local minima and maxima of filtered data
                # get 3rd quartile, threshold is Q3*0.2
                minpt = signal.argrelmin(rr_splf[:, i])[0]
                maxpt = signal.argrelmax(rr_splf[:, i])[0]

                q3 = np.percentile(rr_splf[maxpt, i], 75)  # 3rd quartile (75th percentile)
                thr = 0.2*q3  # threshold

                # step 3 - valid breath cycle is max>thr, min<0, max>thr with no max/min in between
                # local extrema sorted by index
                # local max T/F.  True=>maximum, False=>minimum
                ext, etp = zip(*sorted(zip(np.append(maxpt, minpt), [True]*len(maxpt)+[False]*len(minpt))))
                ext, etp = np.array(ext), np.array(etp)

                brth_cyc = []  # initialize breath cycle array
                rr = []  # initialize respiratory rate array

                for j in range(len(ext)-2):
                    # if goes from local max to local min to local max
                    if etp[j]==True and etp[j+2]==True and etp[j+1]==False:
                        # if the maximums are above the threshold and minimum below 0
                        if rr_splf[ext[j], i] > thr and rr_splf[ext[j+2], i] > thr and rr_splf[ext[j+1], i] < 0:
                            brth_cyc.append([ext[j], ext[j+2]])
                            # append time and breathing frequency
                            rr.append([self.rr_spl[key][ext[j+1], 0], 1/(self.rr_spl[key][ext[j+2], 0]-
                                                                         self.rr_spl[key][ext[j], 0])])
                self.rr[key][param] = np.array(rr)

                if debug:
                    ax[i].plot(self.rr_spl[key][minpt, 0], rr_splf[minpt, i], '*')
                    ax[i].plot(self.rr_spl[key][maxpt, 0], rr_splf[maxpt, i], 'o')
                    ax[i].axhline(thr)
                    for st, sp in brth_cyc:
                        ax[i].plot(self.rr_spl[key][st:sp, 0], rr_splf[st:sp, i], 'r', linewidth=2, alpha=0.5)

                        ax[i].legend(title=f'Paramter: {param}')
                        ax[i].set_ylabel('R-R peak times [s]')

        if debug:
            f.tight_layout()
            f.subplots_adjust(hspace=0)
            ax[0].set_xlabel('Time [s]')

    def CountAdv(self, debug=False):
        """
        Implementation of Advanced Count Method from
        Axel Schafer, Karl Kratky.  "Estimation of Breathing Rate from Respiratory
        Sinus Arrhythmia: Comparison of Various Methods."  Ann. of Biomed. Engr.
        Vol 36 No 3, 2008.
        """
        self.rr = dict()  # allocate array for respiratory rate estimates
        self.brth_cyc = dict()  # allocate storage for breath cycle
        for key in self.vd:
            self.rr[key] = dict()  # another dictionary since lengths of arrays
            # for different parameters will possibly be different sizes

            # step 1 - bandpass filter with pass region between 0.1-0.5Hz
            fs = 1/(self.rr_spl[key][1, 0]-self.rr_spl[key][0, 0])  # spline data frequency
            wl = 0.1/(0.5*fs)  # low cutoff frequency as % of nyquist frequency
            wh = 0.5/(0.5*fs)  # high cutoff freq as % of nyquist freq

            b, a = signal.butter(5, [wl, wh], 'bandpass')  # filtfilt, -> order is 2x given

            rr_splf = np.zeros_like(self.rr_spl[key][:, 1:])
            mn = np.mean(self.rr_spl[key][:, 1:], axis=0)

            # remove mean and apply filter to input spline data
            # rr_splf[:, 0] = signal.filtfilt(b, a, self.rr_spl[key][:, 1]-mn[0])
            # rr_splf[:, 1] = signal.filtfilt(b, a, self.rr_spl[key][:, 2]-mn[1])
            # rr_splf[:, 2] = signal.filtfilt(b, a, self.rr_spl[key][:, 3]-mn[2])

            if debug:
                f, ax = pl.subplots(3, figsize=(16, 9), sharex=True)

            for i, param in zip(range(3), ['fm', 'am', 'bw']):
                # remove mean and apply filter to input spline data
                inter = signal.filtfilt(b, a, self.rr_spl[key][:, i+1]-mn[i])
                self.rr_spl[key][:, i+1] = inter

                # step 2 - find local minima and maxima of filtered data
                # get 3rd quartile, threshold is Q3*0.2
                minpt = signal.argrelmin(self.rr_spl[key][:, i+1])[0]
                maxpt = signal.argrelmax(self.rr_spl[key][:, i+1])[0]

                # step 3 - calculate absolute value diff. between subsequent local extrema
                # 3rd quartile of differences, threshold = 0.1*Q3
                ext, etp = zip(*sorted(zip(np.append(maxpt, minpt), [True]*len(maxpt)+[False]*len(minpt))))
                ext, etp = np.array(ext), np.array(etp)

                ext_diff = np.zeros(len(ext)-1)
                for j in range(len(ext)-1):
                    if etp[j] != etp[j+1]:  # ie min followed by max or max followed by min
                        ext_diff[j] = abs(self.rr_spl[key][ext[j], i+1]-self.rr_spl[key][ext[j+1], i+1])

                thr = 0.1*np.percentile(ext_diff, 75)  # threshold value

                if debug:
                    ax[i].plot(self.rr_spl[key][:, 0], self.rr_spl[key][:, i+1])
                    ax[i].plot(self.rr_spl[key][minpt, 0], self.rr_spl[key][minpt, i+1], 'k+')
                    ax[i].plot(self.rr_spl[key][maxpt, 0], self.rr_spl[key][maxpt, i+1], 'k+')

                rem = 0  # removed indices counter
                # step 4 - remove any sets whose difference is less than the threshold
                for j in range(len(ext)-1):
                    if etp[j-rem] != etp[j-rem+1]:
                        if abs(self.rr_spl[key][ext[j-rem], i+1]-self.rr_spl[key][ext[j-rem+1], i+1]) < thr:
                            ext = np.delete(ext, [j-rem, j-rem+1])
                            etp = np.delete(etp, [j-rem, j-rem+1])
                            rem += 2

                # step 5 - breath cycles are now minimum -> maximum -> minimum = 1 cycle
                brth_cyc = []  # initialize breath cycle array
                brth_cyc_xnx = []  # try maX-miN-maX cycles to see if more than min-max-min

                for j in range(len(ext)-2):
                    if not etp[j]and etp[j+1] and not etp[j+2]:
                        brth_cyc.append([ext[j], ext[j+2]])
                    elif etp[j] and not etp[j+1] and etp[j+2]:
                        brth_cyc_xnx.append([ext[j], ext[j+2]])

                self.rr[key][param] = np.zeros((max([len(brth_cyc), len(brth_cyc_xnx)]), 2))

                if len(brth_cyc) < len(brth_cyc_xnx):
                    brth_cyc = brth_cyc_xnx

                for j in range(len(self.rr[key][param][:, 0])):
                    self.rr[key][param][j] = [(self.rr_spl[key][brth_cyc[j][1], 0]+self.rr_spl[key][brth_cyc[j][0], 0])
                                              / 2, 1/(self.rr_spl[key][brth_cyc[j][1], 0] -
                                                      self.rr_spl[key][brth_cyc[j][0], 0])]
                self.brth_cyc[key] = brth_cyc  # store in class attributes
                if debug:
                    ax[i].plot(self.rr_spl[key][ext, 0], self.rr_spl[key][ext, i+1], 'ro', alpha=0.5, markersize=0.5)

                    for st, sp in brth_cyc:
                        ax[i].plot(self.rr_spl[key][st:sp, 0], self.rr_spl[key][st:sp, i+1], 'r', alpha=0.25,
                                   linewidth=5)
        if debug:
            f.tight_layout()
            f.subplots_adjust(hspace=0)

    def SmartModFusion(self, use_given_time=True, plot=True):
        """
        Modulation smart fusion of bandwidth, AM, and FM respiratory estimates.
        From Karlen et al, 2013.  Respiratory rate is output of standard deviation of
        estimated rate between BW, AM, and FM estimated rates is less than 4 BPM.

        Parameters
        ---------
        use_given_time : bool
            Use timings from parameter resp. rate estimates (True) or use 0.2s interpolation (False).  Defaults to True.
        plot : bool
        Plot resulting fused respiratory rate estimates
        """
        self.rr_fuse = dict()  # allocate dictionary for fused resp. rate estimates
        self.lqi = dict()
        self.st_dev = dict()
        for key in self.vd:
            # convert to BPM
            for param in self.rr[key].keys():
                self.rr[key][param][:, 1] *= 60

            # calculate spline functions
            fmsf = interpolate.CubicSpline(self.rr[key]['fm'][:, 0], self.rr[key]['fm'][:, 1])
            amsf = interpolate.CubicSpline(self.rr[key]['am'][:, 0], self.rr[key]['am'][:, 1])
            bwsf = interpolate.CubicSpline(self.rr[key]['bw'][:, 0], self.rr[key]['bw'][:, 1])

            if use_given_time:
                x = np.unique(np.concatenate((self.rr[key]['fm'][:, 0], self.rr[key]['am'][:, 0],
                                              self.rr[key]['bw'][:, 0])))
            else:
                tmin = max([self.rr[key]['fm'][0, 0], self.rr[key]['am'][0, 0], self.rr[key]['bw'][0, 0]])
                tmax = min([self.rr[key]['fm'][-1, 0], self.rr[key]['am'][-1, 0], self.rr[key]['bw'][-1, 0]])
                x = np.arange(tmin, tmax, 0.2)
            self.lqi[key] = np.array([False]*len(x))  # Low Quality Index

            fms = fmsf(x)
            ams = amsf(x)
            bws = bwsf(x)

            self.st_dev[key] = np.std(np.array([bws, ams, fms]), axis=0)
            rr = np.mean(np.array([bws, ams, fms]), axis=0)

            for i in range(len(self.st_dev[key])):
                if self.st_dev[key][i] > 4:
                    self.lqi[key][i] = True  # if Standard Dev > 4 BPM Low Quality Index

            self.rr_fuse[key] = np.zeros((len(rr), 2))
            self.rr_fuse[key][:, 1] = rr
            self.rr_fuse[key][:, 0] = x

            if plot:
                f, ax = pl.subplots(figsize=(16, 5))
                line1, = ax.plot(self.rr_fuse[key][:, 0], self.rr_fuse[key][:, 1], label='Fused Est.')

                ind = np.argwhere(self.lqi[key][1:] != self.lqi[key][:-1]).flatten() + 1
                ind = np.append(ind, len(self.lqi[key]))
                ind = np.insert(ind, 0, 0)
                for i1, i2 in zip(ind[:-1], ind[1:]+1):
                    ax.fill_between(x[i1:i2], rr[i1:i2]-self.st_dev[key][i1:i2], rr[i1:i2]+self.st_dev[key][i1:i2],
                                    alpha=0.5, color='red' if self.lqi[key][i1] else 'blue')
                ax.set_title(f'{key}')

                b_p = mpatches.Patch(color='blue', alpha=0.5, label='St. Dev. < 4 BPM')
                r_p = mpatches.Patch(color='red', alpha=0.5, label='St. Dev. > 4 BPM')
                ax.legend(handles=[b_p, r_p, line1])


# f = open('..\\sample_EcgAnalysis_v2_data.pickle', 'rb')
# data = pickle_load(f)
# test = ECGAnalysis(data['KJ'])
