"""
V0.1
September 12/29/2017

Lukas Adamowicz

Python 3.6.3 on Windows 10 with 64-bit Anaconda
"""
import numpy as np
import matplotlib.pyplot as pl
from scipy import signal, interpolate

class TSN(object):
	"""
	Class for storing Threshold, Signal, Noise values for R-peak detection
	"""
	def __init__(self):
		self.thr = None #threshold
		self.spk = None #signal peak moving average
		self.npk = None #noise peak moving average

class ECGAnalysis(object):
	def __init__(self,fs):
		"""
		Class defining steps for analyzing ECG data to obtain an estimation 
		of Respiratory Rate.
		
		Parameters
		---------
		fs : float
			Sampling frequency of input data
		"""
		
		self.fs = fs
		self.f_nyq = 0.5*self.fs #nyquist frequency is half sampling frequency
	
	#Step 1 - eliminate low frequencies
	def ElimLowFreq(self,data,cutoff=3,N=1,debug=False):
		"""
		Eliminate low frequencies for ECG data.
		
		Parameters
		---------
		data : ndarray
			ECG voltage data to filter
		cutoff : float, optional
			Cuttoff frequency for filter, -3dB limit.  Defaults to 3Hz
		N : int, optional
			filter order, higher is sharper cutoff.  Defaults to 1
		debug : bool, optional
			Print output graphs for debugging.  Defaults to False
		
		Returns
		------
		data_filt : ndarray
			High pass filtered data array
		"""
		w_cut = cutoff/self.f_nyq #cutoff frequency as percentage of nyquist frequency
		
		b,a = signal.butter(N,w_cut,'highpass') #setup highpass filter
		data_filt = signal.filtfilt(b,a,data) #filter data
		
		if debug==True:
			f,ax = pl.subplots(figsize=(9,5))
			ax.plot(data,label='initial')
			ax.plot(data_filt,label='filtered')
			ax.set_xlabel('Sample No.')
			ax.set_ylabel('Voltage [mV]')
		
		return data_filt
	
	#Step 2 - Eliminate very high frequencies
	def ElimVeryHighFreq(self,data,cutoff=20,N=1,detrend=True,debug=False):
		"""
		Eliminate very high frequencies for ECG data.
		
		Parameters
		---------
		data : ndarray
			ECG voltage data after Step 1
		cutoff : float, optional
			Cutoff frequency for filter, -3dB limit.  Defaults to 20 Hz
		N : int, optional
			Filter order, higher is sharper cutoff.  Defaults to 1
		detrend : bool, optional
			Detrend data.  Defaults to True
		debug : bool, optional
			Print output graphs for debugging.   Defaults to False
		
		Returns
		-------
		data_filt : ndarray
			Low-pass filtered data array
		"""
		w_cut = cutoff/self.f_nyq #cutoff frequency as percentage of nyquist frequency
		
		b,a = signal.butter(N,w_cut,'lowpass') #setup filter parameters
		
		#detrend - remove mean, linear associations
		if detrend==True:
			data_det = signal.detrend(data) #detrend data
			data_filt = signal.filtfilt(b,a,data_det) #filter detrended data
		else:
			data_filt = signal.filtfilt(b,a,data) #filter input data
			
		if debug==True:
			f,ax = pl.subplots(figsize=(9,5))
			ax.plot(data,label='initial')
			ax.plot(data_filt,label='filtered')
			ax.set_xlabel('Sample No.')
			ax.set_ylabel('Voltage [mV]')
			
		return data_filt
	
	#Step 3 - Eliminate Mains frequency (frequency of electrical signals - 60Hz for US)
	def ElimMainsFreq(self,data,cutoff=60,Q=10,detrend=True,debug=False):
		"""
		Eliminate Mains frequency caused by electronics.
		60Hz in Americas.  50Hz in Europe.
		
		Parameters
		---------
		data : float
			ECG voltage data after Step 2
		cutoff : float, optional
			Cutoff frequency for notch filter.  Defaults to 60 Hz (US)
		Q : int, optional
			Filter order, higher is sharper cutoff.  Defaults to 10
		detrend : bool, optional
			Detrend data.  Defaults to True
		debug : bool, optional
			Print output graphs for debugging.  Defaults to False
		
		Returns
		-------
		data_filt : ndarray
			Notch filtered data
		"""
		w0 = cutoff/self.f_nyq
		b,a = signal.iirnotch(w0,Q)
		
		if detrend==True:
			data_det = signal.detrend(data) #detrend data
			data_filt = signal.filtfilt(b,a,data_det) #filter detrended data
		else:
			data_filt = signal.filtfilt(b,a,data) #detrend input data
		
		if debug==True:
			f,ax = pl.subplots(figsize=(9,5))
			ax.plot(data,label='initial')
			ax.plot(data_filt,label='filtered')
			ax.set_xlabel('Sample No.')
			ax.set_ylabel('Voltage [mV]')
			
		return data_filt
	
	#Step 3.5 - Lead Inversion Check
	def CheckLeadInversion(self,data,debug=False):
		"""
		Check for lead inversion by examining percentage of local extrema that are negative.
		R-peaks (maximum local extrema) should be positive.
		
		Parameters
		---------
		data : ndarray
			ECG volage data.  Usually after step 3, but can be after import
		debug : bool, optional
			Print peak values for debugging.  Defaults to False
		
		Returns
		-------
		data_check : ndarray
			Correct orientation float of voltage data for ECG signal
		lead_inv : bool
			Boolean if leads were inverted.  False if not inverted, True if inverted
		"""
		
		d2 = data**2 #square data for all positive and prominent peaks
		d2_max = max(d2) #maximum of squared data
		d2_max_pks = np.zeros_like(d2) #allocate array for peaks
		
		#indices where squared data is greater than 20% of squared max
		inds = np.argwhere(d2>0.2*d2_max) 
		d2_max_pks[inds] = d2[inds] #squared values greater than 20% max, 0s elsewhere
		d2_pks = signal.argrelmax(d2_max_pks)[0] #locations of maximum peaks
		
		d_pks = data[d2_pks] #data values at local extrema (found from squard values)
		
		#percentage of peaks with values that are negative
		p_neg_pks = len(np.where(d_pks<0)[0])/len(d_pks) 
		
		if debug==True:
			x = np.array([i for i in range(len(data))])
			f,ax = pl.subplots(figsize=(9,5))
			ax.plot(x,data)
			ax.plot(x[d2_pks],data[d2_pks],'+')
			ax.set_xlabel('Sample No.')
			ax.set_ylabel('Voltage [mV]')
		
		if p_neg_pks>=0.5:
			return -data, True
		elif p_neg_pks<0.5:
			return data, False
	
	#Step 4 - Eliminate sub-cardiac frequencies
	def ElimSubCardiacFreq(self,data,cutoff=0.5,N=4,debug=False):
		"""
		Eliminate sub-cardiac frequencies (30 BPM - 0.5Hz)
		
		Parameters
		---------
		data : ndarray
			ECG voltage data, correctly oriented and passed through stages 1-3 of filtering
		cutoff : float, optional
			Cutoff frequency for high-pass filter.  Defaults to 0.5 Hz (30 BPM)
		N : int, optional
			Filter order.  Defaults to 4
		debug : bool
			Graph input and output data.  Defaults to False
		
		Returns
		-------
		data_filt : ndarray
			High-pass filtered data
		"""
		w_cut = cutoff/self.f_nyq #define cutoff frequency as % of nyq. freq.
		b,a = signal.butter(N,w_cut,'highpass') #setup high pass filter
		
		data_filt = signal.filtfilt(b,a,data) #backwards-forwards filter
		
		if debug==True:
			f,ax = pl.subplots(figsize=(9,5))
			ax.plot(data,label='initial')
			ax.plot(data_filt,label='filtered')
			ax.legend()
			ax.set_xlabel('Sample No.')
			ax.set_ylabel('Voltage [mV]')
		
		return data_filt
	
	def DerivativeFilter(self,data,time,plot=False):
		"""
		Calculated derivative of data.
		
		Parameters
		---------
		data : ndarray
			ECG voltage data after steps 1-4 filtering and lead inversion check
		time : ndarray
			ECG sample timings
		plot : bool
			Plot resulting data.  Defaults to False
		
		Returns
		-------
		data_der : ndarray
			Derivative of input data
		time_der : ndarray
			Associated timings.  Some cut by derivative operation
		"""
		dt = time[1]-time[0] #timestep
		data_der = (-data[:-4]-2*data[1:-3]+2*data[3:-1]+data[4:])/(8*dt) #derivative
		
		if plot==True:
			f,ax = pl.subplots(figsize=(9,5))
			ax.plot(time[2:-2],data_der,label='Derivative')
			ax.legend()
		
		return data_der, time[2:-2] #timings are shortened at start and end by 2 samples
	
	def _SquaredFilter(self,data):
		"""
		Square values of data.
		
		Parameters
		---------
		data : ndarray
			ECG voltage data after derivative filter applied
		
		Returns
		------
		data_sq : ndarray
			Square filter data
		"""
		return data**2
	
	def IntegratedAverageFilter(self,data,time,width=150,plot=False):
		"""
		Average by integration of data.
		
		Parameters
		---------
		data : ndarray
			ECG voltage data after squaring
		time : ndarray
			Timings for samples associated with ECG data in seconds
		width : int, optional
			Time width (ms) of integration window.  Defaults to 150ms (0.15s)
		plot : bool
			Plot output data. Defaults to False
			
		Returns
		-------
		data_int : ndarray
			Average by integration data
		time_int : ndarray
			Associated timings, cut by proper amount for integration-average filter
		"""
		dt = time[1]-time[0] #timestep
		
		#number of samples to use for integration average window
		#window width in seconds/timestep
		N = int((width/1000)/dt)
		
		data_int = (1/N)*np.array([sum(data[i-N:i+1]) for i in range(N,len(data))])
		
		if plot==True:
			pl.figure(figsize=(9,5))
			pl.plot(time[N:],data_int,label='Integrated')
			pl.legend()
			pl.xlabel('Time [s]')
		
		return data_int, time[N:] #timings shortened only in the beginning
		
	def FindPeaksLearning(self,v_f,t_f,v_int,v_der,width=150,t_learn=8,delay=175,debug=False):
		"""
		Learning function for R-peak detection.
		
		Parameters
		---------
		v_f : ndarray
			FilteredECG voltage signal (ie after Sub-cardiac freq elimination)
		t_f : ndarray
			Timings associated with bandpassed ECG signal
		v_int : ndarray
			Average by integration voltage signal.
		v_der : ndarray
			Derivative voltage signal
		width : float, optional
			Integration window width in milliseconds.  Defaults to 150ms
		t_learn : float, optional
			Amount of time (seconds) for learning algorithm to work on.  Defaults to 8s
		t_delay : float, optional
			Amount of time (ms) to wait before declaring peak if 1/2 amplitude not found.  
			Defaults to 175ms
		
		Returns
		-------
		rr8 : float
			Initial average R-R peak times
		tsn_i : TSN
			Integrated data thresholds and signal and noise moving averages
		tsn_f : 
			Filtered data thresholds and signal and noise moving averages
		"""
		dt = t_f[1]-t_f[0]
		n_dly = int(delay/1000*self.fs) #number of samples corresponding to delay
		#number of samples in 25,125,225ms
		n_25,n_125,n_225 = int(0.025*self.fs),int(0.125*self.fs),int(0.225*self.fs)
		n_lrn = int(t_learn*self.fs) #number of samples in t_learn seconds
		
		#append 0s to front of integrated signal
		v_int = np.append([0]*(len(v_f)-len(v_int)-2),v_int)
		#number of 0s needs to be changed if different derivative scheme
		v_der = np.append([0,0],v_der)
		
		v_f = v_f[:n_lrn] #want only beginning t_learn seconds
		v_int = v_int[:n_lrn]
		v_der = v_der[:n_lrn]
		t_f = t_f[:n_lrn]
		
		#peaks in region with width equal to integration width
		ii_pks = signal.argrelmax(v_int,order=int(0.5*width/1000/dt))[0] 
		
		#remove any peaks that are less than 0.1% of mean of the peaks
		ii_pks = ii_pks[np.where(v_int[ii_pks]>0.001*np.mean(v_int[ii_pks]))[0]]
		
		#get integrated signal values
		vi_pks = v_int[ii_pks]
		ti_pks = t_f[ii_pks]
		
		m_pos = [] #maximum positive slope in the t_delay region around each peak (+- t_delay)
		pki_h = [] #descending half peak value points
		band_pk = [] #max points in v_f preceeding descending half peak value points
		
		r_pk_b = [] #Boolean array for if the point is an R peak or not
		
		for i in range(len(ii_pks)):
			#append max slope for each peak
			m_pos.append(max(v_der[ii_pks[i]-n_dly:ii_pks[i]+n_dly])) 
			try: #to find descending half peak value for each peak
				pki_h.append(np.where(v_int[ii_pks[i]:]<0.5*vi_pks[i])[0][0]+ii_pks[i])
				longwave = False #not a long QRS wave
			except: #if not found, set integration half peak value to max slope + delay 
				pki_h.append(np.argmax(v_der[ii_pks[i]-n_dly:ii_pks[i]+n_dly])[0]\
											 + (ii_pks-n_dly) + n_dly)
				longwave = True
			
			if longwave==False: #search for max filt. peak in preceeding 125-225 ms
				band_pk.append(np.argmax(v_f[pki_h[i]-n_225:pki_h[i]-n_125])+pki_h[i]-n_225)
			elif longwave==True: #if longwave, search in preceeding 150-250ms
				band_pk.append(np.argmax(v_f[pki_h[i]-n_225-n_25:pki_h[i]-n_125-n_25])\
								   +pki_h[i]-n_225-n_25)
		band_pk = np.array(band_pk)
		
		#determing if R peak or T wave
		for i in range(0,len(ii_pks),2):
			#if time between peaks is less than 200ms, first is R, second is T
			#Due to being faster than physiologically possible
			if t_f[band_pk[i+1]]-t_f[band_pk[i]]<0.2:
				r_pk_b.append([True,False])
			#if time between peaks is less than 360ms but greater than 200ms
			elif t_f[band_pk[i+1]]-t_f[band_pk[i]]<0.36:
				#then check the slope.  If slope i is twice as steep as slope i+1, 
				#point i is the qrs complex, point i+1 a t complex
				if 0.5*m_pos[i] > m_pos[i+1]:
					r_pk_b.append([True,False])
				#if slope i is less than half as steep as slope i+1, 
				#then i+1 is a qrs complex, i a t-wave
				elif m_pos[i] < 0.5*m_pos[i+1]:
					r_pk_b.append([False,True])
				#if neither slope is smaller than half the other, they are both qrs complexes
				elif 0.5*m_pos[i] < m_pos[i+1] and m_pos[i] > 0.5*m_pos[i+1]:
					r_pk_b.append([True,True])
			#if time between peaks is greater than 360ms, both are qrs peaks
			elif t_f[band_pk[i+1]]-t_f[band_pk[i]]>0.36:
				r_pk_b.append([True,True])
		
		#assign r-peak values and timings, as well as t-wave values and timings		
		r_v = v_f[band_pk[np.argwhere(np.array(r_pk_b).flatten()==True).flatten()]]
		r_t = t_f[band_pk[np.argwhere(np.array(r_pk_b).flatten()==True).flatten()]]
		t_v = v_f[band_pk[np.argwhere(np.array(r_pk_b).flatten()==False).flatten()]]
		t_t = t_f[band_pk[np.argwhere(np.array(r_pk_b).flatten()==False).flatten()]]
		
		#Create R-R average arrays
		if len(r_t)<=8:
			rr8 = np.ones(8)*r_t[1]-r_t[0]
			rr8[-len(r_t)+1:] = np.array(r_t[1:])-np.array(r_t[:-1])
		elif len(r_t)>8:
			rr8 = np.array(r_t[len(r_t)-8:])-np.array(r_t[len(r_t)-9:-1])
		
		tsn_i = TSN() #tsn for integrated data
		tsn_f = TSN() #tsn for filtered data
		
		#initialize signal peak for integrated data
		tsn_i.spk = 0.125*vi_pks[np.where(v_f[band_pk]==r_v[0])[0][0]]
		#initialize noise peak for integrated data
		tsn_i.npk = 0.125*vi_pks[np.where(v_f[band_pk]==t_v[0])[0][0]]
		tsn_f.spk = 0.125*r_v[0] #initialize signal peak for filtered data
		tsn_f.npk = 0.125*t_v[0] #initialize noise peak for filtered data
		for i in range(1,len(r_v)):
			tsn_i.spk = 0.125*vi_pks[np.where(v_f[band_pk]==r_v[i])[0][0]] + 0.875*tsn_i.spk
			tsn_i.npk = 0.125*vi_pks[np.where(v_f[band_pk]==t_v[0])[0][0]] + 0.875*tsn_i.npk
			tsn_f.spk = 0.125*r_v[i] + 0.875*tsn_f.spk
			tsn_f.npk = 0.125*t_v[i] + 0.875*tsn_f.npk
			tsn_i.t = tsn_i.npk + 0.25*(tsn_i.spk-tsn_i.npk) #threshold for integrated data
			tsn_f.t = tsn_f.npk + 0.25*(tsn_f.spk-tsn_f.npk) #threshold for filtered data
			
		if debug==True:
			f,ax = pl.subplots(2,figsize=(9,5),sharex=True)
			ax[0].plot(t_f,v_int)
			ax[0].plot(ti_pks,vi_pks,'o')
			ax[0].plot(t_f[pki_h],v_int[pki_h],'o')
			ax[0].axhline(tsn_i.t,linestyle='--',color='k')
			ax[1].plot(t_f,v_f)
			ax[1].plot(r_t,r_v,'ro')
			ax[1].plot(t_t,t_v,'go')
			ax[1].axhline(tsn_f.t,linestyle='--',color='k')
			pl.tight_layout()
			
		return rr8, tsn_i, tsn_f
	
	@staticmethod
	def _UpdateAvgRR(rr_int,rr8,rr8_lim):
		"""
		Update R-R 8 value storage arrays for R-peak detection
		
		Parameters
		---------
		rr_int : float
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
		if rr_int>0.92*np.mean(rr8) and rr_int<1.16*np.mean(rr8):
			rr8_lim[:-1] = rr8_lim[1:]
			rr8_lim[-1] = rr_int
			
		rr8[:-1] = rr8[1:] #move values down 1 (ie remove oldest)
		rr8[-1] = rr_int #insert newest value in last spot in list/array
		
		return rr8, rr8_lim
	
	@staticmethod
	def _UpdateThresholds(i_peak,f_peak,tsn_i,tsn_f,signal=True):
		"""
		Update thresholds for R-peak detection
		
		Parameters
		----------
		i_peak : float
			Integrated signal peak
		f_peak : float
			Filtered signal peak
		tsn_i : TSN
			Threshold, signal and noise moving averages for integrated data
		tsn_f : TSN
			Threshold, signal and noise moving averages for integrated data
		signal : bool
			Is the peak data provided a signal(True) or noise(False) peak.  Defaults to True
			
		Returns
		-------
		tsn_i : TSN
			Updated threshold, signal and noise moving averages for integrated data
		tsn_f : TSN
			Updated threshold, signal and noise moving averages for integrated data
		"""
		if signal==True:
			tsn_i.spk = 0.125*i_peak + 0.875*tsn_i.spk #update integrated signal peak value
			tsn_f.spk = 0.125*f_peak + 0.875*tsn_f.spk #update filtered signal peak value
		elif signal==False:
			tsn_i.npk = 0.125*i_peak+ 0.875*tsn_i.npk #update integrated noise peak value
			tsn_f.npk = 0.125*f_peak + 0.875*tsn_f.npk #update filtered noise peak value

		tsn_i.t = tsn_i.npk + 0.25*(tsn_i.spk-tsn_i.npk) #update integrated signal threshold
		tsn_f.t = tsn_f.npk + 0.25*(tsn_f.spk-tsn_f.npk) #update filtered signal threshold
		
		return tsn_i,tsn_f
	
	def FindRPeaks(self,v_f,t_f,v_i,v_d,rr8,tsn_i,tsn_f,width=150,delay=175,debug=False):
		"""
		Algorithm implementation to find R-peaks in ECG of Hamilton and Tompkins, 1986.  
		"Quantitative Investigation of QRS Detection Rules Using the 
		MIT/BIH Arrhythmia Database"
		
		Parameters
		----------
		v_f : ndarray
			Filtered ECG data
		t_f : ndarray
			Timings associated with filtered ecg data
		v_i : ndarray
			Integrated average data
		v_d : ndarray
			Derivative signal data
		rr8 : list of floats
			List/array of R-R timings from the FindPeaksLearning function output
		tsn_i : TSN
			Threshold, signal, noise peak values for integrated data from 
			FindPeaksLearning output
		tsn_f : TSN
			Threshold, signal, noise peak values for filtered data from 
			FindPeaksLearning output
		width : float, optional
			TIme (ms) of integration window for integration step.  Defaults to 150ms
		delay : float, optional
			Time (ms) to wait before declaring peak if descending 
			half max peak value not found.  Defaults to 175ms
		
		Returns
		-------
		r_pks : ndarray
			N-by-2 array of [r-peak voltage,r-peak timestamp]
		q_trs : ndarray
			N-by-2 array of [q-trough voltage, q-trough timestamp]
		"""
		
		v_d = np.append(v_d,[0,0])
		v_d = np.append([0,0],v_d)
		v_i = np.append(v_i,[0]*2) #append 2 zeros for derivative change
		#append zeros to front for integration and d/dx change
		v_i = np.append([0]*int(len(v_f)-len(v_i)),v_i)
		
		dt = t_f[1]-t_f[0] #time difference between samples
		
		#number of samples in 25,125,225ms
		n25,n125,n225 = int(0.025*self.fs),int(0.125*self.fs),int(0.225*self.fs)  
		nw = int(width/1000*self.fs) #number of samples in width ms
		nd = int(delay/1000*self.fs) #number of samples in delay ms
		
		rr8_lim = np.zeros(8) #initialize limited R-R array
		
		#peak indices in region with width equal to integration window width
		vi_pts = signal.argrelmax(v_i,order=int(0.5*width/1000/dt))[0]
		#descending half peak value initilization.  Stores indices 
		vi_hpt = np.zeros_like(vi_pts)
		#maximum values and indices for each peak in filtered data
		vf_pks = np.zeros((len(vi_pts),2))
		
		#maximum slopes and indices for each peak.  
		#Initialized as +1 due to comparison with previous slope later
		m_pos = np.zeros((len(vi_pts)+1,2))
		r_pks = np.zeros((1,2)) #initialize vector for R peak values and timestamps
		
		for i in range(len(vi_pts)):
			#finding maximum slope in the +- width surrounding each peak
			if vi_pts[i]-nw > 0 and vi_pts[i]+nw<len(v_i): #if +- window is fully in data range
				m_pos[i+1] = [max(v_d[vi_pts[i]-nw:vi_pts[i]+nw])+int(vi_pts[i]-nw),\
								 np.argmax(v_d[vi_pts[i]-nw:vi_pts[i]+nw])+int(vi_pts[i]-nw)]
			elif vi_pts[i]-nw < 0: #if the window goes before data range
				m_pos[i+1] = [max(v_d[0:vi_pts[i]+nw]),np.argmax(v_d[0:vi_pts[i]+nw])]
			elif vi_pts[i]+nw > len(v_i): #if the window goes outside end of data range
				m_pos[i+1] = [max(v_d[vi_pts[i]-nw:]),np.argmax(v_d[vi_pts[i]-nw:])]
			
			#finding descending half peak value
			try:
				vi_hpt[i] = np.where(v_i[vi_pts[i]:int(m_pos[i+1,1])+nd] \
											 <0.5*v_i[vi_pts[i]])[0][0] + vi_pts[i]
				longwave=False #not a long QRS wave complex
			except:
				vi_hpt[i] = vi_pts[i] + nd
				longwave=True #long QRS wave complex
			
			#find maximum values in filtered data preceeding descending half peak values
			if longwave==False: #if not a long wave search preceeding 225 to 125ms
				vf_pks[i] = [max(v_f[vi_hpt[i]-n225:vi_hpt[i]-n125]),\
								  np.argmax(v_f[vi_hpt[i]-n225:vi_hpt[i]-n125])+vi_hpt[i]-n225]
			elif longwave==True: #if long wave, search preceeding 250 to 150 ms
				vf_pks[i] = [max(v_f[vi_hpt[i]-n225-n25:vi_hpt[i]-n125-n25]),\
					  np.argmax(v_f[vi_hpt[i]-n225-n25:vi_hpt[i]-n125-n25])+vi_hpt[i]-n225-n25]
				
			#Determine type of peak (R,T, etc)
			#if the peaks are above the thresholds and time between is greater than 0.36s		
			if v_i[vi_pts[i]] > tsn_i.t and vf_pks[i,0] > tsn_f.t and \
										(t_f[int(vf_pks[i,1])]-r_pks[-1,1])>=0.36:
				r_pks = np.append(r_pks,[[vf_pks[i,0],t_f[int(vf_pks[i,1])]]],axis=0)
				j = i+1 #assign key value.  Index of the last detected r_peak
				tsn_i,tsn_f = self._UpdateThresholds(v_i[vi_pts[i]],r_pks[-1,0],\
																 tsn_i,tsn_f,signal=True)
			
			#peaks above thresholds, time between is greater than 0.2s but less than 0.36s
			elif v_i[vi_pts[i]] > tsn_i.t and vf_pks[i,0] > tsn_f.t and \
										(t_f[vf_pks[i,1]]-r_pks[-1,1])>0.2:
				#if the maximum associated slope is greater than half the previous 
				#detected R wave
				if m_pos[i+1,0] > 0.5*m_pos[j,0]: #it is a R peak
					r_pks = np.append(r_pks,[vf_pks[i,0],t_f[v_f[i,1]]],axis=0)
					j = i+1
					tsn_i,tsn_f = self._UpdateThresholds(v_i[vi_pts[i]],r_pks[-1,0],\
																	  tsn_i,tsn_f,signal=True)
				else: #it is a peak
					tsn_i,tsn_f = self._UpdateThresholds(v_i[vi_pts[i]],v_f[vi_pts[i]],\
																	  tsn_i,tsn_f,signal=False)
			
			else: #if not above the thresholds it is a noise peak
				tsn_i,tsn_f = self._UpdateThresholds(v_i[vi_pts[i]],v_f[vi_pts[i]],\
																 tsn_i,tsn_f,signal=False)

			##########################################################################
			#     Missing check for missing r-peaks by looking at peaks between      #
			#     consecutive R-peaks that the time between is greater than          #
			#     166% of the limited average: avg(rr8_lim)  	                       #
			##########################################################################
			
			if len(r_pks)>2: #if there have been 2 R-peaks detected
				rr8,rr8_lim = self._UpdateAvgRR(r_pks[-1,1]-r_pks[-2,1],rr8,rr8_lim)
			
		r_pks = r_pks[1:,:] #trim off first initialized entry
		q_trs = np.zeros_like(r_pks) #initialize Q-troughs array
		for i in range(len(r_pks[:,0])):
			i_rpk = np.where(t_f==r_pks[i,1])[0][0] #index of R-peak
			#look in preceeding 0.1s of R-peak for the minimum
			q_trs[i] = [min(v_f[i_rpk-int(0.1*self.fs):i_rpk]),\
						 t_f[np.argmin(v_f[i_rpk-int(0.1*self.fs):i_rpk])+i_rpk-int(0.1*self.fs)]]
		
		if debug==True:
			f,ax = pl.subplots(2,figsize=(9,5),sharex=True)
			ax[0].plot(t_f,v_f,label='filtered')
			ax[0].plot(r_pks[:,1],r_pks[:,0],'r+')
			ax[0].plot(q_trs[:,1],q_trs[:,0],'k+')
			ax[0].legend(loc='best')
			
			ax[1].plot(t_f,v_i,label='integrated')
			try:
				ax[1].plot(t_f[vi_hpt],v_i[vi_hpt],'ko')
			except:
				ax[1].plot(t_f[vi_hpt[:-1]],v_i[vi_hpt[:-1]],'ko')
			ax[1].plot(t_f[vi_pts],v_i[vi_pts],'ro')
			ax[1].legend(loc='best')
			ax[1].set_xlabel('Time [s]')
			
			f.tight_layout()
			f.subplots_adjust(hspace=0)
		
		return r_pks,q_trs
	
	def RespRateExtraction(self,r_pks,q_trs):
		"""
		Compute parameters for respiratory rate estimation.
		
		Parameters
		---------
		r_pks : ndarray
			N-by-2 array of [R-peak voltage, R-peak timestamp]
		q_trs : ndarray
			N-by-2 array of [Q-trough voltage, Q-trough timestamp]
		
		Returns
		-------
		bw : ndarray
			Array of N voltages.  Mean of associated troughs and peaks
		am : ndarray
			Array of N voltages.  Difference between associated troughs and peaks
		fm : ndarray
			Array of N-1 voltages.  Difference in time between consecutive R-peaks
		t_fm : ndarray
			Array of N-1 times associated with 'fm' values
		"""
		bw = np.mean([r_pks[:,0],q_trs[:,0]],axis=0) #X_b1 from Charlton paper
		am = r_pks[:,0]-q_trs[:,0] #X_b2
		fm = r_pks[1:,1]-r_pks[:-1,1] #X_b3
		t_fm = (r_pks[1:,1]+r_pks[:-1,1])/2
		
		return bw, am, fm, t_fm
	
	def FMSplineInterpolate(self,fm,t_fm,debug=False):
		"""
		Spline Interpolation of frequency modulated data
		
		Parameters
		---------
		fm : ndarray
			Array of N-1 voltages.  Difference in time between consecutive R-peaks
		t_fm : ndarray
			Array of N-1 times associated with 'fm' values
		
		Returns
		-------
		xs : ndarray
			Array of x-values for computed spline
		ys : ndarray
			Array of y-values for computed spline
		"""
		cs = interpolate.CubicSpline(t_fm,fm) #setup spline function
		xs = np.arange(t_fm[0],t_fm[-1],0.2) #setup x values.  0.2s intervals
		
		#if x-values don't include end time, add it to the array
		if xs[-1] != t_fm[-1]:
			xs = np.append(xs,t_fm[-1])
			
		if debug==True:
			pl.figure()
			pl.plot(t_fm,fm,'o',label='fm values')
			pl.plot(xs,cs(xs),label='spline')
			pl.legend()
		
		return xs, cs(xs) #xy, ys
	
	def CountOriginal(self,data,t,debug=False):
		"""
		Implementation of Original Count Method from 
		Axel Schafer, Karl Kratky.  "Estimation of Breathing Rate from Respiratory
		Sinus Arrhythmia: Comparison of Various Methods."  Ann. of Biomed. Engr.
		Vol 36 No 3, 2008.
		
		Parameters
		----------
		data : ndarray
			Array of data of interest.  Originally used for R-R interval timings.
		t : ndarray
			Timings for data of interest
			
		Returns
		------
		rr : ndarray
			N-by-2 array of respiration frequencies (Beats per sec) and timings
		"""
		#step 1 - bandpass filter with pass region between 0.1-0.5Hz
		fs = 1/(t[1]-t[0]) #spline data frequency
		wl = 0.1/(0.5*fs) #low cutoff frequency as % of nyquist frequency
		wh = 0.5/(0.5*fs) #high cutoff freq as % of nyquist freq
		
		b,a = signal.butter(5,[wl,wh],'bandpass') #filtfilt, -> order is 2x given
		
		data -= np.mean(data) #remove mean from data
		df = signal.filtfilt(b,a,data) #apply filter to input data
		
		#step 2 - find local minima and maxima of filtered data
		# get 3rd quartile, threshold is Q3*0.2
		minpt = signal.argrelmin(df)[0]
		maxpt = signal.argrelmax(df)[0]
		
		q3 = np.percentile(df[maxpt],75) #3rd quartile (75th percentile)
		thr = 0.2*q3 #threshold
		
		#step 3 - valid breath cycle is max>thr, min<0,max>thr with no max/min in between
		#local extrema sorted by index
		#local max T/F.  True=>maximum, False=>minimum
		ext,etp = zip(*sorted(zip(np.append(maxpt,minpt),\
										[True]*len(maxpt)+[False]*len(minpt))))
		ext,etp = np.array(ext),np.array(etp)
		
		brth_cyc = [] #initialize breath cycle array
		rr = np.zeros((1,2)) #initialize respiratory rate array
		
		for i in range(len(ext)-2):
			if etp[i]==True and etp[i+2]==True and etp[i+1]==False:
				if df[ext[i]]>thr and df[ext[i+2]]>thr and df[ext[i+1]]<0:
					brth_cyc.append([ext[i],ext[i+2]])
					rr = np.append(rr,[[1/(t[ext[i+2]]-t[ext[i]]),t[ext[i+1]]]],axis=0)
		rr = rr[1:,:]			
		print(rr)
		if debug==True:
			f,ax = pl.subplots(figsize=(9,5))
			ax.plot(t,data,'k--',alpha=0.5,label='initial')
			ax.plot(t,df,'b',label='filtered')
			ax.plot(t[minpt],df[minpt],'*')
			ax.plot(t[maxpt],df[maxpt],'o')
			ax.axhline(thr)
			for st,sp in brth_cyc:
				ax.plot(t[st:sp],df[st:sp],'r',linewidth=2,alpha=0.5)
			ax.plot(rr[:,1],rr[:,0],'+',label='Resp Rate')
			ax.legend()
		
		return rr
	
	def CoutAdv(self,data,t,debug=False):
		"""
		Implementation of Advanced Count Method from 
		Axel Schafer, Karl Kratky.  "Estimation of Breathing Rate from Respiratory
		Sinus Arrhythmia: Comparison of Various Methods."  Ann. of Biomed. Engr.
		Vol 36 No 3, 2008.
		
		Parameters
		----------
		data : ndarray
			Array of data of interest.  Originally used for R-R interval timings.
		t : ndarray
			Timings for data of interest
			
		Returns
		------
		rr : ndarray
			N-by-2 array of respiration frequencies (Beats per sec) and timings
		"""

t,v = np.genfromtxt('C:\\Users\\Lukas Adamowicz\\Dropbox\\Masters\\Project'+\
				  '\\RespiratoryRate_HeartRate\\Python RRest\\sample_ecg.csv',\
				  skip_header=0,unpack=True,delimiter=',')
t -= t[0]
t /= 1000

#v,t = v[250:int(len(v)/24)],t[250:int(len(v)/24)]
v,t = v[250:20000],t[250:20000]

test = ECGAnalysis(500)
v_1 = test.ElimLowFreq(v,debug=False)
v_2 = test.ElimVeryHighFreq(v_1,debug=False)
v_3 = test.ElimMainsFreq(v_2,debug=False)
v_3c,l_inv = test.CheckLeadInversion(v_3,debug=False)
v_4 = test.ElimSubCardiacFreq(v_3c,debug=False)

v_d,t_d = test.DerivativeFilter(v_4,t)
v_s = v_d**2
v_i,t_i = test.IntegratedAverageFilter(v_s,t_d)

rr8,tsn_i,tsn_f = test.FindPeaksLearning(v_4,t,v_i,v_d,debug=False)

r_pk,q_tr = test.FindRPeaks(v_4,t,v_i,v_d,rr8,tsn_i,tsn_f,debug=False)
bw,am,fm,fmt = test.RespRateExtraction(r_pk,q_tr)
fmts,fms = test.FMSplineInterpolate(fm,fmt,debug=False)
RR = test.CountOriginal(fms,fmts,debug=True)