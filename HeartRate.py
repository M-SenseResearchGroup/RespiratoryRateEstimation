"""
V0.1
September 12/29/2017

Lukas Adamowicz

Python 3.6.3 on Windows 10 with 64-bit Anaconda
"""
import numpy as np
import matplotlib.pyplot as pl
from scipy import signal, interpolate

class ECGAnalysis(object):
	def __init__(self,fs):
		"""
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
		Parameters
		---------
		data : float ndarray
			ECG voltage data to filter
		cutoff : float, optional
			Cuttoff frequency for filter, -3dB limit.  Defaults to 3Hz
		N : int, optional
			filter order, higher is sharper cutoff.  Defaults to 1
		debug : bool, optional
			Print output graphs for debugging.  Defaults to False
		
		Returns
		------
		data_filt : float ndarray
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
		Parameters
		---------
		data : float ndarray
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
		data_filt : float ndarray
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
		data_filt : float ndarray
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
		Parameters
		---------
		data : float ndarray
			ECG volage data.  Usually after step 3, but can be after import
		debug : bool, optional
			Print peak values for debugging.  Defaults to False
		
		Returns
		-------
		data_check : float ndarray
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
		Parameters
		---------
		data : float ndarray
			ECG voltage data, correctly oriented and passed through stages 1-3 of filtering
		cutoff : float, optional
			Cutoff frequency for high-pass filter.  Defaults to 0.5 Hz (30 BPM)
		N : int, optional
			Filter order.  Defaults to 4
		debug : bool
			Graph input and output data.  Defaults to False
		
		Returns
		-------
		data_filt : float ndarray
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
		Parameters
		---------
		data : float ndarray
			ECG voltage data after steps 1-4 filtering and lead inversion check
		time : float ndarray
			ECG sample timings
		plot : bool
			Plot resulting data.  Defaults to False
		
		Returns
		-------
		data_der : float ndarray
			Derivative of input data
		time_der : float ndarray
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
		Parameters
		---------
		data : float ndarray
			ECG voltage data after derivative filter applied
		
		Returns
		------
		data_sq : float ndarray
			Square filter data
		"""
		return data**2
	
	def IntegratedAverageFilter(self,data,time,width=150,plot=False):
		"""
		Parameters
		---------
		data : float ndarray
			ECG voltage data after squaring
		time : float ndarray
			Timings for samples associated with ECG data in seconds
		width : int, optional
			Time width (ms) of integration window.  Defaults to 150ms (0.15s)
		plot : bool
			Plot output data. Defaults to False
			
		Returns
		-------
		data_int : float ndarray
			Average by integration data
		time_int : float ndarray
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
		
	def FindPeaksLearning(self,v_band,t_band,v_int,v_der,width=150,t_learn=8,delay=175,debug=False):
		"""
		Parameters
		---------
		v_band : float ndarray
			Bandpassed ECG voltage signal (ie after Sub-cardiac freq elimination)
		t_band : float ndarray
			Timings associated with bandpassed ECG signal
		v_int : float ndarray
			Average by integration voltage signal.
		v_der : float ndarray
			Derivative voltage signal
		width : float, optional
			Integration window width in milliseconds.  Defaults to 150ms
		t_learn : float, optional
			Amount of time (seconds) for learning algorithm to work on.  Defaults to 8s
		t_delay : float, optional
			Amount of time (ms) to wait before declaring peak if 1/2 amplitude not found.  Defaults to 175ms
		
		Returns
		-------
		rr_avg : float
			Initial average R-R peak times
		thr_int : list of floats
			Integrated data thresholds and associated values.  3 values
		thr_fil : list of floats
			Filtered data thresholds and associated values.  3 values
		"""
		dt = t_band[1]-t_band[0]
		n_dly = int(delay/1000*self.fs) #number of samples corresponding to delay
		n_25,n_125,n_225 = int(0.025*self.fs),int(0.125*self.fs),int(0.225*self.fs) #number of samples in 25,125,225ms
		n_lrn = int(t_learn*self.fs) #number of samples in t_learn seconds
		
		#append 0s to front of integrated signal
		v_int = np.append([0]*(len(v_band)-len(v_int)-2),v_int)
		v_der = np.append([0,0],v_der) #number of 0s needs to be changed if different derivative scheme
		
		v_band = v_band[:n_lrn] #want only beginning t_learn seconds
		v_int = v_int[:n_lrn]
		v_der = v_der[:n_lrn]
		t_band = t_band[:n_lrn]
		
		#peaks in region with width equal to integration width
		ii_pks = signal.argrelmax(v_int,order=int(0.5*width/1000/dt))[0] 
		
		#remove any peaks that are less than 0.1% of mean of the peaks
		ii_pks = ii_pks[np.where(v_int[ii_pks]>0.001*np.mean(v_int[ii_pks]))[0]]
		
		#get integrated signal values
		vi_pks = v_int[ii_pks]
		ti_pks = t_band[ii_pks]
		
		m_pos = [] #maximum positive slope in the t_delay region around each peak (+- t_delay)
		pki_h = [] #descending half peak value points
		band_pk = [] #max points in v_band preceeding descending half peak value points
		
		r_pk_b = [] #Boolean array for if the point is an R peak or not
		
		for i in range(len(ii_pks)):
			m_pos.append(max(v_der[ii_pks[i]-n_dly:ii_pks[i]+n_dly])) #append max slope for each peak
			try: #to find descending half peak value for each peak
				pki_h.append(np.where(v_int[ii_pks[i]:]<0.5*vi_pks[i])[0][0]+ii_pks[i])
				longwave = False #not a long QRS wave
			except: #if not found, set integration half peak value to max slope + delay 
				pki_h.append(np.argmax(v_der[ii_pks[i]-n_dly:ii_pks[i]+n_dly])[0] + (ii_pks-n_dly) + n_dly)
				longwave = True
			
			if longwave==False: #if not a long wave, search for max filtered peak in preceeding 125-225 ms
				band_pk.append(np.argmax(v_band[pki_h[i]-n_225:pki_h[i]+n_125])+pki_h[i]-n_225)
			elif longwave==True: #if longwave, search in preceeding 150-250ms
				band_pk.append(np.argmax(v_band[pki_h[i]-n_225+n_25:pki_h[i]+n_125+n_25])+pki_h[i]-n_225+n_25)
		band_pk = np.array(band_pk)
		
		#determing if R peak or T wave
		for i in range(0,len(ii_pks),2):
			#if time between peaks is less than 200ms, first is R, second is T
			#Due to being faster than physiologically possible
			if t_band[band_pk[i+1]]-t_band[band_pk[i]]<0.2:
				r_pk_b.append([True,False])
			#if time between peaks is less than 360ms but greater than 200ms
			elif t_band[band_pk[i+1]]-t_band[band_pk[i]]<0.36:
				#then check the slope.  If slope i is twice as steep as slope i+1, point i is the qrs complex, point i+1 a t complex
				if 0.5*m_pos[i] > m_pos[i+1]:
					r_pk_b.append([True,False])
				#if slope i is less than half as steep as slope i+1, then i+1 is a qrs complex, i a t-wave
				elif m_pos[i] < 0.5*m_pos[i+1]:
					r_pk_b.append([False,True])
				#if neither slope is smaller than half the other, they are both qrs complexes
				elif 0.5*m_pos[i] < m_pos[i+1] and m_pos[i] > 0.5*m_pos[i+1]:
					r_pk_b.append([True,True])
			#if time between peaks is greater than 360ms, both are qrs peaks
			elif t_band[band_pk[i+1]]-t_band[band_pk[i]]>0.36:
				r_pk_b.append([True,True])
		
		#assign r-peak values and timings, as well as t-wave values and timings		
		r_v = v_band[band_pk[np.argwhere(np.array(r_pk_b).flatten()==True).flatten()]]
		r_t = t_band[band_pk[np.argwhere(np.array(r_pk_b).flatten()==True).flatten()]]
		t_v = v_band[band_pk[np.argwhere(np.array(r_pk_b).flatten()==False).flatten()]]
		t_t = t_band[band_pk[np.argwhere(np.array(r_pk_b).flatten()==False).flatten()]]
		
		#Create R-R average arrays
		if len(r_t)<=8:
			rr_avg = np.ones(8)*r_t[1]-r_t[0]
			rr_avg[-len(r_t)+1:] = np.array(r_t[1:])-np.array(r_t[:-1])
		elif len(r_t)>8:
			rr_avg = np.array(r_t[len(r_t)-8:])-np.array(r_t[len(r_t)-9:-1])
			
		spk_i = 0.125*vi_pks[np.where(v_band[band_pk]==r_v[0])[0][0]] #initialize signal peak for integrated data
		npk_i = 0.125*vi_pks[np.where(v_band[band_pk]==t_v[0])[0][0]] #initialize noise peak for integrated data
		spk_f = 0.125*r_v[0] #initialize signal peak for filtered data
		npk_f = 0.125*t_v[0] #initialize noise peak for filtered data
		for i in range(1,len(r_v)):
			spk_i = 0.125*vi_pks[np.where(v_band[band_pk]==r_v[i])[0][0]] + 0.875*spk_i
			npk_i = 0.125*vi_pks[np.where(v_band[band_pk]==t_v[0])[0][0]] + 0.875*npk_i
			spk_f = 0.125*r_v[i] + 0.875*spk_f
			npk_f = 0.125*t_v[i] + 0.875*npk_f
			thr_i = npk_i + 0.25*(spk_i-npk_i) #threshold for integrated data
			thr_f = npk_f + 0.25*(spk_f-npk_f) #threshold for filtered data
			
		if debug==True:
			f,ax = pl.subplots(2,figsize=(9,5),sharex=True)
			ax[0].plot(t_band,v_int)
			ax[0].plot(ti_pks,vi_pks,'o')
			ax[0].plot(t_band[pki_h],v_int[pki_h],'o')
			ax[0].axhline(thr_i,linestyle='--',color='k')
			ax[1].plot(t_band,v_band)
			ax[1].plot(r_t,r_v,'ro')
			ax[1].plot(t_t,t_v,'go')
			ax[1].axhline(thr_f,linestyle='--',color='k')
			pl.tight_layout()
			
		return rr_avg, [thr_i,spk_i,npk_i], [thr_f,spk_f,npk_f]
	
	@staticmethod
	def _UpdateAvgRR(rr_int,rr_avg,rr_avg_lim):
		"""
		Parameters
		---------
		
		
v = np.genfromtxt('C:\\Users\\Lukas Adamowicz\\Dropbox\\Masters\\Project'+\
				  '\\RespiratoryRate_HeartRate\\Python RRest\\sample_ecg.csv',\
				  skip_header=0,unpack=True,delimiter=',')
v = v[250:]
t = np.array([1/500*i for i in range(len(v))])

test = ECGAnalysis(500)
v_1 = test.ElimLowFreq(v,debug=False)
v_2 = test.ElimVeryHighFreq(v_1,debug=False)
v_3 = test.ElimMainsFreq(v_2,debug=False)
v_3c,l_inv = test.CheckLeadInversion(v_3,debug=False)
v_4 = test.ElimSubCardiacFreq(v_3c,debug=False)

v_d,t_d = test.DerivativeFilter(v_4,t)
v_s = v_d**2
v_i,t_i = test.IntegratedAverageFilter(v_s,t_d)

rr_avg,tsn_i,tsn_f = test.FindPeaksLearning(v_4,t,v_i,v_d,debug=True)