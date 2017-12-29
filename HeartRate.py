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

