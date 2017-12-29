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
	
	#Step 1
	def ElimLowFreq(self,data,cutoff=3,N=1,debug=False):
		"""
		Parameters
		---------
		data : float
			ECG voltage data to filter
		cutoff : float
			Cuttoff frequency for filter, -3dB limit
		N : int
			filter order, higher is sharper cutoff
		debug : bool
			Print output graphs for debugging
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
	
	#Step 2
	def ElimVeryHighFreq(self,data,cutoff=20,N=1,detrend=True,debug=False):
		"""
		Parameters
		---------
		data : float
			ECG voltage data after Step 1
		cutoff : float
			Cutoff frequency for filter, -3dB limit
		N : int
			Filter order, higher is sharper cutoff
		debug : bool
			Print output graphs for debugging
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
	
	#Step 3
	def ElimMainsFreq(self,data,cutoff=60,Q=10,detrend=True,debug=False):
		"""
		Parameters
		---------
		data : float
			ECG voltage data after Step 2
		cutoff : float
			Cutoff frequency for notch filter
		Q : int
			Filter order, higher is sharper cutoff
		debug : bool
			Print output graphs for debugging
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
			
		
	
v = np.genfromtxt('C:\\Users\\Lukas Adamowicz\\Dropbox\\Masters\\Project'+\
				  '\\RespiratoryRate_HeartRate\\Python RRest\\sample_ecg.csv',\
				  skip_header=0,unpack=True,delimiter=',')
v = v[250:]
t = np.array([1/500*i for i in range(len(v))])

test = ECGAnalysis(500)
v_1 = test.ElimLowFreq(v,debug=False)
v_2 = test.ElimVeryHighFreq(v_1,debug=False)
v_3 = test.ElimMainsFreq(v_2,debug=True)

