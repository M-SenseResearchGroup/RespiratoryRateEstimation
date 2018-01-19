"""
V0.2.0
January 19, 2018

Lukas Adamowicz

Python 3.6.3 on Windows 10 with 64-bit Anaconda
"""
import numpy as np
import matplotlib.pyplot as pl
from scipy import signal, interpolate

from timeit import default_timer as timer

# TODO remove this when done testing
class EcgData:
	def __init__(self):
		self.t = dict()
		self.v = dict()
		
class TSN:
	"""
	Class for storing Threshold, Signal, Noise values for R-peak detection
	"""
	def __init__(self):
		self.thr = None #threshold
		self.spk = None #signal peak moving average
		self.npk = None #noise peak moving average

class ECGAnalysis(object):
	def __init__(self,master):
		"""
		Class defining steps for analyzing ECG data to obtain an estimation 
		of Respiratory Rate.
		
		Parameters
		---------
		master : EcgData
			EcgData class (DataStructures) with voltage and timing dictionaries for
			desired events to be analyzed
		"""
		self.v = master.v #should reference the same data so no duplicates
		self.t = master.t
		
		self.vd = list(master.v.keys())
		self.td = list(master.t.keys()) #these 2 should be the same
		
		#calculate sampling frequency
		self.fs = round(1/(self.t[self.td[0]][1] - self.t[self.td[0]][0]))
		master.fs = self.fs #assign this to master for possible future use
		
		self.nyq = 0.5*self.fs #Nyquist frequency is half sampling frequency
		
#		int_avg_width=150,t_learn=8,delay=175
		
	def ElimLowFreq(self,cutoff=3,N=1,debug=False):
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
		if debug==True:
			self.v_old = self.v.copy() #store previous step's voltage for reference
			
		w_cut = cutoff/self.nyq #cutoff frequency as percentage of nyquist frequency
		
		b,a = signal.butter(N,w_cut,'highpass') #setup highpass filter
		
		#perform filtering on the data
		for key in self.vd:
			self.v[key] = signal.filtfilt(b,a,self.v[key]) 
		
		if debug==True:
			n = len(self.vd)
			f,ax = pl.subplots(n,figsize=(16,8))
			for i,key in zip(range(n),self.vd):
				ax[i].plot(self.t[key],self.v_old[key],label='initial')
				ax[i].plot(self.t[key],self.v[key],label='Step 1')
				ax[i].set_xlabel('Time [s]')
				ax[i].set_ylabel('Voltage [mV]')
				ax[i].legend(title=f'{key}')
			ax[0].set_title('1 - Eliminate Low Frequencies')
			pl.tight_layout()
	
	def ElimVeryHighFreq(self,cutoff=20,N=1,detrend=True,debug=False):
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
		if debug==True:
			#copy old voltages for reference/debugging
			self.v_old = self.v.copy()
			
		#cutoff frequency as percentage of nyquist frequency	
		w_cut = cutoff/self.nyq
		
		#setup filter parameters
		b,a = signal.butter(N,w_cut,'lowpass') 
		
		for key in self.vd:
			#detrend - remove mean, linear associations
			if detrend==True:
				data_det = signal.detrend(self.v[key])
				self.v[key] = signal.filtfilt(b,a,data_det)
			else:
				#filter input data
				self.v[key] = signal.filtfilt(b,a,self.v[key])
			
		if debug==True:
			n = len(self.vd)
			f,ax = pl.subplots(n,figsize=(16,8))
			for i,key in zip(range(n),self.vd):
				ax[i].plot(self.t[key],self.v_old[key],label='Step 1')
				ax[i].plot(self.t[key],self.v[key],label='Step 2')
				ax[i].set_xlabel('Time [s]')
				ax[i].set_ylabel('Voltage [mV]')
				ax[i].legend(title=f'{key}')
			ax[0].set_title('2 - Eliminate Very High Frequencies')
			pl.tight_layout()
	
	def ElimMainsFreq(self,cutoff=60,Q=10,detrend=True,debug=False):
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
		if debug==True:
			self.v_old = self.v.copy() 
		
		w0 = cutoff/self.nyq
		b,a = signal.iirnotch(w0,Q)
		
		for key in self.vd:
			if detrend==True:
				data_det = signal.detrend(self.v[key])
				self.v[key] = signal.filtfilt(b,a,data_det)
			else:
				self.v[key] = signal.filtfilt(b,a,self.v[key])
		
		if debug==True:
			n = len(self.vd)
			f,ax = pl.subplots(n,figsize=(16,8))
			for i,key in zip(range(n),self.vd):
				ax[i].plot(self.t[key],self.v_old[key],label='Step 2')
				ax[i].plot(self.t[key],self.v[key],label='Step 3')
				ax[i].set_xlabel('Time [s]')
				ax[i].set_ylabel('Voltage [mV]')
				ax[i].legend(title=f'{key}')
			ax[0].set_title('3 - Eliminate Mains Frequencies')
			pl.tight_layout()
	
	def CheckLeadInversion(self,debug=False):
		"""
		Step 4: Check for lead inversion by examining percentage of local 
		extrema that are negative. R-peaks (maximum local extrema) should be positive.
		Run after ElimMainsFreq
		"""
		
		d2 = self.v[self.vd[0]]**2 #square data for all positive and prominent peaks
		d2_max = max(d2) #maximum of squared data
		d2_max_pks = np.zeros_like(d2) #allocate array for peaks
		
		#indices where squared data is greater than 20% of squared max
		inds = np.argwhere(d2>0.2*d2_max) 
		d2_max_pks[inds] = d2[inds] #squared values greater than 20% max, 0s elsewhere
		d2_pks = signal.argrelmax(d2_max_pks)[0] #locations of maximum peaks
		
		#data values at local extrema (found from squard values)
		d_pks = self.v[self.vd[0]][d2_pks]
		
		#percentage of peaks with values that are negative
		p_neg_pks = len(np.where(d_pks<0)[0])/len(d_pks) 
		
		if debug==True:
			x = np.array([i for i in range(len(self.v[self.vd[0]]))])
			f,ax = pl.subplots(figsize=(9,5))
			ax.plot(x,self.v[self.vd[0]])
			ax.plot(x[d2_pks],self.v[self.vd[0]][d2_pks],'+')
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
	
	def ElimSubCardiacFreq(self,cutoff=0.5,N=4,debug=False):
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
		if debug==True:
			self.v_old = self.v.copy()
		
		w_cut = cutoff/self.nyq #define cutoff frequency as % of nyq. freq.
		b,a = signal.butter(N,w_cut,'highpass') #setup high pass filter
		
		for key in self.vd:
			self.v[key] = signal.filtfilt(b,a,self.v[key])
		
		if debug==True:
			n = len(self.vd)
			f,ax = pl.subplots(n,figsize=(16,8))
			for i,key in zip(range(n),self.vd):
				ax[i].plot(self.t[key],self.v_old[key],label='Step 4')
				ax[i].plot(self.t[key],self.v[key],label='Step 5')
				ax[i].set_xlabel('Time [s]')
				ax[i].set_ylabel('Voltage [mV]')
				ax[i].legend(title=f'{key}')
			ax[0].set_title('5 - Eliminate Sub-Cardiac Frequencies')
			pl.tight_layout()
		
		if self.v_old != None:
			self.v_old = None #remove from memory
	
	def DerivativeFilter(self,plot=False):
		"""
		Calculated derivative of filtered data, as well as squaring data before next step
		
		Class Results
		-------------
		v_der : ndarray
			Derivative of filtered voltage signal data
		t_der : ndarray
			Timestamps for derivative voltage data
		"""
		self.dt = self.t[1]-self.t[0] #timestep
		self.v_der = (-self.v[:-4]-2*self.v[1:-3]+2*self.v[3:-1]+self.v[4:])/(8*self.dt)
		self.t_der = self.t[2:-2] #timsteps are cut short by 2 on either end
		
		self.v_sq = self.v_der**2
		
		if plot==True:
			f,ax = pl.subplots(2,figsize=(9,5),sharex=True)
			ax[0].plot(self.t_der,self.v_der,label='Derivative')
			ax[1].plot(self.t_der,self.v_sq,label='Squared')
			ax[0].legend()
			ax[1].legend()
			f.tight_layout()
			f.subplots_adjust(hspace=0)
			ax.set_title('Derivative and squared filter')
	
	def MovingAverageFilter(self,plot=False):
		"""
		Moving average of squared data.
		
		Class Returns
		-------------
		v_int : ndarray
			Moving average resulting signal
		t_int : ndarray
			Timestamps for resulting average signal
		"""
		#number of samples to use for moving average window
		#window width in seconds/timestep
		N = int(round((self.iaw/1000)/self.dt))
		
		cs = np.cumsum(np.insert(self.v_sq,0,0))
		self.v_int = (cs[N:]-cs[:-N])/N
		self.t_int = self.t_der[N-1:]
		
		#append 0s in places where arrays were shortened
		self.v_int = np.append([0]*(N+1),self.v_int)
		self.v_int = np.append(self.v_int,[0,0])
		
		self.v_der = np.append([0]*2,self.v_der)
		self.v_der = np.append(self.v_der,[0]*2)
		
		if plot==True:
			pl.figure(figsize=(9,5))
			pl.plot(self.t_int,self.v_int,label='Moving Average')
			pl.legend()
			pl.xlabel('Time [s]')
			pl.set_title('Moving average filter')
		
	def FindPeaksLearning(self,debug=False):
		"""
		Learning function for R-peak detection.
		
		Class Returns
		-------
		rr8 : float
			Initial average R-R peak times
		tsn_i : TSN
			Integrated data thresholds and signal and noise moving averages
		tsn_f : TSN
			Filtered data thresholds and signal and noise moving averages
		"""
		n_dly = int(round(self.delay/1000*self.fs)) #number of samples corresponding to delay
		#number of samples in 25,125,225ms
		n_25,n_125 = int(round(0.025*self.fs)),int(round(0.125*self.fs))
		n_225 = int(round(0.225*self.fs))
		n_lrn = int(round(self.tlearn*self.fs)) #number of samples in t_learn seconds
		
		v_f = self.v[:n_lrn] #want only beginning t_learn seconds
		v_int = self.v_int[:n_lrn]
		v_der = self.v_der[:n_lrn]
		t_f = self.t[:n_lrn]
		
		#peaks in region with width equal to integration width
		ii_pks = signal.argrelmax(v_int,order=int(round(0.5*self.iaw/1000/self.dt)))[0] 
		
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
			m_pos.append(max(v_der[ii_pks[i]-n_dly if ii_pks[i]-n_dly>=0 else 0\
						  :ii_pks[i]+n_dly])) 
			try: #to find descending half peak value for each peak
				pki_h.append(np.where(v_int[ii_pks[i]:]<0.5*vi_pks[i])[0][0]+ii_pks[i])
				longwave = False #not a long QRS wave
			except: #if not found, set integration half peak value to max slope + delay
				pki_h.append(np.argmax(v_der[ii_pks[i]-n_dly if ii_pks[i]-n_dly>0 else 0\
								 :ii_pks[i]+n_dly])+ (ii_pks[i]-n_dly) + n_dly)
				longwave = True
			
			if longwave==False: #search for max filt. peak in preceeding 125-225 ms
				band_pk.append(np.argmax(v_f[pki_h[i]-n_225 if pki_h[i]-n_225>0 else 0\
								 :pki_h[i]-n_125])+pki_h[i]-n_225)
			elif longwave==True: #if longwave, search in preceeding 150-250ms
				band_pk.append(np.argmax(v_f[pki_h[i]-n_225-n_25 if pki_h[i]-n_225-n_25>0 else 0\
								 :pki_h[i]-n_125-n_25])+pki_h[i]-n_225-n_25)
		band_pk = np.array(band_pk)
		#determing if R peak or T wave
		for i in range(0,len(ii_pks) if len(ii_pks)%2==0 else len(ii_pks)-1,2):
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
			self.rr8 = np.ones(8)*r_t[1]-r_t[0]
			self.rr8[-len(r_t)+1:] = np.array(r_t[1:])-np.array(r_t[:-1])
		elif len(r_t)>8:
			self.rr8 = np.array(r_t[len(r_t)-8:])-np.array(r_t[len(r_t)-9:-1])
		
		self.tsn_i = TSN() #tsn for integrated data
		self.tsn_f = TSN() #tsn for filtered data
		
		#initialize signal peak for integrated data
		self.tsn_i.spk = 0.125*vi_pks[np.where(v_f[band_pk]==r_v[0])[0][0]]
		#initialize noise peak for integrated data
		self.tsn_i.npk = 0.125*vi_pks[np.where(v_f[band_pk]==t_v[0])[0][0]]
		self.tsn_f.spk = 0.125*r_v[0] #initialize signal peak for filtered data
		self.tsn_f.npk = 0.125*t_v[0] #initialize noise peak for filtered data
		
		for i in range(1,len(r_v)):
			self.tsn_i.spk = 0.125*vi_pks[np.where(v_f[band_pk]==r_v[i])[0][0]]\
					 + 0.875*self.tsn_i.spk
					 
			self.tsn_i.npk = 0.125*vi_pks[np.where(v_f[band_pk]==t_v[0])[0][0]]\
					 + 0.875*self.tsn_i.npk
					 
			self.tsn_f.spk = 0.125*r_v[i] + 0.875*self.tsn_f.spk
			self.tsn_f.npk = 0.125*t_v[i] + 0.875*self.tsn_f.npk
			
			#threshold for integrated data
			self.tsn_i.t = self.tsn_i.npk + 0.25*(self.tsn_i.spk-self.tsn_i.npk)
			#threshold for filtered data 
			self.tsn_f.t = self.tsn_f.npk + 0.25*(self.tsn_f.spk-self.tsn_f.npk) 
			
		if debug==True:
			f,ax = pl.subplots(2,figsize=(9,5),sharex=True)
			ax[0].plot(self.t,self.v_int)
			ax[0].plot(ti_pks,vi_pks,'o')
			ax[0].plot(self.t[pki_h],self.v_int[pki_h],'o')
			ax[0].axhline(self.tsn_i.t,linestyle='--',color='k')
			ax[1].plot(self.t,self.v)
			ax[1].plot(r_t,r_v,'ro')
			ax[1].plot(t_t,t_v,'go')
			ax[1].axhline(self.tsn_f.t,linestyle='--',color='k')
			ax[0].set_title('Find R-Peaks Learning')
			pl.tight_layout()
	
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
	
	def FindRPeaks(self,debug=False):
		"""
		Algorithm implementation to find R-peaks in ECG of Hamilton and Tompkins, 1986.  
		"Quantitative Investigation of QRS Detection Rules Using the 
		MIT/BIH Arrhythmia Database"
		
		Class Returns
		-------
		r_pks : ndarray
			N-by-2 array of [r-peak timestamp,r-peak voltage]
		q_trs : ndarray
			N-by-2 array of [q-trough timestamp, q-trough voltage]
		"""
		
		#number of samples in 25,125,225ms
		n25,n125 = int(round(0.025*self.fs)),int(round(0.125*self.fs))
		n225 = int(round(0.225*self.fs))
		nw = int(round(self.iaw/1000*self.fs)) #number of samples in integration window width
		nd = int(round(self.delay/1000*self.fs)) #number of samples in delay ms
		
		rr8_lim = np.zeros(8) #initialize limited R-R array
		
		#peak indices in region with width equal to integration window width
		vi_pts = signal.argrelmax(self.v_int,order=int(round(0.5*self.iaw/1000/self.dt)))[0]
		#descending half peak value initilization.  Stores indices 
		vi_hpt = np.zeros_like(vi_pts)
		#maximum values and indices for each peak in filtered data
		vf_pks = np.zeros((len(vi_pts),2))
		
		#maximum slopes and indices for each peak.  
		#Initialized as +1 due to comparison with previous slope later
		m_pos = np.zeros((len(vi_pts)+1,2))
		self.r_pks = np.zeros((1,2)) #initialize vector for R peak values and timestamps
		
		for i in range(len(vi_pts)):
			#finding maximum slope in the +- width surrounding each peak
			#if +- window is fully in data range
			if vi_pts[i]-nw > 0 and vi_pts[i]+nw<len(self.v_int): 
				m_pos[i+1] = [max(self.v_der[vi_pts[i]-nw:vi_pts[i]+nw])+\
								 int(round(vi_pts[i]-nw)),np.argmax(self.v_der[vi_pts[i]-\
									nw:vi_pts[i]+nw])+int(round(vi_pts[i]-nw))]
			elif vi_pts[i]-nw < 0: #if the window goes before data range
				m_pos[i+1] = [max(self.v_der[0:vi_pts[i]+nw]),\
									 np.argmax(self.v_der[0:vi_pts[i]+nw])]
			elif vi_pts[i]+nw > len(self.v_int): #if the window goes outside end of data range
				m_pos[i+1] = [max(self.v_der[vi_pts[i]-nw:]),\
									 np.argmax(self.v_der[vi_pts[i]-nw:])]
			
			#finding descending half peak value
			try:
				vi_hpt[i] = np.where(self.v_int[vi_pts[i]:int(round(m_pos[i+1,1]))+nd] \
											 <0.5*self.v_int[vi_pts[i]])[0][0] + vi_pts[i]
				longwave=False #not a long QRS wave complex
			except:
				vi_hpt[i] = vi_pts[i] + nd
				longwave=True #long QRS wave complex
			
			#find maximum values in filtered data preceeding descending half peak values
			if longwave==False: #if not a long wave search preceeding 225 to 125ms
				vf_pks[i] = [max(self.v[vi_hpt[i]-n225:vi_hpt[i]-n125]),\
								  np.argmax(self.v[vi_hpt[i]-n225:vi_hpt[i]-n125])+vi_hpt[i]-n225]
			elif longwave==True: #if long wave, search preceeding 250 to 150 ms
				vf_pks[i] = [max(self.v[vi_hpt[i]-n225-n25:vi_hpt[i]-n125-n25]),\
					  np.argmax(self.v[vi_hpt[i]-n225-n25:vi_hpt[i]-n125-n25])+vi_hpt[i]-n225-n25]
				
			#Determine type of peak (R,T, etc)
			#if the peaks are above the thresholds and time between is greater than 0.36s		
			if self.v_int[vi_pts[i]] > self.tsn_i.t and vf_pks[i,0] > self.tsn_f.t and \
										(self.t[int(round(vf_pks[i,1]))]-self.r_pks[-1,0])>=0.36:
				self.r_pks = np.append(self.r_pks,[[self.t[int(round(vf_pks[i,1]))],\
											   vf_pks[i,0]]],axis=0)
				j = i+1 #assign key value.  Index of the last detected r_peak
				self.tsn_i,self.tsn_f = self._UpdateThresholds(self.v_int[vi_pts[i]],\
												   self.r_pks[-1,1],self.tsn_i,self.tsn_f,signal=True)
			
			#peaks above thresholds, time between is greater than 0.2s but less than 0.36s
			elif self.v_int[vi_pts[i]] > self.tsn_i.t and vf_pks[i,0] > self.tsn_f.t and \
										(self.t[vf_pks[i,1]]-self.r_pks[-1,0])>0.2:
				#if the maximum associated slope is greater than half the previous 
				#detected R wave
				if m_pos[i+1,0] > 0.5*m_pos[j,0]: #it is a R peak
					self.r_pks = np.append(self.r_pks,[self.t[int(round(vf_pks[i,1]))],\
											   vf_pks[i,0]],axis=0)
					j = i+1
					self.tsn_i,self.tsn_f = self._UpdateThresholds(self.v_int[vi_pts[i]],\
													self.r_pks[-1,0],self.tsn_i,self.tsn_f,signal=True)
				else: #it is a peak
					self.tsn_i,self.tsn_f = self._UpdateThresholds(self.v_int[vi_pts[i]],\
													self.v[vi_pts[i]],self.tsn_i,self.tsn_f,signal=False)
			
			else: #if not above the thresholds it is a noise peak
				self.tsn_i,self.tsn_f = self._UpdateThresholds(self.v_int[vi_pts[i]],\
												   self.v[vi_pts[i]],self.tsn_i,self.tsn_f,signal=False)

			##########################################################################
			#     Missing check for missing r-peaks by looking at peaks between      #
			#     consecutive R-peaks that the time between is greater than          #
			#     166% of the limited average: avg(rr8_lim)  	                       #
			##########################################################################
			
			if len(self.r_pks[:,0])>2: #if there have been 2 R-peaks detected
				self.rr8,rr8_lim = self._UpdateAvgRR(self.r_pks[-1,0]-self.r_pks[-2,0],\
										 self.rr8,rr8_lim)
			
		self.r_pks = self.r_pks[1:,:] #trim off first initialized entry
		self.q_trs = np.zeros_like(self.r_pks) #initialize Q-troughs array
		for i in range(len(self.r_pks[:,0])):
			i_rpk = np.where(self.t==self.r_pks[i,0])[0][0] #index of R-peak
			#look in preceeding 0.1s of R-peak for the minimum
			self.q_trs[i] = [self.t[np.argmin(self.v[i_rpk-int(round(0.1*self.fs)):i_rpk])\
							  +i_rpk-int(round(0.1*self.fs))],\
								min(self.v[i_rpk-int(round(0.1*self.fs)):i_rpk])]
		
		if debug==True:
			f,ax = pl.subplots(2,figsize=(9,5),sharex=True)
			ax[0].plot(self.t,self.v,label='filtered')
			ax[0].plot(self.r_pks[:,0],self.r_pks[:,1],'r+')
			ax[0].plot(self.q_trs[:,0],self.q_trs[:,1],'k+')
			ax[0].legend(loc='best')
			
			ax[1].plot(self.t,self.v_int,label='integrated')
			try:
				ax[1].plot(self.t[vi_hpt],self.v_int[vi_hpt],'ko')
			except:
				ax[1].plot(self.t[vi_hpt[:-1]],self.v_int[vi_hpt[:-1]],'ko')
			ax[1].plot(self.t[vi_pts],self.v_int[vi_pts],'ro')
			ax[1].legend(loc='best')
			ax[1].set_xlabel('Time [s]')
			
			ax[0].set_title('Find R-Peaks')
			
			f.tight_layout()
			f.subplots_adjust(hspace=0)
	
	def RespRateExtraction(self):
		"""
		Compute parameters for respiratory rate estimation.
		
		Parameters
		---------
		r_pks : ndarray
			N-by-2 array of [R-peak voltage, R-peak timestamp]
		q_trs : ndarray
			N-by-2 array of [Q-trough voltage, Q-trough timestamp]
		
		Class Returns
		-------
		bw : ndarray
			N by 2 array of timestamps and voltages.  Mean of associated troughs and peaks
		am : ndarray
			N by 2 array timestamps and voltages.  
			Difference between associated troughs and peaks
		fm : ndarray
			N-1 by 2 array of timestamps and voltages.  
			Difference in time between consecutive R-peaks
		"""
		self.bw = np.copy(self.r_pks)
		#X_b1 from Charlton paper
		self.bw[:,1] = np.mean([self.r_pks[:,1],self.q_trs[:,1]],axis=0) 
		
		self.am = np.copy(self.r_pks)
		self.am[:,1] = self.r_pks[:,1]-self.q_trs[:,1] #X_b2
		
		self.fm = np.zeros((len(self.r_pks[:,0])-1,2))
		self.fm[:,1] = self.r_pks[1:,0]-self.r_pks[:-1,0] #X_b3
		self.fm[:,0] = (self.r_pks[1:,0]+self.r_pks[:-1,0])/2
	
	@staticmethod
	def SplineInterpolate(data,debug=False):
		"""
		Spline Interpolation of data
		
		Parameters
		---------
		data : ndarray
			N-1 by 2 array of timestamps and data
		
		Returns
		-------
		spl : ndarray
			N by 2 array of x and y values of spline
		"""
		cs = interpolate.CubicSpline(data[:,0],data[:,1]) #setup spline function
		xs = np.arange(data[0,0],data[-1,0],0.2) #setup x values.  0.2s intervals
		
		#if x-values don't include end time, add it to the array
		if xs[-1] != data[-1,0]:
			xs = np.append(xs,data[-1,0])
			
		spl = np.zeros((len(xs),2))
		spl[:,1] = cs(xs)
		spl[:,0] = xs
			
		if debug==True:
			pl.figure()
			pl.plot(data[:,0],data[:,1],'o',label='fm values')
			pl.plot(xs,cs(xs),label='spline')
			pl.legend()
			pl.title('Spline Interpolation')
		
		return spl
	
	@staticmethod
	def CountOriginal(data,debug=False):
		"""
		Implementation of Original Count Method from 
		Axel Schafer, Karl Kratky.  "Estimation of Breathing Rate from Respiratory
		Sinus Arrhythmia: Comparison of Various Methods."  Ann. of Biomed. Engr.
		Vol 36 No 3, 2008.
		
		Parameters
		----------
		data : ndarray
			N by 2 array of timestamps and data
			
		Returns
		------
		rr : ndarray
			N by 2 array of respiration timings and frequencies (beats per second)
		"""
		#step 1 - bandpass filter with pass region between 0.1-0.5Hz
		fs = 1/(data[1,0]-data[0,0]) #spline data frequency
		wl = 0.1/(0.5*fs) #low cutoff frequency as % of nyquist frequency
		wh = 0.5/(0.5*fs) #high cutoff freq as % of nyquist freq
		
		b,a = signal.butter(5,[wl,wh],'bandpass') #filtfilt, -> order is 2x given
		
		data[:,1] -= np.mean(data[:,1]) #remove mean from data
		df = signal.filtfilt(b,a,data[:,1]) #apply filter to input data
		
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
					rr = np.append(rr,[[data[ext[i+1],0],1/(data[ext[i+2],0]\
						 -data[ext[i],0])]],axis=0)
		rr = rr[1:,:]			
		
		if debug==True:
			f,ax = pl.subplots(2,figsize=(9,5),sharex=True)
			ax[0].plot(data[:,0],data[:,1],'k--',alpha=0.5,label='initial')
			ax[0].plot(data[:,0],df,'b',label='filtered')
			ax[0].plot(data[minpt,0],df[minpt],'*')
			ax[0].plot(data[maxpt,0],df[maxpt],'o')
			ax[0].axhline(thr)
			for st,sp in brth_cyc:
				ax[0].plot(data[st:sp,0],df[st:sp],'r',linewidth=2,alpha=0.5)
			ax[1].plot(rr[:,0],rr[:,1]*60,'+')
			ax[0].legend()
			
			ax[1].set_xlabel('Time [s]')
			ax[1].set_ylabel('Resp. Rate [BPM]')
			ax[0].set_ylabel('R-R peak times [s]')
			
			ax[0].set_title('Count Original')
			
			f.tight_layout()
			f.subplots_adjust(hspace=0)
		
		return rr
	
	@staticmethod
	def CountAdv(data,debug=False):
		"""
		Implementation of Advanced Count Method from 
		Axel Schafer, Karl Kratky.  "Estimation of Breathing Rate from Respiratory
		Sinus Arrhythmia: Comparison of Various Methods."  Ann. of Biomed. Engr.
		Vol 36 No 3, 2008.
		
		Parameters
		----------
		data : ndarray
			N by 2 array of timesteps and data.  Originally used for R-R interval timings.
			
		Returns
		------
		rr : ndarray
			N by 2 array of respiration timesteps and frequencies (beats per second)
		"""
		
		#step 1 - bandpass filter with pass region between 0.1-0.5Hz
		fs = 1/(data[1,0]-data[0,0]) #spline data frequency
		wl = 0.1/(0.5*fs) #low cutoff frequency as % of nyquist frequency
		wh = 0.5/(0.5*fs) #high cutoff freq as % of nyquist freq
		
		b,a = signal.butter(5,[wl,wh],'bandpass') #filtfilt, -> order is 2x given
		
		data[:,1] -= np.mean(data[:,1]) #remove mean from data
		df = signal.filtfilt(b,a,data[:,1]) #apply filter to input data
		
		#step 2 - find local minima and maxima of filtered data
		# get 3rd quartile, threshold is Q3*0.2
		minpt = signal.argrelmin(df)[0]
		maxpt = signal.argrelmax(df)[0]
		
		#step 3 - calculate absolute value diff. between subsequent local extrema
		# 3rd quartile of differences, threshold = 0.1*Q3
		ext,etp = zip(*sorted(zip(np.append(maxpt,minpt),\
							[True]*len(maxpt)+[False]*len(minpt))))
		ext,etp = np.array(ext),np.array(etp)
		
		ext_diff = np.zeros(len(ext)-1)
		for i in range(len(ext)-1):
			if etp[i] != etp[i+1]: #ie min followed by max or max followed by min
				ext_diff[i] = abs(df[ext[i]]-df[ext[i+1]])
				
		thr = 0.1*np.percentile(ext_diff,75) #threshold value
		
		if debug==True:
			f,ax = pl.subplots(2,figsize=(9,5),sharex=True)
			ax[0].plot(data[:,0],df)
			ax[0].plot(data[minpt,0],df[minpt],'k+')
			ax[0].plot(data[maxpt,0],df[maxpt],'k+')
		
		rem = 0 #removed indices counter
		#step 4 - remove any sets whose difference is less than the threshold
		for i in range(len(ext)-1):
			if etp[i-rem] != etp[i-rem+1] and abs(df[ext[i-rem]]-df[ext[i-rem+1]])<thr:
				ext = np.delete(ext,[i-rem,i-rem+1])
				etp = np.delete(etp,[i-rem,i-rem+1])
				rem += 2
		
		#step 5 - breath cycles are now minimum -> maximum -> minimum = 1 cycle
		brth_cyc = [] #initialize breath cycle array
		#rr = np.zeros((1,2)) #initialize array for respiratory rate and timings
		
		#############################################################
		#   	                 YES/NO?                              #
		#############################################################
		brth_cyc_xnx = [] #try maX-miN-maX cycles to see if more than min-max-min
		
		for i in range(len(ext)-2):
			if etp[i]==False and etp[i+1]==True and etp[i+2]==False:
				brth_cyc.append([ext[i],ext[i+2]])
			elif etp[i]==True and etp[i+1]==False and etp[i+2]==True:
				brth_cyc_xnx.append([ext[i],ext[i+2]])
				
		rr = np.zeros((max([len(brth_cyc),len(brth_cyc_xnx)]),2))
		
		if len(brth_cyc)<len(brth_cyc_xnx):
			brth_cyc = brth_cyc_xnx
			
		for i in range(len(rr[:,0])):
			rr[i] = [(data[brth_cyc[i][1],0]+data[brth_cyc[i][0],0])/2\
				 ,1/(data[brth_cyc[i][1],0]-data[brth_cyc[i][0],0])]
		
		if debug==True:
			ax[0].plot(data[ext,0],df[ext],'ro',alpha=0.5,markersize=0.5)
			
			for st,sp in brth_cyc:
				ax[0].plot(data[st:sp,0],df[st:sp],'r',alpha=0.25,linewidth=5)
			
			ax[1].plot(rr[:,0],rr[:,1]*60,'+',markersize=10)
			ax[1].set_ylabel('Breath Frequency [BPM]')
			ax[0].set_ylabel('R-R peak time [s]')
			ax[1].set_xlabel('Time [s]')
			
			ax[0].set_title('Advanced Count')
			
			f.tight_layout()
			f.subplots_adjust(hspace=0)
		
		return rr
	
	@staticmethod
	def SmartModFusion(bw,am,fm,use_given_time=True,plot=True):
		"""
		Modulation smart fusion of bandwidth, AM, and FM respiratory estimates.
		From Karlen et al, 2013.  Respiratory rate is output of standard deviation of 
		estimated rate between BW, AM, and FM estimated rates is less than 4 BPM
		
		Parameters
		---------
		bw : ndarray
			N by 2 array of respiratory rate estimations and timings via bandwidth parameter
		am : ndarray
			N by 2 array of respiratory rate estimations and timings via AM parameter
		fm : ndarray
			N by 2 array of respiratory rate estimations and timings via FM parameter
		use_given_time : bool
			Use timings from BW/AM/FM data, or based on 0.2s timings.  Defaults to True.
		plot : bool
			Plot resulting data.  Defaults to True.
		
		Returns
		-------
		rr_est : ndarray
			N by 2 array of respiratory rate estimates and timings
		"""
		#convert to BPM
		bw[:,1] *= 60
		am[:,1] *= 60
		fm[:,1] *= 60
		
		bwsf = interpolate.CubicSpline(bw[:,0],bw[:,1])
		amsf = interpolate.CubicSpline(am[:,0],am[:,1])
		fmsf = interpolate.CubicSpline(fm[:,0],fm[:,1])
		
		if use_given_time==True:
			x = np.unique(np.append(np.append(bw[:,0],am[:,0]),fm[:,0]))
		else:
			tmin = max([bw[0,0],am[0,0],fm[0,0]])
			tmax = min([bw[-1,0],am[-1,0],fm[-1,0]])
			x = np.arange(tmin,tmax,0.2)
		lqi = np.array([False]*len(x)) #Low Quality Index
		
		bws = bwsf(x)
		ams = amsf(x)
		fms = fmsf(x)
		
		st_dev = np.std(np.array([bws,ams,fms]),axis=0)
		rr = np.mean(np.array([bws,ams,fms]),axis=0)
		
		for i in range(len(st_dev)):
			if st_dev[i]>4:
				lqi[i] = True #if Standard Dev > 4 BPM Low Quality Index
		
		rr_est = np.zeros((len(rr),2))
		rr_est[:,1] = rr
		rr_est[:,0] = x
		
		if plot==True:
			f,ax = pl.subplots(figsize=(9,5))
			ax.plot(rr_est[:,0],rr_est[:,1],label='Fused Est.')
#			ax.plot(bw[:,0],bw[:,1],'--',alpha=0.5,label='BW Est.')
#			ax.plot(am[:,0],am[:,1],'--',alpha=0.5,label='AM Est.')
#			ax.plot(fm[:,0],fm[:,1],'--',alpha=0.5,label='FM Est.')
			
			ax.fill_between(x,rr-st_dev,rr+st_dev,alpha=0.5)
			
			ax.legend()
			
		
		return rr_est,lqi

#s = timer()

fprefix = 'C:\\Users\\Lukas Adamowicz'+\
	  '\\Dropbox\\Masters\\Project\\RespiratoryRate_HeartRate\\Python RRest\\'
data = EcgData()
data.t['middle'],data.v['middle'] = np.genfromtxt(fprefix+'middle_ecg.csv',\
				  skip_header=0,unpack=True,delimiter=',')

data.t['back'],data.v['back'] = np.genfromtxt(fprefix+'back_ecg.csv',\
				  skip_header=0,unpack=True,delimiter=',')

data.t['forward'],data.v['forward'] = np.genfromtxt(fprefix+'forward_ecg.csv',\
				  skip_header=0,unpack=True,delimiter=',')

test = ECGAnalysis(data)
#print(f'Import/Setup time: {timer()-s:.5f} s')
