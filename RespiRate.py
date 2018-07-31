"""
V0.2.0
June 14, 2018

Lukas Adamowicz

Python 3.6.5 on Windows 10 with 64-bit Anaconda
"""

import sys
from os import getcwd
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QAction, qApp, QMessageBox, QMenu, \
    QLabel, QLineEdit, QHBoxLayout, QVBoxLayout, QGroupBox, QDialog, QCheckBox, QPushButton, QWidget, QComboBox, \
    QRadioButton, QTableWidget, QTableWidgetItem, QTabWidget
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import pyqtSlot, Qt
from ECGAnalysis_v2 import ECGAnalysis, EcgData
from numpy import loadtxt, array, argmin, argwhere, append, insert, savetxt
from pickle import load as Pload, dump as Pdump
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Patch


# TODO add a saved settings file to store settings between sessions.
class RespiRate(QMainWindow):
    def __init__(self):
        super().__init__()

        self.sbar = self.statusBar()  # new shorthand for status bar
        self.annots = None  # allocate for annotations to allow for checking later on
        self.count_run = False  # set that the count method has not been run.  used for fusion run toggle with settings

        self.boldFont = QFont('FontFamily', weight=75)  # create option for bold font

        self.loc = getcwd()  # get the location of the file.

        self.filtSet = FilterSettingsWindow(self)
        self.settings = SettingsWindow(self)
        self.rpkTab = TableWindow(self)
        self.rrTab = TableWindow(self)

        self.initUI()

        self.form_widget = FormWidget(self)
        self.setCentralWidget(self.form_widget)

    def initUI(self):
        self.setGeometry(100, 100, 850, 650)
        self.setWindowTitle('RespiRate')

        # save imported raw data
        self.saveData = QAction('Save', self)
        self.saveData.setDisabled(True)  # disable initially because we have no imported data to save
        self.saveData.setShortcut('Ctrl+S')
        self.saveData.setStatusTip('Save imported raw data')
        self.saveData.triggered.connect(self.save_data)

        # Exit menu action
        exitAct = QAction(QIcon('next.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        # import annotations menu action
        importAnnot = QAction('Import Annotations', self)
        importAnnot.setStatusTip('Import Annotation File')
        importAnnot.triggered.connect(self.open_annotation_mc10)

        # import raw data
        importRaw = QAction("Import Raw Data", self)
        importRaw.setStatusTip('Import Raw ECG Data')
        importRaw.triggered.connect(self.open_lead_mc10)

        # import annotation/raw data submenu
        importMenu = QMenu('Import Data', self)
        importMenu.addAction(importAnnot)
        importMenu.addAction(importRaw)

        # open previously saved (serialized data) action
        openData = QAction('Open', self)
        openData.setShortcut('Ctrl+O')
        openData.setStatusTip('Open saved data')
        openData.triggered.connect(self.open_serialized_data)

        # Open filter settings window
        filtSetAct = QAction(QIcon('Icons\\filter.png'), 'Filter Settings', self)
        filtSetAct.setStatusTip('Open filter settings dialog')
        filtSetAct.triggered.connect(self.open_filter_settings)

        # Open settings window
        setAct = QAction(QIcon('Icons\\configure.png'), 'Settings', self)
        setAct.setStatusTip('Open settings dialog')
        setAct.triggered.connect(self.open_settings)

        # Open HR table window
        self.rpkTabAct = QAction(QIcon('Icons\\next.png'), 'R-Peak Table', self)
        self.rpkTabAct.setStatusTip('View R-Peak times')
        self.rpkTabAct.setDisabled(True)
        self.rpkTabAct.triggered.connect(self.open_rpkTab)

        # Open RR table window
        self.rrTabAct = QAction(QIcon('Icons\\next.png'), 'Breaths Table', self)
        self.rrTabAct.setStatusTip('View Breath times')
        self.rrTabAct.setDisabled(True)
        self.rrTabAct.triggered.connect(self.open_rrTab)

        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        runmenu = menubar.addMenu('&Analysis')
        datamenu = menubar.addMenu('&Data')
        settingsmenu = menubar.addMenu('&Settings')

        filemenu.addMenu(importMenu)
        filemenu.addAction(openData)
        filemenu.addAction(self.saveData)
        filemenu.addAction(exitAct)

        datamenu.addAction(self.rpkTabAct)
        datamenu.addAction(self.rrTabAct)

        settingsmenu.addAction(filtSetAct)
        settingsmenu.addAction(setAct)

        # toolbar actions
        self.allAct = QAction(QIcon('Icons\\arrow-continue.png'), 'Run All', self)
        self.allAct.setToolTip('Run all analysis')
        self.allAct.setDisabled(True)
        self.allAct.triggered.connect(self.runALL)

        # TODO change icon
        self.elfAct = QAction(QIcon('Icons\\next.png'), 'Eliminate Low Frequencies', self)
        self.elfAct.setToolTip('Eliminate Low Frequencies')
        self.elfAct.setDisabled(True)
        self.elfAct.triggered.connect(self.runELF)

        # TODO change icon
        self.evhfAct = QAction(QIcon('Icons\\next.png'), 'Eliminate Very High Frequencies', self)
        self.evhfAct.setToolTip('Eliminate Very High Frequencies')
        self.evhfAct.setDisabled(True)
        self.evhfAct.triggered.connect(self.runEVHF)

        # TODO change icon
        self.emfAct = QAction(QIcon('Icons\\next.png'), 'Eliminate Mains Frequencies', self)
        self.emfAct.setToolTip('Eliminate Mains Frequencies')
        self.emfAct.setDisabled(True)
        self.emfAct.triggered.connect(self.runEMF)

        # TODO change icon
        self.cliAct = QAction(QIcon('Icons\\next.png'), 'Check Lead Inversion', self)
        self.cliAct.setToolTip('Check for lead inversion')
        self.cliAct.setDisabled(True)
        self.cliAct.triggered.connect(self.runCLI)

        # TODO change icon
        self.escAct = QAction(QIcon('Icons\\next.png'), 'Eliminate Sub-cardiac Frequencies', self)
        self.escAct.setToolTip('Eliminate Sub-cardiac Frequencies')
        self.escAct.setDisabled(True)
        self.escAct.triggered.connect(self.runESC)

        # TODO change icon
        self.derAct = QAction(QIcon('Icons\\next.png'), 'Derivative Filter', self)
        self.derAct.setToolTip('Calculate Derivative Filter')
        self.derAct.setDisabled(True)
        self.derAct.triggered.connect(self.runDER)

        # TODO change icon
        self.maAct = QAction(QIcon('Icons\\next.png'), 'Moving Average Filter', self)
        self.maAct.setToolTip('Calculating Moving Average')
        self.maAct.setDisabled(True)
        self.maAct.triggered.connect(self.runMA)

        # TODO change icon
        self.rpkAct = QAction(QIcon('Icons\\next.png'), 'Detect R-Peaks', self)
        self.rpkAct.setToolTip('Determine R-peak locations')
        self.rpkAct.setDisabled(True)
        self.rpkAct.triggered.connect(self.runRPK)

        # TODO change icon
        self.rrpAct = QAction(QIcon('Icons\\next.png'), 'Extract RR Parameters', self)  # Respiratory rate params
        self.rrpAct.setToolTip('Extract Respiratory Rate Calculation Parameters')
        self.rrpAct.setDisabled(True)
        self.rrpAct.triggered.connect(self.runRRP)

        # TODO change icon
        self.cntAct = QAction(QIcon('Icons\\next.png'), 'Perform Count', self)  # count adv/orig
        self.cntAct.setToolTip('Perform count method set in settings')
        self.cntAct.setDisabled(True)
        self.cntAct.triggered.connect(self.runCNT)

        # TODO change icon
        self.fusAct = QAction(QIcon('Icons\\next.png'), 'Fuse Estimates', self)
        self.fusAct.setToolTip('Fuse Respiratory Rate estimates')
        self.fusAct.setDisabled(True)
        self.fusAct.triggered.connect(self.runFUS)

        runmenu.addAction(self.allAct)
        runmenu.addSeparator()
        runmenu.addAction(self.elfAct)
        runmenu.addAction(self.evhfAct)
        runmenu.addAction(self.emfAct)
        runmenu.addAction(self.cliAct)
        runmenu.addAction(self.escAct)
        runmenu.addAction(self.derAct)
        runmenu.addAction(self.maAct)
        runmenu.addAction(self.rpkAct)
        runmenu.addAction(self.rrpAct)
        runmenu.addAction(self.cntAct)
        runmenu.addAction(self.fusAct)

        # toolbar setup
        self.toolbar = self.addToolBar('Process Data')
        self.toolbar.addAction(self.allAct)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.elfAct)
        self.toolbar.addAction(self.evhfAct)
        self.toolbar.addAction(self.emfAct)
        self.toolbar.addAction(self.cliAct)
        self.toolbar.addAction(self.escAct)
        self.toolbar.addAction(self.derAct)
        self.toolbar.addAction(self.maAct)
        self.toolbar.addAction(self.rpkAct)
        self.toolbar.addAction(self.rrpAct)
        self.toolbar.addAction(self.cntAct)
        self.toolbar.addAction(self.fusAct)

        self.show()

    def runALL(self):
        self.runELF()
        self.runEVHF()
        self.runEMF()
        self.runCLI()
        self.runESC()
        self.runDER()
        self.runMA()
        self.runRPK()
        self.runRRP()
        self.runCNT()
        if self.settings.fuse:
            self.runFUS()

    def runELF(self):
        self.sbar.showMessage('Eliminating low frequencies...')
        self.ECG.ElimLowFreq(cutoff=self.filtSet.elf_cut, N=self.filtSet.elf_N)
        self.evhfAct.setDisabled(False)

        if 'Filtered' not in [self.form_widget.stepChoice.itemText(i) for i in
                              range(self.form_widget.stepChoice.count())]:
            self.form_widget.stepChoice.addItem('Filtered')
        self.sbar.clearMessage()

    def runEVHF(self):
        self.sbar.showMessage('Eliminating very high frequencies...')
        self.ECG.ElimVeryHighFreq(cutoff=self.filtSet.evhf_cut, N=self.filtSet.evhf_N, detrend=self.filtSet.evhf_det)
        self.emfAct.setDisabled(False)

        self.sbar.clearMessage()

    def runEMF(self):
        self.sbar.showMessage('Eliminating mains frequencies...')
        self.ECG.ElimMainsFreq(cutoff=self.filtSet.emf_cut, Q=self.filtSet.emf_Q, detrend=self.filtSet.emf_det)
        self.cliAct.setDisabled(False)

        self.sbar.clearMessage()

    def runCLI(self):
        self.sbar.showMessage('Checking and fixing lead inversion...')
        self.ECG.CheckLeadInversion()
        self.escAct.setDisabled(False)

        self.sbar.clearMessage()

    def runESC(self):
        self.sbar.showMessage('Eliminating sub-cardiac frequencies...')
        self.ECG.ElimSubCardiacFreq(cutoff=self.filtSet.esc_cut, N=self.filtSet.esc_N)
        self.derAct.setDisabled(False)

        self.sbar.clearMessage()

    def runDER(self):
        self.sbar.showMessage('Calculating derivative filter...')
        self.ECG.delay = self.settings.delay  # set the correct delay based on the settings dialog
        self.ECG.DerivativeFilter()
        self.maAct.setDisabled(False)

        if 'Derivative Filter' not in [self.form_widget.stepChoice.itemText(i) for i in
                                       range(self.form_widget.stepChoice.count())]:
            self.form_widget.stepChoice.addItem('Derivative Filter')

        self.sbar.clearMessage()

    def runMA(self):
        self.sbar.showMessage('Calculating moving average filter...')
        self.ECG.mfl = self.filtSet.mov_len  # set the moving average length based on the filter settings dialog
        self.ECG.MovingAverageFilter()
        self.rpkAct.setDisabled(False)

        if 'Moving Average' not in [self.form_widget.stepChoice.itemText(i) for i in
                                    range(self.form_widget.stepChoice.count())]:
            self.form_widget.stepChoice.addItem('Moving Average')

        self.sbar.clearMessage()

    def runRPK(self):
        self.sbar.showMessage('Detecting R-peaks...')
        self.ECG.DetectRPeaks()
        self.rrpAct.setDisabled(False)
        self.rpkTabAct.setDisabled(False)

        if 'R-Peaks' not in [self.form_widget.stepChoice.itemText(i) for i in
                             range(self.form_widget.stepChoice.count())]:
            self.form_widget.stepChoice.addItem('R-Peaks')
        if 'Heart Rate (2-point)' not in [self.form_widget.stepChoice.itemText(i) for i in
                                          range(self.form_widget.stepChoice.count())]:
            self.form_widget.stepChoice.addItem('Heart Rate (2-point)')
        ind = self.form_widget.stepChoice.findText('Heart Rate (2-point)')
        self.form_widget.stepChoice.setItemData(ind, self.boldFont, Qt.FontRole)

        self.sbar.clearMessage()

    def runRRP(self):
        self.sbar.showMessage('Extracting Respiratory Rate Parameters...')
        self.ECG.RespRateExtraction()
        self.cntAct.setDisabled(False)

        self.sbar.clearMessage()

    def runCNT(self):
        steps = [self.form_widget.stepChoice.itemText(i) for i in range(self.form_widget.stepChoice.count())]
        adv_steps = ['AM Respiratory Rate (Adv)', 'AM Respiratory Signal (Adv)',
                     'FM Respiratory Rate (Adv)', 'FM Respiratory Signal (Adv)',
                     'BW Respiratory Rate (Adv)', 'BW Respiratory Signal (Adv)']
        orig_steps = ['AM Respiratory Rate (Orig)', 'AM Respiratory Signal (Orig)',
                      'FM Respiratory Rate (Orig)', 'FM Respiratory Signal (Orig)',
                      'BW Respiratory Rate (Orig)', 'BW Respiratory Signal (Orig)']

        if self.settings.count:
            self.sbar.showMessage('Performing Count Advanced method...')
            self.ECG.CountAdv()

            for astep in adv_steps:
                if astep not in steps:
                    self.form_widget.stepChoice.addItem(astep)
            for ostep in orig_steps:
                if ostep in steps:
                    ind = self.form_widget.stepChoice.findText(ostep)
                    self.form_widget.stepChoice.removeItem(ind)
        else:
            self.sbar.showMessage('Performing Count Original method...')
            self.ECG.CountOriginal()

            for ostep in orig_steps:
                if ostep not in steps:
                    self.form_widget.stepChoice.addItem(ostep)
            for astep in adv_steps:
                if astep in steps:
                    ind = self.form_widget.stepChoice.findText(astep)
                    self.form_widget.stepChoice.removeItem(ind)
        if self.settings.fuse:
            self.fusAct.setDisabled(False)
        else:
            self.fusAct.setDisabled(True)

        self.count_run = True  # set value saying count method has been run
        self.sbar.clearMessage()

    def runFUS(self):
        steps = [self.form_widget.stepChoice.itemText(i) for i in range(self.form_widget.stepChoice.count())]

        if self.settings.fuse:
            self.sbar.showMessage('Fusing Respiratory Rate Estimates...')
            self.ECG.SmartModFusion(use_given_time=self.settings.fuse_time, plot=False)

            if 'Fused Respiratory Rate' not in steps:
                self.form_widget.stepChoice.addItem('Fused Respiratory Rate')
            ind = self.form_widget.stepChoice.findText('Fused Respiratory Rate')
            self.form_widget.stepChoice.setItemData(ind, self.boldFont, Qt.FontRole)

        self.sbar.clearMessage()

    def save_data(self):
        """
        Serialize the data using Pickle for easy importing later
        """

        saveFile, _ = QFileDialog.getSaveFileName(self, "Save Raw Data", self.loc, "ECG (*.ecg);;All Files (*.*)",
                                                  options=QFileDialog.Options())

        if saveFile:
            self.sbar.showMessage('Saving data...')
            fid = open(saveFile, 'wb')
            Pdump(self.data, fid)
            fid.close()
            self.sbar.clearMessage()

    def open_serialized_data(self):
        """
        Open previously serialized data
        """
        openFile, _ = QFileDialog.getOpenFileName(self, "Open Data File", self.loc, "ECG (*.ecg);;All Files (*.*)",
                                                  options=QFileDialog.Options())

        if openFile:
            self.sbar.showMessage('Opening data file...')
            fid = open(openFile, 'rb')
            self.data = Pload(fid)
            fid.close()
            self.sbar.clearMessage()

            self.ECG = ECGAnalysis(self.data, moving_filter_len=self.filtSet.mov_len)

            self.saveData.setDisabled(False)
            self.allAct.setDisabled(False)
            self.elfAct.setDisabled(False)
            self.form_widget.stepChoice.setDisabled(False)
            self.form_widget.eventChoice.setDisabled(False)
            self.form_widget.plotButton.setDisabled(False)

            self.form_widget.stepChoice.addItem('Raw Data')
            ind = self.form_widget.stepChoice.findText('Raw Data')
            self.form_widget.stepChoice.setItemData(ind, self.boldFont, Qt.FontRole)

            for ev in self.data.t.keys():
                self.form_widget.eventChoice.addItem(ev)

    def open_annotation_mc10(self):
        """
        Opens MC10 activity annotation file and then displays how many activities were found
        """

        # TODO remove home_dir, which is used only for testing
        home_dir = "C:\\Users\\Lukas Adamowicz\\Dropbox\\Masters\\Project - Bike Study\\RespiratoryRate_HeartRate\\" +\
                   "RespiratoryRateEstimation\\Validation_Data\\RespRate_PPG_Phone\\DN"

        annotationFile, _ = QFileDialog.getOpenFileName(self, "Open MC10 Annotation File", '',
                                                        "Comma Separated Value (*.csv);;All Files (*.*)",
                                                        options=QFileDialog.Options())

        if annotationFile:
            self.sbar.showMessage('Importing annotations...')  # show message while importing annotations
            # import annotation data
            events, starts, stops = loadtxt(annotationFile, delimiter=',', usecols=(2, 4, 5), skiprows=1, unpack=True,
                                            dtype=str)
            # save the annotation data to a dictionary
            self.annots = {ev[1:-1]: array([start[1:-1], stop[1:-1]]).astype(float) for ev, start, stop in zip(events, starts, stops)}

            # display a message showing how many events were found
            QMessageBox.question(self, "Annotation Import", f"Found {len(self.annots.keys())} events.", QMessageBox.Ok)

        self.sbar.clearMessage()  # clear the message when done importing

    def open_lead_mc10(self):
        """
        Opens MC10 ECG lead data
        """

        # TODO remove home_dir, which is used only for testing
        home_dir = "C:\\Users\\Lukas Adamowicz\\Dropbox\\Masters\\Project - Bike Study\\RespiratoryRate_HeartRate\\" + \
                   "RespiratoryRateEstimation\\Validation_Data\\RespRate_PPG_Phone\\DN\\ecg_lead_ii\\d5la7xul\\" +\
                   "2018-01-24T03-18-36-159Z"

        leadFile, _ = QFileDialog.getOpenFileName(self, "Open MC10 ECG file", '',
                                                  "Comma Separated Value (*.csv);;All Files (*.*)",
                                                  options=QFileDialog.Options())

        if leadFile:
            self.sbar.showMessage('Importing raw data...')  # show status bar message while importing data
            self.data = EcgData()  # object for storing ECG data
            if isinstance(self.annots, type(None)):  # check if self.annots was defined by importing a file or not
                self.data.t['all'], self.data.v['all'] = loadtxt(leadFile, delimiter=',', skiprows=1, unpack=True,
                                                                 dtype=str)
                QMessageBox.question(self, "Raw Data Import", "Imported all raw data", QMessageBox.Ok)
            else:
                t, v = loadtxt(leadFile, delimiter=',', skiprows=1, unpack=True, dtype=str)

                for event in self.annots.keys():
                    ind_start = argmin(abs(t.astype(float)-self.annots[event][0]))  # get the start index of the event
                    ind_stop = argmin(abs(t.astype(float)-self.annots[event][1]))  # get the stop index of the event

                    self.data.t[event] = t[ind_start:ind_stop].astype(float)  # separate and allocate the timings
                    self.data.v[event] = v[ind_start:ind_stop].astype(float)  # separate and allocate the voltages

                QMessageBox.question(self, "Raw Data Import", f"Imported raw data and segmented into " +\
                                     f"{len(self.data.t.keys())} events.", QMessageBox.Ok)

            for ev in self.data.t.keys():
                self.data.t[ev] -= self.data.t[ev][0]
                self.data.t[ev] /= 1000

            self.ECG = ECGAnalysis(self.data, moving_filter_len=self.filtSet.mov_len)
            self.saveData.setDisabled(False)  # allow data saving
            self.allAct.setDisabled(False)  # allow all data processing
            self.elfAct.setDisabled(False)  # allow step-by-step data processing

            self.form_widget.stepChoice.setDisabled(False)
            self.form_widget.eventChoice.setDisabled(False)
            self.form_widget.plotButton.setDisabled(False)

            if 'Raw Data' not in [self.form_widget.stepChoice.itemText(i) for i in
                                  range(self.form_widget.stepChoice.count())]:
                self.form_widget.stepChoice.addItem('Raw Data')

            ind = self.form_widget.stepChoice.findText('Raw Data')
            self.form_widget.stepChoice.setItemData(ind, self.boldFont, Qt.FontRole)

            for ev in self.data.t.keys():
                if ev not in [self.form_widget.eventChoice.itemText(i) for i in
                              range(self.form_widget.eventChoice.count())]:
                    self.form_widget.eventChoice.addItem(ev)

        self.sbar.clearMessage()  # clear the status bar message when done importing

    def open_filter_settings(self):
        self.filtSet.show()

    def open_settings(self):
        self.settings.show()

    def open_rpkTab(self):
        self.rpkTab.populateFromDict(self.ECG.r_pks)
        self.rpkTab.show()

    def open_rrTab(self):
        pass


class FormWidget(QWidget):
    def __init__(self, parent):
        super(FormWidget, self).__init__(parent)

        self.parent = parent
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setContentsMargins(0, 0, 0, 0)

        self.canvas_tb = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.canvas_tb)
        self.layout.addWidget(self.canvas)

        self.dropDownLayout = QHBoxLayout(self)

        self.stepChoice = QComboBox(self)
        self.eventChoice = QComboBox(self)
        self.stepChoice.setDisabled(True)
        self.eventChoice.setDisabled(True)

        self.plotButton = QPushButton('Plot', self)
        self.plotButton.setDisabled(True)
        self.plotButton.clicked.connect(self.plot)

        self.dropDownLayout.addWidget(self.stepChoice)
        self.dropDownLayout.addWidget(self.eventChoice)
        self.dropDownLayout.addWidget(self.plotButton)

        self.layout.addLayout(self.dropDownLayout)

    def plot(self):
        self.axes.clear()
        step = str(self.stepChoice.currentText())
        event = str(self.eventChoice.currentText())

        if step == 'Raw Data':
            self.axes.plot(self.parent.data.t[event], self.parent.data.v[event])
            self.axes.set_ylabel('Voltage (mV)')
            self.axes.set_xlabel('Time [s]')
        elif step == 'Filtered':
            self.axes.plot(self.parent.ECG.t[event], self.parent.ECG.v[event])
            self.axes.set_ylabel('Voltage (mV)')
            self.axes.set_xlabel('Time [s]')
        elif step == 'Derivative Filter':
            self.axes.plot(self.parent.ECG.t_der[event], self.parent.ECG.v_der[event])
            self.axes.set_ylabel('Voltage (mV)')
            self.axes.set_xlabel('Time [s]')
        elif step == 'Moving Average':
            self.axes.plot(self.parent.ECG.t_ma[event], self.parent.ECG.v_ma[event])
            self.axes.set_ylabel('Voltage (mV)')
            self.axes.set_xlabel('Time [s]')
        elif step == 'R-Peaks':
            self.axes.plot(self.parent.ECG.t[event], self.parent.ECG.v[event])
            self.axes.plot(self.parent.ECG.r_pks[event][:, 0], self.parent.ECG.r_pks[event][:, 1], 'ko')
            self.axes.set_ylabel('Voltage (mV)')
            self.axes.set_xlabel('Time [s]')
        elif step == 'Heart Rate (2-point)':
            x = self.parent.ECG.r_pks[event][1:, 0] - self.parent.ECG.r_pks[event][0, 0]
            y = 60/(self.parent.ECG.r_pks[event][1:, 0] - self.parent.ECG.r_pks[event][:-1, 0])
            self.axes.plot(x, y)
            self.axes.set_ylabel('2 point Instantaneous Heart Rate [BPM]')
            self.axes.set_xlabel('Time [s]')
        elif 'FM Respiratory Rate' in step:
            self.axes.plot(self.parent.ECG.rr[event]['fm'][:, 0], self.parent.ECG.rr[event]['fm'][:, 1], 'ko--')
            self.axes.set_xlabel('Time [s]')
            self.axes.set_ylabel('FM Respiratory Rate [BPM]')
        elif 'FM Respiratory Signal' in step:
            line1, = self.axes.plot(self.parent.ECG.rr_spl[event][:, 0], self.parent.ECG.rr_spl[event][:, 1],
                                    label='FM Signal')
            if self.parent.settings.plot_cntRange:
                for st, sp in self.parent.ECG.brth_cyc[event]:
                    self.axes.plot(self.parent.ECG.rr_spl[event][st:sp, 0], self.parent.ECG.rr_spl[event][st:sp, 1],
                                   'r', alpha=0.25, linewidth=5)
                red = Patch(color='red', alpha=0.25, label='Breath count range')
                self.axes.legend(handles=[line1, red])
            self.axes.set_ylabel('Signal')
            self.axes.set_xlabel('Time [s]')
        elif 'AM Respiratory Rate' in step:
            self.axes.plot(self.parent.ECG.rr[event]['am'][:, 0], self.parent.ECG.rr[event]['am'][:, 1], 'ko--')
            self.axes.set_xlabel('Time [s]')
            self.axes.set_ylabel('AM Respiratory Rate [BPM]')
        elif 'AM Respiratory Signal' in step:
            line1, = self.axes.plot(self.parent.ECG.rr_spl[event][:, 0], self.parent.ECG.rr_spl[event][:, 2],
                                    label='AM Signal')
            if self.parent.settings.plot_cntRange:
                for st, sp in self.parent.ECG.brth_cyc[event]:
                    self.axes.plot(self.parent.ECG.rr_spl[event][st:sp, 0], self.parent.ECG.rr_spl[event][st:sp, 2],
                                   'r', alpha=0.25, linewidth=5)
                red = Patch(color='red', alpha=0.25, label='Breath count range')
                self.axes.legend(handles=[line1, red])
            self.axes.set_ylabel('Signal')
            self.axes.set_xlabel('Time [s]')
        elif 'BW Respiratory Rate' in step:
            self.axes.plot(self.parent.ECG.rr[event]['bw'][:, 0], self.parent.ECG.rr[event]['bw'][:, 1], 'ko--')
            self.axes.set_xlabel('Time [s]')
            self.axes.set_ylabel('BW Respiratory Rate [BPM]')
        elif 'BW Respiratory Signal' in step:
            line1, = self.axes.plot(self.parent.ECG.rr_spl[event][:, 0], self.parent.ECG.rr_spl[event][:, 3],
                                    label='BW Signal')
            if self.parent.settings.plot_cntRange:
                for st, sp in self.parent.ECG.brth_cyc[event]:
                    self.axes.plot(self.parent.ECG.rr_spl[event][st:sp, 0], self.parent.ECG.rr_spl[event][st:sp, 3],
                                   'r', alpha=0.25, linewidth=5)
                red = Patch(color='red', alpha=0.25, label='Breath count range')
                self.axes.legend(handles=[line1, red])
            self.axes.set_ylabel('Signal')
            self.axes.set_xlabel('Time [s]')
        elif step == 'Fused Respiratory Rate':
            line1, = self.axes.plot(self.parent.ECG.rr_fuse[event][:, 0], self.parent.ECG.rr_fuse[event][:, 1],
                                    label='Fused Estimate')
            if self.parent.settings.plot_std:
                ind = argwhere(self.parent.ECG.lqi[event][1:] != self.parent.ECG.lqi[event][:-1]).flatten() + 1
                ind = append(ind, len(self.parent.ECG.lqi[event]))
                ind = insert(ind, 0, 0)

                for i1, i2 in zip(ind[:-1], ind[1:]+1):
                    self.axes.fill_between(self.parent.ECG.rr_fuse[event][i1:i2][:, 0],
                                           self.parent.ECG.rr_fuse[event][i1:i2][:, 1] -
                                           self.parent.ECG.st_dev[event][i1:i2],
                                           self.parent.ECG.rr_fuse[event][i1:i2][:, 1] +
                                           self.parent.ECG.st_dev[event][i1:i2],
                                           alpha=0.5, color='red' if self.parent.ECG.lqi[event][i1] else 'blue')

                    blue = Patch(color='blue', alpha=0.5, label='St. Dev. < 4 BPM')
                    red = Patch(color='red', alpha=0.5, label='St. Dev. > 4 BPM')

                    self.axes.legend(handles=[blue, red, line1], loc='best')
            else:
                self.axes.legend(loc='best')

            self.axes.set_xlabel('Time [s]')
            self.axes.set_ylabel('Fused Respiratory Rate [BPM]')

        self.canvas.draw()


class SettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent = parent

        # set defaults
        self.del_def = 175  # search delay for R-peak detection in milliseconds
        self.count_def = True  # true == count advanced, false == count original
        self.fuse_def = True  # perform fusion, defaults to True
        self.fuse_time_def = True  # use timings from respiratory rate parameter extraction

        # plotting defaults
        self.plot_std_def = True  # plot standard deviation/low quality index on fused estimate
        self.plot_cntRange_def = True  # plot the range over which count method is detecting breaths

        self.setDefaults()

        self.initUI()

    def initUI(self):
        self.setGeometry(400, 400, 250, 150)
        self.setWindowTitle('Settings')

        del_gb, self.del_w = self.createRPeakOptions()  # delay options

        cnt_gb, self.cnt_w = self.createCountOptions()  # count method options

        fus_gb, self.fus_w = self.createFuseOptions()  # create smart fusion options

        plt_gb, self.plt_w = self.createPlotOptions()  # create plot options

        resetDefaults = QPushButton('Reset Defaults', self)
        resetDefaults.setAutoDefault(False)
        resetDefaults.clicked.connect(lambda: self.setDefaults(clicked=True))

        windowLayout = QVBoxLayout()
        windowLayout.addWidget(del_gb)
        windowLayout.addWidget(cnt_gb)
        windowLayout.addWidget(fus_gb)
        windowLayout.addWidget(plt_gb)
        windowLayout.addWidget(resetDefaults)
        self.setLayout(windowLayout)

    def setDefaults(self, clicked=False):
        if not clicked:
            self.delay = self.del_def
            self.count = self.count_def
            self.fuse = self.fuse_def
            self.fuse_time = self.fuse_time_def
            self.plot_std = self.plot_std_def
            self.plot_cntRange = self.plot_cntRange_def
        else:
            self.del_w.setText(str(self.del_def))
            self.cnt_w[0].setChecked(self.count_def)
            self.fus_w[0].setChecked(self.fuse_def)
            self.fus_w[1].setChecked(self.fuse_time_def)
            self.plt_w[0].setChecked(self.plot_cntRange_def)
            self.plt_w[1].setChecked(self.plot_std_def)

    def createRPeakOptions(self):
        vGroupBox = QGroupBox('R-Peak Detection')
        layout = QVBoxLayout()

        line1 = QHBoxLayout()
        label1 = QLabel('Search Delay (ms):', self)
        label1.setToolTip('Delay in which if a descending half-peak is not found, declare that it is.')
        txtbox1 = QLineEdit(self)
        txtbox1.setText(str(self.delay))
        txtbox1.textChanged.connect(lambda text: self.changeIntVals(text, 'delay'))

        line1.addWidget(label1)
        line1.addWidget(txtbox1)

        layout.addLayout(line1)

        vGroupBox.setLayout(layout)

        return vGroupBox, txtbox1

    def createCountOptions(self):
        vGroupBox = QGroupBox('Count Methods')
        layout = QVBoxLayout()

        adv = QRadioButton('Count Advanced', self)
        adv.setToolTip('Use Count Advanced method to determine breaths')
        adv.toggled.connect(lambda: setattr(self, 'count', True))
        orig = QRadioButton('Count Original', self)
        orig.setToolTip('Use Count Original method to determine breaths')
        orig.toggled.connect(lambda: setattr(self, 'count', False))

        if self.count:
            adv.setChecked(True)
        else:
            orig.setChecked(True)

        layout.addWidget(orig)
        layout.addWidget(adv)

        vGroupBox.setLayout(layout)

        return vGroupBox, [adv, orig]

    def createFuseOptions(self):
        vGroupBox = QGroupBox('Respiratory Rate Fusion Options')
        layout = QVBoxLayout()

        timing = QCheckBox('Use RR timing', self)
        timing.setToolTip('Use Respiratory Rate parameter determined timings.  If not checked, will use 0.2s '
                          'intervals extracted from spline interpolation')
        timing.setDisabled(not self.fuse)
        timing.setChecked(self.fuse_time)
        timing.stateChanged.connect(self.changeTimeVals)

        fuse = QCheckBox('Fuse RR estimates', self)
        fuse.setToolTip('Fuse FM, AM, and BW Respiratory Rate estimates together')
        fuse.setChecked(self.fuse)
        fuse.stateChanged.connect(lambda state: self.changeFuseVals(state, timing))

        layout.addWidget(fuse)
        layout.addWidget(timing)

        vGroupBox.setLayout(layout)

        return vGroupBox, [fuse, timing]

    def createPlotOptions(self):
        vGroupBox = QGroupBox('Plotting Options')
        layout = QVBoxLayout()

        cb_cnt = QCheckBox('Plot count range', self)
        cb_cnt.setToolTip('Plot the range over which the count method is detecting breaths over the Respiratory signal')
        cb_cnt.setChecked(self.plot_cntRange)
        cb_cnt.stateChanged.connect(lambda: setattr(self, 'plot_cntRange', cb_cnt.isChecked()))

        cb_std = QCheckBox('Plot Fusion st. dev.', self)
        cb_std.setToolTip('Plot the standard deviation of the FM, AM, and BW estimates of respiratory rate along with '
                          'the fused estimate')
        cb_std.setChecked(self.plot_std)
        cb_std.stateChanged.connect(lambda: setattr(self, 'plot_std', cb_std.isChecked()))

        layout.addWidget(cb_cnt)
        layout.addWidget(cb_std)

        vGroupBox.setLayout(layout)

        return vGroupBox, [cb_cnt, cb_std]

    def changeIntVals(self, text, var_name):
        try:
            self.__dict__[var_name] = int(text)
        except ValueError:
            pass

    def changeFuseVals(self, cboxval, time_box):
        if cboxval == 0:
            self.fuse = False
            time_box.setDisabled(True)
            if self.parent.count_run:
                self.parent.fusAct.setDisabled(True)
        else:
            self.fuse = True
            time_box.setDisabled(False)
            if self.parent.count_run:
                self.parent.fusAct.setDisabled(False)

    def changeTimeVals(self, val):
        if val == 0:
            self.fuse_time = False
        else:
            self.fuse_time = True


class FilterSettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # set defaults
        self.mov_len_def = 150  # moving average filter length (samples) default

        self.elf_cut_def = 3.0  # Eliminate low frequency default cutoff (Hz)
        self.elf_N_def = 1  # Eliminate low frequency default filter order

        self.evhf_cut_def = 20.0  # eliminate very high frequency default cutoff (Hz)
        self.evhf_N_def = 1  # eliminate very high frequency default filter order
        self.evhf_det_def = True  # detrend input data to eliminate very high frequency default

        self.emf_cut_def = 60.0  # eliminate mains frequency default cutoff (Hz)
        self.emf_Q_def = 10  # eliminate mains frequency default notch filter quality
        self.emf_det_def = True  # detrend input data to eliminate mains frequency default

        self.esc_cut_def = 0.5  # eliminate sub-cardiac frequency default cutoff (Hz)
        self.esc_N_def = 4  # eliminate sub-cardiac frequency default order

        # set values
        self.setDefaults()

        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Filter Settings')

        elf_gb, self.elf_w = self.createFilterOptions('Eliminate Low Frequencies', 'elf_cut', 'elf_N')
        evhf_gb, self.evhf_w = self.createFilterOptions('Eliminate Very High Frequencies', 'evhf_cut', 'evhf_N',
                                                        'evhf_det')
        emf_gb, self.emf_w = self.createFilterOptions('Eliminate Mains Frequencies', 'emf_cut', 'emf_Q', 'emf_det')
        esc_gb, self.esc_w = self.createFilterOptions('Eliminate Sub-Cardiac Frequencies', 'esc_cut', 'esc_N')

        maf_gb = QGroupBox('Moving Average Filter')
        mafLayout = QHBoxLayout()
        mafTxt = QLineEdit(self)
        mafTxt.setText(str(self.mov_len))
        mafTxt.textEdited.connect(lambda text: self.changeIntVals(text, 'mov_len'))

        mafLayout.addWidget(QLabel('Length: ', self))
        mafLayout.addWidget(mafTxt)
        maf_gb.setLayout(mafLayout)

        resetDefaults = QPushButton('Reset Defaults', self)
        resetDefaults.setAutoDefault(False)
        resetDefaults.clicked.connect(lambda: self.setDefaults(clicked=True))

        windowLayout = QVBoxLayout()
        windowLayout.addWidget(elf_gb)
        windowLayout.addWidget(evhf_gb)
        windowLayout.addWidget(emf_gb)
        windowLayout.addWidget(esc_gb)
        windowLayout.addWidget(maf_gb)
        windowLayout.addWidget(resetDefaults)
        self.setLayout(windowLayout)

    def setDefaults(self, clicked=False):
        if not clicked:
            self.mov_len = self.mov_len_def
            self.elf_cut = self.elf_cut_def
            self.elf_N = self.elf_N_def
            self.evhf_cut = self.evhf_cut_def
            self.evhf_N = self.evhf_N_def
            self.evhf_det = self.evhf_det_def
            self.emf_cut = self.emf_cut_def
            self.emf_Q = self.emf_Q_def
            self.emf_det = self.emf_det_def
            self.esc_cut = self.esc_cut_def
            self.esc_N = self.esc_N_def
        else:
            self.elf_w[0].setText(str(self.elf_cut_def))
            self.elf_w[1].setText(str(self.elf_N_def))
            self.evhf_w[0].setText(str(self.evhf_cut_def))
            self.evhf_w[1].setText(str(self.evhf_N_def))
            self.evhf_w[2].setChecked(self.evhf_det_def)
            self.emf_w[0].setText(str(self.emf_cut_def))
            self.emf_w[1].setText(str(self.emf_Q_def))
            self.emf_w[2].setChecked(self.emf_det_def)
            self.esc_w[0].setText(str(self.esc_cut_def))
            self.esc_w[1].setText(str(self.esc_N_def))

    def createFilterOptions(self, title, cut_name, N_name, det_name=None):
        vGroupBox = QGroupBox(title)
        layout = QVBoxLayout()

        input_widgets = []

        line1 = QHBoxLayout()
        label1 = QLabel('Cutoff (Hz):', self)
        txtbox1 = QLineEdit(self)
        txtbox1.setText(str(self.__dict__[cut_name]))
        txtbox1.textChanged.connect(lambda text: self.changeFloatVals(text, cut_name))
        line1.addWidget(label1)
        line1.addWidget(txtbox1)

        input_widgets.append(txtbox1)
        layout.addLayout(line1)

        line2 = QHBoxLayout()
        label2 = QLabel('Order:', self)
        txtbox2 = QLineEdit(self)
        txtbox2.setText(str(self.__dict__[N_name]))
        txtbox2.textChanged.connect(lambda text: self.changeIntVals(text, N_name))
        line2.addWidget(label2)
        line2.addWidget(txtbox2)

        input_widgets.append(txtbox2)
        layout.addLayout(line2)

        if not isinstance(det_name, type(None)):
            line3 = QHBoxLayout()
            label3 = QLabel('Detrend?', self)
            chbox = QCheckBox(self)
            chbox.setChecked(self.__dict__[det_name])
            chbox.stateChanged.connect(lambda chboxval: self.changeCBoxVals(chboxval, det_name))
            line3.addWidget(label3)
            line3.addWidget(chbox)

            input_widgets.append(chbox)
            layout.addLayout(line3)

        vGroupBox.setLayout(layout)

        return vGroupBox, input_widgets

    def changeIntVals(self, text, name):
        try:
            self.__dict__[name] = int(text)
        except ValueError:
            pass

    def changeFloatVals(self, text, name):
        try:
            self.__dict__[name] = float(text)
        except ValueError:
            pass

    def changeCBoxVals(self, cboxval, name):
        if cboxval == 0:
            self.__dict__[name] = False
        else:
            self.__dict__[name] = True


class TableWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.initUI()

    def initUI(self):
        self.setWindowTitle('R-Peaks')
        self.setGeometry(300, 400, 300, 500)

        self.tab = QTabWidget()

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.tab)
        self.setLayout(self.layout)

    def populateFromDict(self, data):
        self.tabs = []  # append tabs to this list
        self.tables = []  # append tables to this list
        self.tab.clear()  # clear any old data/widgets

        eButtons = []  # store export buttons in here

        for ev in data.keys():
            self.tabs.append(QWidget())  # create a tab for the event
            self.tab.addTab(self.tabs[-1], ev)  # add the tab to the tab widget with the event name

            self.tabs[-1].layout = QVBoxLayout()  # create the layout for the tab

            m, _ = data[ev].shape  # get number of rows

            self.tables.append(self.createTable(m, 2))  # create and append the m-by-2 table for time and voltage
            self.tables[-1].setHorizontalHeaderLabels('Time [s];Voltage [mv]'.split(';'))  # set appropriate headers

            # set the data in the table
            for i in range(m):
                self.tables[-1].setItem(i, 0, QTableWidgetItem(str(data[ev][i, 0])))
                self.tables[-1].setItem(i, 1, QTableWidgetItem(str(data[ev][i, 1])))

            # create the export button
            eButtons.append(QPushButton('Export', self))
            eButtons[-1].clicked.connect(lambda: self.exportData(data[ev]))

            self.tables[-1].move(0, 0)
            self.tabs[-1].layout.addWidget(self.tables[-1])
            self.tabs[-1].layout.addWidget(eButtons[-1])
            self.tabs[-1].setLayout(self.tabs[-1].layout)

        # TODO add export button and function for R-peak data

    def exportData(self, data):
        """
        Export tabulated data to the chosen file.  If a csv file is chosen, delimiter is comma, if .txt, a space
        """
        saveFile, _ = QFileDialog.getSaveFileName(self, "Export Data", "", "CSV (*.csv);;TXT (*.txt);;All Files (*.*)",
                                                  options=QFileDialog.Options())

        if saveFile:
            if '.csv' in saveFile:
                savetxt(saveFile, data, delimiter=',', fmt='%.8f', header='Time [s], Voltage[mv]')
            else:
                savetxt(saveFile, data, delimiter=' ', fmt='%.8f', header='Time [s], Voltage[mv]')

    def createTable(self, m, n):
        table = QTableWidget()
        table.setRowCount(m)
        table.setColumnCount(n)

        # self.table.setItem(0,0, QTableWidgetItem("Cell (1,1)"))

        # table selection change
        table.doubleClicked.connect(self.on_dclick)

        return table

    @pyqtSlot()
    def on_dclick(self):
        # do stuff
        pass


# below code allows exceptions to be reported for debugging the program
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook


def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook

if __name__ == '__main__':
    app = QApplication(sys.argv)
    RR = RespiRate()
    sys.exit(app.exec_())
