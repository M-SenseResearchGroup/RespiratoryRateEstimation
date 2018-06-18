"""
V0.2.0
June 14, 2018

Lukas Adamowicz

Python 3.6.5 on Windows 10 with 64-bit Anaconda
"""

import sys
from os import getcwd
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QAction, qApp, QMessageBox, QMenu, QToolTip, \
    QLabel, QLineEdit, QHBoxLayout, QVBoxLayout, QGroupBox, QDialog, QCheckBox, QPushButton
from PyQt5.QtGui import QIcon
from ECGAnalysis_v2 import ECGAnalysis, EcgData
from numpy import loadtxt, array, argmin
from pickle import load as Pload, dump as Pdump


class RespiRate(QMainWindow):
    def __init__(self):
        super().__init__()

        self.sbar = self.statusBar()  # new shorthand for status bar
        self.annots = None  # allocate for annotations to allow for checking later on

        self.loc = getcwd()  # get the location of the file.

        self.filtSet= FilterSettingsWindow(self)

        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 650, 450)
        self.setWindowTitle('RespiRate')

        # save imported raw data
        self.saveData = QAction('Save', self)
        self.saveData.setDisabled(True)  # disable initially because we have no imported data to save
        self.saveData.setShortcut('Ctrl+S')
        self.saveData.setStatusTip('Save imported raw data')
        self.saveData.triggered.connect(self.save_data)

        # Exit menu action
        exitAct = QAction(QIcon('exit.png'), '&Exit', self)
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

        # Open settings window
        filtSetAct = QAction('Filter Settings', self)
        filtSetAct.setStatusTip('Open filter settings dialog')
        filtSetAct.triggered.connect(self.open_settings)

        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        settingsmenu = menubar.addMenu('&Settings')

        filemenu.addMenu(importMenu)
        filemenu.addAction(openData)
        filemenu.addAction(self.saveData)
        filemenu.addAction(exitAct)

        settingsmenu.addAction(filtSetAct)

        self.show()

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
            fid = open(openFile,'rb')
            self.data = Pload(fid)
            fid.close()
            self.sbar.clearMessage()

            self.saveData.setDisabled(False)

    def open_annotation_mc10(self):
        """
        Opens MC10 activity annotation file and then displays how many activities were found
        """

        # TODO remove home_dir, which is used only for testing
        home_dir = "C:\\Users\\Lukas Adamowicz\\Dropbox\\Masters\\Project - Bike Study\\RespiratoryRate_HeartRate\\" +\
                   "RespiratoryRateEstimation\\Validation_Data\\RespRate_PPG_Phone\\DN"

        annotationFile, _ = QFileDialog.getOpenFileName(self, "Open MC10 Annotation File", home_dir,
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

        leadFile, _ = QFileDialog.getOpenFileName(self, "Open MC10 ECG file", home_dir,
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
            self.saveData.setDisabled(False)
        self.sbar.clearMessage()  # clear the status bar message when done importing

    def open_settings(self):
        self.filtSet.show()


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
        if clicked == False:
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
