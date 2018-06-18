"""
V0.2.0
June 14, 2018

Lukas Adamowicz

Python 3.6.5 on Windows 10 with 64-bit Anaconda
"""

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QAction, qApp, QMessageBox, QMenu
from PyQt5.QtGui import QIcon
from ECGAnalysis_v2 import ECGAnalysis, EcgData
from numpy import loadtxt, array, argmin


class RespiRate(QMainWindow):
    def __init__(self):
        super().__init__()

        self.sbar = self.statusBar()  # new shorthand for status bar
        self.annots = None  # allocate for annotations to allow for checking later on

        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 650, 450)
        self.setWindowTitle('RespiRate')

        exitAct = QAction(QIcon('exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        openAnnot = QAction('Import Annotations', self)
        openAnnot.setStatusTip('Import Annotation File')
        openAnnot.triggered.connect(self.open_annotation_mc10)

        openRaw = QAction("Import Raw Data", self)
        openRaw.setStatusTip('Import Raw ECG Data')
        openRaw.triggered.connect(self.open_lead_mc10)

        importMenu = QMenu('Import Data', self)
        importMenu.addAction(openAnnot)
        importMenu.addAction(openRaw)

        self.statusBar()

        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')

        filemenu.addMenu(importMenu)
        filemenu.addAction(exitAct)

        self.show()

    def open_annotation_mc10(self):
        """
        Opens MC10 activity annotation file and then displays how many activities were found
        """

        # for easy testing
        home_dir = "C:\\Users\\Lukas Adamowicz\\Dropbox\\Masters\\Project - Bike Study\\RespiratoryRate_HeartRate\\" +\
                   "RespiratoryRateEstimation\\Validation_Data\\RespRate_PPG_Phone\\DN"

        annotationFile, _ = QFileDialog.getOpenFileName(self, "Open MC10 Annotation File", home_dir,
                                                        "Comma Separated Value (*.csv);;All Files (*.*)",
                                                        options=QFileDialog.Options())

        if annotationFile:
            # import annotation data
            events, starts, stops = loadtxt(annotationFile, delimiter=',', usecols=(2, 4, 5), skiprows=1, unpack=True,
                                            dtype=str)
            # save the annotation data to a dictionary
            self.annots = {ev[1:-1]: array([start[1:-1], stop[1:-1]]).astype(float) for ev, start, stop in zip(events, starts, stops)}

            # display a message showing how many events were found
            QMessageBox.question(self, "Annotation Import", f"Found {len(self.annots.keys())} events.", QMessageBox.Ok)

    def open_lead_mc10(self):
        """
        Opens MC10 ECG lead data
        """

        # for easy testing
        home_dir = "C:\\Users\\Lukas Adamowicz\\Dropbox\\Masters\\Project - Bike Study\\RespiratoryRate_HeartRate\\" + \
                   "RespiratoryRateEstimation\\Validation_Data\\RespRate_PPG_Phone\\DN\\ecg_lead_ii\\d5la7xul\\" +\
                   "2018-01-24T03-18-36-159Z"

        leadFile, _ = QFileDialog.getOpenFileName(self, "Open MC10 ECG file", home_dir,
                                                  "Comma Separated Value (*.csv);;All Files (*.*)",
                                                  options=QFileDialog.Options())

        if leadFile:
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
