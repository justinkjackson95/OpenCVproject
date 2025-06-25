#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import pyqtSlot

CaptureTool2UI, CaptureTool2WindowBase = uic.loadUiType("captureTool2.ui")

class CaptureTool2(CaptureTool2UI, CaptureTool2WindowBase):
    def __init__(self, parent : QtWidgets.QWidget = None):
        CaptureTool2WindowBase.__init__(self, parent=parent)
        self.setupUi(self)

        # UI items event connection        
        #lambda is used when passing extras args to method
        self.exitButton.clicked.connect(self.closescr)

    def closescr(self, CaptureTool2):
        # hide the screen on exit button click
        self.hide()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    captureTool2 = CaptureTool2()
    captureTool2.show()
    sys.exit(app.exec_())
