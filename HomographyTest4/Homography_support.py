#! /usr/bin/env python3
#  -*- coding: utf-8 -*-
#
# Support module generated by PAGE version 8.0
#  in conjunction with Tcl version 8.6
#    Apr 09, 2024 12:59:20 PM JST  platform: Windows NT

import sys
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *

import Homography

_debug = True # False to eliminate debug printing from callback functions.

def main(*args):
    '''Main entry point for the application.'''
    global root
    root = tk.Tk()
    root.protocol( 'WM_DELETE_WINDOW' , root.destroy)
    # Creates a toplevel widget.
    global _top1, _w1
    _top1 = root
    _w1 = Homography.mainWindow(_top1)
    root.mainloop()

def buttonLoad_onClick(*args):
    if _debug:
        print('Homography_support.buttonLoad_onClick')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def buttonSave_onClick(*args):
    if _debug:
        print('Homography_support.buttonSave_onClick')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def checkBorder_onChange(*args):
    if _debug:
        print('Homography_support.checkBorder_onChange')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

if __name__ == '__main__':
    Homography.start_up()



