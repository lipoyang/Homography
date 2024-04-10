#! /usr/bin/env python3
#  -*- coding: utf-8 -*-
#
# Support module generated by PAGE version 8.0
#  in conjunction with Tcl version 8.6
#    Apr 09, 2024 12:59:20 PM JST  platform: Windows NT

import sys
import os
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
import tkinter.messagebox
from tkinter.constants import *
import threading

import Homography
import HomographyTest4

# 初期化ずみフラグ
hasInitialized = False

# メイン関数
def main(*args):
    '''Main entry point for the application.'''
    global root, _w1
    root = tk.Tk()
    root.protocol( 'WM_DELETE_WINDOW' , root.destroy)
    # Creates a toplevel widget.
    global _top1, _w1
    _top1 = root
    _w1 = Homography.mainWindow(_top1)

    # 初期化コード追加
    _w1.canvasMain.bind("<Button-1>",        HomographyTest4.mouse_down)
    _w1.canvasMain.bind('<ButtonRelease-1>', HomographyTest4.mouse_up  )
    _w1.canvasMain.bind('<Motion>',          HomographyTest4.mouse_move)
    _w1.canvasMain.bind("<Map>", canvas_onMap) # キャンバスが描画されたとき

    root.mainloop()

# キャンバスが描画されたとき(サイズが確定したとき)
def canvas_onMap(event):
    # 初期化ずみでなれば初期化処理
    global _w1, hasInitialized
    if hasInitialized: return
    HomographyTest4.initialize(_w1.canvasMain)
    hasInitialized = True

# Loadボタンクリック時
def buttonLoad_onClick(*args):
    global file_name, sub_window
    fTyp = [("画像ファイル", "*.bmp;*.png;*.jpg")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if file_name != "":
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()
        if ext != ".bmp" and ext != ".png" and ext != ".jpg":
            tk.messagebox.showerror("エラー", "拡張子が不正です")
            return
        # print(f"file_name={file_name}")
        x = root.winfo_x() + 100
        y = root.winfo_y() + 100
        sub_window = tkinter.Toplevel(root)
        sub_window.geometry(f"200x80+{x}+{y}")
        sub_window.resizable(0,  0)
        sub_window.title(" ")
        sub_window.attributes("-topmost", True)
        label = tkinter.Label(sub_window, text = "画像読み込み中")
        label.pack(pady=(10,0), padx=10)
        progress_bar = ttk.Progressbar(sub_window, orient="horizontal", length=200, mode="indeterminate")
        progress_bar.pack(pady=10, padx=10)
        progress_bar.start(5)
        thread1 = threading.Thread(target=loadImage)
        thread1.start()

# 画像読み込み/描画処理 (重いので別スレッド)
def loadImage():
    global file_name, sub_window
    HomographyTest4.loadImage(file_name)
    sub_window.destroy()

# Saveボタンクリック時
def buttonSave_onClick(*args):
    if not HomographyTest4.hasImageLoaded: return
    fTyp = [("PNGファイル", "*.png")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tkinter.filedialog.asksaveasfilename(filetypes=fTyp, initialdir=iDir, defaultextension = "png")
    if file_name != "":
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()
        if ext != ".png":
            tk.messagebox.showerror("エラー", "拡張子が不正です")
            return
        # print(f"file_name={file_name}")
        HomographyTest4.saveImage(file_name)

# Borderチェックボックス変更時
def checkBorder_onChange(*args):
    global _w1
    check = _w1.checkBorderVal.get()
    # print(f"check={check}")
    HomographyTest4.setFrameValid(check)

if __name__ == '__main__':
    Homography.start_up()
