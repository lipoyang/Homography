from numba import jit # 要 pip install scipy
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import os
import math
import time #デバッグ用

# ホモグラフィ行列の計算
# (xi_, yi_) : 写像後の点
@jit(nopython=True, cache=True)
def calcH(x0_, y0_, x1_, y1_, x2_, y2_, x3_, y3_):

    # ホモグラフィ行列 H の計算
    h02 = x0_
    h12 = y0_
    h22 = 1

    A = x1_ - x2_; B = x3_ - x2_; C = x0_ - x1_ + x2_ - x3_
    D = y1_ - y2_; E = y3_ - y2_; F = y0_ - y1_ + y2_ - y3_
    h20 = (C*E - B*F) / (A*E - B*D)
    h21 = (A*F - C*D) / (A*E - B*D)

    h00 = x1_ - x0_ + x1_*h20
    h10 = y1_ - y0_ + y1_*h20
    h01 = x3_ - x0_ + x3_*h21
    h11 = y3_ - y0_ + y3_*h21

    # H の逆行列 H^-1 の計算
    det = h00*h11*h22 + h01*h12*h20 + h02*h10*h21 - h02*h11*h20 - h00*h12*h21 - h01*h10*h22
    h00_ = (h11*h22 - h12*h21)/det;  h01_ = (h02*h21 - h01*h22)/det;  h02_ = (h01*h12 - h02*h11)/det
    h10_ = (h12*h20 - h10*h22)/det;  h11_ = (h00*h22 - h02*h20)/det;  h12_ = (h02*h10 - h00*h12)/det
    h20_ = (h10*h21 - h11*h20)/det;  h21_ = (h01*h20 - h00*h21)/det;  h22_ = (h00*h11 - h01*h10)/det

    H_ = np.array([[h00_, h01_, h02_],
                   [h10_, h11_, h12_],
                   [h20_, h21_, h22_]])
    return H_

#  ホモグラフィ変換の画像描画
@jit(nopython=True, cache=True)
def drawHomography(p_, dst_data, fine):

    # ホモグラフィ行列
    H_ = calcH(p_[0, 0], p_[0, 1], p_[1, 0], p_[1, 1], p_[2, 0], p_[2, 1], p_[3, 0], p_[3, 1])
    h00 = H_[0,0]; h01 = H_[0,1]; h02 = H_[0,2];  
    h10 = H_[1,0]; h11 = H_[1,1]; h12 = H_[1,2];  
    h20 = H_[2,0]; h21 = H_[2,1]; h22 = H_[2,2];  

    # 描画範囲
    x1_ = int(min(p_[0, 0], p_[1, 0], p_[2, 0], p_[3, 0]))
    x2_ = int(max(p_[0, 0], p_[1, 0], p_[2, 0], p_[3, 0]))
    y1_ = int(min(p_[0, 1], p_[1, 1], p_[2, 1], p_[3, 1]))
    y2_ = int(max(p_[0, 1], p_[1, 1], p_[2, 1], p_[3, 1]))

    # 写像先の各々の点について
    for y_ in range(y1_, y2_+1):
        for x_ in range(x1_, x2_+1):
            # (x, y) = (X/W, Y/W), (X, Y, W) = H^-1 (x', y', 1)
            X = h00 * x_ + h01 * y_ + h02
            Y = h10 * x_ + h11 * y_ + h12
            W = h20 * x_ + h21 * y_ + h22
            x = (SrcW-1) * X / W
            y = (SrcH-1) * Y / W

            # 範囲の判定
            xc = math.ceil(x-0.01)
            yc = math.ceil(y-0.01)
            xf = math.floor(x)
            yf = math.floor(y)
            if 0 <= xf and xc < SrcW and 0 <= yf and yc < SrcH:
                if fine:
                    # 線形補間で色を取得
                    dst_data[y_, x_] = interpolation(x,y)
                else:
                    # 荒く高速描画
                    X = int(x)
                    Y = int(y)
                    dst_data[y_, x_, 0] = src_data[Y, X, 0]
                    dst_data[y_, x_, 1] = src_data[Y, X, 1]
                    dst_data[y_, x_, 2] = src_data[Y, X, 2]
                    dst_data[y_, x_, 3] = 255

# 線形補間
@jit(nopython=True, cache=True)
def interpolation(x, y):
    R = np.zeros((2, 2), dtype=float)
    G = np.zeros((2, 2), dtype=float)
    B = np.zeros((2, 2), dtype=float)

    X = int(x)
    Y = int(y)
    for i in range(2):
        for j in range(2):
            _x = X + i
            if _x >= SrcW: _x = X
            _y = Y + j
            if _y >= SrcH: _y = Y
            R[i, j] = src_data[_y, _x, 0]
            G[i, j] = src_data[_y, _x, 1]
            B[i, j] = src_data[_y, _x, 2]            
    dX = x - X
    dY = y - Y
    MdX = 1 - dX
    MdY = 1 - dY
    r = round(MdX * (MdY * R[0, 0] + dY * R[0, 1]) + dX * (MdY * R[1, 0] + dY * R[1, 1]))
    g = round(MdX * (MdY * G[0, 0] + dY * G[0, 1]) + dX * (MdY * G[1, 0] + dY * G[1, 1]))
    b = round(MdX * (MdY * B[0, 0] + dY * B[0, 1]) + dX * (MdY * B[1, 0] + dY * B[1, 1]))
    return r, g, b, 255

# 描画
def draw(fine):
    global img_tk

    start_time = time.time()
    
    # ホモグラフィ画像の描画
    dst_data = np.zeros((WinH, WinW, 4), dtype=np.uint8)
    drawHomography(p_, dst_data, fine)

    dst_img = Image.fromarray(dst_data, 'RGBA')
    img_tk = ImageTk.PhotoImage(dst_img)
    canvas.delete("all")
    canvas.create_image(0, 0, image = img_tk, anchor='nw') # 'nw':アンカー位置左上

    # 枠線
    if isFrameValid:
        canvas.create_line(p_[0, 0], p_[0, 1], p_[1, 0], p_[1, 1], fill="red")
        canvas.create_line(p_[1, 0], p_[1, 1], p_[2, 0], p_[2, 1], fill="red")
        canvas.create_line(p_[2, 0], p_[2, 1], p_[3, 0], p_[3, 1], fill="red")
        canvas.create_line(p_[3, 0], p_[3, 1], p_[0, 0], p_[0, 1], fill="red")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"time: {elapsed_time} sec")

# 凸四角形判定
def isConvexQuad():
    for i in range(4):
        x1, y1 = p_[i]
        x2, y2 = p_[(i + 1) % 4]
        x3, y3 = p_[(i + 2) % 4]
        cross_product = float((x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2))
        if cross_product < 0:
            return False
    return True

# 画面内判定
def isInWindow(x, y):
    ret = 0 <= x and x < WinW and 0 <= y and y < WinH
    return ret 

# マウスイベント関連
R2 = 10*10  # 頂点の近傍閾値(半径ピクセル数の自乗)
p_sel = -1  # 選択中の頂点 (0～3, -1は未選択状態)

# マウス左ボタン押したとき
def mouse_down(event):
    global p_sel, p_
    x, y = event.x, event.y
    if x < 0 or x >= WinW or y < 0 or y >= WinH: return
    for i in range(4):
        X = p_[i, 0]; Y = p_[i, 1]
        r2 = (X-x)*(X-x) + (Y-y)*(Y-y)
        if r2 < R2:
            p_sel = i
            p_[i, 0] = x
            p_[i, 1] = y
            # 画面内で凸四角形か判定
            if isInWindow(x, y) and isConvexQuad():
                draw(False)
            else:
                p_ = p_prev.copy()
                p_sel = -1
            break

# マウス左ボタン離したとき
def mouse_up(event):
    global p_sel, p_, p_prev
    if p_sel >= 0:
        x, y = event.x, event.y
        p_[p_sel, 0] = x
        p_[p_sel, 1] = y
        # 画面内で凸四角形か判定
        if isInWindow(x, y) and isConvexQuad():
            p_prev = p_.copy()
        else:
            p_ = p_prev.copy()
        draw(True)
    p_sel = -1

# マウスドラッグ時
def mouse_move(event):
    global p_sel, p_
    button_state = event.state
    if button_state & 0x100:  # 左ボタン状態
        if p_sel >= 0:
            x, y = event.x, event.y
            p_[p_sel, 0] = x
            p_[p_sel, 1] = y
            # 画面内で凸四角形か判定
            if not (isInWindow(x, y) and isConvexQuad()):
                p_ = p_prev.copy()
                p_sel = -1
            draw(False)

# キー入力判定
isFrameValid = True
def key_press(event):
    global isFrameValid
    if event.keysym == "f" or event.keysym == "F":
        isFrameValid = not isFrameValid
        draw(True)

# 元画像を開く
dir_path = os.path.dirname(__file__)
image_path = os.path.join(dir_path, "lena_alt.bmp")
src_img = Image.open(image_path)
src_data = np.asarray(src_img)
SrcW, SrcH = src_img.size # 画像のサイズ

# 四隅の座標の初期値
p_ = np.array([[50, 50], [50+SrcW-1, 50], [50+SrcW-1, 50+SrcH-1], [50, 50+SrcH-1]], dtype=np.float64)
p_prev = p_.copy()

# ウィンドウ
WinW = 800; WinH = 600 # ウィンドウサイズ
root = tk.Tk()
root.title(u"ホモグラフィ変換")
root.geometry(str(WinW) + "x" + str(WinH))
root.bind("<Button-1>",        mouse_down)
root.bind('<ButtonRelease-1>', mouse_up  )
root.bind('<Motion>',          mouse_move)
root.bind("<KeyPress>",        key_press)
canvas = tk.Canvas(root, width = WinW, height = WinH)
canvas.place(x=0, y=0)

# 最初の描画
draw(True)

# メインループ
root.mainloop()
