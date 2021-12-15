from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from ttkbootstrap import Style
from tkinter import ttk
import cv2
import math
import scipy.signal as ss
from tkinter import filedialog

matplt = False
original_img = None
board = None
axis = None
figure = None
toolbar = None


def remove_oldPlot():
    global board
    global toolbar
    if matplt:
        toolbar.pack_forget()
        board.get_tk_widget().pack_forget()


def open_File(parent):
    global original_img
    global board
    global matplt
    global figure
    global toolbar
    filename = filedialog.askopenfilename()
    print(filename)
    original_img = cv2.imread(filename,1)
    remove_oldPlot()
    figure = plt.Figure(figsize=(7.83, 4.85))
    axis = figure.add_subplot(111)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)
    axis.imshow(original_img)
    figure.set_facecolor('#98D6EA')
    board = FigureCanvasTkAgg(figure, master=parent)
    board.draw()
    board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    toolbar = NavigationToolbar2Tk(board, parent)
    toolbar.update()
    board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)
    matplt = True

def add_Noise(parent):
    if matplt:
        global original_img
        global board
        global figure
        global toolbar
        gauss = np.random.normal(0, 1, original_img.size)
        gauss = gauss.reshape(original_img.shape[0], original_img.shape[1], original_img.shape[2]).astype('uint8')
        original_img = original_img + original_img * gauss
        remove_oldPlot()
        figure = plt.Figure(figsize=(7.83, 4.85))
        axis = figure.add_subplot(111)
        axis.imshow(original_img)
        figure.set_facecolor('#98D6EA')
        board = FigureCanvasTkAgg(figure, master=parent)
        board.draw()
        board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(board, parent)
        toolbar.update()
        board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)

Sliders = {}
Labels_Sliders = {}
Params_Sliders = {}
Btn_Sliders = {}

# Gamma


def Gamma(value, parent):
    if matplt:
        global board
        global toolbar

        invGamma = 1.0 / float(value)
        table = np.array([((i / 255.0) ** invGamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
        Gamma_img = cv2.LUT(original_img, table)

        figure = plt.Figure(figsize=(7.83, 4.85))
        axs = figure.add_subplot(1, 1, 1)
        axs.imshow(Gamma_img)
        remove_oldPlot()
        figure.set_facecolor('#98D6EA')
        board = FigureCanvasTkAgg(figure, master=parent)
        board.draw()
        board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(board, parent)
        toolbar.update()
        board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)
        print(value)


def UpdateGamma(val):
    Params_Sliders['Gamma'][0].delete(0, END)
    Params_Sliders['Gamma'][0].insert(0, val)
    # print(Params_Sliders['Gamma'][0].get())


def Gamma_Slide(LFrame, win_show):
    if 'Gamma' not in Sliders:
        remove_Sliders()
        Gamma_Var = StringVar()
        Gamma_Var.set("1.00")
        Gamma_Param = ttk.Entry(
            LFrame, style="primary.TEntry", width=4, textvariable=Gamma_Var)
        Gamma_Label = ttk.Label(LFrame, text="Gamma:", font=2)
        Gamma_Slider = Scale(LFrame, from_=1.0, to=2.0, resolution=0.01,
                             length=100, orient="horizontal", command=lambda value: UpdateGamma(value))
        Gamma_Btn = Button(LFrame, text="OK", height=1,
                           command=lambda: Gamma(Gamma_Var.get(), win_show))
        Gamma_Slider.grid(row=2, column=0)
        Gamma_Label.grid(row=1, column=0, sticky=W,padx=(6.5, 0))
        Gamma_Param.grid(row=2, column=1,pady=(0,3))
        Gamma_Btn.grid(row=1, column=1, ipadx=6)
        Sliders['Gamma'] = Gamma_Slider
        Labels_Sliders['Gamma'] = Gamma_Label
        Params_Sliders['Gamma'] = [Gamma_Param, Gamma_Var]
        Btn_Sliders['Gamma'] = Gamma_Btn


# Histogram
def Histogram(parent):
    if matplt:
        global board
        global toolbar
        remove_Sliders()
        color = ('b', 'g', 'r')
        figure = plt.Figure(figsize=(7.83, 4.85))
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=100)
        transform_img = clahe.apply(gray_img)
        axs = figure.add_subplot(2, 2, 1)
        axs.imshow(transform_img, cmap='gray')
        axs.set_title("cv2.createCLAHE")
        for i, col in enumerate(color):
            axs = figure.add_subplot(2, 2, i+2)
            hist = cv2.calcHist(images=[original_img], channels=[
                                i], mask=None, histSize=[256], ranges=[0, 256])
            axs.plot(hist, color=col)
        remove_oldPlot()
        figure.set_facecolor('#98D6EA')
        board = FigureCanvasTkAgg(figure, master=parent)
        board.draw()
        board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(board, parent)
        toolbar.update()
        board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)

# Log

def Log(value, parent):
    if matplt:
        global board
        global toolbar
        
        cl = float(value) /float(np.log(256))
        Log_img = cl * np.log(original_img.astype(float)+1)
        Log_img = np.array(Log_img,dtype=np.uint8)

        figure = plt.Figure(figsize=(7.83, 4.85))
        axs = figure.add_subplot(1, 1, 1)
        axs.imshow(Log_img)
        remove_oldPlot()
        figure.set_facecolor('#DFD8CA')
        board = FigureCanvasTkAgg(figure, master=parent)
        board.draw()
        board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(board, parent)
        toolbar.update()
        board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)
        print(value)

def UpdateLog(val):
    Params_Sliders['Log'][0].delete(0, END)
    Params_Sliders['Log'][0].insert(0, val)
    # print(Params_Sliders['Gamma'][0].get())

def Log_Slide(LFrame, win_show):
    if 'Log' not in Sliders:
        remove_Sliders()
        Log_Var = StringVar()
        Log_Var.set("0")
        Log_Param = ttk.Entry(
            LFrame, style="primary.TEntry", width=4, textvariable=Log_Var)
        Log_Label = ttk.Label(LFrame, text="LoG:", font=2)
        Log_Slider = Scale(LFrame, from_=1.0, to=256.0,
                             length=100, orient="horizontal", command=lambda value: UpdateLog(value))
        Log_Btn = Button(LFrame, text="OK", height=1,
                           command=lambda: Log(Log_Var.get(), win_show))
        Log_Slider.grid(row=2, column=0)
        Log_Label.grid(row=1, column=0
        ,sticky=W,padx=(6.5,0))
        Log_Param.grid(row=2, column=1, pady=(0, 3))
        Log_Btn.grid(row=1, column=1, ipadx=6)
        Sliders['Log'] = Log_Slider
        Labels_Sliders['Log'] = Log_Label
        Params_Sliders['Log'] = [Log_Param, Log_Var]
        Btn_Sliders['Log'] = Log_Btn

# Power

def Power(value, parent):
    if matplt:
        global board
        global toolbar
        cp = 225/(255*float(value))
        Power_img = cp*(original_img.astype(np.float)**float(value))
        Power_img = np.array(Power_img, dtype=np.uint8)

        figure = plt.Figure(figsize=(7.83, 4.85))
        axs = figure.add_subplot(1, 1, 1)
        axs.imshow(Power_img)
        remove_oldPlot()
        figure.set_facecolor('#98D6EA')
        board = FigureCanvasTkAgg(figure, master=parent)
        board.draw()
        board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(board, parent)
        toolbar.update()
        board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)
        print(value)

def UpdatePower(val):
    Params_Sliders['Power'][0].delete(0, END)
    Params_Sliders['Power'][0].insert(0, val)
    # print(Params_Sliders['Gamma'][0].get())

def Power_Slide(LFrame, win_show):
    if 'Power' not in Sliders:
        remove_Sliders()
        Power_Var = StringVar()
        Power_Var.set("0")
        Power_Param = ttk.Entry(
            LFrame, style="primary.TEntry", width=4, textvariable=Power_Var)
        Power_Label = ttk.Label(LFrame, text="Power:", font=2)
        Power_Slider = Scale(LFrame, from_=1.0, to=10, resolution=0.01,
                             length=100, orient="horizontal", command=lambda value: UpdatePower(value))
        Power_Btn = Button(LFrame, text="OK", height=1,
                           command=lambda: Power(Power_Var.get(), win_show))
        Power_Slider.grid(row=2, column=0)
        Power_Label.grid(row=1, column=0
        ,sticky=W,padx=(6.5,0))
        Power_Param.grid(row=2, column=1, pady=(0, 3))
        Power_Btn.grid(row=1, column=1, ipadx=6)
        Sliders['Power'] = Power_Slider
        Labels_Sliders['Power'] = Power_Label
        Params_Sliders['Power'] = [Power_Param, Power_Var]
        Btn_Sliders['Power'] = Power_Btn


# Quantization

def Quanz(value, parent):
    if matplt:
        global board
        global toolbar
        numberOfLevels = 2.**float(value)
        levelGap = 256 / numberOfLevels
        Quanz_img = np.ceil(original_img/levelGap)*levelGap-1
        Quanz_img = np.array(Quanz_img, dtype=np.uint8)
        figure = plt.Figure(figsize=(7.83, 4.85))
        axs = figure.add_subplot(1, 1, 1)
        axs.imshow(Quanz_img)
        remove_oldPlot()
        figure.set_facecolor('#98D6EA')
        board = FigureCanvasTkAgg(figure, master=parent)
        board.draw()
        board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(board, parent)
        toolbar.update()
        board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)
        print(value)

def UpdateQuanz(val):
    Params_Sliders['Quanz'][0].delete(0, END)
    Params_Sliders['Quanz'][0].insert(0, val)
    # print(Params_Sliders['Gamma'][0].get())

def Quanz_Slide(LFrame, win_show):
    if 'Quanz' not in Sliders:
        remove_Sliders()
        Quanz_Var = StringVar()
        Quanz_Var.set("1")
        Quanz_Param = ttk.Entry(
            LFrame, style="primary.TEntry", width=4, textvariable=Quanz_Var)
        Quanz_Label = ttk.Label(LFrame, text="Bit(s):", font=2)
        Quanz_Slider = Scale(LFrame, from_=1.0, to=8,
                             length=100, orient="horizontal", command=lambda value: UpdateQuanz(value))
        Quanz_Btn = Button(LFrame, text="OK", height=1,
                           command=lambda: Quanz(Quanz_Var.get(), win_show))
        Quanz_Slider.grid(row=2, column=0)
        Quanz_Label.grid(row=1, column=0
        ,sticky=W,padx=(6.5,0))
        Quanz_Param.grid(row=2, column=1, pady=(0, 3))
        Quanz_Btn.grid(row=1, column=1, ipadx=6)
        Sliders['Quanz'] = Quanz_Slider
        Labels_Sliders['Quanz'] = Quanz_Label
        Params_Sliders['Quanz'] = [Quanz_Param, Quanz_Var]
        Btn_Sliders['Quanz'] = Quanz_Btn


# Sub_bg

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def Sub_bg(value, parent):
    if matplt:
        value=float(value)
        global board
        global toolbar
        alpha = 0.95
        background = 1
        im = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        im = im2double(im)
        background = alpha*background+(1-alpha)*im
        diffImg = np.abs(im-background)
        threaImg = diffImg > value
        threaImg = threaImg.astype('int') * 255
        threaImg = np.array(threaImg, dtype=np.uint8)
        figure = plt.Figure(figsize=(7.83, 4.85))
        axs = figure.add_subplot(1, 1, 1)
        axs.imshow(threaImg,cmap='gray')
        remove_oldPlot()
        figure.set_facecolor('#98D6EA')
        board = FigureCanvasTkAgg(figure, master=parent)
        board.draw()
        board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(board, parent)
        toolbar.update()
        board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)
        print(value)

def UpdateSub_bg(val):
    Params_Sliders['Sub_bg'][0].delete(0, END)
    Params_Sliders['Sub_bg'][0].insert(0, val)
    # print(Params_Sliders['Gamma'][0].get())

def Sub_bg_Slide(LFrame, win_show):
    if 'Sub_bg' not in Sliders:
        remove_Sliders()
        Sub_bg_Var = StringVar()
        Sub_bg_Var.set("0")
        Sub_bg_Param = ttk.Entry(
            LFrame, style="primary.TEntry", width=4, textvariable=Sub_bg_Var)
        Sub_bg_Label = ttk.Label(LFrame, text="Thresh:", font=2)
        Sub_bg_Slider = Scale(LFrame, from_=0, to=1,resolution=0.01,
                             length=100, orient="horizontal", command=lambda value: UpdateSub_bg(value))
        Sub_bg_Btn = Button(LFrame, text="OK", height=1,
                           command=lambda: Sub_bg(Sub_bg_Var.get(), win_show))
        Sub_bg_Slider.grid(row=2, column=0)
        Sub_bg_Label.grid(row=1, column=0
        ,sticky=W,padx=(6.5,0))
        Sub_bg_Param.grid(row=2, column=1, pady=(0, 3))
        Sub_bg_Btn.grid(row=1, column=1, ipadx=6)
        Sliders['Sub_bg'] = Sub_bg_Slider
        Labels_Sliders['Sub_bg'] = Sub_bg_Label
        Params_Sliders['Sub_bg'] = [Sub_bg_Param, Sub_bg_Var]
        Btn_Sliders['Sub_bg'] = Sub_bg_Btn

# Directional filtering


def Direct(kernel,rotate, parent):
    if matplt:
        kernel = int(kernel)
        rotate = int(rotate)
        global board
        global toolbar
        kernels = np.zeros((kernel, kernel))
        kernels[:, 0] += -1
        kernels[:, -1] += 1
        if(rotate == 0):
            kernels_rotation = kernels
        else:
            kernels_rotation = np.rot90(kernels, k=rotate//90)
        Direct_img = cv2.filter2D(original_img, ddepth=-1, kernel=kernels_rotation)
        Direct_img = np.array(Direct_img, dtype=np.uint8)
        figure = plt.Figure(figsize=(7.83, 4.85))
        axs = figure.add_subplot(1, 1, 1)
        axs.imshow(Direct_img)
        remove_oldPlot()
        figure.set_facecolor('#98D6EA')
        board = FigureCanvasTkAgg(figure, master=parent)
        board.draw()
        board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(board, parent)
        toolbar.update()
        board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)


def UpdateDirect(kernel,rotate):
    Params_Sliders['Direct'][0].delete(0, END)
    Params_Sliders['Direct'][0].insert(0, kernel)
    Params_Sliders['Direct'][1].delete(0, END)
    Params_Sliders['Direct'][1].insert(0, rotate)
    # print(Params_Sliders['Gamma'][0].get())


def Direct_Slide(LFrame, win_show):
    if 'Direct' not in Sliders:
        remove_Sliders()
        Direct_K_Var = StringVar()
        Direct_R_Var = StringVar()
        Direct_K_Var.set("3")
        Direct_R_Var.set("0")
        #Kernel
        Direct_K_Param = ttk.Entry(
            LFrame, style="primary.TEntry", width=4, textvariable=Direct_K_Var)
        Direct_K_Label = ttk.Label(LFrame, text="Kernel:", font=0.5)
        Direct_K_Slider = Scale(LFrame, from_=3, to=9,
                             length=100, orient="horizontal", command=lambda value: UpdateDirect(value,Direct_R_Var.get()))
        #Rotation
        Direct_R_Param = ttk.Entry(
            LFrame, style="primary.TEntry", width=4, textvariable=Direct_R_Var)
        Direct_R_Label = ttk.Label(LFrame, text="Rotate kernel:", font=0.5)
        Direct_R_Slider = Scale(LFrame, from_=0, to=360,resolution=90,
                             length=100, orient="horizontal", command=lambda value: UpdateDirect(Direct_K_Var.get(),value))
        Direct_Btn = Button(LFrame, text="OK", height=1,
                           command=lambda: Direct(Direct_K_Var.get(),Direct_R_Var.get(), win_show))
        Direct_Btn.grid(row=1, column=1, ipadx=6)
        
        Direct_K_Label.grid(row=1, column=0, sticky=W, padx=(6.5, 0))
        Direct_K_Slider.grid(row=2, column=0)
        Direct_K_Param.grid(row=2, column=1)

        Direct_R_Label.grid(row=3, column=0, sticky=W, padx=(6.5, 0))
        Direct_R_Slider.grid(row=4, column=0)
        Direct_R_Param.grid(row=4, column=1,pady=(0,2))

        Sliders['Direct'] = [Direct_K_Slider,Direct_R_Slider]
        Labels_Sliders['Direct'] = [Direct_K_Label,Direct_R_Label]
        Params_Sliders['Direct'] = [Direct_K_Param,Direct_R_Param, Direct_K_Var,Direct_R_Var]
        Btn_Sliders['Direct'] = Direct_Btn


# Threshold Median


def Median(kernel,threshold, parent):
    if matplt:
        kernel = int(kernel)
        threshold = int(threshold)
        global board
        global toolbar
        if(len(original_img.shape) == 3):
            img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = original_img
        w, h = img_gray.shape
        img_new = ss.medfilt2d(img_gray, kernel_size=kernel)
        img_old = img_gray.astype('float')
        Median_img = img_new.astype('float')
        for i in range(1, w):
            for j in range(1, h):
                if math.fabs(img_old[i][j] - Median_img[i][j]) <= threshold:
                    Median_img[i][j] = img_old[i][j]
        Median_img = np.array(Median_img, dtype=np.uint8)
        figure = plt.Figure(figsize=(7.83, 4.85))
        axs = figure.add_subplot(1, 1, 1)
        axs.imshow(Median_img,cmap='gray')
        remove_oldPlot()
        figure.set_facecolor('#98D6EA')
        board = FigureCanvasTkAgg(figure, master=parent)
        board.draw()
        board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(board, parent)
        toolbar.update()
        board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)


def UpdateMedian(kernel,thresh):
    Params_Sliders['Median'][0].delete(0, END)
    Params_Sliders['Median'][0].insert(0, kernel)
    Params_Sliders['Median'][1].delete(0, END)
    Params_Sliders['Median'][1].insert(0, thresh)
    # print(Params_Sliders['Gamma'][0].get())


def Median_Slide(LFrame, win_show):
    if 'Median' not in Sliders:
        remove_Sliders()
        Median_K_Var = StringVar()
        Median_T_Var = StringVar()
        Median_K_Var.set("3")
        Median_T_Var.set("0")
        #Kernel
        Median_K_Param = ttk.Entry(
            LFrame, style="primary.TEntry", width=4, textvariable=Median_K_Var)
        Median_K_Label = ttk.Label(LFrame, text="Kernel:", font=0.5)
        Median_K_Slider = Scale(LFrame, from_=3.0, to=11.0,
                             length=100, orient="horizontal", command=lambda value: UpdateMedian(value,Median_T_Var.get()))
        #Threshold
        Median_T_Param = ttk.Entry(
            LFrame, style="primary.TEntry", width=4, textvariable=Median_T_Var)
        Median_T_Label = ttk.Label(LFrame, text="Threshold:", font=0.5)
        Median_T_Slider = Scale(LFrame, from_=0, to=255,
                             length=100, orient="horizontal", command=lambda value: UpdateMedian(Median_K_Var.get(),value))
        Median_Btn = Button(LFrame, text="OK", height=1,
                           command=lambda: Median(Median_K_Var.get(),Median_T_Var.get(), win_show))
        Median_Btn.grid(row=1, column=1, ipadx=6)
        
        Median_K_Label.grid(row=1, column=0, sticky=W, padx=(6.5, 0))
        Median_K_Slider.grid(row=2, column=0)
        Median_K_Param.grid(row=2, column=1)

        Median_T_Label.grid(row=3, column=0, sticky=W, padx=(6.5, 0))
        Median_T_Slider.grid(row=4, column=0)
        Median_T_Param.grid(row=4, column=1,pady=(0,2))

        Sliders['Median'] = [Median_K_Slider,Median_T_Slider]
        Labels_Sliders['Median'] = [Median_K_Label,Median_T_Label]
        Params_Sliders['Median'] = [Median_K_Param,Median_T_Param, Median_K_Var,Median_T_Var]
        Btn_Sliders['Median'] = Median_Btn


# Gaussian Blur


def Gaussian(kernel,sigma, parent):
    if matplt:
        kernel = int(kernel)
        sigma = int(sigma)
        global board
        global toolbar
        Gaussian_img = cv2.GaussianBlur(original_img, (kernel, kernel),
                               sigma, None, sigma, cv2.BORDER_DEFAULT)
        
        Gaussian_img = np.array(Gaussian_img, dtype=np.uint8)
        figure = plt.Figure(figsize=(7.83, 4.85))
        axs = figure.add_subplot(1, 1, 1)
        axs.imshow(Gaussian_img)
        remove_oldPlot()
        figure.set_facecolor('#98D6EA')
        board = FigureCanvasTkAgg(figure, master=parent)
        board.draw()
        board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(board, parent)
        toolbar.update()
        board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)


def UpdateGaussian(kernel,sigma):
    Params_Sliders['Gaussian'][0].delete(0, END)
    Params_Sliders['Gaussian'][0].insert(0, kernel)
    Params_Sliders['Gaussian'][1].delete(0, END)
    Params_Sliders['Gaussian'][1].insert(0, sigma)


def Gaussian_Slide(LFrame, win_show):
    if 'Gaussian' not in Sliders:
        remove_Sliders()
        Gaussian_K_Var = StringVar()
        Gaussian_S_Var = StringVar()
        Gaussian_K_Var.set("3")
        Gaussian_S_Var.set("0")
        #Kernel
        Gaussian_K_Param = ttk.Entry(
            LFrame, style="primary.TEntry", width=4, textvariable=Gaussian_K_Var)
        Gaussian_K_Label = ttk.Label(LFrame, text="Kernel:", font=0.5)
        Gaussian_K_Slider = Scale(LFrame, from_=3.0, to=11.0,
                             length=100, orient="horizontal", command=lambda value: UpdateGaussian(value,Gaussian_S_Var.get()))
        #Sigma
        Gaussian_S_Param = ttk.Entry(
            LFrame, style="primary.TEntry", width=4, textvariable=Gaussian_S_Var)
        Gaussian_S_Label = ttk.Label(LFrame, text="Sigma:", font=0.5)
        Gaussian_S_Slider = Scale(LFrame, from_=0, to=20,
                             length=100, orient="horizontal", command=lambda value: UpdateGaussian(Gaussian_K_Var.get(),value))
        Gaussian_Btn = Button(LFrame, text="OK", height=1,
                           command=lambda: Median(Gaussian_K_Var.get(),Gaussian_S_Var.get(), win_show))
        Gaussian_Btn.grid(row=1, column=1, ipadx=6)
        
        Gaussian_K_Label.grid(row=1, column=0, sticky=W, padx=(6.5, 0))
        Gaussian_K_Slider.grid(row=2, column=0)
        Gaussian_K_Param.grid(row=2, column=1)

        Gaussian_S_Label.grid(row=3, column=0, sticky=W, padx=(6.5, 0))
        Gaussian_S_Slider.grid(row=4, column=0)
        Gaussian_S_Param.grid(row=4, column=1,pady=(0,2))

        Sliders['Gaussian'] = [Gaussian_K_Slider,Gaussian_S_Slider]
        Labels_Sliders['Gaussian'] = [Gaussian_K_Label,Gaussian_S_Label]
        Params_Sliders['Gaussian'] = [Gaussian_K_Param,Gaussian_S_Param, Gaussian_K_Var,Gaussian_S_Var]
        Btn_Sliders['Gaussian'] = Gaussian_Btn


# ZeroCrossing of Lapcian


def ZeroCross(parent):
    if matplt:
        remove_Sliders()
        global board
        global toolbar
        if (len(original_img.shape) == 3):
            img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        else:
            img = original_img
        LoG = cv2.Laplacian(img, cv2.CV_16S)
        minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3, 3)))
        maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3, 3)))
        zeroCross_img = np.logical_or(np.logical_and(
            minLoG < 0, LoG > 0), np.logical_and(maxLoG > 0, LoG < 0))
        zeroCross_img = np.array(zeroCross_img*255, dtype=np.uint8)
        figure = plt.Figure(figsize=(7.83, 4.85))
        axs = figure.add_subplot(1, 1, 1)
        axs.imshow(zeroCross_img,cmap='gray')
        remove_oldPlot()
        figure.set_facecolor('#98D6EA')
        board = FigureCanvasTkAgg(figure, master=parent)
        board.draw()
        board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(board, parent)
        toolbar.update()
        board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)

# Canny


def Canny(high, low, parent):
    if matplt:
        high = int(high)
        low = int(low)
        global board
        global toolbar
        Canny_img = cv2.Canny(original_img, high, low)
        Canny_img = np.array(Canny_img, dtype=np.uint8)
        figure = plt.Figure(figsize=(7.83, 4.85))
        axs = figure.add_subplot(1, 1, 1)
        axs.imshow(Canny_img,cmap='gray')
        remove_oldPlot()
        figure.set_facecolor('#98D6EA')
        board = FigureCanvasTkAgg(figure, master=parent)
        board.draw()
        board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(board, parent)
        toolbar.update()
        board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)


def UpdateCanny(hight, low):
    Params_Sliders['Canny'][0].delete(0, END)
    Params_Sliders['Canny'][0].insert(0, hight)
    Params_Sliders['Canny'][1].delete(0, END)
    Params_Sliders['Canny'][1].insert(0, low)


def Canny_Slide(LFrame, win_show):
    if 'Gaussian' not in Sliders:
        remove_Sliders()
        Canny_H_Var = StringVar()
        Canny_L_Var = StringVar()
        Canny_H_Var.set("0")
        Canny_L_Var.set("0")
        #Hight Threshold
        Canny_H_Param = ttk.Entry(
            LFrame, style="primary.TEntry", width=4, textvariable=Canny_H_Var)
        Canny_H_Label = ttk.Label(LFrame, text="Hight threshold:", font=0.2)
        Canny_H_Slider = Scale(LFrame, from_=0.0, to=225.0,
                                  length=100, orient="horizontal", command=lambda value: UpdateCanny(value, Canny_L_Var.get()))
        #Low Threshold
        Canny_L_Param = ttk.Entry(
            LFrame, style="primary.TEntry", width=4, textvariable=Canny_L_Var)
        Canny_L_Label = ttk.Label(LFrame, text="Low threshold:", font=0.2)
        Canny_L_Slider = Scale(LFrame, from_=0, to=225,
                                  length=100, orient="horizontal", command=lambda value: UpdateCanny(Canny_H_Var.get(), value))
        Canny_Btn = Button(LFrame, text="OK", height=1,
                              command=lambda: Canny(Canny_H_Var.get(), Canny_L_Var.get(), win_show))
        Canny_Btn.grid(row=1, column=1, ipadx=6)

        Canny_H_Label.grid(row=1, column=0, sticky=W, padx=(6.5, 0))
        Canny_H_Slider.grid(row=2, column=0)
        Canny_H_Param.grid(row=2, column=1)

        Canny_L_Label.grid(row=3, column=0, sticky=W, padx=(6.5, 0))
        Canny_L_Slider.grid(row=4, column=0)
        Canny_L_Param.grid(row=4, column=1, pady=(0, 2))

        Sliders['Canny'] = [Canny_H_Slider, Canny_L_Slider]
        Labels_Sliders['Canny'] = [Canny_H_Label, Canny_L_Label]
        Params_Sliders['Canny'] = [Canny_H_Param,
                                      Canny_L_Param, Canny_H_Var, Canny_L_Var]
        Btn_Sliders['Canny'] = Canny_Btn

# Hough Transform


def HoughTransform(parent):
    if matplt:
        remove_Sliders()
        global board
        global toolbar
        gray = cv2.cvtColor(original_img.copy(), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                minLineLength=100, maxLineGap=250)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(original_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        Hough_img = np.array(original_img.copy(), dtype=np.uint8)
        figure = plt.Figure(figsize=(7.83, 4.85))
        axs = figure.add_subplot(1, 1, 1)
        axs.imshow(Hough_img, cmap='gray')
        remove_oldPlot()
        figure.set_facecolor('#98D6EA')
        board = FigureCanvasTkAgg(figure, master=parent)
        board.draw()
        board.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(board, parent)
        toolbar.update()
        board.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)

def remove_Sliders():
    global Sliders
    global Labels_Sliders
    global Params_Sliders
    global Btn_Sliders
    for key in Sliders.keys():
        if key not in ['Direct','Median','Gaussian','Canny']:
            Sliders[key].grid_remove()
            Labels_Sliders[key].grid_remove()
            Params_Sliders[key][0].grid_remove()
            Btn_Sliders[key].grid_remove()
        else:
            Sliders[key][0].grid_remove()
            Sliders[key][1].grid_remove()
            Labels_Sliders[key][0].grid_remove()
            Labels_Sliders[key][1].grid_remove()
            Params_Sliders[key][0].grid_remove()
            Params_Sliders[key][1].grid_remove()
            Btn_Sliders[key].grid_remove()

    Sliders = {}
    Labels_Sliders = {}
    Params_Sliders = {}
    Btn_Sliders = {}
