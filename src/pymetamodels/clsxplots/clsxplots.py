#!/usr/bin/python

import os, sys, traceback, datetime
import io
import gc
from multiprocessing import *
import multiprocessing as mpp
import inspect
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import ticker

from matplotlib.cbook import get_sample_data
#from matplotlib._png import read_png
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.patches import Circle
from PIL import Image

import pickle

import pymetamodels.clsxplots.cls_bars as clsbars
import pymetamodels.clsxplots.cls_2D_contour as cls_2D_contour
import pymetamodels.clsxplots.cls_3D_plot as cls_3D_plot

class xplot(object):

    """
        .. _xplot:

        **XPlots**

        :platform: Unix, Windows
        :synopsis: Generator of plots. Internal class
        :author: FLC

        :Dependences: Matplot

    """

    def __init__(self, out_path, layout = 0, eps = False, no_logo = False):

        self.lst = []
        self.out_path = out_path
        self._vlayout = layout #0 standard, 1 paper
        self._eps = eps # eps files True False
        self._no_logo = no_logo # no logo true false
        self._logo = r'ITA'
        self._logo_color = '#005189'
        self._logo_style = 'normal' #'italic', 'oblique'
        self._version = 0
        if layout == 0:
            self.font_size = 13
        else:
            self.font_size = 20

    def __del__(self):

        plt.close()
        gc.collect()

    def save_as_file(self, file_path, file_name):

        #Saves full object as a file

        filename = os.path.join(file_path, file_name)

        self.saved_timestap = datetime.datetime.now()
        self.saved_version = self._version

        with open(filename, 'wb') as output:

            cPickle.dump(self, output, cPickle.HIGHEST_PROTOCOL)

        return filename

    def load_as_file(self, file_path, file_name, inherit_properties = True):

        #Loads full object as a file

        filename = os.path.join(file_path, file_name)

        with open(filename, 'rb') as input:

            obj = cPickle.load(input)

        if inherit_properties:
            obj.out_path = self.out_path
            obj._vlayout = self._vlayout
            obj._eps = self._eps
            obj._no_logo = self._no_logo

        return obj

    def _layout(self):

        #self.font_size = 13
        plt.rcParams['font.size']            = r"%.1f" % self.font_size
        plt.rcParams['figure.figsize']       = '11.6929134, 8.26771654'    # A4
        plt.rcParams['figure.subplot.right'] = '0.975'
        plt.rcParams['savefig.dpi']          = '300'
        plt.rcParams['legend.fontsize']      = 'small'
        plt.rcParams['legend.fancybox']      = 'True'
        plt.rcParams['legend.numpoints']     = '1'
        plt.rcParams['axes.grid']            = 'True'
        plt.rcParams['grid.linestyle']       = ':'
        plt.rcParams['mathtext.default']     = 'regular'
        plt.rc('font', size=self.font_size)

        if self._vlayout == 1: # paper
            #self.font_size = 20

            plt.rcParams['savefig.dpi'] = '600'
            plt.rc('font', size=self.font_size)
            plt.rc('font', family='serif')
            plt.rc('font', serif='Times New Roman')
            #plt.rc('text', usetex=True)


    def mplot(self, ID, type_plot = 0, output_folder = None):

        ### check ID
        if len(str(ID)) == 0:

            raise ValueError('\n Plot ID not valid ' + ID + '\n')

        ### check first
        if len(self.lst) == 0:

            if type_plot == 11:
                self.lst.append(cls_2D_contour._cls_2D_contour(ID, type_plot, self, output_folder))
                print("Plot create ID other: " + ID)
            elif type_plot == 10:
                self.lst.append(clsbars._cls_bars(ID, type_plot, self, output_folder))
                print("Plot create ID other: " + ID)
            elif type_plot == 12:
                self.lst.append(cls_3D_plot._cls_3D_plot(ID, type_plot, self, output_folder))
                print("Plot create ID other: " + ID)
            else:
                self.lst.append(_xplot(ID, type_plot, self, output_folder))
                print("Plot create ID: " + ID)
            return self.lst[-1]

        else:
            ### give first existent
            for ll in self.lst:

                if ll.ID == str(ID):

                    return ll

            ### create new
            if type_plot == 11:
                self.lst.append(cls_2D_contour._cls_2D_contour(ID, type_plot, self, output_folder))
            elif type_plot == 10:
                self.lst.append(clsbars._cls_bars(ID, type_plot, self, output_folder))
            elif type_plot == 12:
                self.lst.append(cls_3D_plot._cls_3D_plot(ID, type_plot, self, output_folder))
            else:
                self.lst.append(_xplot(ID, type_plot, self, output_folder))
            return self.lst[-1]

    def make_pages(self, run_eps = False, run_ppt = False, empty = False, multi = False, run_img = False):

        ### print all plots

        self._layout()

        for ll in self.lst:

            if multi:
                run_function_as_another_process(ll.make_page, run_eps, run_ppt, run_img)
            else:
                if ll.type_plot == 11:
                    ll.make_page(run_eps, run_ppt, run_img)
                elif ll.type_plot == 10:
                    ll.make_page(run_eps, run_ppt)
                elif ll.type_plot == 12:
                    ll.make_page(run_eps, run_ppt)
                else:
                    ll.make_page(run_eps, run_ppt, run_img)

        if empty: self.lst = []

class _xplot(object):

    def __init__(self, ID, type_plot, parent, output_folder):

        self.parent = parent
        self.ID = str(ID)
        self.type_plot = type_plot

        self.consider_zeros = False

        self.title = "empty"
        self.listtplot = []
        self.listtplot_log = 0
        self.limits = ['', '' , '' , '', 'line', 'best', '', '']
        self.sci_axis = True
        self.auto_max_min = [False, False, 7., 7.]

        self.annotate = False
        self.anotates_series = []
        self.annotate_int = False

        self.extra_text_data=[[0,0,""]]
        self.extra_text = 0

        self.extra_image = 0
        self.extra_add_image = []

        self.xlabel = "empty"
        self.ylabel = "empty"
        self.y2label = "empty"
        self.two_legends = False

        self._x_axes_off = False
        self._y_axes_off = False

        self.grid = True
        self.gridy2 = False

        self._error_bars = False
        self.avoid_scintific_x = True
        self.avoid_scintific_y = True

        self._facecolors = None

        self.config = self._default()

        self.output_folder = output_folder

    ##### CONFIG

    def x_min_max(self):

        return self.limits[0], self.limits[1]

    def y_min_max(self):

        return self.limits[2], self.limits[3]

    def x_max(self, value):

        self.limits[1] = value

    def x_min(self, value):

        self.limits[0] = value

    def y_min(self, value):

        self.limits[2] = value

    def y_max(self, value):

        self.limits[3] = value

    def y2_min(self, value):

        self.limits[6] = value

    def y2_max(self, value):

        self.limits[7] = value

    def max_min_auto(self, auto_x = True , auto_y = True, div_x = 7., div_y = 7.):

        self.auto_max_min = [auto_x, auto_y, div_x, div_y]

    def x_axes_off(self, value = False):

        self._x_axes_off = value

    def y_axes_off(self, value = False):

        self._y_axes_off = value

    def scientific_axis(self, value):

        self.sci_axis = values

    def legend_position(self, legend):

        ### check valids
        self.limits[5] = legend

    def general_marker(self, marker):

        ### check valids
        self.limits[4] = marker

    def extra_text_add(self, pos_x, pos_y, txt):

        if self.extra_text == 0:
            self.extra_text_data=[[0,0,""]]

        self.extra_text_data.append([0,0,""])

        self.extra_text_data[self.extra_text][0] = pos_x
        self.extra_text_data[self.extra_text][1] = pos_y
        self.extra_text_data[self.extra_text][2] = txt

        self.extra_text = self.extra_text + 1

    def extra_add_image(self, pos_x, pos_y, image_file_png, zoom = 1.):
        ### Note: Problems in eps images

        if self.extra_image == 0:
            self.extra_add_image=[[0,0,0,""]]

        self.extra_add_image.append([0,0,0,""])

        self.extra_add_image[self.extra_image][0] = pos_x
        self.extra_add_image[self.extra_image][1] = pos_y
        self.extra_add_image[self.extra_image][2] = zoom
        self.extra_add_image[self.extra_image][3] = image_file_png

        self.extra_image = self.extra_image + 1

    def empty_facecolors(self, face_color = 'none'):

        self._facecolors = face_color

    def append_data(self, x , y , legend, txt="", linestyle="", marker="", color = "", \
                    annotate_serie = "nop", consider_zeros = False, y2 = None, error_y = None, error_x = None):
        ### Appends the signal

        self.consider_zeros = consider_zeros

        ### Case of single nambers
        def check(_x):
            if type(_x) != type(np.zeros(2)):
                _xx = np.asarray([_x])
            else:
                _xx = _x
            return _xx
        xx = check(x)
        yy = check(y)
        if y2 is not None:
            yy2 = check(y2)
        else:
            yy2 = None
        if error_y is not None:
            yerror_y = check(error_y)
            self._error_bars = True
        else:
            yerror_y = None
        if error_x is not None:
            xerror_x = check(error_x)
            self._error_bars = True
        else:
            xerror_x = None

        ### Check length equal
        if len(xx) != len(yy):
            raise ValueError("\n Plot data different length " + self.ID + " \n ")
            return False

        ### No legends
        if legend == "":
            legend =  "_nolegend_"

        self.listtplot.append([str(legend),xx,yy,linestyle, marker, color, txt, yy2, yerror_y, xerror_x])

        ### Annotate
        if annotate_serie != "nop":

            if type(annotate_serie) != type(np.zeros(2)):
                zzz = np.asarray([annotate_serie])
            else:
                zzz = annotate_serie

            self.anotates_series.append(zzz)
        else:
            self.anotates_series.append(np.asarray([]))
            pass

        return True

    def annotate_data(self, value, just_integers = False, align = "right" ):

        self.annotate = value
        self.annotate_int = just_integers
        self.annotate_align = align

    ##### PLOT

    def make_page(self, run_eps = True, run_ppt = True, run_img = False):

        if self.type_plot == 0:
            ### Normal plots

            self._0_make_page(run_eps, run_ppt, run_img)

        else:

            self._0_make_page(run_eps, run_ppt, run_img)

    def _0_make_page(self, run_eps = True, run_ppt = True, run_img = False):

        ### Creates the plot
        plt.ioff()

        plt.rcParams['figure.subplot.top'] = '0.90'

        if self.parent._vlayout == 1: # paper
            plt.rcParams['figure.subplot.right'] = '0.95'
            #plt.rcParams['figure.subplot.left'] = '0.17'
            if self.parent.font_size <= 17:
                plt.rcParams['figure.subplot.top'] = '0.90'
            else:
                plt.rcParams['figure.subplot.top'] = '0.95'
                plt.rcParams['figure.subplot.bottom'] = '0.12'

            fig = plt.figure(figsize = [8.3,5.85])  #size in inches
            plot1 = self.append_axes(cols = 1, rows = 1, wspace = 0.1, hspace = 0.05)
        else:
            fig = plt.figure(figsize = [8.3,5.85])  #size in inches
            plot1 = self.append_axes(cols = 1, rows = 1, wspace = 0.1, hspace = 0.05)

        ax = plt.subplot(111)
        ax_y2 = None
        axes_add = []
        axes_add_no_legend = []

        plt.suptitle(self.title, fontsize = 11, ha = 'left', x = plt.rcParams['figure.subplot.left'] )

        if self.limits[4] == "line":
            self.limits[4] = ""
            line_style = "solid"
        else:
            line_style = ""

        if len(self.listtplot)>0:

            max_x, min_x = "", ""
            max_y, min_y = "", ""

            for ixi in range(0,len(self.listtplot)):

                if self.listtplot[ixi][0] == "_nolegend_":
                    marker=""
                    line_style = "solid"
                else:
                    marker = self.limits[4]

                ### clean array
                if self.listtplot[ixi][7] is not None:
                    # with y2 axis
                    aux1, aux3 = self.no_nan(self.listtplot[ixi][1],self.listtplot[ixi][7])
                    aux2 = None
                    if ax_y2 is None:
                        ax_y2 = ax.twinx()
                        plt.subplots_adjust(right=0.9)
                else:
                    aux1, aux2 = self.no_nan(self.listtplot[ixi][1], self.listtplot[ixi][2])
                    aux3 = None
                    if len(aux1) == 0:
                        print("\n Plot empty data " + self.ID + " \n ")
                        raise ValueError("\n Plot empty data " + self.ID + " \n ")

                ### line styles
                if self.listtplot[ixi][3] != "":
                    line_style_f = self.listtplot[ixi][3]
                else:
                    line_style_f = line_style

                ### marker styles
                if self.listtplot[ixi][5] != "":
                    marker_f = self.listtplot[ixi][4]
                else:
                    marker_f = self.limits[4]

                ### color
                if self.listtplot[ixi][5] != "" or self.listtplot[ixi][5] != "_same_as_previous_":
                    coll = self.listtplot[ixi][5]
                else:
                    coll = ""

                ### text
                if self.listtplot[ixi][6] != "":
                    txt = self.listtplot[ixi][6]
                else:
                    txt = ""

                ### plot
                if self.listtplot_log==1:
                    if txt == "":
                        if aux3 is not None:
                            axx = ax_y2.loglog(aux1, aux3, label = self.listtplot[ixi][0], linestyle=line_style_f, marker = marker_f)
                        else:
                            axx = ax.loglog(aux1, aux2, label = self.listtplot[ixi][0], linestyle=line_style_f, marker = marker_f)
                    else:
                        if aux3 is not None:
                            axx = ax_y2.loglog(aux1, aux3, txt, label = self.listtplot[ixi][0])
                        else:
                            axx = ax.loglog(aux1, aux2, txt, label = self.listtplot[ixi][0])
                    if coll != "": axx[0].set_color(coll)
                    axes_add = axes_add + axx
                    if self.listtplot[ixi][0] != "_nolegend_": axes_add_no_legend = axes_add_no_legend + axx
                elif self.listtplot_log==0:
                    if txt == "":
                        if aux3 is not None:
                            axx = ax_y2.plot(aux1, aux3, label = self.listtplot[ixi][0], linestyle=line_style_f, marker = marker_f)
                        else:
                            axx = ax.plot(aux1, aux2, label = self.listtplot[ixi][0], linestyle=line_style_f, marker = marker_f)
                    else:
                        if aux3 is not None:
                            axx = ax_y2.plot(aux1, aux3, txt, label = self.listtplot[ixi][0])
                        else:
                            axx = ax.plot(aux1, aux2, txt, label = self.listtplot[ixi][0])
                    if coll != "": axx[0].set_color(coll)
                    axes_add = axes_add + axx
                    if self.listtplot[ixi][0] != "_nolegend_": axes_add_no_legend = axes_add_no_legend + axx
                elif self.listtplot_log==2:
                    if txt == "":
                        if aux3 is not None:
                            axx = ax_y2.semilogx(aux1, aux3, label = self.listtplot[ixi][0], linestyle=line_style_f, marker = marker_f)
                        else:
                            axx = ax.semilogx(aux1, aux2, label = self.listtplot[ixi][0], linestyle=line_style_f, marker = marker_f)
                    else:
                        if aux3 is not None:
                            axx = ax_y2.semilogx(aux1, aux3, txt, label = self.listtplot[ixi][0])
                        else:
                            axx = ax.semilogx(aux1, aux2, txt, label = self.listtplot[ixi][0])
                    if coll != "": axx[0].set_color(coll)
                    axes_add = axes_add + axx
                    if self.listtplot[ixi][0] != "_nolegend_": axes_add_no_legend = axes_add_no_legend + axx
                elif self.listtplot_log==3:
                    if txt == "":
                        if aux3 is not None:
                            axx = ax_y2.semilogy(aux1, aux3, label = self.listtplot[ixi][0], linestyle=line_style_f, marker = marker_f)
                        else:
                            axx = ax.semilogy(aux1, aux2, label = self.listtplot[ixi][0], linestyle=line_style_f, marker = marker_f)
                    else:
                        if aux3 is not None:
                            axx = ax_y2.semilogy(aux1, aux3, txt, label = self.listtplot[ixi][0])
                        else:
                            axx = ax.semilogy(aux1, aux2, txt, label = self.listtplot[ixi][0])
                    if coll != "": axx[0].set_color(coll)
                    axes_add = axes_add + axx
                    if self.listtplot[ixi][0] != "_nolegend_": axes_add_no_legend = axes_add_no_legend + axx

                ### no legend cases
                if self.listtplot[ixi][5] == "_same_as_previous_":
                    if self.listtplot[ixi][0] == "_nolegend_":
                        axx[0].set_color(colorr)
                    else:
                        pass
                colorr = axx[0].get_color()

                ### facecolors
                if self._facecolors is not None:
                    #axx.set_markerfacecolor(self._facecolors)
                    pass

                ### error bars
                if self._error_bars == True:

                    if aux3 is not None:
                        if self.listtplot[ixi][8] is not None:
                            noth,eaux1 = self.no_nan(self.listtplot[ixi][1],self.listtplot[ixi][8])
                            ax_y2.errorbar(aux1, aux3, yerr=eaux1 , fmt=None, ecolor=axx[0].get_color())

                        if self.listtplot[ixi][9] is not None:
                            noth,eaux2 = self.no_nan(self.listtplot[ixi][1],self.listtplot[ixi][9])
                            ax_y2.errorbar(aux1, aux3, xerr=eaux2, fmt=None, ecolor=axx[0].get_color())
                    else:
                        if self.listtplot[ixi][8] is not None:
                            noth,eaux1 = self.no_nan(self.listtplot[ixi][1],self.listtplot[ixi][8])
                            ax.errorbar(aux1, aux2, yerr=eaux1 , fmt=None, ecolor=axx[0].get_color())

                        if self.listtplot[ixi][9] is not None:
                            noth,eaux2 = self.no_nan(self.listtplot[ixi][1],self.listtplot[ixi][9])
                            ax.errorbar(aux1, aux2, txt, xerr=eaux2, fmt=None, ecolor=axx[0].get_color())

                ### max_min
                if self.auto_max_min[0]:
                    if max_x == "":
                        max_x, min_x = np.max(aux1), np.min(aux1)
                    else:
                        if max_x < np.max(aux1): max_x = np.max(aux1)
                        if min_x > np.min(aux1): min_x = np.min(aux1)

                if self.auto_max_min[1] and aux2 is not None:
                    if max_y == "":
                        max_y, min_y = np.max(aux2), np.min(aux2)
                    else:
                        if max_y < np.max(aux2): max_y = np.max(aux2)
                        if min_y > np.min(aux2): min_y = np.min(aux2)

                ### axes off
                if self._y_axes_off:
                    ax.axes.get_yaxis().set_visible(False)
                if self._x_axes_off:
                    ax.axes.get_xaxis().set_visible(False)

                ### anotate

                if self.annotate and \
                        len(self.listtplot[ixi][1]) == len(self.anotates_series[ixi]) and \
                        len(self.listtplot[ixi][2]) == len(self.anotates_series[ixi]) and aux2 is not None:

                    #aux1, zzz = self.no_nan(self.listtplot[ixi][1], self.anotates_series[ixi])
                    #aux2, zzz1 = self.no_nan(self.listtplot[ixi][2], self.anotates_series[ixi])
                    aux1, aux2, zzz = self.no_nan3(self.listtplot[ixi][1], self.listtplot[ixi][2], self.anotates_series[ixi])

                    #if len(zzz) == len(zzz1) and len(zzz) == len(aux1) and len(zzz) == len(aux2):
                    if len(zzz) == len(aux1) and len(zzz) == len(aux2):

                        if self.annotate_int == False:

                            for X, Y, Z in zip(aux1, aux2, zzz):
                                # Annotate the points 5 _points_ above and to the left of the vertex
                                ax.annotate('{}'.format(Z), xy=(X,Y), xytext=(-5, 5), ha=self.annotate_align,
                                            textcoords='offset points')
                        else:

                            iintegers = zzz % 1 < 1e-6
                            for (X, Y, Z) in zip(aux1[iintegers], aux2[iintegers], zzz[iintegers]):
                                ax.annotate('{:.0f}'.format(Z), xy=(X,Y), xytext=(-10, 10), ha=self.annotate_align,
                                            textcoords='offset points',
                                            arrowprops=dict(arrowstyle='->', shrinkA=0))

            ### setup limits
            if self.sci_axis:
                xfm = ax.xaxis.get_major_formatter()
                if self.listtplot_log==0: xfm.set_powerlimits([ -3, 3])
                yfm = ax.yaxis.get_major_formatter()
                if self.listtplot_log==0: yfm.set_powerlimits([ -3, 3])

            ### check that is not text "" isntead of only type the self.limits[0]
            if type(self.limits[0]) != type(''): ax.set_xlim(xmin=self.limits[0])
            if type(self.limits[1]) != type(''): ax.set_xlim(xmax=self.limits[1])
            if type(self.limits[2]) != type(''): ax.set_ylim(ymin=self.limits[2])
            if type(self.limits[3]) != type(''): ax.set_ylim(ymax=self.limits[3])
            if ax_y2 is not None:
                if type(self.limits[6]) != type(''): ax_y2.set_ylim(ymin=self.limits[6])
                if type(self.limits[7]) != type(''): ax_y2.set_ylim(ymax=self.limits[7])

            ### auto max min
            if self.auto_max_min[0]:
                max_xx = max_x + ((max_x-min_x)/self.auto_max_min[2])
                min_xx = min_x - ((max_x-min_x)/self.auto_max_min[2])
                ax.set_xlim(xmin = min_xx)
                ax.set_xlim(xmax = max_xx)
            else:
                min_xx, max_xx = ax.get_xlim()
                if type(self.limits[0]) != type(''): min_xx = float(self.limits[0])
                if type(self.limits[1]) != type(''): max_xx = float(self.limits[1])

            if self.auto_max_min[1]:
                max_yy = max_y + ((max_y-min_y)/self.auto_max_min[3])
                min_yy = min_y - ((max_y-min_y)/self.auto_max_min[3])
                ax.set_ylim(ymin = min_yy)
                ax.set_ylim(ymax = max_yy)
            else:
                min_yy, max_yy = ax.get_ylim()
                if type(self.limits[2]) != type(''): min_yy = float(self.limits[2])
                if type(self.limits[3]) != type(''): max_yy = float(self.limits[3])

            ### avoid scintific notation
            if self.avoid_scintific_x:
                if (self.listtplot_log == 0 or self.listtplot_log == 3):
                    #ax.get_xaxis().get_major_formatter().set_useOffset(False)
                    ax.get_xaxis().get_major_formatter().set_scientific(False)
                else:
                    #ax.xaxis.set_major_locator(plt.AutoLocator())
                    #ax.xaxis.set_major_formatter(plt.ScalarFormatter())
                    minorLocator = matplotlib.ticker.MultipleLocator(np.abs((max_xx-min_xx)/10))
                    ax.xaxis.set_minor_locator(minorLocator)
                    ax.xaxis.set_minor_formatter(plt.ScalarFormatter())
                    ax.xaxis.set_major_formatter(plt.NullFormatter())

            if self.avoid_scintific_y:
                if (self.listtplot_log == 0 or self.listtplot_log == 2):
                    #ax.get_yaxis().get_major_formatter().set_useOffset(False)
                    ax.get_yaxis().get_major_formatter().set_scientific(False)
                else:
                    #ax.yaxis.set_major_locator(plt.AutoLocator())
                    #ax.yaxis.set_major_formatter(plt.ScalarFormatter())
                    minorLocator = matplotlib.ticker.MultipleLocator(np.abs((max_yy-min_yy)/10))
                    ax.yaxis.set_minor_locator(minorLocator)
                    ax.yaxis.set_minor_formatter(plt.ScalarFormatter())
                    ax.yaxis.set_major_formatter(plt.NullFormatter())

            ### labels
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)
            ax.grid(self.grid)

            if ax_y2 is not None:
                ax_y2.set_ylabel(self.y2label)
                if self.two_legends:
                    ax.legend(loc = self.limits[5])
                    ax_y2.legend(loc = self.limits[5])
                else:
                    labs = [l.get_label() for l in axes_add_no_legend]
                    ax.legend(axes_add_no_legend, labs, loc=0)
                ax_y2.grid(self.gridy2)
            else:
                ax.legend(loc = self.limits[5])

            if self.extra_text > 0:
                for cccc in range(0,len(self.extra_text_data)-1):
                    xrt = self.extra_text_data[cccc]
                    pos_x, pos_y, txt = xrt[0], xrt[1], xrt[2]
                    #print cccc, pos_x, pos_y, txt
                    #plt.text(pos_x, pos_y, txt, horizontalalignment='left', transform=ax.transAxes,zorder=10)
                    plt.annotate(txt,xy=(pos_x,pos_y),xytext=(pos_x,pos_y),xycoords="axes fraction",textcoords="axes fraction", zorder=10)

            if self.extra_image > 0:
                for cccc in range(0,len(self.extra_add_image)-1):
                    xrt = self.extra_add_image[cccc]
                    pos_x, pos_y, _zoom, txt = xrt[0], xrt[1], xrt[2], xrt[3]

                    im = Image.open(txt)
                    im_w = im.size[0]
                    im_h = im.size[1]
                    img = im.resize((int(im_w*_zoom), int(im_h*_zoom)), Image.ANTIALIAS)
                    im_w = im.size[0]
                    im_h = im.size[1]

                    _w = (float(plt.rcParams['savefig.dpi'])*8.3/1) *  pos_x
                    _h = (float(plt.rcParams['savefig.dpi'])*5.85/1) * pos_y
                    #print cccc, pos_x, pos_y, _zoom, _w, _h

                    #fig.figimage(img, _w, _h,zorder=1)

                    #fn = get_sample_data("grace_hopper.png", asfileobj=False)
                    #arr_lena = read_png(fn)

                    imagebox = OffsetImage(img, zoom=1.0)

                    _w = (float(plt.rcParams['savefig.dpi'])*8.3/1)
                    _h = (float(plt.rcParams['savefig.dpi'])*5.85/1)
                    _w = im_w / (2.*_w)
                    _h = im_h / (2.*_h)
                    _w, _h = 0., 0.

                    ab = AnnotationBbox(imagebox, xy=(pos_x+_w,pos_y+_h),
                                        xybox=(pos_x+_w,pos_y+_h),
                                        xycoords="axes fraction",
                                        boxcoords="axes fraction",
                                        pad=0.5, frameon=False)

                    ax.add_artist(ab)

            # add text
            if self.config["extra_text"] > 0:
                for cccc in range(0,len(self.config["extra_text_data"])-1):
                    xrt = self.config["extra_text_data"][cccc]
                    pos_x, pos_y, txt = xrt[0], xrt[1], xrt[2]
                    #print cccc, pos_x, pos_y, txt
                    #plt.text(pos_x, pos_y, txt, horizontalalignment='left', transform=ax.transAxes,zorder=10)
                    plt.annotate(txt,xy=(pos_x,pos_y),xytext=(pos_x,pos_y),xycoords="axes fraction",textcoords="axes fraction", zorder=10)

            # add arrow
            if self.config["extra_arrow"] > 0:
                for cccc in range(0,len(self.config["extra_arrow_data"])-1):
                    xrt = self.config["extra_arrow_data"][cccc]
                    pos_x, pos_y, to_pos_x, to_pos_y, _width, _head = xrt[0], xrt[1], xrt[2], xrt[3], xrt[4], xrt[5]

                    plt.annotate("",xy=(pos_x,pos_y),xytext=(to_pos_x, to_pos_y), \
                                xycoords="axes fraction",textcoords="axes fraction", zorder=10, \
                                arrowprops=dict(arrowstyle=_head,linewidth=_width))


            ### draw
            route = r'xplot_' + str(self.ID).replace(" ", "_") + '.jpg'
            self.savefig(route)

            if run_img:
                self.savefig(route, run_img = run_img)

            if self.parent._eps or run_eps:
                _was_dpi = plt.rcParams['savefig.dpi']
                plt.rcParams['savefig.dpi'] = '100'
                route = r'xplot_' + str(self.ID).replace(" ", "_") + '.eps'
                self.savefig(route)
                route = r'xplot_' + str(self.ID).replace(" ", "_") + '.pdf'
                self.savefig(route)
                plt.rcParams['savefig.dpi'] = _was_dpi

            plt.ion()
            #print "Saving plot: " + 'plot_alot_' + self.ID

            return 'xplot_alot_' + self.ID

        else:

            return ""

    def _default(self):

        config = {}

        config["extra_text"] = 0
        config["extra_text_data"] = []
        config["extra_arrow"] = 0
        config["extra_arrow_data"] = []

        return config

    def extra_text_add(self, pos_x, pos_y, txt):

        if self.config["extra_text"] == 0:
            self.config["extra_text_data"]=[[0,0,""]]

        self.config["extra_text_data"].append([0,0,""])

        self.config["extra_text_data"][self.config["extra_text"]][0] = pos_x
        self.config["extra_text_data"][self.config["extra_text"]][1] = pos_y
        self.config["extra_text_data"][self.config["extra_text"]][2] = txt

        self.config["extra_text"] = self.config["extra_text"] + 1

    def extra_arrow_add(self, pos_x, pos_y, to_pos_x, to_pos_y, width = None, head_style = "->"):

        if self.config["extra_arrow"] == 0:
            self.config["extra_arrow_data"]=[[0,0,0,0,width,head_style]]

        self.config["extra_arrow_data"].append([0,0,0,0,width,head_style])

        self.config["extra_arrow_data"][self.config["extra_arrow"]][0] = pos_x
        self.config["extra_arrow_data"][self.config["extra_arrow"]][1] = pos_y
        self.config["extra_arrow_data"][self.config["extra_arrow"]][2] = to_pos_x
        self.config["extra_arrow_data"][self.config["extra_arrow"]][3] = to_pos_y
        self.config["extra_arrow_data"][self.config["extra_arrow"]][4] = width
        self.config["extra_arrow_data"][self.config["extra_arrow"]][5] = head_style

        self.config["extra_arrow"] = self.config["extra_arrow"] + 1

    def savefig(self, inName,w=11.6929134,h=8.26771654, run_img = False):

        if self.output_folder is None:
            route = os.path.join(self.parent.out_path,inName)
        else:
            route = os.path.join(self.output_folder,inName)

        plt.ioff()
        plt.draw()

        if not self.parent._no_logo:
            plt.text(0.995, 0.005 , self.parent._logo,
                     style=self.parent._logo_style, weight='bold', size='small',
                     color=self.parent._logo_color, horizontalalignment='right',
                     transform = plt.gca().transAxes)

        self.buf = None
        if run_img:
            self.buf = io.BytesIO()
            plt.savefig(self.buf, format='png')
            self.buf.seek(0)
        else:
            plt.savefig(route)
        if not self.parent._no_logo:
            del(plt.gca().texts[-1])

        plt.ion()

    def close_fig(self):

        plt.close()
        gc.collect()

    def append_axes(self, rows = 1, cols=1, h = 1, hspace = 0.05, wspace = 0.1):

        #get the setting from rcParams
        rcleft = plt.rcParams['figure.subplot.left']
        rcright = plt.rcParams['figure.subplot.right']
        rcwidth = rcright - rcleft
        rctop = plt.rcParams['figure.subplot.top']
        rcbottom = plt.rcParams['figure.subplot.bottom']
        rcheight = rctop - rcbottom

        #get the bottom y coordinate of each axes in the current figure (gcf() creates a figure if none)
        axbottoms = [ax.get_position().get_points()[0][1] for ax in plt.gcf().get_axes()]

        #top of new axes is below present axes
        if axbottoms: top = np.min(axbottoms) - hspace
        else:         top = rctop

        bottom = np.max((top - h, rcbottom))
        if ( top - (rows - 1) * hspace ) <= bottom:
            raise(ValueError, 'no space available below lowest axes')

        #create lists with coordinates and dimensions for new axes
        height = ((top - bottom) - (rows - 1) * hspace)/rows
        bottoms = [(bottom + i * (height + hspace)) for i in range(rows)]
        width = (rcwidth - (cols-1) * wspace)/cols
        lefts = [(rcleft + i * (width + wspace)) for i in range(cols)]

        #return a list of axes instances
        return [plt.axes([lefts[j],bottoms[i], width, height]) for i in range(rows-1,-1,-1) for j in range(cols) ]

    def no_nan(self, array1, array2):
        ### clean arrays

        arr1 = array1.copy()
        arr2 = array2.copy()

        index = np.where(np.isnan(arr1))[0]
        arr1 = np.delete(arr1, index)
        arr2 = np.delete(arr2, index)

        index = np.where(np.isnan(arr2))[0]
        arr1 = np.delete(arr1, index)
        arr2 = np.delete(arr2, index)

        index = np.where(np.isinf(arr1))[0]
        arr1 = np.delete(arr1, index)
        arr2 = np.delete(arr2, index)

        index = np.where(np.isinf(arr2))[0]
        arr1 = np.delete(arr1, index)
        arr2 = np.delete(arr2, index)

        if not self.consider_zeros:
            index = np.where(arr1 == 0.)[0]
            arr1 = np.delete(arr1, index)
            arr2 = np.delete(arr2, index)

            index = np.where(arr2 == 0.)[0]
            arr1 = np.delete(arr1, index)
            arr2 = np.delete(arr2, index)

        return arr1, arr2

    def no_nan3(self, array1, array2, array3):
        ### clean arrays

        arr1 = array1.copy()
        arr2 = array2.copy()
        arr3 = array3.copy()

        index = np.where(np.isnan(arr1))[0]
        arr1 = np.delete(arr1, index)
        arr2 = np.delete(arr2, index)
        arr3 = np.delete(arr3, index)

        index = np.where(np.isnan(arr2))[0]
        arr1 = np.delete(arr1, index)
        arr2 = np.delete(arr2, index)
        arr3 = np.delete(arr3, index)

        index = np.where(np.isinf(arr1))[0]
        arr1 = np.delete(arr1, index)
        arr2 = np.delete(arr2, index)
        arr3 = np.delete(arr3, index)

        index = np.where(np.isinf(arr2))[0]
        arr1 = np.delete(arr1, index)
        arr2 = np.delete(arr2, index)
        arr3 = np.delete(arr3, index)

        if not self.consider_zeros:
            index = np.where(arr1 == 0.)[0]
            arr1 = np.delete(arr1, index)
            arr2 = np.delete(arr2, index)
            arr3 = np.delete(arr3, index)

            index = np.where(arr2 == 0.)[0]
            arr1 = np.delete(arr1, index)
            arr2 = np.delete(arr2, index)
            arr3 = np.delete(arr3, index)

        return arr1, arr2, arr3

##############################################
### Run as separate process

def _wrap_fun1(fun, *args, **kwds):
    ### wrap function for the function that is call
    try:
        out = fun(*args, **kwds)
    except Exception:
        ex_type, ex_value, tb = sys.exc_info()
        error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
        out = None
    else:
        error = None

def _wrap_fun(q, fun, *args, **kwds):
    ### wrap function for the function that is call
    try:
        out = fun(*args, **kwds)
    except Exception:
        ex_type, ex_value, tb = sys.exc_info()
        error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
        out = None
    else:
        error = None

    q.put((out, error))

def _wrap_cls(q, cls, fun, *args, **kwds):
    ### wrap function for the calss method that is call
    try:
        methodToCall = getattr(cls, fun)
        out =  methodToCall(*args, **kwds)
    except Exception:
        ex_type, ex_value, tb = sys.exc_info()
        error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
        out = None
    else:
        error = None

    q.put((out, error))

def _apply_async_with_callback_cls(cls, fun_name, *args, **kwds):
    ### asyncronous call for clas method, output is return trough a queue
    q = mpp.Queue()

    args =(q, cls, fun_name,) + args

    p = Process(target =_wrap_cls, args=args, kwargs=kwds)

    p.start()
    out, error = q.get()
    p.join()

    if error:
        ex_type, ex_value, tb_str = error
        message = '%s (in subprocess cls)\n%s' % (ex_value.message, tb_str)
        lib_error(txt=message, out=True)

    return out

def _apply_async_with_callback_fun( fun_name, *args, **kwds):
    ### asyncronous call for function, output is return trough a queue
    q = mpp.Queue()

    args =(q, fun_name,) + args

    p = Process(target =_wrap_fun, args=args, kwargs=kwds)

    p.start()
    out, error = q.get()
    p.join()

    if error:
        ex_type, ex_value, tb_str = error
        message = '%s (in subprocess fun)\n%s' % (ex_value.message, tb_str)
        #lib_error(txt=message, out=True)
        print(message)

    return out

def __apply_async_with_callback_cls_async(cls, fun_name, *args, **kwds):

    result_list = []
    def log_result(result):
        # This is called whenever foo_pool(i) returns a result.
        # result_list is modified only by the main process, not the pool workers.
        result_list.append(result)

    args =(cls, fun_name,) + args

    pool = mpp.Pool()

    pool.apply_async( _wrap_fun, args , kwds, callback = log_result)

    pool.close()
    pool.join()

    return result_list[0]

def __apply_async_with_callback_fun_async( fun_name, *args, **kwds):

    result_list = []
    def log_result(result):
        # This is called whenever foo_pool(i) returns a result.
        # result_list is modified only by the main process, not the pool workers.
        result_list.append(result)

    args =(fun_name,) + args

    pool = mpp.Pool()

    pool.apply_async( _wrap_fun1, args , kwds, callback = log_result)

    pool.close()
    pool.join()

    return result_list[0]

def run_function_as_another_process(fun_name, *args, **kwds):

    """

    **Synopsis:**
        * Runs function in separete process and waits until finish

    :Dependences: multiprocessing

    .. _run_function_as_another_process:

    **Args:**
        * fun_name: can be a class function cls.fun(), or a function fun()
        * args: arguments
        * kwds: optional arguments

    **Optional parameters:**
        * None

    **Returns:**
        * Function results

    **Error:**

        * Print error info and terminate global process

    .. note::

        * Use multiprocessing, works on win and unix
        * Runs as another PID
        * Can be use for memory problems leaks
        * Do not work with functions declare inside functions, they can not be pickle

    """

    try:
        cls_obj = fun_name.im_self
        raw_fun = True
    except:
        raw_fun = False

    if not inspect.isfunction(fun_name) and raw_fun:
        ### Case is a class method cls.fun()

        cls_obj = fun_name.im_self
        name = fun_name.__name__

        return _apply_async_with_callback_cls(cls_obj, name, *args, **kwds)

    else:
        ### Case is a function fun()
        return _apply_async_with_callback_fun(fun_name, *args, **kwds)

def run_in_separate_process_unix(func, *args, **kwds):

    """

    **Synopsis:**
        * Runs function in separete process and waits until finish

    :Dependences: Cpickle, pipe

    .. _run_in_separate_process:

    **Args:**
        * fun_name: can be a class function cls.fun(), or a function fun()
        * args: arguments
        * kwds: optional arguments

    **Optional parameters:**
        * None

    **Returns:**
        * Function results

    **Error:**

        None

    .. note::

        * Only works on Unix
        * `From <http://code.activestate.com/recipes/511474-run-function-in-separate-process/>`_.
    """

    pread, pwrite = os.pipe()
    pid = os.fork() ### Does not exits in windows

    if pid > 0:

        os.close(pwrite)
        with os.fdopen(pread, 'rb') as f:
            status, result = cPickle.load(f)
        os.waitpid(pid, 0)

        if status == 0:
            return result
        else:
            raise result

    else:

        os.close(pread)
        try:
            result = func(*args, **kwds)
            status = 0
        except Exception(exc):
            result = exc
            status = 1
        with os.fdopen(pwrite, 'wb') as f:
            try:
                cPickle.dump((status,result), f, cPickle.HIGHEST_PROTOCOL)
            except cPickle.PicklingError(exc):
                cPickle.dump((2,exc), f, cPickle.HIGHEST_PROTOCOL)

        os._exit(0)

if __name__ == '__main__':

    print("Is not a runtime")
