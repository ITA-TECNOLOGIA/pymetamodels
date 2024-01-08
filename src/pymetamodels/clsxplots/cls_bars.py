#!/usr/bin/python

import os, sys, traceback, datetime
from multiprocessing import *
import multiprocessing as mpp
import inspect
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import ticker

import pickle

from pymetamodels.clsxplots.obj_func import test_empty

class _cls_bars(object):

    def __init__(self, ID, type_plot, parent, output_folder):

        self.parent = parent
        self.ID = str(ID)
        self.type_plot = type_plot

        self.consider_zeros = True

        self.title = ""

        self.series = []

        self.config = self._default()

        self.output_folder = output_folder

    ##### ADD DATA
    def append_data(self, y , legend, color = "", \
                          annotate_serie = False, y_std = [], y_opposite = False, x_axis = [], \
                          alpha = 1., hatch = "", edgecolor="", separation_annotate = 0.03, errorbar_color="black"):

        config = self._default()

        config["name_label"]= legend
        config["annotate_serie"]= annotate_serie
        config["separation_annotate"]= float(separation_annotate)
        
        if test_empty(x_axis): config["x_axis"] = np.asarray(x_axis)
        if test_empty(y): config["y_axis"] = np.asarray(y, dtype = float)
        if test_empty(y_std): config["y_std"]= np.asarray(y_std, dtype = float)

        config["y_opposite"]= y_opposite
        if color != "": config["color"] = color

        config["alpha"]= float(alpha)

        lst = ['//', '/', '-', '+', 'x', '\\', '*', 'o', 'O', '.']
        if hatch in lst:
            config["hatch"]= hatch
            if edgecolor == "":
                config["edgecolor"]= color
                config["alpha"] = config["alpha"] * 0.5
            else:
                config["edgecolor"]= edgecolor

        config["errorbar_color"]= errorbar_color

        self.series.append(config)

    ##### PLOT
    def make_page(self, run_eps = True, run_ppt = True):

        self._0_make_page(run_eps, run_ppt)

    def _0_make_page(self, run_eps = True, run_ppt = True):

        ### Creates the plot
        plt.ioff()

        plt.rcParams['figure.subplot.top'] = '0.90'

        if self.parent._vlayout == 1: # paper
            plt.rcParams['figure.subplot.right'] = '0.95'
            if self.parent.font_size <= 17:
                plt.rcParams['figure.subplot.top'] = '0.90'
            else:
                plt.rcParams['figure.subplot.top'] = '0.95'
                plt.rcParams['figure.subplot.bottom'] = '0.12'
            fig = plt.figure(figsize = [8.3,5.85])  #size in inches
            plot1 = self.append_axes(cols = 1, rows = 1, wspace = 0.2, hspace = 0.05) 
        else:
            plt.rcParams['figure.subplot.left'] = '0.15'
            plt.rcParams['figure.subplot.right'] = '0.95'
            fig = plt.figure(figsize = [8.3,5.85])  #size in inches
            plot1 = self.append_axes(cols = 1, rows = 1, wspace = 0.1, hspace = 0.05)

        ax = plt.subplot(111)
        ax_y2 = None

        #Get max size
        max = 0
        for serie in self.series:
            if len(serie["y_axis"]) > max:
                max = len(serie["y_axis"])
        ind = np.arange(max)

        #Loop series
        ii = 1
        _rects_lst = []
        _lbl_pos = []
        _lbl_names = []

        #Width bars
        _width, _height = self.get_ax_size(ax,fig)
        self.config["width"] =  _width / float(len(self.series[0]["x_axis"])*3.*float(len(self.series)))

        for serie in self.series:

            if serie["y_opposite"]:
                if ax_y2 is None: ax_y2 = ax.twinx()
                axx = ax_y2
            else:
                axx = ax

            arg = [ind,serie["y_axis"], self.config["width"]]

            #Arguments
            karg = {}

            if self.config["islog"]:
                karg["log"] = 1
                karg["bottom"] = self.config["islog_min"]

            if serie["color"] != "":
                karg["color"] = serie["color"]
                karg["error_kw"] = dict(ecolor=serie["errorbar_color"])
            if serie["y_std"] != []: karg["yerr"] = serie["y_std"]
            if serie["alpha"] != 1.: karg["alpha"] = serie["alpha"]
            if serie["hatch"] != "": karg["hatch"] = serie["hatch"]
            if serie["edgecolor"] != "": karg["edgecolor"] = serie["edgecolor"]
            #karg["align"] = 'center'

            #Add bar
            rects = axx.bar(*arg, **karg)

            _rects_lst.append(rects)
            _lbl_pos.append(rects[0])
            if serie["name_label"] != "" and serie["name_label"] != "_nolegend_":
                _lbl_names.append(serie["name_label"])

            #Add labels
            if serie["annotate_serie"]:

                __max = 0
                for rect in rects:
                    if abs(rect.get_height()) > __max:
                        __max = abs(rect.get_height())

                ccc =0
                for rect in rects:

                    _xloc = rect.get_x() + (rect.get_width()/2.)
                    _yloc = serie["y_axis"][ccc]
                    _yloc = rect.get_y() + rect.get_height() + (__max*serie["separation_annotate"])

                    axx.annotate(serie["y_axis"][ccc],
                                (_xloc, _yloc),
                                va="bottom", ha="center")

                    ccc = ccc + 1

            #End
            if ii == len(self.series):
                pass
            else:
                ind = ind + self.config["width"]
                ii = ii + 1

        # add some text for labels, title and axes ticks
        axx.set_ylabel(self.config["name_y_axis"])
        axx.set_xlabel(self.config["name_x_axis"])

        axx.set_title(self.title)
        #axx.set_xticks(ind)
        #axx.set_xticklabels( serie["x_axis"] )

        _phase = self.config["width"] * float(len(self.series))/2.
        _phase = np.arange(len(serie["x_axis"])) + _phase
        plt.xticks(  _phase, tuple(serie["x_axis"]))

        ax.legend( _lbl_pos, _lbl_names, loc=0 )

        # limits
        if self.config["range_x_down"] != "":
            ax.set_xlim(xmin = self.config["range_x_down"])

        if self.config["range_x_up"] != "":
            ax.set_xlim(xmax = self.config["range_x_up"])

        if self.config["range_y_down"] != "":
            ax.set_ylim(ymin = self.config["range_y_down"])

        if self.config["range_y_up"] != "":
            ax.set_ylim(ymax = self.config["range_y_up"])

        if self.config["x_ticks_rotate"] != 0.:
            plt.xticks(rotation = self.config["x_ticks_rotate"])

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
        #fig.tight_layout() ## Matplotlib version change for >=3.8.0 bbox_inches='tight' in savefig

        route = r'xplot_barplot_' + str(self.ID).replace(" ", "_") + '.jpg'
        self.savefig(route)

        if self.parent._eps or run_eps:
            route = r'xplot_barplot_' + str(self.ID).replace(" ", "_") + '.eps'
            self.savefig(route)
            route = r'xplot_barplot_' + str(self.ID).replace(" ", "_") + '.pdf'
            self.savefig(route)

        plt.ion()

        return 'xplot_alot_' + self.ID

    def savefig(self, inName,w=11.6929134,h=8.26771654):

        ###

        if self.output_folder is None:
            route = os.path.join(self.parent.out_path,inName)
        else:
            route = os.path.join(self.output_folder,inName)

        plt.ioff()

        if not self.parent._no_logo:
            plt.text(0.995, 0.005 , self.parent._logo,
                     style=self.parent._logo_style, weight='bold', size='small',
                     color=self.parent._logo_color, horizontalalignment='right',
                     transform = plt.gca().transAxes)
        
        plt.savefig(route, bbox_inches='tight')
        
        if not self.parent._no_logo:
            try:
                del(plt.gca().texts[-1])
            except:
                pass

        plt.ion()

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
            raise ValueError('no space available below lowest axes')

        #create lists with coordinates and dimensions for new axes
        height = ((top - bottom) - (rows - 1) * hspace)/rows
        bottoms = [(bottom + i * (height + hspace)) for i in range(rows)]
        width = (rcwidth - (cols-1) * wspace)/cols
        lefts = [(rcleft + i * (width + wspace)) for i in range(cols)]

        #return a list of axes instances
        return [plt.axes([lefts[j],bottoms[i], width, height], visible = False) for i in range(rows-1,-1,-1) for j in range(cols) ] ## error on Matplotlib verion >=3.8.0 visible=False

    def get_ax_size(self, _ax, _fig, inDPI = False):
        bbox = _ax.get_window_extent().transformed(_fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        if inDPI:
            width *= _fig.dpi
            height *= _fig.dpi
        return width, height

    ##### CONFIG
    def _default(self):

        config = {}

        config["name_x_axis"] = ""
        config["name_y_axis"] = ""
        config["range_x_down"]= ""
        config["range_x_up"]= ""
        config["range_y_down"]= ""
        config["range_y_up"]= ""
        config["name_label"]= ""
        config["annotate_serie"]= False
        config["x_axis"]= []
        config["y_axis"]= []
        config["y_std"]= []
        config["y_opposite"]= False
        config["x_opposite"]= False
        config["width"]= 0.35
        config["color"]= ""
        config["alpha"]= 1.
        config["hatch"]= ""
        config["edgecolor"]= ""
        config["islog"]= False
        config["islog_min"]= 1.
        config["x_ticks_rotate"]= 0.

        config["extra_text"] = 0
        config["extra_text_data"] = []
        config["extra_arrow"] = 0
        config["extra_arrow_data"] = []

        return config

    def name_x_axis(self, txt):

        self.config["name_x_axis"] = str(txt)

    def name_y_axis(self, txt):

        self.config["name_y_axis"] = str(txt)

    def range_x_down(self, txt):

        self.config["range_x_down"] = float(txt)

    def range_x_up(self, txt):

        self.config["range_x_up"] = float(txt)

    def range_y_down(self, txt):

        self.config["range_y_down"] = float(txt)

    def range_y_up(self, txt):

        self.config["range_y_up"] = float(txt)

    def x_ticks_rotate(self, angle):

        self.config["x_ticks_rotate"]= float(angle)

    def is_log(self, min_value):

        self.config["islog"] = True
        self.config["islog_min"] = float(min_value)

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
