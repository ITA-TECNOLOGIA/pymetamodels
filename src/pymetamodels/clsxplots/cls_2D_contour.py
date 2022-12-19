#!/usr/bin/python

import os, sys, traceback, datetime
from multiprocessing import *
import multiprocessing as mpp
import inspect
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import ticker

import pickle

class _cls_2D_contour(object):

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
    def append_data(self, x, y, z, max_traingle_size, bar_title, levels = 12, cmap = "", \
                    linewidths = 0.1, colors_lines='k', data_points_size = 0.1, \
                    format_bar = '%.1f', format_contour = '%.1f', format_contour_size = 9 ):

        config = self._default()

        config["bar_title"]= bar_title
        config["format_bar"]= format_bar
        config["format_contour"]= format_contour
        config["format_contour_size"]= format_contour_size

        if x != []: config["x_axis"] = np.asarray(x, dtype = float)
        if y != []: config["y_axis"] = np.asarray(y, dtype = float)
        if z != []: config["z_axis"] = np.asarray(z, dtype = float)
        if not config["max_traingle_size"]:
            config["max_traingle_size"] = max_traingle_size
        else:
            config["max_traingle_size"] = float(max_traingle_size)
        config["levels"] = int(levels)
        if linewidths is None:
            config["linewidths"] = float(0)
        else:
            config["linewidths"] = float(linewidths)
        config["colors_lines"] = colors_lines
        if data_points_size is None:
            config["data_points_size"] = float(0)
        else:
            config["data_points_size"] = float(data_points_size)

        if cmap == "":
            config["cmap"] = plt.cm.rainbow
        else:
            config["cmap"] = cmap

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
            fig = plt.figure(figsize = [8.3,5.85])  #size in inches
            plot1 = self.append_axes(cols = 1, rows = 1, wspace = 0.2, hspace = 0.05)
        else:
            plt.rcParams['figure.subplot.left'] = '0.15'
            plt.rcParams['figure.subplot.right'] = '0.95'
            fig = plt.figure(figsize = [8.3,5.85])  #size in inches
            plot1 = self.append_axes(cols = 1, rows = 1, wspace = 0.1, hspace = 0.05)

        ax = plt.subplot(111)

        #Loop series
        ii = 1

        for serie in self.series:

            ### Triangulation
            x = serie["x_axis"]
            y = serie["y_axis"]
            z = serie["z_axis"]

            # triangulation Delanuy
            triang = tri.Triangulation(x, y)
            #avoind long triangles
            if serie["max_traingle_size"]:
                mask = self.long_edges(x, y, triang.triangles, serie["max_traingle_size"])
                triang.set_mask(mask)

            # clip z-axis
            if self.config["range_z_down"] != "":
                z = np.clip(z,self.config["range_z_down"], np.max(z))
            if self.config["range_z_up"] != "":
                z = np.clip(z,np.min(z),self.config["range_z_up"])

            arg = [triang,z, serie["levels"]]

            #Arguments colour contour plot
            karg = {}
            if serie["cmap"]:
                karg["cmap"] = serie["cmap"]

                CS3c = ax.tricontourf(*arg, **karg)

            #Arguments lines contour plot
            karg = {}
            if serie["linewidths"] > 0:
                karg["linewidths"] = serie["linewidths"]
                karg["colors"] = serie["colors_lines"]

                CS3l = ax.tricontour(*arg, **karg)

                if serie["format_contour"] and serie["format_contour"] != "":
                    ax.clabel(CS3l, fontsize=serie["format_contour_size"], inline=1, fmt=serie["format_contour"])

            if serie["data_points_size"] > 0:
                plt.plot(x, y, 'ko', ms=serie["data_points_size"])

            #Color bar
            if serie["cmap"] and serie["format_bar"] != "" and serie["format_bar"] is not None:
                Z = [[0,0],[0,0]]
                _min, _max, N_colors = np.min(z), np.max(z), serie["levels"]
                cmap_Ticks_arr = np.linspace(_min, _max, num = N_colors).astype(np.float32)
                step = cmap_Ticks_arr[1] - cmap_Ticks_arr[0]
                cmap_arr = np.linspace(_min, _max+step, num = 500).astype(np.float32)
                cmap_Ticks_arr = np.linspace(_min, _max+step, num = N_colors).astype(np.float32)

                clb = fig.colorbar(CS3c, format=serie["format_bar"])
                clb.set_label(serie["bar_title"])

            ii = ii + 1

        # add some text for labels, title and axes ticks
        ax.set_title(self.config["name_label"])

        if not self.config["show_axis"]:
            ax.axis('off')
        else:
            ax.set_ylabel(self.config["name_y_axis"])
            ax.set_xlabel(self.config["name_x_axis"])

            #ax.legend( _lbl_pos, _lbl_names, loc=0 )

            # limits
            if self.config["range_x_down"] != "":
                ax.set_xlim(xmin = self.config["range_x_down"])

            if self.config["range_x_up"] != "":
                ax.set_xlim(xmax = self.config["range_x_up"])

            if self.config["range_y_down"] != "":
                ax.set_ylim(ymin = self.config["range_y_down"])

            if self.config["range_y_up"] != "":
                ax.set_ylim(ymax = self.config["range_y_up"])

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
        route = r'xplot_2Dcontour_' + str(self.ID).replace(" ", "_") + '.jpg'
        self.savefig(route)
        if self.parent._eps or run_eps:
            route = r'xplot_2Dcontour_' + str(self.ID).replace(" ", "_") + '.eps'
            self.savefig(route)
            route = r'xplot_2Dcontour_' + str(self.ID).replace(" ", "_") + '.pdf'
            self.savefig(route)

        plt.ion()

        return 'xplot_alot_' + self.ID

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
        return [plt.axes([lefts[j],bottoms[i], width, height]) for i in range(rows-1,-1,-1) for j in range(cols) ]

    def long_edges(self, x, y, triangles, radio):
        out = []
        for points in triangles:
            #print points
            a,b,c = points
            d0 = np.sqrt( (x[a] - x[b]) **2 + (y[a] - y[b])**2 )
            d1 = np.sqrt( (x[b] - x[c]) **2 + (y[b] - y[c])**2 )
            d2 = np.sqrt( (x[c] - x[a]) **2 + (y[c] - y[a])**2 )
            max_edge = max([d0, d1, d2])
            #print points, max_edge
            if max_edge > radio:
                out.append(True)
            else:
                out.append(False)
        return out

    def savefig(self, inName,w=11.6929134,h=8.26771654):

        ##

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
        plt.savefig(route)
        if not self.parent._no_logo:
            del(plt.gca().texts[-1])
        
        plt.ion()

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
        config["range_z_down"]= ""
        config["range_z_up"]= ""
        config["name_label"]= ""

        config["show_axis"]= True
        config["x_axis"] = []
        config["y_axis"] = []
        config["z_axis"] = []
        config["max_traingle_size"] = 0
        config["cmap"] = plt.cm.rainbow
        config["levels"] = 12
        config["linewidths"] = 0.1
        config["colors_lines"] = 'k'
        config["data_points_size"] = 0.1
        config["bar_title"]= ""
        config["format_bar"]= '%.1f'
        config["format_contour"]= '%.1f'
        config["format_contour_size"]= 9

        config["extra_text"] = 0
        config["extra_text_data"] = []
        config["extra_arrow"] = 0
        config["extra_arrow_data"] = []

        return config

    def name_label(self, val):

        self.config["name_label"] = val

    def show_axis(self, val):

        self.config["show_axis"] = val

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

    def range_z_down(self, txt):

        self.config["range_z_down"] = float(txt)

    def range_z_up(self, txt):

        self.config["range_z_up"] = float(txt)

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
