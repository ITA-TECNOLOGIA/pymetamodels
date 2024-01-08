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
from matplotlib import cm

import pickle

from pymetamodels.clsxplots.obj_func import test_empty

class _cls_3D_plot(object):

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
    def append_scatter_data(self, x, y, z, legend = "", \
                    data_points_size = 20, \
                    marker="", color = "" ):

        config = self._default()

        config["type"] = "scatter"
        config["data_points_size"]= data_points_size
        config["marker"]= marker
        config["color"]= color

        ### No legends
        if legend == "":
            config["legend"] =  "_nolegend_"
        else:
            config["legend"] =  legend

        if test_empty(x): config["x_axis"] = np.asarray(x, dtype = float)
        if test_empty(y): config["y_axis"] = np.asarray(y, dtype = float)
        if test_empty(z): config["z_axis"] = np.asarray(z, dtype = float)

        self.series.append(config)

    def append_surface_data(self, x, y, z, legend = "", \
                    linewidth = 0, \
                    antialiased = False, color = "" ):

        config = self._default()

        config["type"] = "surface"
        config["linewidth"] = linewidth
        config["antialiased"]= antialiased
        config["cmap"]= cm.coolwarm
        config["color"]= color

        ### No legends
        if legend == "":
            config["legend"] =  "_nolegend_"
        else:
            config["legend"] =  legend
        
        if test_empty(x): config["x_axis"] = np.asarray(x, dtype = float)
        if test_empty(y): config["y_axis"] = np.asarray(y, dtype = float)
        if test_empty(z): config["z_axis"] = np.asarray(z, dtype = float)

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

        ax = plt.subplot(111, projection = '3d')

        #Loop series
        ii = 1
        is_legends = False

        for serie in self.series:

            ### Triangulation
            x = serie["x_axis"]
            y = serie["y_axis"]
            z = serie["z_axis"]

            if serie["type"] == "scatter":
                # clip z-axis
                if self.config["range_z_down"] != "":
                    z = np.clip(z,self.config["range_z_down"], np.max(z))
                if self.config["range_z_up"] != "":
                    z = np.clip(z,np.min(z),self.config["range_z_up"])

                arg = [x,y]

                #Arguments colour contour plot
                karg = {}
                karg["zs"] = z
                if serie["data_points_size"] != "":
                    karg["s"] = serie["data_points_size"]
                if serie["color"] != "":
                    karg["c"] = serie["color"]
                if serie["marker"] != "":
                    karg["marker"] = serie["marker"]
                if serie["legend"] !=  "_nolegend_":
                    karg["label"] = serie["legend"]
                    is_legends = True
                karg["depthshade"] = True

                ax.scatter(*arg,**karg)

            if serie["type"] == "surface":
                # clip z-axis
                if self.config["range_z_down"] != "":
                    z = np.clip(z,self.config["range_z_down"], np.max(z))
                if self.config["range_z_up"] != "":
                    z = np.clip(z,np.min(z),self.config["range_z_up"])

                arg = [x,y,z]

                #Arguments colour contour plot
                karg = {}
                if serie["linewidth"] != "":
                    karg["linewidth"] = serie["linewidth"]
                if serie["color"] != "":
                    karg["color"] = serie["color"]
                if serie["cmap"] != "" or serie["cmap"] is None:
                    karg["cmap"] = serie["cmap"]
                if serie["legend"] !=  "_nolegend_":
                    karg["label"] = serie["legend"]
                    is_legends = True
                karg["antialiased"] = serie["antialiased"]

                surf = ax.plot_surface(*arg,**karg)
                if serie["legend"] !=  "_nolegend_":
                    # https://stackoverflow.com/questions/27449109/adding-legend-to-a-surface-plot
                    surf._edgecolors2d = surf._edgecolor3d
                    surf._facecolors2d = surf._facecolor3d

            ii = ii + 1

        # add some text for labels, title and axes ticks
        if self.config["name_label"] != "":
            ax.set_title(self.config["name_label"])

        if is_legends:
            plt.legend(loc = "upper right")

        if not self.config["show_axis"]:
            ax.axis('off')
        else:
            ax.set_ylabel(self.config["name_y_axis"])
            ax.set_xlabel(self.config["name_x_axis"])
            ax.set_zlabel(self.config["name_z_axis"])

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

            if self.config["range_z_down"] != "":
                ax.set_zlim(zmin = self.config["range_z_down"])

            if self.config["range_z_up"] != "":
                ax.set_zlim(zmax = self.config["range_z_up"])

        # add text
        if self.config["extra_text"] > 0:
            for cccc in range(0,len(self.config["extra_text_data"])-1):
                xrt = self.config["extra_text_data"][cccc]
                pos_x, pos_y, txt, fontsize = xrt[0], xrt[1], xrt[2], xrt[3]
                #print cccc, pos_x, pos_y, txt
                #plt.text(pos_x, pos_y, txt, horizontalalignment='left', transform=ax.transAxes,zorder=10)
                if fontsize is None:
                    plt.annotate(txt,xy=(pos_x,pos_y),xytext=(pos_x,pos_y),xycoords="axes fraction",textcoords="axes fraction", zorder=10)
                else:
                    plt.annotate(txt,xy=(pos_x,pos_y),xytext=(pos_x,pos_y),xycoords="axes fraction",textcoords="axes fraction", zorder=10, fontsize = fontsize)

        # add arrow
        if self.config["extra_arrow"] > 0:
            for cccc in range(0,len(self.config["extra_arrow_data"])-1):
                xrt = self.config["extra_arrow_data"][cccc]
                pos_x, pos_y, to_pos_x, to_pos_y, _width, _head = xrt[0], xrt[1], xrt[2], xrt[3], xrt[4], xrt[5]

                plt.annotate("",xy=(pos_x,pos_y),xytext=(to_pos_x, to_pos_y), \
                            xycoords="axes fraction",textcoords="axes fraction", zorder=10, \
                            arrowprops=dict(arrowstyle=_head,linewidth=_width))

        ### draw
        route = r'xplot_3Dplot_' + str(self.ID).replace(" ", "_") + '.jpg'
        self.savefig(route)
        if self.parent._eps or run_eps:
            route = r'xplot_3Dplot_' + str(self.ID).replace(" ", "_") + '.eps'
            self.savefig(route)
            route = r'xplot_3Dplot_' + str(self.ID).replace(" ", "_") + '.pdf'
            self.savefig(route)

        plt.ion()

        return 'xplot_alot_' + self.ID

    def savefig(self, inName,w=11.6929134,h=8.26771654):

        if self.output_folder is None:
            route = os.path.join(self.parent.out_path,inName)
        else:
            route = os.path.join(self.output_folder,inName)

        plt.ioff()

        if not self.parent._no_logo:
            ax = plt.gca()
            ax.text(0.005, 0.995, 0., self.parent._logo, zdir=None,
                     style=self.parent._logo_style, weight='bold', size='small',
                     color=self.parent._logo_color, horizontalalignment='right',
                     transform = plt.gca().transAxes)
        plt.savefig(route)
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
        config["name_z_axis"] = ""
        config["range_x_down"]= ""
        config["range_x_up"]= ""
        config["range_y_down"]= ""
        config["range_y_up"]= ""
        config["range_z_down"]= ""
        config["range_z_up"]= ""
        config["name_label"]= ""

        config["data_points_size"] = ""
        config["color"] = ""
        config["marker"] = ""
        config["label"] = ""

        config["show_axis"]= True
        config["x_axis"] = []
        config["y_axis"] = []
        config["z_axis"] = []

        config["extra_text"] = 0
        config["extra_text_data"] = []
        config["extra_arrow"] = 0
        config["extra_arrow_data"] = []

        return config

    def name_title(self, txt):

        self.config["name_label"] = str(txt)

    def show_axis(self, txt):

        self.config["show_axis"] = bool(txt)

    def name_x_axis(self, txt):

        self.config["name_x_axis"] = str(txt)

    def name_y_axis(self, txt):

        self.config["name_y_axis"] = str(txt)

    def name_z_axis(self, txt):

        self.config["name_z_axis"] = str(txt)

    def range_x_down(self, txt):

        self.config["range_x_down"] = float(txt)

    def range_x_up(self, txt):

        self.config["range_x_up"] = float(txt)

    def range_y_down(self, txt):

        self.config["range_y_down"] = float(txt)

    def range_y_up(self, txt):

        self.config["range_y_up"] = float(txt)

    def range_z_up(self, txt):

        self.config["range_z_up"] = float(txt)

    def range_z_down(self, txt):

        self.config["range_z_down"] = float(txt)

    def extra_text_add(self, pos_x, pos_y, txt, fontsize = None):

        if self.config["extra_text"] == 0:
            self.config["extra_text_data"]=[[0,0,"",None]]

        self.config["extra_text_data"].append([0,0,"",None])

        self.config["extra_text_data"][self.config["extra_text"]][0] = pos_x
        self.config["extra_text_data"][self.config["extra_text"]][1] = pos_y
        self.config["extra_text_data"][self.config["extra_text"]][2] = txt
        self.config["extra_text_data"][self.config["extra_text"]][3] = fontsize

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
