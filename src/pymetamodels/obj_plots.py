#!/usr/bin/env python3

import os,sys
import pymetamodels.clsxplots.clsxplots as xplots
import gc

class objplots(object):

    ## Common functions
    ###################

    def __init__(self, model):

        self.model = model
        self.length_max_2D = 10000.
        self.length_max_3D = 5000.

        pass

    def mlp_start(self, output_folder):

        ## General start of plot class

        return xplots.xplot(output_folder, layout = 0, eps = False, no_logo = False)

    def mlp_end(self, mlp):

        ## Make plot pages

        mlp.make_pages(run_eps = True, run_ppt = True )

    def downsample_1D_array(self, arr, new_length_max = 2500):

        ## Downsample 1D arrays

        length = len(arr)

        if length > (new_length_max+500):

            skip = int(1.*length/new_length_max)

            return arr[0:length:skip].copy()

        return arr.copy()

    ## Plots
    ###################

    def plot_scatter_sensitivity(self, data):

        ## XY scatter plot cross variable sampling relation in the sensitivity analysis
        mlp = self.mlp_start(data["output_folder"])

        ID = data["file name"]
        mlpp = mlp.mplot(ID)

        yy = self.downsample_1D_array(data["ydata"], new_length_max = self.length_max_2D)
        xx = self.downsample_1D_array(data["xdata"], new_length_max = self.length_max_2D)

        mlpp.append_data(xx, yy, data["legend"], data["format"])

        mlpp.ylabel = data["ylabel"]
        mlpp.xlabel = data["xlabel"]

        mlpp.title = ''
        mlpp.legend_position('best')
        mlpp.listtplot_log = 0

        mlpp.x_max('')
        mlpp.x_min('')
        mlpp.y_max('')
        mlpp.y_min('')


        self.mlp_end(mlp)
        del mlp
        gc.collect()

    def plot_scatter_histsensi(self, data):

        ## bar plot for sensitivity indexes
        mlp = self.mlp_start(data["output_folder"])

        ID = data["file name"]
        mlpp = mlp.mplot(ID, type_plot = 10)

        yy = self.downsample_1D_array(data["ydata"], new_length_max = self.length_max_2D)
        xx = self.downsample_1D_array(data["xdata"], new_length_max = self.length_max_2D)

        mlpp.append_data(yy , data["legend"], color = data["color"], \
                          annotate_serie = False, y_std = [], y_opposite = False, x_axis = xx, \
                          alpha = 1., hatch = "", edgecolor="", separation_annotate = 0.03, errorbar_color="black")

        mlpp.name_y_axis(data["ylabel"])
        mlpp.name_x_axis(data["xlabel"])
        mlpp.x_ticks_rotate(45.)

        self.mlp_end(mlp)
        del mlp
        gc.collect()

    def plot_scatterXY_model(self, data):

        ## XY scatter plot cross variable sampling relation in the sensitivity analysis
        mlp = self.mlp_start(data["output_folder"])

        ID = data["file name"]
        mlpp = mlp.mplot(ID)

        yy = self.downsample_1D_array(data["ydata"], new_length_max = self.length_max_2D)
        xx = self.downsample_1D_array(data["xdata"], new_length_max = self.length_max_2D)

        mlpp.append_data(xx, yy, data["legend"], data["format"])

        mlpp.ylabel = data["ylabel"]
        mlpp.xlabel = data["xlabel"]

        if "xdata2" in data:
            yy = self.downsample_1D_array(data["ydata2"], new_length_max = self.length_max_2D)
            xx = self.downsample_1D_array(data["xdata2"], new_length_max = self.length_max_2D)

            mlpp.append_data(xx, yy, data["legend2"], data["format2"])

        mlpp.title = ''
        mlpp.legend_position('best')
        mlpp.listtplot_log = 0

        mlpp.x_max('')
        mlpp.x_min('')
        mlpp.y_max('')
        mlpp.y_min('')


        self.mlp_end(mlp)
        del mlp
        gc.collect()

    def plot_scatterXYZ_model(self, data):

        ## XY scatter plot cross variable sampling relation in the sensitivity analysis
        mlp = self.mlp_start(data["output_folder"])

        ID = data["file name"]
        mlpp = mlp.mplot(ID, type_plot = 12)

        if data["scatter"]:
            xx = self.downsample_1D_array(data["xdata"], new_length_max = self.length_max_3D)
            yy = self.downsample_1D_array(data["ydata"], new_length_max = self.length_max_3D)
            zz = self.downsample_1D_array(data["zdata"], new_length_max = self.length_max_3D)

            mlpp.append_scatter_data(xx, yy, zz, legend = data["legend"], data_points_size = data["data_points_size"], marker=data["marker"], color = data["color"] )

        xx_grid = data["xdata_grid"]
        yy_grid = data["ydata_grid"]
        zz_grid = data["zdata_grid"]
        mlpp.append_surface_data(xx_grid, yy_grid, zz_grid, legend = data["legend_grid"], linewidth = 0, antialiased = False, color = "" )

        mlpp.ylabel = data["ylabel"]
        mlpp.xlabel = data["xlabel"]
        mlpp.zlabel = data["zlabel"]

        mlpp.name_x_axis(data["xlabel"])
        mlpp.name_y_axis(data["ylabel"])
        mlpp.name_z_axis(data["zlabel"])

        if "anotate" in data:
            [txt, pos_x, pos_y, fontsize] = data["anotate"]
            mlpp.extra_text_add(pos_x, pos_y, txt, fontsize = fontsize)

        self.mlp_end(mlp)
        del mlp
        gc.collect()

    #### Other functions
    #####################################

    def error(self, msg):
        print("Error: " + msg)
        if self.model.objlog:
            self.model.objlog.warning("ID03 - %s" % msg)          
        raise ValueError(msg)

    def msg(self, msg):
        print("Msg: " + msg)
        if self.model.objlog:
            self.model.objlog.info("ID03 - %s" % msg)            
