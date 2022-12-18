#!/usr/bin/env python3

## damped oscillator
##

import os, sys, random
import numpy as np
from math import exp, pow, sqrt, sin
import pymetamodels

class model_obj(object):

    """

    .. _model_id_001_damped_oscillator_op:

    **Damped oscillator functions model**

    :platform: Unix, Windows
    :synopsis: Damped oscillator
    :author:

    :Dependences:

    """

    def __init__(self):

        ## Initial variables
        self.doeX = None
        self.doeY = None
        self.case_dict = None
        self.ii = 1

    def run_model(self):

        # variables without constrain
        [m,k,D,Ekin] = [4.86426605e+00, 2.55749192e+01, 2.77792593e-02, 3.89040358e+01] # #without constrains_min_obj_1
        [m,k,D,Ekin] = [4.00821573e+00, 2.03779120e+01, 1.86784180e-02, 3.90331743e+01] # #without constrains_min_obj_6

        # variables with constrain
        [m,k,D,Ekin] = [4.45555556e+00, 2.33333333e+01, 2.77777778e-02, 4.11111111e+01] # #with constrains_min_obj_1
        [m,k,D,Ekin] = [3.1625, 15., 0.04, 20.] # #with constrains_min_obj_6
        [m,k,D,Ekin] = [3.775, 15., 0.04, 10.] # #with constrains_min_obj_6
        [m,k,D,Ekin] = [4.67333333, 23.33333333, 0.034, 26.] # #with constrains_min_obj_6
        [m,k,D,Ekin] = [1.52, 34.33, 0.034, 15.33] # #with constrains_min_obj_6
        [m,k,D,Ekin] = [4.67333333, 23.33333333, 0.034, 26.] # #with constrains_min_obj_6

        # solution
        #[m,k,D,Ekin] = [3.284521484, 15.99609375, 0.020097656, 11.66015625]

        # calculations
        omega_0 = (k/m)**0.5 #w_0
        omega = omega_0 * (1-(D**2.))**0.5 #w
        v_0 = ((2*Ekin)/m)**0.5

        num_steps = 101
        t_max =  10
        t_observe = 5 # x(t>5) -> min

        # time serie
        t_serie = []
        for ii in range(0,num_steps):
            t_serie.append(t_observe+ii*float(t_max-t_observe)/num_steps)

        # ------------------------- #
        #   Results
        # ------------------------- #
        # time serie values
        x_serie = []
        for ii in range(0,num_steps):
            tt_serie = t_serie[ii]
            xx_serie = exp(-1*D*omega_0*tt_serie)*(v_0/omega)*sin(omega*tt_serie)
            x_serie.append(xx_serie)

        # find max value
        ref = None
        for ii in range(0,num_steps):
            if ref is None: ref = abs(x_serie[ii])
            if abs(x_serie[ii]) >= ref: ref = abs(x_serie[ii])

        x_max = ref

        w_damped = omega_0 * (1-(D**2.))**0.5

        # compute simulation - reference absoluta value
        #"""
        dict_row_col = {}
        dict_row_col["time"] = t_serie
        dict_row_col["disp"] = x_serie

        mmita = pymetamodels.load()
        file_name = "%s_%i" % ("test", self.ii)
        path_out = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test")
        if not os.path.exists(path_out): os.makedirs(path_out)
        mmita.save_table_csv_channels(dict_row_col, file_name, path_out, units_chn = None, sort_chn = False)
        """
        mmita = pymetamodels.load()
        file_name = "reference.csv"
        path_out = os.path.dirname(os.path.realpath(__file__))
        reference, units = mmita.read_table_csv_channels(file_name, path_out, units_chn = False)
        x_serie = np.asarray(x_serie,dtype=np.dtype('float64'))
        reference = np.asarray(reference["disp"],dtype=np.dtype('float64'))
        if len(x_serie) - len(reference) != 0: error
        #min_obj = np.sum((np.absolutex_serie**2. - reference**2.))**0.5)
        min_obj = (1.-np.corrcoef(x_serie,reference)[0,1])*2.
        """

if __name__ == "__main__":

    analysis = model_obj()
    print("test")
    analysis.run_model()
