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
        self.ii = None

    def run_model(self):

        # variables
        m = self.doeX["m"][self.ii] #kg [0.1,5.0]
        k = self.doeX["k"][self.ii] # N/m [20,50]

        D = self.doeX["D"][self.ii]
        Ekin = self.doeX["E_{kin}"][self.ii] #10 [Nm]

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

        # compute simulation - reference absolute value
        mmita = pymetamodels.load()
        file_name = "reference.csv"
        path_out = os.path.dirname(os.path.realpath(__file__))
        reference, units = mmita.read_table_csv_channels(file_name, path_out, units_chn = False)
        x_serie = np.asarray(x_serie,dtype=np.dtype('float64'))
        reference = np.asarray(reference["disp"],dtype=np.dtype('float64'))
        if len(x_serie) - len(reference) != 0: error

        # outputs
        self.doeY["x_{max}"][self.ii] = x_max
        self.doeY["w_{damped}"][self.ii] = w_damped
        self.doeY["cons-w_{damped}"][self.ii] = 10.-w_damped
        self.doeY["min-obj"][self.ii] = np.sum(np.square(x_serie-reference))
