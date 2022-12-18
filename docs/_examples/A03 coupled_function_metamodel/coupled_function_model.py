#!/usr/bin/env python3

## coupled function
##

import os, sys, random
import numpy as np
from math import *

class model_obj(object):

    """

    .. _model_id_001_coupled_function:

    **Coupled_function**

    :platform: Unix, Windows
    :synopsis: coupled fuction
    :author: FLC

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
        X1 = self.doeX["X1"][self.ii]
        X2 = self.doeX["X2"][self.ii]
        X3 = self.doeX["X3"][self.ii]
        X4 = self.doeX["X4"][self.ii]
        X5 = self.doeX["X5"][self.ii]

        # calculations
        Y =  0.5*X1+X2+0.5*X1*X2+5*sin(X3)+0.2*X4+0.1*X5

        # outputs
        self.doeY["Y"][self.ii] = Y
        if "Y2" in self.doeY.keys():
            self.doeY["Y2"][self.ii] = Y*1.5

        pass
