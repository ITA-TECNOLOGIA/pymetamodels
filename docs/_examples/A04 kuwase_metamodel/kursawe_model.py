#!/usr/bin/env python3

import os, sys, random
import numpy as np
from math import exp, pow, sqrt, sin

class model_obj(object):

    """

    .. _model_id_001_kursawe_model:

    **Kursawe functions model**

    :platform: Unix, Windows
    :synopsis: kursawe model
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

        # calculations
        kursawe1 = -10 * exp(-0.2*sqrt(X1*X1 + X2*X2) )
        kursawe1 = kursawe1 - 10 * exp(-0.2*sqrt(X2*X2 + X3*X3) )

        kursawe2 = pow(abs(X1),0.8) + 5*pow(sin(X1),3)
        kursawe2 = kursawe2 + pow(abs(X2),0.8) + 5*pow(sin(X2),3)
        kursawe2 = kursawe2 + pow(abs(X3),0.8) + 5*pow(sin(X3),3)

        # outputs
        self.doeY["kursawe1"][self.ii] = kursawe1
        self.doeY["kursawe2"][self.ii] = kursawe2

        pass
