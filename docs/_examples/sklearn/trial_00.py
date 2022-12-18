#!/usr/bin/env python3

import os, sys, random, warnings
import numpy as np
import gc

import pymetamodels.obj_metamodel as obj_metamodel

class analysis(object):

    """

    .. _model_coupled_function_data_struct:

    **Synopsis:**
        * Metamodels trials

    """

    def __init__(self):

        self.script_folder = os.path.dirname(os.path.abspath(__file__))

        self.obj = obj_metamodel.objmetamodel()

    def run_tests(self):

        self.obj.run_tests(self.script_folder)


if __name__ == "__main__":

    analysis = analysis()

    analysis.run_tests()
