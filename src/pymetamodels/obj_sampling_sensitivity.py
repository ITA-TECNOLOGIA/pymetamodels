#!/usr/bin/env python3

import os, sys, random, warnings
import numpy as np

import pymetamodels.obj_data as obj_data


class objsamplingsensitivity(object):

    """Python class representing the sampling and sensitivity analysis

        :platform: Windows
        :synopsis: object optimization calculation

        :Dependences: SALib

        :ivar version: optimization model version
        :ivar conf_level: the confidence interval level (default 0.95)

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`

        |

    """

    ## Common functions
    ###################

    def __init__(self, model):

        self.version = 0
        self.model = model

        self.conf_level = 0.95

    def ini_analisis_type(self):

        ## Initialise and return dictionary with the sampling and sensitivity anaylis info

        out = obj_data.objdata()

        ID = "Sobol"
        out[ID] = obj_data.objdata()
        out[ID]["Analysis name"] = "Sobol sensitivity analysis"
        out[ID]["Sampling name"] = "Saltelli sampling"
        out[ID]["References"] = ["SOBOL2001271","Saltelli2010","Campolongo2011a","Saltelli2002"]

        ID = "Morris"
        out[ID] = obj_data.objdata()
        out[ID]["Analysis name"] = "Morris Analysis"
        out[ID]["Sampling name"] = "Method of Morris"
        out[ID]["References"] = ["Ruano2012","Campolongo2007","Morris1991"]

        ID = "RBD-Fast"
        out[ID] = obj_data.objdata()
        out[ID]["Analysis name"] = "Latin hypercube sampling (LHS)"
        out[ID]["Sampling name"] = "RBD-FAST - Random balance designs fourier amplitude sensitivity test"
        out[ID]["References"] = ["McKay2000a","Iman1981","Tarantola2006","Plischke2010","Tissot2012"]

        ID = "Fast"
        out[ID] = obj_data.objdata()
        out[ID]["Analysis name"] = "FAST - Fourier amplitude sensitivity Test"
        out[ID]["Sampling name"] = "Fourier amplitude sensitivity test model (eFAST)"
        out[ID]["References"] = ["Cukier1973","Saltelli1999"]

        ID = "Delta-MIM"
        out[ID] = obj_data.objdata()
        out[ID]["Analysis name"] = "Delta moment-independent measure"
        out[ID]["Sampling name"] = "Latin hypercube sampling (LHS)"
        out[ID]["References"] = ["McKay2000a","Iman1981","Borgonovo2007","Plischke2013"]

        ID = "DGSM"
        out[ID] = obj_data.objdata()
        out[ID]["Analysis name"] = "Derivative-based Global Sensitivity Measure (DGSM)"
        out[ID]["Sampling name"] = "Sampling for derivative-based global sensitivity measure (DGSM)"
        out[ID]["References"] = ["Sobol2009","Sobol2010"]

        ID = "Factorial"
        out[ID] = obj_data.objdata()
        out[ID]["Analysis name"] = "Fractional factorial"
        out[ID]["Sampling name"] = "Fractional factorial sampling"
        out[ID]["References"] = ["Saltelli2008"]

        ID = "PAWN"
        out[ID] = obj_data.objdata()
        out[ID]["Analysis name"] = "PAWN sensitivity analysis"
        out[ID]["Sampling name"] = "Latin hypercube sampling (LHS)"
        out[ID]["References"] = ["Pianosi2015","Pianosi2018","Baroni2020"]

        """
        # Not working
        ID = "HDMR"
        out[ID] = obj_data.objdata()
        out[ID]["Analysis name"] = "High-dimensional model representation (HDMR)"
        out[ID]["Sampling name"] = "Latin hypercube sampling (LHS)"
        out[ID]["References"] = ["Rabitz2010"]
        """

        return out

    def _run_analisis(self, case, sensitivity = True, sampling = True):

        ## Run the sesitivity or sampling routines

        # Grab the sensitivity method
        sensitivity_type = self.sensitivity_type(case)

        # Define the problem
        self.problem_definition(case)

        # Run sensitivity anlysis
        if sensitivity_type == "Sobol":
            if sampling: self.sampling_sobol(case)
            if sensitivity: self.sensi_sobol(case)
        elif sensitivity_type == "Morris":
            if sampling: self.sampling_morris(case)
            if sensitivity: self.sensi_morris(case)
        elif sensitivity_type == "RBD-Fast":
            if sampling: self.sampling_RBD_Fast(case)
            if sensitivity: self.sensi_RBD_Fast(case)
        elif sensitivity_type == "Fast":
            if sampling: self.sampling_Fast(case)
            if sensitivity: self.sensi_Fast(case)
        elif sensitivity_type == "Delta-MIM":
            if sampling: self.sampling_delta(case)
            if sensitivity: self.sensi_delta(case)
        elif sensitivity_type == "DGSM":
            if sampling: self.sampling_dgsm(case)
            if sensitivity: self.sensi_dgsm(case)
        elif sensitivity_type == "Factorial":
            if sampling: self.sampling_factorial(case)
            if sensitivity: self.sensi_factorial(case)
        elif sensitivity_type == "PAWN":
            if sampling: self.sampling_PAWN(case)
            if sensitivity: self.sensi_PAWN(case)
        elif sensitivity_type == "HDMR":
            if sampling: self.sampling_hdmr(case)
            if sensitivity: self.sensi_hdmr(case)
        else:
            self.error("Sensibility analyisis type unknown %s" % sensitivity_type)

    def normalize_sensitivity(self, case, out_var, var_in):

        ## Method of sensitivity analysis normalisation
        # doi::10.1007/s12273-015-0245-4

        # Grab the sensitivity method
        sensitivity_type = self.sensitivity_type(case)

        [s1,cov] = self.model.SiY_key_vals(case)

        # get arr
        lst = self.model.vars_keys(case, not_cte = True)
        s1_arr = np.asarray(self.model.case[case][self.model.si_out][out_var][s1].copy())
        cov_arr = np.asarray(self.model.case[case][self.model.si_out][out_var][cov].copy())

        np.seterr(divide='ignore', invalid='ignore')
        if sensitivity_type == "PAWN":
            cov_arr1 = cov_arr
        else:
            cov_arr1 = cov_arr / s1_arr
        np.seterr(divide='warn', invalid='warn')
        s1_arr1 = np.abs(s1_arr) / np.max(np.abs(s1_arr))

        # list of order of importance
        lsto = s1_arr1.tolist()
        lst_ord= sorted(range(len(lsto)), key=lambda k: lsto[k])
        for ii in range(0,len(lst_ord)):
            lsto[lst_ord[ii]] = len(lst_ord)-ii
        s1_arr1_ord = np.asarray(lsto, dtype=type(float(0)))

        #print(s1_arr1,s1_arr1_ord, len(s1_arr1)==len(s1_arr1_ord))

        #
        indx = lst.index(var_in)

        return (s1_arr1[indx], cov_arr1[indx], s1_arr[indx], s1_arr1_ord[indx])

    def problem_definition(self, case):

        ## Builds the problem definition

        lst_vars = self.model.vars_keys(case, not_cte = True)

        num_vars = len(lst_vars)

        bounds = self.model.vars_bounds(case, lst_vars)

        dist = self.model.vars_dist(case, lst_vars)

        problem = {}
        problem['num_vars'] = num_vars
        problem['names'] = lst_vars
        problem['bounds'] = bounds
        problem['dists'] = dist

        self.model.case[case]["problem"] = problem

    ##

    def sensitivity_type(self, case):

        ## Returns the sensitivity type analysis

        sensitivity_type = self.model.case[case][self.model.sensitivity_method]

        if sensitivity_type not in self.model.v_analisis_type.keys():

            self.error("The sensitivity type is not recognize: %s" % sensitivity_type)

        return sensitivity_type

    def add_Si_inputs(self, case, Si, key):

        ##Add Si analysis to data, one per Y variables

        if not self.model.si_out in self.model.case[case].keys():
            self.model.case[case][self.model.si_out] = obj_data.objdata()

        self.model.case[case][self.model.si_out][key] = Si

    def add_doe_inputs(self, case, X):

        ## Add sampling arrays to data doe as numpy arrays

        # doe_in

        self.model.case[case][self.model.doe_in] = obj_data.objdata()

        ii = 0
        for key in self.model.case[case]["problem"]["names"]:

            self.model.case[case][self.model.doe_in][key] = X[:,ii].copy()
            ii = ii + 1

        for key in self.model.vars_keys(case, not_cte = False):

            if self.model.case[case][self.model.vars_key][key] is None:
                vval = np.zeros(X.shape[0])
                vval[:] = np.nan
            else:
                vval = np.zeros(X.shape[0]) + self.model.case[case][self.model.vars_key][key]
            self.model.case[case][self.model.doe_in][key] = vval

        # doe_out

        self.model.case[case][self.model.doe_out] = obj_data.objdata()

        for key in self.model.vars_out_keys(case):
            
            if self.model.case[case][self.model.vars_out_key][key] is None:
                vval = np.zeros(X.shape[0])
                vval[:] = np.nan
            elif not self.model.is_number(self.model.case[case][self.model.vars_out_key][key]):
                vval = np.zeros(X.shape[0])
                vval[:] = np.nan                
            else:
                vval = np.zeros(X.shape[0]) + self.model.case[case][self.model.vars_out_key][key]
            self.model.case[case][self.model.doe_out][key] = vval

    def doe_inputs_X(self, case):

        ## Return sampling arrays in the X format

        # obtain the shape of the X format array
        lst = []
        ii = 0
        for key in self.model.case[case]["problem"]["names"]:
            lst1 = []
            lst1.append(key)
            lst1.append(self.model.case[case][self.model.doe_in][key].shape[0])
            lst1.append(ii)
            ii = ii + 1
            lst.append(lst1)

        rows = None
        for ll in lst:
            if rows is None:
                rows = ll[1]
            else:
                if rows != ll[1]:
                    self.error("Error on arrays shape")
        cols = len(lst)

        # generate the X array
        X= np.empty((rows,cols))

        ii = 0
        for key in self.model.case[case]["problem"]["names"]:
            X[:,ii] = self.model.case[case][self.model.doe_in][key][:]
            ii = ii + 1

        return X

    def sampling_sobol(self, case):

        import SALib.sample.saltelli as saltelli

        ## Runs DOE build up

        problem = self.model.case[case]["problem"]
        N = self.nextPowerOf2(int(self.model.case[case][self.model.samples]))
        calc_second_order = True
        skip_values = None

        X = saltelli.sample(problem, N,
            calc_second_order = calc_second_order, skip_values=skip_values)

        self.msg("Case %s sampling value %i by Sobol method. DOEX shape %s" % (case,N,str(X.shape)))

        self.add_doe_inputs(case, X)

        return X

    def sensi_sobol(self, case):

        import SALib.analyze.sobol as sobol

        ## Runs semsitivity analysis SOBOL

        problem = self.model.case[case]["problem"]
        calc_second_order = True
        skip_values = None
        num_resamples = 100
        conf_level = self.conf_level
        print_to_console = False

        for key in self.model.vars_out_keys(case):

            Y = self.model.case[case][self.model.doe_out][key]

            Si = sobol.analyze(problem, Y, calc_second_order,
                 num_resamples=num_resamples, conf_level=conf_level, print_to_console=print_to_console,
                 parallel=False, n_processors=None, keep_resamples=False, seed=None)

            self.add_Si_inputs(case, Si, key)

            pass

    def sampling_morris(self, case):

        import SALib.sample.morris as morris

        ## Runs DOE build up

        problem = self.model.case[case]["problem"]
        N = self.nextPowerOf2(int(self.model.case[case][self.model.samples]))
        num_levels = 4

        X = morris.sample(problem, N, num_levels = num_levels,
            optimal_trajectories = None, local_optimization = True, seed = None)

        self.msg("Case %s sampling value %i by Morris method. DOEX shape %s" % (case,N,str(X.shape)))

        self.add_doe_inputs(case, X)

        return X

    def sensi_morris(self, case):

        import SALib.analyze.morris as morris

        ## Runs semsitivity analysis Morris

        problem = self.model.case[case]["problem"]
        num_resamples = 100
        conf_level = self.conf_level
        print_to_console = False
        num_levels = 4

        for key in self.model.vars_out_keys(case):

            X = self.doe_inputs_X(case)
            Y = self.model.case[case][self.model.doe_out][key]

            Si = morris.analyze(problem, X, Y, num_resamples = num_resamples,
                conf_level = conf_level, print_to_console = print_to_console,
                num_levels = num_levels, seed=None)

            self.add_Si_inputs(case, Si, key)

            pass

    def sampling_RBD_Fast(self, case):

        import SALib.sample.latin as latin

        ## Runs DOE build up

        problem = self.model.case[case]["problem"]
        N = self.nextPowerOf2(int(self.model.case[case][self.model.samples]))

        X = latin.sample(problem, N, seed = None)

        self.msg("Case %s sampling value %i by latin Hypercube for RBD-FAST. DOEX shape %s" % (case,N,str(X.shape)))

        self.add_doe_inputs(case, X)

        return X

    def sensi_RBD_Fast(self, case):

        import SALib.analyze.rbd_fast as rbd_fast

        ## Runs semsitivity analysis RBd_Fast

        problem = self.model.case[case]["problem"]
        M = 10
        num_resamples = 100
        conf_level = self.conf_level
        print_to_console = False

        for key in self.model.vars_out_keys(case):

            X = self.doe_inputs_X(case)
            Y = self.model.case[case][self.model.doe_out][key]

            Si = rbd_fast.analyze(problem, X, Y, M = M, num_resamples = num_resamples,
                conf_level = conf_level, print_to_console = print_to_console, seed=None)

            self.add_Si_inputs(case, Si, key)

            pass

    def sampling_Fast(self, case):

        import SALib.sample.fast_sampler as fast_sampler

        ## Runs DOE build up

        problem = self.model.case[case]["problem"]
        N = self.nextPowerOf2(int(self.model.case[case][self.model.samples]))
        M = 4

        X = fast_sampler.sample(problem, N, M = M, seed = None)

        self.msg("Case %s sampling value %i by latin Hypercube for FAST. DOEX shape %s" % (case,N,str(X.shape)))

        self.add_doe_inputs(case, X)

        return X

    def sensi_Fast(self, case):

        import SALib.analyze.fast as fast

        ## Runs semsitivity analysis Fast

        problem = self.model.case[case]["problem"]
        M = 4
        num_resamples = 100
        conf_level = self.conf_level
        print_to_console = False

        for key in self.model.vars_out_keys(case):

            X = self.doe_inputs_X(case)
            Y = self.model.case[case][self.model.doe_out][key]

            Si = fast.analyze(problem, Y, M=M, num_resamples=num_resamples, conf_level=conf_level,
                print_to_console=print_to_console, seed=None)

            self.add_Si_inputs(case, Si, key)

            pass

    def sampling_delta(self, case):

        import SALib.sample.latin as latin

        ## Runs DOE build up

        problem = self.model.case[case]["problem"]
        N = self.nextPowerOf2(int(self.model.case[case][self.model.samples]))

        X = latin.sample(problem, N, seed = None)

        self.msg("Case %s sampling value %i by latin Hypercube for Delta-MIM. DOEX shape %s" % (case,N,str(X.shape)))

        self.add_doe_inputs(case, X)

        return X

    def sensi_delta(self, case):

        import SALib.analyze.delta as delta

        ## Runs semsitivity analysis delta

        problem = self.model.case[case]["problem"]
        M = 4
        num_resamples = 100
        conf_level = self.conf_level
        print_to_console = False

        for key in self.model.vars_out_keys(case):

            X = self.doe_inputs_X(case)
            Y = self.model.case[case][self.model.doe_out][key]

            Si = delta.analyze(problem, X, Y, num_resamples=num_resamples, conf_level=conf_level,
                 print_to_console=print_to_console, seed=None)

            self.add_Si_inputs(case, Si, key)

            pass

    def sampling_dgsm(self, case):

        import SALib.sample.finite_diff as finite_diff

        ## Runs DOE build up

        problem = self.model.case[case]["problem"]
        N = self.nextPowerOf2(int(self.model.case[case][self.model.samples]))
        delta = 0.01
        skip_values = 1024

        X = finite_diff.sample(problem, N, delta = delta, seed = None, skip_values = skip_values)

        self.msg("Case %s sampling value %i by DGSM. DOEX shape %s" % (case,N,str(X.shape)))

        self.add_doe_inputs(case, X)

        return X

    def sensi_dgsm(self, case):

        import SALib.analyze.dgsm as dgsm

        ## Runs semsitivity analysis dgsm

        problem = self.model.case[case]["problem"]
        M = 4
        num_resamples = 100
        conf_level = self.conf_level
        print_to_console = False

        for key in self.model.vars_out_keys(case):

            X = self.doe_inputs_X(case)
            Y = self.model.case[case][self.model.doe_out][key]

            Si = dgsm.analyze(problem, X, Y, num_resamples=num_resamples, conf_level=conf_level,
                 print_to_console=print_to_console, seed=None)

            self.add_Si_inputs(case, Si, key)

            pass

    def sampling_factorial(self, case):

        import SALib.sample.ff as ff

        ## Runs DOE build up

        problem = self.model.case[case]["problem"]
        N = self.nextPowerOf2(int(self.model.case[case][self.model.samples]))

        X = ff.sample(problem, seed = None)

        self.msg("Case %s sampling value %i by Factorial. DOEX shape %s" % (case,N,str(X.shape)))

        self.add_doe_inputs(case, X)

        return X

    def sensi_factorial(self, case):

        import SALib.analyze.ff as ffa

        ## Runs semsitivity analysis factorial

        problem = self.model.case[case]["problem"]
        second_order = False
        print_to_console = False

        for key in self.model.vars_out_keys(case):

            X = self.doe_inputs_X(case)
            Y = self.model.case[case][self.model.doe_out][key]

            Si = ffa.analyze(problem, X, Y, second_order=second_order,
                 print_to_console=print_to_console, seed=None)

            self.add_Si_inputs(case, Si, key)

            pass

    def sampling_PAWN(self, case):

        import SALib.sample.latin as latin

        ## Runs DOE build up

        problem = self.model.case[case]["problem"]
        N = self.nextPowerOf2(int(self.model.case[case][self.model.samples]))

        X = latin.sample(problem, N, seed = None)

        self.msg("Case %s sampling value %i by latin Hypercube for PAWN. DOEX shape %s" % (case,N,str(X.shape)))

        self.add_doe_inputs(case, X)

        return X

    def sensi_PAWN(self, case):

        import SALib.analyze.pawn as pawn

        ## Runs semsitivity analysis pawn

        problem = self.model.case[case]["problem"]
        S = 10
        print_to_console = False

        for key in self.model.vars_out_keys(case):

            X = self.doe_inputs_X(case)
            Y = self.model.case[case][self.model.doe_out][key]

            Si = pawn.analyze(problem, X, Y, S = S,
                 print_to_console=print_to_console, seed=None)

            self.add_Si_inputs(case, Si, key)

            pass

    def sampling_hdmr(self, case):

        import SALib.sample.latin as latin

        ## Runs DOE build up

        problem = self.model.case[case]["problem"]
        N = self.nextPowerOf2(int(self.model.case[case][self.model.samples]))

        X = latin.sample(problem, N, seed = None)

        self.msg("Case %s sampling value %i by latin Hypercube for HDMR. DOEX shape %s" % (case,N,str(X.shape)))

        self.add_doe_inputs(case, X)

        return X

    def sensi_hdmr(self, case):

        import SALib.analyze.hdmr as hdmr

        ## Runs semsitivity analysis pawn

        problem = self.model.case[case]["problem"]
        maxorder = 2
        maxiter = 100
        m = 2
        K = 20
        print_to_console = False

        for key in self.model.vars_out_keys(case):

            X = self.doe_inputs_X(case)
            Y = self.model.case[case][self.model.doe_out][key]

            Si = hdmr.analyze(problem, X, Y, maxorder = maxorder, maxiter = maxiter, m = m, K = K,
                R = None, alpha = 0.95, lambdax = 0.01,
                print_to_console = print_to_console, seed = None)

            self.add_Si_inputs(case, Si, key)

            pass

    #### Other functions
    #####################################

    def error(self, msg):
        print("Error: " + msg)
        if self.model.objlog:
            self.model.objlog.warning("ID04 - %s" % msg)          
        raise ValueError(msg)

    def msg(self, msg):
        print("Msg: " + msg)
        if self.model.objlog:
            self.model.objlog.info("ID04 - %s" % msg)           

    def nextPowerOf2(self, n):

        # Finds next power of two
        # for n. If n itself is a
        # power of two then returns n

        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n += 1

        return n
