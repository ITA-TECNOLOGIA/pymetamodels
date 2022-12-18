#!/usr/bin/env python3

import os, sys, random, warnings
import numpy as np
import gc
import time
import pickle
import itertools
import math

import pymetamodels.obj_logging as obj_log

import scipy.optimize as scp_opt

class objoptimization(object):

    """Python class representing the optimization calculation

        :platform: Windows
        :synopsis: object optimization calculation

        :Dependences: numpy, scipy

        :ivar version: optimization model version
        :ivar tol: (default 1e-6) optimization methods tolerance value
        :ivar rel_tol_val_grid_methods: (default 5e-4) optimization grid methods tolerance value
        :ivar max_size_grid_methods: (default 5e-3) optimization grid methods max size of the grid
        :ivar tolerance_check_bounds_constrains: (default 1e-3) tolerance to check bounds and contrains limits

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and for example :ref:`tutorial A06 <ex_kuwase_optimize>`

        |

    """

    ## Common functions
    ###################

    def __init__(self, logging_path = None):

        self.version = 0
        self.pickle_protocol = 3
        self.file_extension = ".optita"

        self.logging_path = logging_path
        if logging_path:
            self.objlog = obj_log.objlogging(self.logging_path)
        else:
            self.objlog = None        

        self.verbose_testing = False
        self.warnings = False

        ## Options
        self.tol = 1e-6
        self.max_size_grid_methods = 5e3
        self.rel_tol_val_grid_methods = 5e-4
        self.tolerance_check_bounds_constrains = 1e-3

        ## Vars
        self._min_fun = None
        self._ini_guess_x = None
        self._min_fun_ls_model = None
        self._doeY_guess = None

        self._min_var = None
        self._type_op_min = None
        self._doeX_np_varX = None
        self._doeY_np_varY = None

    def _data_obj(self):

        obj = {}

        obj["version"] = self.version
        obj["pickle_protocol"] = self.pickle_protocol
        obj["file_extension"] = self.file_extension

        obj["verbose_testing"] = self.verbose_testing
        obj["warnings"] = self.warnings

        obj["tol"] = self.tol
        obj["max_size_grid_methods"] = self.max_size_grid_methods
        obj["rel_tol_val_grid_methods"] = self.rel_tol_val_grid_methods
        obj["tolerance_check_bounds_constrains"] = self.tolerance_check_bounds_constrains

        obj["_min_fun"] = self._min_fun
        obj["_ini_guess_x"] = self._ini_guess_x
        obj["_min_fun_ls_model"] = self._min_fun_ls_model
        obj["_doeY_guess"] = self._doeY_guess

        obj["_min_var"] = self._min_var
        obj["_type_op_min"] = self._type_op_min
        obj["_doeX_np_varX"] = self._doeX_np_varX
        obj["_doeY_np_varY"] = self._doeY_np_varY

        return obj

    def _load_data_obj(self, data_obj):

        if data_obj["version"] == 0:

            self.version = data_obj["version"]
            self.pickle_protocol = data_obj["pickle_protocol"]
            self.file_extension = data_obj["file_extension"]

            self.verbose_testing = data_obj["verbose_testing"]
            self.warnings = data_obj["warnings"]

            ## Options
            self.tol = data_obj["tol"]
            self.max_size_grid_methods = data_obj["max_size_grid_methods"]
            self.rel_tol_val_grid_methods = data_obj["rel_tol_val_grid_methods"]
            self.tolerance_check_bounds_constrains = data_obj["tolerance_check_bounds_constrains"]

            ## Vars
            self._min_fun = data_obj["_min_fun"]
            self._ini_guess_x = data_obj["_ini_guess_x"]
            self._min_fun_ls_model = data_obj["_min_fun_ls_model"]
            self._doeY_guess = data_obj["_doeY_guess"]

            self._min_var = data_obj["_min_var"]
            self._type_op_min = data_obj["_type_op_min"]
            self._doeX_np_varX = data_obj["_doeX_np_varX"]
            self._doeY_np_varY = data_obj["_doeY_np_varY"]

        else:

            self.error("Version %i unknown" % data_obj["version"])

    @property
    def min_fun(self):

        """
        Min value of objective function found

        :getter: Returns Min value of objective function found
        :type: float

        |

        """

        return self._min_fun

    @property
    def DOEX_min_func(self):

        """
        DOEX array of Min value of objective function

        :getter: Returns DOEX array of Min value of objective function
        :type: array

        |

        """

        return self._ini_guess_x

    @property
    def min_func_model(self):

        """
        Optimization model use for Min value of objective function

        :getter: Returns Optimization model
        :type: string

        |

        """

        return self._min_fun_ls_model

    @property
    def DOEY_min_func(self):

        """
        DOEY array of Min value of objective function

        :getter: Returns DOEY array of Min value of objective function
        :type: array

        |

        """

        return self._doeY_guess

    @property
    def doeX_varX(self):

        """
        List of var names corresponding to the optimized doeX

        :getter: Returns list of var names corresponding to the optimized doeX
        :type: tuple

        |

        """

        return self._doeX_np_varX

    @property
    def doeY_varY(self):

        """
        List of var names corresponding to the optimized doeY

        :getter: Returns list of var names corresponding to the optimized doeY
        :type: tuple

        |

        """

        return self._doeY_np_varY

    def doeY_index(self, varY_name):

        ## Returns the index in the doeY array for a varY name

        varsY = self.doeY_varY

        ii = 0
        for varY in varsY:

            if varY == varY_name:
                return ii

            ii = ii + 1

        return None

    def doeX_index(self, varX_name):

        ## Returns the index in the doeY array for a varX name

        varsX = self.doeX_varX

        ii = 0
        for varX in varsX:

            if varX == varX_name:
                return ii

            ii = ii + 1

        return None

    ###########################################################
    ## Optimization routines

    def save_to_file(self, folder, file_name):

        """

        .. _objoptimization_save_to_file:

        **Synopsis:**
            * Save the optimization data to a file with .optita extension

        **Args:**
            * folder: folder path
            * file_name: file name

        **Optional parameters:**
            * None

        **Returns:**
            * The file path

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`

        |

        """

        file_path = os.path.join(folder, file_name + self.file_extension)

        with open(file_path, 'wb') as f:

            pickle.dump(self._data_obj(), f, protocol = self.pickle_protocol)

        return file_path

    def load_file(self, file_path):

        """

        .. _objoptimization_load_file:

        **Synopsis:**
            * Loads the optimization data to a file with .optita extension

        **Args:**
            * file_path: path to the file

        **Optional parameters:**
            * None

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`

        |

        """

        with open(file_path, 'rb') as f:

            data_obj = pickle.load(f)

        self._load_data_obj(data_obj)

    def run_optimization(self, obj_metamodel, min_vars, type_op_min, bounds, constrains_vars = None, scheme = None):

        """

        .. _objoptimization_run_optimization:

        **Synopsis:**
            * Execute the optimization routines to minimize a DOEY metamodel variable

        **Args:**
            * obj_metamodel: metamodel object
            * min_var: variable to be minimize
            * type_op_min: type variable to be minimize 1: minimize 2:equal to zero
            * bounds: DOEX variables bounds

        **Optional parameters:**
            * constrains_vars = None:
            * scheme=None: scheme method to be applied. The availables schemes are: "general", "general_fast", "general_with_constrains", "global", "minimize", "grid_method", "iter_grid_method"

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A06 <ex_kuwase_optimize>`
            * The kind of sampling routines available specified metamodel configuration spreadsheet. :ref:`See metamodel configuration spreadsheet <pymetamodels_configuration>`

        |

        """

        if len(min_vars) == 0:
            self.error("No DOEY variables has been declare as op_min in the configuration sheet")
        elif len(min_vars) > 1:
            self.error("More than one DOEY variable has been declare as op_min in the configuration sheet. Vars: ", min_vars)

        return self._opt_model(obj_metamodel, min_vars, type_op_min, bounds = bounds, constrains_vars = constrains_vars, scheme = scheme)

    def _opt_model(self, obj_metamodel, min_vars, type_op_min, bounds, constrains_vars = None, scheme = None):

        ## Optimize the model

        # Choose scheme
        lst_models = self.type_scheme(scheme)

        # Iterate models
        resume = {}
        _time_ini_0 = time.process_time()
        _ini_guess_x = None
        _min_fun = None
        _min_fun_ls_model = None

        self.msg("Start optimization scheme %s" % (scheme), type = 10)

        for ls_model in lst_models:

            _time_ini = time.process_time()

            (result_x , result_fun) = self._opt_model_call(ls_model, obj_metamodel, min_vars[0], type_op_min, bounds = bounds, constrains_vars = constrains_vars, ini_guess_x = _ini_guess_x)

            _elapsed_time = time.process_time() - _time_ini

            # check bounds
            if result_x is not None:
                if not self.check_bounds(bounds, result_x, tolerance = self.tolerance_check_bounds_constrains):
                    self.msg("Bounds: %s" % bounds, type = 10)
                    self.msg("DoeY: %s" % obj_metamodel.predict(np.asarray([result_x.tolist()])), type = 10)
                    self.msg("Out of bounds: %s model. Min: %.4f. Result: %s" % (ls_model, result_fun, result_x), type = 10)
                    (result_x, result_fun) = (None, None)

            # check constraints
            if result_x is not None:
                if not self.check_constraints(constrains_vars, obj_metamodel, result_x, tolerance = self.tolerance_check_bounds_constrains):
                    self.msg("Constrains: %s" % constrains_vars, type = 10)
                    self.msg("DoeY: %s" % obj_metamodel.predict(np.asarray([result_x.tolist()])), type = 10)
                    self.msg("Out of constrains: %s model. Min: %.4f. Result: %s" % (ls_model, result_fun, result_x), type = 10)
                    (result_x, result_fun) = (None, None)

            # populate resume dict
            if result_x is not None:
                key = ls_model
                resume[key] = {}
                resume[key]["result_x"] = result_x
                resume[key]["result_fun"] = result_fun
                resume[key]["elapsed_time"] = _elapsed_time
                self.msg("Opt method: %s. Min: %.4f Time elapsed %.2f [s]. Result: %s" % (ls_model, result_fun, _elapsed_time, result_x), type = 10)
                self.msg("DoeY: %s" % obj_metamodel.predict(np.asarray([result_x.tolist()])), type = 10)

                # find best one
                if _min_fun is None:
                    _min_fun = result_fun
                    _min_fun_ls_model = key
                    _ini_guess_x = result_x
                else:
                    if _min_fun > result_fun:
                        _min_fun = result_fun
                        _min_fun_ls_model = key
                        _ini_guess_x = result_x
                        self.msg("Updated best guess with %s model. Min: %.4f. Result: %s" % (ls_model, result_fun, result_x), type = 10)

        _elapsed_time = time.process_time() - _time_ini_0

        # Return
        if _min_fun is None:
            self._min_fun = None
            self._ini_guess_x = None
            self._min_fun_ls_model = None
            self._doeY_guess = None

            self._min_var = min_vars[0]
            self._type_op_min = type_op_min
            self._doeX_np_varX = obj_metamodel.doeX_varX
            self._doeY_np_varY = obj_metamodel.doeY_varY

            self.msg("Solution for the optimization problem was not found for the given objective and constrains")
            return None, None

        self.msg("Min var: %s.Best opt method: %s. Min: %.4f Time elapsed %.2f [s]. Result: %s" % (min_vars, _min_fun_ls_model, _min_fun, _elapsed_time, _ini_guess_x))
        _doeY_guess = obj_metamodel.predict(np.asarray([_ini_guess_x.tolist()]))
        self.msg("DoeY: %s" % _doeY_guess)

        # Sove to object
        self._min_fun = _min_fun
        self._ini_guess_x = _ini_guess_x
        self._min_fun_ls_model = _min_fun_ls_model
        self._doeY_guess = _doeY_guess

        self._min_var = min_vars[0]
        self._type_op_min = type_op_min
        self._doeX_np_varX = obj_metamodel.doeX_varX
        self._doeY_np_varY = obj_metamodel.doeY_varY

        return _min_fun, _ini_guess_x

    def type_scheme(self, scheme):

        ## Builds the type of schemes

        if scheme is None or scheme == "general":
            lst_models = ["iter_grid_method", "shgo", "shgo_slow", "diff_evol", "min_gen", "Powell", "Nelder-Mead", "TNC", "COBYLA", "SLSQP"]
        elif scheme == "general_fast":
            lst_models = ["iter_grid_method", "shgo", "diff_evol", "min_gen", "Powell", "Nelder-Mead", "TNC", "COBYLA", "SLSQP"]
        elif scheme == "general_with_constrains":
            lst_models = ["iter_grid_method", "COBYLA", "SLSQP", "shgo"]
        elif scheme == "global":
            lst_models = ["iter_grid_method", "shgo", "shgo_slow", "diff_evol"]
        elif scheme == "minimize":
            lst_models = ["min_gen", "Powell", "Nelder-Mead", "TNC", "COBYLA", "SLSQP"]
        elif scheme == "grid_method":
            lst_models = ["grid_method"]
        elif scheme == "iter_grid_method":
            lst_models = ["iter_grid_method"]
        else:
            lst_models = self.type_scheme("general_fast")

        return lst_models

    def _opt_model_call(self, ls_model, obj_metamodel, min_var, type_op_min, bounds = None, constrains_vars = (), ini_guess_x = None):

        ## Resolve optimization according SHGO

        # Build the objective
        def _func_obj_metamodel(*args, **kwargs):
            # def _func_obj_metamodel1(x, min_var, type_op_min, _obj_metamodel):

            def _function_template(*args, **kwargs):
                # def _function_template1(x, min_var, type_op_min, _obj_metamodel):

                #return _obj_metamodel.predict_1D(min_var, x)
                if args[2] == 1:
                    return args[3].predict_1D(args[1], args[0])
                elif args[2] == 2:
                    return abs(args[3].predict_1D(args[1], args[0]))
                else:
                    error_value_type_op_min

            return _function_template

        func_obj_metamodel = _func_obj_metamodel()

        # Initialize minimize functions
        if ini_guess_x is None:
            #x0 = self._bounds_mid_point(bounds)
            result = self.grid_method(obj_metamodel, min_var, type_op_min, bounds, constraints=constrains_vars, max_size = 5e3)
            x0 = result[0]
        else:
            x0 = ini_guess_x

        # Ini contrains
        if (constrains_vars is not None) and (constrains_vars != ()):
            ii = 0
            for con in constrains_vars:
                constrains_vars[ii]["fun"] = func_obj_metamodel
                constrains_vars[ii]["args"][2] = obj_metamodel
                ii = ii + 1
            with_constraints = True
            self.tol = 1e-7
        else:
            with_constraints = False

        ## Selector opt models
        with warnings.catch_warnings():
            if not self.warnings: warnings.filterwarnings("ignore")

            if ls_model == "brute":
                # Not constraints
                if with_constraints is True: return None, None
                result = scp_opt.brute(func_obj_metamodel, bounds, args=(min_var, type_op_min, obj_metamodel), full_output=True, finish=scp_opt.fmin)
                (result_x , result_fun) = (result[0], result[1])

            if ls_model == "shgo":
                result = scp_opt.shgo(func_obj_metamodel, bounds, args=(min_var, type_op_min, obj_metamodel), constraints=constrains_vars, n=150, sampling_method='simplicial')
                (result_x , result_fun) = (result.x, result.fun)

            if ls_model == "shgo_slow":
                result = scp_opt.shgo(func_obj_metamodel, bounds, args=(min_var, type_op_min, obj_metamodel), constraints=constrains_vars, iters=3, sampling_method='simplicial')
                (result_x , result_fun) = (result.x, result.fun)

            if ls_model == "diff_evol":
                # Not constraints
                if with_constraints is True: return None, None
                result = scp_opt.differential_evolution(func_obj_metamodel, bounds, args=(min_var, type_op_min, obj_metamodel), constraints=(), seed=1)
                (result_x , result_fun) = (result.x, result.fun)

            if ls_model == "grid_method":
                result = self.grid_method(obj_metamodel, min_var, type_op_min, bounds, constraints=constrains_vars, max_size = self.max_size_grid_methods)
                (result_x , result_fun) = (result[0], result[1])

            if ls_model == "iter_grid_method":
                result = self.iter_grid_method(obj_metamodel, min_var, type_op_min, bounds, constraints=constrains_vars, max_size = self.max_size_grid_methods, rel_tol_val = self.rel_tol_val_grid_methods, iter_max = 10)
                (result_x , result_fun) = (result[0], result[1])

            if ls_model == "min_gen":
                # Not constraints
                if with_constraints is True: return None, None
                result = scp_opt.minimize(func_obj_metamodel, x0, args=(min_var, type_op_min, obj_metamodel), method=None, bounds=bounds, constraints=constrains_vars, tol=self.tol, callback=None, options=None)
                (result_x , result_fun) = (result.x, result.fun)

            if ls_model == "Nelder-Mead":
                # Not constraints
                if with_constraints is True: return None, None
                result = scp_opt.minimize(func_obj_metamodel, x0, args=(min_var, type_op_min, obj_metamodel), method='Nelder-Mead', bounds=bounds, constraints=constrains_vars, tol=self.tol, callback=None, options=None)
                (result_x , result_fun) = (result.x, result.fun)

            if ls_model == "Powell":
                # Not constraints
                if with_constraints is True: return None, None
                result = scp_opt.minimize(func_obj_metamodel, x0, args=(min_var, type_op_min, obj_metamodel), method='Powell', bounds=bounds, constraints=constrains_vars, tol=self.tol, callback=None, options=None)
                (result_x , result_fun) = (result.x, result.fun)

            if ls_model == "TNC":
                # Not constraints
                if with_constraints is True: return None, None
                result = scp_opt.minimize(func_obj_metamodel, x0, args=(min_var, type_op_min, obj_metamodel), method='TNC', bounds=bounds, constraints=constrains_vars, tol=self.tol, callback=None, options=None)
                (result_x , result_fun) = (result.x, result.fun)

            if ls_model == "COBYLA":
                # does not handle bounds
                result = scp_opt.minimize(func_obj_metamodel, x0, args=(min_var, type_op_min, obj_metamodel), method='COBYLA', bounds=bounds, constraints=constrains_vars, tol=self.tol, callback=None, options=None)
                (result_x , result_fun) = (result.x, result.fun)

            if ls_model == "SLSQP":
                result = scp_opt.minimize(func_obj_metamodel, x0, args=(min_var, type_op_min, obj_metamodel), method='SLSQP', bounds=bounds, constraints=constrains_vars, tol=self.tol, callback=None, options=None)
                (result_x , result_fun) = (result.x, result.fun)

        return result_x , result_fun

    def _bounds_mid_point(self, bounds):

        ## Mid-point bounds point
        out = []
        for ele in bounds:
            val = (ele[1]-ele[0])+ele[0]
            out.append(val)

        return out

    def check_bounds(self, bounds, result_x, tolerance = 1e-5):

        # Check bounds
        ii = 0
        for ele in bounds:

            if result_x[ii] < ele[0]-tolerance: return False
            if result_x[ii] > ele[1]+tolerance: return False

            ii = ii + 1

        return True

    def check_constraints(self, constraints, obj_metamodel, result_x, tolerance = 1e-5):

        # Check constraints
        if (constraints is not None) and (constraints != ()):
            doeY_np_shape = obj_metamodel.doeY_np_shape
            doeY = obj_metamodel.predict(np.asarray([result_x.tolist()]))[0]

            for cons in constraints:

                min_var_cons = obj_metamodel.doeY_index(cons["args"][0])

                if cons["type"] == 'ineq':
                    if doeY[min_var_cons] < 0. - tolerance: return False
                elif cons["type"] == 'eq':
                    if doeY[min_var_cons] < 0. - tolerance: return False
                    if doeY[min_var_cons] > 0. + tolerance: return False

            return True
        else:
            return True

    ###########################################################
    ## Internal methods

    def iter_grid_method(self, obj_metamodel, min_var, type_op_min, bounds, constraints=None, max_size = 5e3, rel_tol_val = 5e-3, iter_max = 10):

        ## Minimization base on grid division (iterative MOPs)
        iter_max = iter_max
        tol_val = rel_tol_val

        # initialize
        (best_x_val, best_fun_val, lst_min_distance) = self.grid_method(obj_metamodel, min_var, type_op_min, bounds, constraints=constraints, max_size = max_size*1.5, return_lst_min_distance = True)

        if best_fun_val is None: return None, None

        iter= 0
        tolerance = None
        ref_best_fun_val = best_fun_val

        # loop
        while True:

            new_bounds = []
            ii = 0
            for ele in best_x_val:
                new_bounds.append([ele-lst_min_distance[ii],ele+lst_min_distance[ii]])
                ii = ii + 1

            (best_x_val_old, best_fun_val_old) = (best_x_val, best_fun_val)

            (best_x_val, best_fun_val, lst_min_distance) = self.grid_method(obj_metamodel, min_var, type_op_min, new_bounds, constraints=constraints, max_size = max_size, return_lst_min_distance = True)

            if best_fun_val is None:
                self.msg("Method iter_grid_method end before max iteration", type = 10)
                return best_x_val_old, best_fun_val_old
                #return None, None
                pass

            # check tolerance
            tolerance = abs(best_fun_val-ref_best_fun_val) #/ np.min([abs(best_fun_val),abs(ref_best_fun_val)])

            if tolerance < tol_val:

                break
            else:
                ref_best_fun_val = best_fun_val

            iter += 1
            if iter > iter_max: break

        return best_x_val, best_fun_val

    def grid_method(self, obj_metamodel, min_var, type_op_min, bounds, constraints=None, max_size = 1e4, return_lst_min_distance = False):

        ## Minimization base on grid division (adaptative MOPs), one pass
        x_val, fun_val = None, None

        # Build cartesina product DOEX
        len_doeX = len(bounds)
        num = int((max_size)**(1./len_doeX))

        tup = ()
        lst_min_distance = []
        for bd in bounds:
            arr = np.linspace(bd[0],bd[1],num=num)
            tup = tup + (arr,)
            lst_min_distance.append(arr[1]-arr[0])

        vector = list(itertools.product(*tup,repeat=1))
        vector = np.asarray(vector, dtype=np.dtype('float64'))

        doeY = obj_metamodel.predict(vector)

        # apply constraints
        if (constraints is not None) and (constraints != ()):
            doeY_np_shape = obj_metamodel.doeY_np_shape

            for cons in constraints:

                min_var_cons = obj_metamodel.doeY_index(cons["args"][0])

                if cons["type"] == 'ineq':
                    mask = np.ma.masked_less_equal(doeY[:,min_var_cons], 0.)
                    mask = mask.mask
                    if len(mask.shape) == 0: ## Not available data with that constrain
                        if return_lst_min_distance:
                            return None, None, lst_min_distance
                        else:
                            return None, None
                    else:
                        mask = np.column_stack(tuple([mask]*doeY_np_shape[1]))
                        doeY = np.ma.array(doeY, mask = mask)
                elif cons["type"] == 'eq':
                    mask = np.ma.masked_not_equal(doeY[:,min_var_cons], 0.)
                    mask = mask.mask
                    if len(mask.shape) == 0: ## Not available data with that constrain
                        if return_lst_min_distance:
                            return None, None, lst_min_distance
                        else:
                            return None, None
                    else:
                        mask = np.column_stack(tuple([mask]*doeY_np_shape[1]))
                        doeY = np.ma.array(doeY, mask = mask)

        # find min
        doeY_np_shape = obj_metamodel.doeY_np_shape
        min_var_ii = obj_metamodel.doeY_index(min_var)

        if type_op_min == 1:
            if len(doeY_np_shape) == 1:
                row_min = np.argmin(doeY[:])
            else:
                row_min = np.argmin(doeY[:,min_var_ii])
        elif type_op_min == 2:
            if len(doeY_np_shape) == 1:
                row_min = np.argmin(np.absolute(doeY[:]))
            else:
                row_min = np.argmin(np.absolute(doeY[:,min_var_ii]))
        else:
            self.error("Wrong type_op_min value")

        x_val = vector[row_min,:]
        if len(doeY_np_shape) == 1:
            fun_val = doeY[row_min]
        else:
            fun_val = doeY[row_min,min_var_ii]

        if type(fun_val) is np.ma.core.MaskedConstant:
            if return_lst_min_distance:
                return None, None, lst_min_distance
            else:
                return None, None
        else:
            if return_lst_min_distance:
                return x_val, float(fun_val), lst_min_distance
            else:
                return x_val, float(fun_val)

    ###########################################################
    ## Other functions

    def error(self, msg):
        print("Error: " + msg)
        if self.objlog:
            self.objlog.warning("ID06 - %s" % msg)            
        raise ValueError(msg)

    def msg(self, msg, type = 0):

        if type == 0:
            print("Msg: " + msg)
        elif type == 10:
            if self.verbose_testing:
                print("Msg: " + msg)
            pass
        else:
            print("Msg: " + msg)
        if self.objlog:
            self.objlog.info("ID06 - %s" % msg)              
