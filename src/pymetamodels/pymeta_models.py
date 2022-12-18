#!/usr/bin/env python3

import os, sys
import xlrd, xlwt
import numpy as np
import pymetamodels.obj_data as obj_data
import pymetamodels.obj_plots as obj_plots
import pymetamodels.obj_conf as obj_conf
import pymetamodels.obj_metamodel as obj_metamodel
import pymetamodels.obj_optimization as obj_optimization
import pymetamodels.obj_sampling_sensitivity as obj_samp
import pymetamodels.obj_logging as obj_log
from xlutils.copy import copy as xlcopy
import csv


#### Initial functions
#####################################

def load():

    """

    .. _pymetamodels_load:

    **Synopsis:**
        * Returns an empty metamodel() object

    **Args:**
        * None

    **Optional parameters:**
        * None

    **Returns:**
        * Returns an empty metamodel() object

    .. note::

        * Is equivalent to :code:`mita = pymetamodels.metamodel()`

    |

    """    

    return metamodel()

def newconf():

    """

    .. _pymetamodels_objconf_new:

    **Synopsis:**
        * Returns an empty objconf() object

    **Args:**
        * None

    **Optional parameters:**
        * None

    **Returns:**
        * Returns a new objconf() object

    .. note::

        * Is equivalent to :code:`mita = pymetamodels.metamodel().objconf()`

    |

    """  

    return metamodel().objconf()  

#### Main class
#####################################

class metamodel(object):

    """Python class to perform design of experiments, analysis, metamodels and optimizations

        :platform: Windows
        :synopsis: object definition of optimization variables, read and save from Excel files, sensitivity analysis

        :Dependences: numpy, SALib, sklearn, scipy, xlrd, xlwt

        :ivar case: global data object case
        :ivar plt: object plot

        |

    """

    def __init__(self):

        ## ctes
        # sheet cases
        self.case_tit = "case"
        self.vars_sheet = "vars sheet"
        self.output_sheet = "output sheet"
        self.samples = "samples"
        self.sensitivity_method = "sensitivity method"
        self.comment = "comment"
        
        self.variable = "variable"
        self.value = "value"
        self.min_bound = "min"
        self.max_bound = "max"        
        self.distribution = "distribution"
        self.is_variable = "is_range"
        self.cov_un = "cov_un"
        self.ud = "ud"
        self.alias = "alias"        

        self.as_array = "array"
        self.op_min = "op_min"
        self.op_min_eq = "op_min=0"
        self.op_ineq = "ineq_(>=0)"
        self.op_eq = "eq_(=0)"     

        self.vars_key = "vars"
        self.vars_out_key = "vars_out"
        self.op_key = "op_"
        self.doe_name = "DOE"
        self.doe_in = "%s_X" % self.doe_name
        self.doe_out = "%s_Y" % self.doe_name
        self.si_out = "Si_Y"

        self.objmetamodel = "objmeta"
        self.objoptimization = "objopti"

        # sensitivity analysis
        self.si_anly_case = "case"
        self.si_anly_si = "S_n"
        self.si_anly_cov = "COV_{S_n}"
        self.si_anly_s1 = "S_1"
        self.si_anly_sio = "O_n"

        # vars
        self.case = obj_data.objdata()
        self.plt = obj_plots.objplots(self)
        self.conf = obj_conf.objconf(self)
        self.samp = obj_samp.objsamplingsensitivity(self)        

        self.v_analisis_type = self.samp.ini_analisis_type()

        self.logging_path = None
        self.objlog = None

    @property
    def version(self):
        
        import pymetamodels as obj

        return obj.__version__

    def logging_start(self, logging_path):

        """

        .. _pymetamodels_logging_start:

        **Synopsis:**
            * Initialize the logging to a external file named as logging_pymetamodels.log

        **Args:**
            * logging_path: the output folder path to the logging 

        **Optional parameters:**
            * None

        **Returns:**
            * If the path exists

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`

        |

        """

        if os.path.exists(logging_path):

            self.logging_path = logging_path
            self.objlog = obj_log.objlogging(self.logging_path)
            return True

        else:

            return False

    def keys(self):

        """

        .. _pymetamodels_keys:

        **Synopsis:**
            * Return the list of names and id cases

        **Args:**
            * None

        **Optional parameters:**
            * None

        **Returns:**
            * List of names and id cases

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`

        |

        """

        return self.case.keys()

    def objconf(self):

        """

        .. _pymetamodels_objconf_load:

        **Synopsis:**
            * Return the objconf object for programatically build the configuration spreadsheet

        **Args:**
            * None

        **Optional parameters:**
            * None

        **Returns:**
            * objconf object

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * See :ref:`objconf <pymetamodels_objconf>`

        |

        """

        return self.conf

    #### Sampling, sensitivity
    #####################################

    def obj_samplig_sensi(self):

        """

        .. _pymetamodels_obj_samplig_sensi:

        **Synopsis:**
            * Return the objsamplingsensitivity object

        **Args:**
            * None

        **Optional parameters:**
            * None

        **Returns:**
            * objsamplingsensitivity object

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * See :ref:`obj_samplig_sensi() <pymetamodels_objsamp>`

        |

        """

        return self.samp

    def sensitivity_type(self, case):

        """

        .. _pymetamodels_obj_samp_sensi:

        **Synopsis:**
            * Returns the sensitivity type analysis

        **Args:**
            * None

        **Optional parameters:**
            * None

        **Returns:**
            * sensitivity type analysis name string

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`

        |

        """

        return self.samp.sensitivity_type(case)


    def doe_inputs_X(self, case):

        ## Return sampling arrays in the X format

        return self.samp.doe_inputs_X(case)

    def run_sampling_routine(self, case):

        """

        .. _pymetamodels_run_sampling_routine:

        **Synopsis:**
            * Execute the sampling routines to generate the DOEX for each case

        **Args:**
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * The kind of sampling routines available specified metamodel configuration spreadsheet. :ref:`See metamodel configuration spreadsheet <pymetamodels_configuration>`

        |

        """

        ## Run the sampling genration

        return self.samp._run_analisis(case, sensitivity = False, sampling = True)

    def run_sensitivity_analisis(self, case):

        """

        .. _pymetamodels_run_sensitivity_analisis:

        **Synopsis:**
            * Execute the sensitivity analysis routines to generate the sensitivity indexes S_i

        **Args:**
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * The kind of sampling routines available specified metamodel configuration spreadsheet. :ref:`See metamodel configuration spreadsheet <pymetamodels_configuration>`

        |

        """

        ## Run the sesitivity test

        return self.samp._run_analisis(case, sensitivity = True, sampling = False)

    def run_sensitivity_normalization(self):

        """

        .. _pymetamodels_run_sensitivity_normalization:

        **Synopsis:**
            * Execute the normalization of the sensitivity indexes

        **Args:**
            * None

        **Optional parameters:**
            * None

        **Returns:**
            * Adds to the metamodel data structure the sensitivity indexes normalize

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * This calculation is performed as described by Nguyen :cite:`Nguyen2015`

        |

        """

        ## Run a sensitivity normalisation for all calculated sensitiviies
        # doi::10.1007/s12273-015-0245-4

        # get all Y vars
        lst_out_var = []
        for key_case in self.case.keys():
            lst = self.vars_out_keys(key_case)
            for ll in lst:
                if ll not in lst_out_var: lst_out_var.append(ll)

        # get all X vars
        lst_in_var = []
        for key_case in self.case.keys():
            lst = self.vars_keys(key_case, not_cte = True)
            for ll in lst:
                if ll not in lst_in_var: lst_in_var.append(ll)

        # construct normalization dictionary
        out = obj_data.objdata()
        tx_case = self.si_anly_case
        tx_si = self.si_anly_si
        tx_cov = self.si_anly_cov
        tx_s1 = self.si_anly_s1
        tx_sio = self.si_anly_sio

        for out_var in lst_out_var:

            out[out_var] = obj_data.objdata()

            for var in lst_in_var:

                out[out_var][var] = obj_data.objdata()
                out[out_var][var][tx_case] = []
                out[out_var][var][tx_si] = []
                out[out_var][var][tx_cov] = []
                out[out_var][var][tx_s1] = []
                out[out_var][var][tx_sio] = []

                for key_case in self.case.keys():

                    if self.si_out in  self.case[key_case].keys():

                        if (out_var in self.vars_out_keys(key_case)) and (var in self.vars_keys(key_case, not_cte = True)):

                            (s1_arr, cov_arr, s1, s1_arr1_ord) = self.samp.normalize_sensitivity(key_case, out_var, var)

                            out[out_var][var][tx_case].append(key_case)
                            out[out_var][var][tx_si].append(s1_arr)
                            out[out_var][var][tx_cov].append(cov_arr)
                            out[out_var][var][tx_s1].append(s1)
                            out[out_var][var][tx_sio].append(s1_arr1_ord)

                        else:
                            out[out_var][var][tx_case].append(key_case)
                            out[out_var][var][tx_si].append("")
                            out[out_var][var][tx_cov].append("")
                            out[out_var][var][tx_s1].append("")
                            out[out_var][var][tx_sio].append("")

        """
        print(lst_out_var)
        print(lst_in_var)
        print(out.print())
        """

        return out

    #### Metamodel construction
    #####################################

    def run_metamodel_construction(self, case, scheme = None):

        """

        .. _pymetamodels_run_metamodel_construction:

        **Synopsis:**
            * Execute the metamodelling regression routines to generate a predictor of DOEY values

        **Args:**
            * case: case name to be execute

        **Optional parameters:**
            * scheme=None: designate the type of metamodel search scheme that will be carried out to find the most optimal ML metamolde. The available schemes are: None, general, general_fast, general_fast_nonpol, linear, gaussian, polynomial (see :numref:`pymetamodels_conf_metamodel`)

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A3 <ex_coupled_function_metamodel>`
            * The kind metamodel configuration spreadsheet. :ref:`See metamodel configuration spreadsheet <pymetamodels_configuration>`
            * The data object structure can be seen in :ref:`objmetamodel() <pymetamodels_objmetamodel>`

        |

        """

        ## metamodelling regression routine

        self.add_metamodel_data(case, scheme = scheme)

    def obj_metamodel(self, case):

        """

        .. _pymetamodels_obj_metamodel:

        **Synopsis:**
            * Returns a data structure object correspoding to the train metamodel object (see :ref:`objmetamodel() <pymetamodels_objmetamodel>`)

        **Args:**
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * Pointer to data structure object correspoding to the :ref:`objmetamodel() <pymetamodels_objmetamodel>`

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A2 <ex_coupled_function_data_struct>`
            * The data object structure can be seen in :ref:`objmetamodel() <pymetamodels_objmetamodel>`

        |

        """

        ## Returns the doeX inputs variable arrays per case

        return self.case[case][self.objmetamodel]

    def save_metamodel(self, folder_path, case):

        """

        .. _pymetamodels_save_metamodel:

        **Synopsis:**
            * Save metamodels objects as a .metaita file which can be later read
            * .metaita files are placed in a sub-folder of the output folder

        **Args:**
            * folder_path: path to the folder where to save the files
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * Path to the file

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * Relates to :ref:`save_to_file() <objmetamodel_save_to_file>`

        |

        """

        ## Save metaita files
        ##

        # new folder
        folder = os.path.join(folder_path, "metamodel_files")
        if not os.path.exists(folder): os.makedirs(folder)

        file_name = r"metamodel_%s" % (case)

        return self.obj_metamodel(case).save_to_file(folder, file_name)

    def load_metamodel(self, folder_path, case):

        """

        .. _pymetamodels_load_metamodel:

        **Synopsis:**
            * Load metamodels object as a .metaita file for each case with the function :ref:`save_metamodel() <pymetamodels_save_metamodel>`

        **Args:**
            * folder_path: path to the folder where to save the files
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * True if the metamodel was loaded

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * Relates to :ref:`save_metamodel() <pymetamodels_save_metamodel>`

        |

        """

        ## Save metaita files
        ##

        # new folder
        folder = os.path.join(folder_path, "metamodel_files")

        file_name = r"metamodel_%s" % (case)

        obj = obj_metamodel.objmetamodel(logging_path = self.logging_path)

        obj.verbose_testing = False

        file_path = os.path.join(folder, file_name + obj.file_extension)

        if os.path.exists(file_path):
            obj.load_file(file_path)
            self.case[case][self.objmetamodel] = obj
            return True
        else:
            return False

    def add_metamodel_data(self, case, scheme = None):

        ## Add metamodelling regression routine

        if not self.objmetamodel in self.case[case].keys():
            self.case[case][self.objmetamodel] = obj_data.objdata()

        obj = obj_metamodel.objmetamodel(logging_path = self.logging_path)

        obj.verbose_testing = False

        doeY_train, var_keysY = self.doeY_asnp(case, return_keysY = True)

        doeX_train, var_keysX = self.doeX_asnp(case, return_keysX = True)

        obj.fit_model(doeX_train, doeY_train, var_keysX, var_keysY, doeX_test = None, doeY_test = None, scheme = scheme, with_test = True)

        self.case[case][self.objmetamodel] = obj

    def doeY_asnp(self, case, return_keysY = False):

        ## Returns doeY as a numpy array (n_samples,n_targets)

        doeY = self.doeY(case)

        n_targets = len(list(doeY.keys()))
        first_key = list(doeY.keys())[0]
        n_samples = len(doeY[first_key])

        if n_targets == 0:
            self.error("Wrong dimension")
        elif n_targets == 1:
            arr = np.ndarray((n_samples),dtype=np.dtype('float64'))
        else:
            arr = np.ndarray((n_samples,n_targets),dtype=np.dtype('float64'))

        ii = 0
        for key in doeY.keys():
            if n_targets == 1:
                arr[:] = doeY[key][:]
            else:
                arr[:,ii] = doeY[key][:]

            ii = ii + 1

        if return_keysY:
            return arr, list(doeY.keys())
        else:
            return arr

    def doeX_asnp(self, case, return_keysX = False):

        ## Returns doeX as a numpy array (n_samples,n_features)

        doeX = self.doeX(case)
        keysX = self.vars_keys(case, not_cte = True)

        n_features = len(keysX)
        first_key = list(doeX.keys())[0]
        n_samples = len(doeX[first_key])

        if n_features == 0:
            self.error("Wrong dimension")
        elif n_features == 1:
            arr = np.ndarray((n_samples, 1),dtype=np.dtype('float64'))
        else:
            arr = np.ndarray((n_samples,n_features),dtype=np.dtype('float64'))

        ii = 0
        for key in keysX:
            if n_features == 1:
                arr[:] = doeX[key][:]
            else:
                arr[:,ii] = doeX[key][:]

            ii = ii + 1

        if return_keysX:
            return arr, keysX
        else:
            return arr

    #### Optimzation problems
    #####################################

    def obj_optimization(self, case):

        """

        .. _pymetamodels_obj_optimization:

        **Synopsis:**
            * Returns a data structure object correspoding to the optimization object (see :ref:`objoptimization() <pymetamodels_objoptimization>`)

        **Args:**
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * Pointer to data structure object correspoding to the :ref:`objoptimization() <pymetamodels_objoptimization>`

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A06 <ex_kuwase_optimize>`
            * The data object structure can be seen in :ref:`objoptimization() <pymetamodels_objoptimization>`

        |

        """

        ## Returns the doeX inputs variable arrays per case

        return self.case[case][self.objoptimization]

    def save_optimization(self, folder_path, case):

        """

        .. _pymetamodels_save_optimization:

        **Synopsis:**
            * Save the optimization data as a .optita file which can be later read
            * .optita files are placed in a sub-folder of the output folder

        **Args:**
            * folder_path: path to the folder where to save the files
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * Path to the file

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * Relates to :ref:`save_to_file() <objoptimization_save_to_file>`

        |

        """

        ## Save metaita files
        ##

        # new folder
        folder = os.path.join(folder_path, "metamodel_files")
        if not os.path.exists(folder): os.makedirs(folder)

        file_name = r"optimization_%s" % (case)

        return self.obj_optimization(case).save_to_file(folder, file_name)

    def load_optimization(self, folder_path, case):

        """

        .. _pymetamodels_load_optimization:

        **Synopsis:**
            * Load optimization object as a .optita file for each case

        **Args:**
            * folder_path: path to the folder where to save the files
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * True if the optimization was loaded

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * Relates to :ref:`save_optimization() <pymetamodels_save_optimization>`

        |

        """

        ## Save metaita files
        ##

        # new folder
        folder = os.path.join(folder_path, "metamodel_files")

        file_name = r"optimization_%s" % (case)

        obj = obj_optimization.objoptimization(logging_path = self.logging_path)

        obj.verbose_testing = False

        file_path = os.path.join(folder, file_name + obj.file_extension)

        if os.path.exists(file_path):
            obj.load_file(file_path)
            self.case[case][self.objoptimization] = obj
            return True
        else:
            return False

    def run_optimization_problem(self, case, scheme = None, max_size_grid_methods = None, rel_tol_val_grid_methods = None, verbose_testing = False):

        """

        .. _pymetamodels_run_optimization_problem:

        **Synopsis:**
            * Execute optimization problem routines for a case according the configuration spreadsheet

        **Args:**
            * case: case name to be execute

        **Optional parameters:**
            * scheme = None: designate the type of optimization solver scheme that will be carried out (see :numref:`pymetamodels_conf_optimization`). The availables schemes are: "general", "general_fast", "general_with_constrains", "global", "minimize", "grid_method", "iter_grid_method"
            * max_size_grid_methods = None: value for max_size_grid_methods ivar, diemsion of the grid for grid methods
            * rel_tol_val_grid_methods = None: tolerance value for rel_tol_val_grid_methods ivar, tolerance of the iterative maximum error
            * verbose_testing = False: verbose routines messages

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A06 <ex_kuwase_optimize>`
            * The optimization configuration spreadsheet. :ref:`See optimization configuration spreadsheet <pymetamodels_configuration>`
            * The data object structure can be seen in :ref:`objoptimization() <pymetamodels_objoptimization>`

        |

        """

        ## Add metamodelling regression routine

        if not self.objoptimization in self.case[case].keys():
            self.case[case][self.objoptimization] = None

        obj = obj_optimization.objoptimization(logging_path = self.logging_path)

        obj.verbose_testing = verbose_testing

        obj_meta = self.obj_metamodel(case)

        min_vars, type_op_min = self.lst_min_vars(case)

        bounds = self.opt_bounds(case)

        constrains = self.opt_constrains_vars(case, type_op_min)

        if max_size_grid_methods is not None:
            obj.max_size_grid_methods = max_size_grid_methods

        if rel_tol_val_grid_methods is not None:
            obj.rel_tol_val_grid_methods = rel_tol_val_grid_methods

        obj.run_optimization(obj_meta, min_vars, type_op_min, bounds = bounds, constrains_vars = constrains, scheme = scheme)

        self.case[case][self.objoptimization] = obj

    def lst_min_vars(self, case):

        ## Output list with the min_vars output variables

        vars_out = self.vars_out_parameter_matrix(case)

        lst_min_vars = []
        type_op_min = 0

        for var in vars_out.keys():
            if bool(vars_out[var][self.op_min]):
                lst_min_vars.append(var)
                type_op_min = 1

        for var in vars_out.keys():
            if bool(vars_out[var][self.op_min_eq]):
                lst_min_vars.append(var)
                type_op_min = 2

        return lst_min_vars, type_op_min

    def opt_bounds(self, case):

         ## Outputs the bounds sequence

         vars_x = self.obj_metamodel(case).doeX_varX

         bounds = []

         for var_x in vars_x:
             boundX = self.var_bounds(case, var_x)

             bounds.append((boundX[0], boundX[1]))

         return bounds

    def opt_constrains_vars(self, case, type_op_min):

        ## Outputs the constrains sequence

        vars_out = self.vars_out_parameter_matrix(case)

        constrains = []

        for var in vars_out.keys():

            if not bool(vars_out[var][self.op_min]):

                if bool(vars_out[var][self.op_eq]) and bool(vars_out[var][self.op_ineq]):
                    self.error("Double constraints not properly define for %s" % var)
                elif bool(vars_out[var][self.op_ineq]):
                    dct = {}
                    dct['type'] = 'ineq'
                    dct['fun'] = None
                    dct['args'] = [var, type_op_min, None]
                    constrains.append(dct)
                elif bool(vars_out[var][self.op_eq]):
                    dct = {}
                    dct['type'] = 'eq'
                    dct['fun'] = None
                    dct['args'] = [var, type_op_min, None]
                    constrains.append(dct)
                else:
                    pass
        if len(constrains) == 0:
            return ()
        else:
            return constrains

    #### Data functions
    #####################################

    def print_structure(self, case = None):

        """

        .. _pymetamodels_print_structure:

        **Synopsis:**
            * Recursively data object structure case and all the nested data objects dictionaries

        **Args:**
            * dictionary: dictionary object to recursively be printed

        **Optional parameters:**
            * case: case name to be execute, if None prints the full case object

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A2 <ex_coupled_function_data_struct>`

        |

        """

        ## Print the structure of the data dictionary

        self.msg('Data dict structure')

        if case is None:
            self.print_dict(self.case, ident = '', braces=1)
        else:
            self.print_dict(self.case[case], ident = '', braces=1)

    def list_of_parameters(self):

        ## List of parameters
        return [self.value, self.min_bound, self.max_bound, self.distribution, self.is_variable, self.cov_un, self.ud, self.alias]

    def vars_parameter_matrix(self, case):

        """

        .. _pymetamodels_vars_parameter_matrix:

        **Synopsis:**
            * Returns a data structure object correspoding to the parameters values for each input variables from the metamodel configuration

        **Args:**
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * Pointer to data structure object correspoding to the parameters values for each input variable from the metamodel configuration

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A2 <ex_coupled_function_data_struct>`
            * The data object structure can be seen using the function :ref:`print_dict() <pymetamodels_print_dict>`

        |

        """

        ## Returns the parameter value for an input variable
        vars_in = self.vars_keys(case, not_cte = False) + self.vars_keys(case, not_cte = True)
        params = self.list_of_parameters()

        dct = {}
        for ff in vars_in:
            dct[ff] = {}
            for cc in params:
                dct[ff][cc] = self.case[case][self.vars_key][self.op_key + ff][cc]

        return dct

    def vars_out_parameter_matrix(self, case):

        """

        .. _pymetamodels_vars_out_parameter_matrix:

        **Synopsis:**
            * Returns a data structure object correspoding to the parameters values for each output variables from the metamodel configuration

        **Args:**
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * Pointer to data structure object correspoding to the parameters values for each output variable from the metamodel configuration

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A2 <ex_coupled_function_data_struct>`
            * The data object structure can be seen using the function :ref:`print_dict() <pymetamodels_print_dict>`

        |

        """

        ## Returns the parameter value for an input variable
        vars_out = self.vars_out_keys(case)

        params = []
        for vvar in vars_out:
            lst = self.var_param_out(case, vvar).keys()
            params = list(params | lst)

        dct = {}
        for ff in vars_out:
            dct[ff] = {}
            for cc in params:
                dct[ff][cc] = self.case[case][self.vars_out_key][self.op_key+ff][cc]

        return dct

    def units_dictionary(self, case):

        ## Return the units dictionary with var names

        vars_in = self.vars_parameter_matrix(case)
        vars_out = self.vars_out_parameter_matrix(case)

        dct = {}
        for var in vars_in.keys():
            if var not in dct.keys():
                dct[var] = vars_in[var][self.ud]
            else:
                self.error("Inconsistent variables name")

        for var in vars_out.keys():
            if var not in dct.keys():
                dct[var] = vars_out[var][self.ud]
            else:
                self.error("Inconsistent variables name")

        return dct

    def vars_dist(self, case, lst_vars):

        ## Returns the bounds of a list of vars

        lst = []

        for ivar in lst_vars:

            lst.append(self.case[case][self.vars_key][self.op_key + ivar][self.distribution])

        return lst

    def vars_bounds(self, case, lst_vars):

        ## Returns the bounds of a list of vars

        bounds = []

        for ivar in lst_vars:

            lst = self.var_bounds(case, ivar)
            bounds.append(lst)

        return bounds

    def var_bounds(self, case, varname):

        ## Returns the bounds of a var name

        lst = []
        lst.append(self.case[case][self.vars_key][self.op_key + varname][self.min_bound])
        lst.append(self.case[case][self.vars_key][self.op_key + varname][self.max_bound])

        return lst

    def vars_keys(self, case, not_cte = True):

        ## Returns the list of variables which are not constant

        lst_ctes = []
        lst_vars = []

        for key in self.case[case][self.vars_key].keys():

            if key[:len(self.op_key)] == self.op_key:

                key_name = key[len(self.op_key):]

                if self.case[case][self.vars_key][key][self.is_variable]:
                    lst_vars.append(key_name)
                else:
                    lst_ctes.append(key_name)

        if not_cte:
            return lst_vars
        else:
            return lst_ctes

    def vars_out_keys(self, case):

        ## Returns the list of output variables

        lst_vars = []

        for key in self.case[case][self.vars_out_key].keys():

            if key[:len(self.op_key)] == self.op_key:

                key_name = key[len(self.op_key):]

                lst_vars.append(key_name)

        return lst_vars

    def var_param(self, case, var_name):

        ### Returns vals parameters as dictionary

        return self.case[case][self.vars_key][self.op_key+var_name]

    def var_param_out(self, case, var_name):

        ### Returns vals parameters output as dictionary

        return self.case[case][self.vars_out_key][self.op_key+var_name]

    def doeX(self, case):

        """

        .. _pymetamodels_doeX:

        **Synopsis:**
            * Returns a data structure object correspoding to the DOEX inputs variable arrays

        **Args:**
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * Pointer to data structure object correspoding to the DOEX

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A2 <ex_coupled_function_data_struct>`
            * The data object structure can be seen using the function :ref:`print_dict() <pymetamodels_print_dict>`

        |

        """

        ## Returns the doeX inputs variable arrays per case

        return self.case[case][self.doe_in]

    def doeY(self, case):

        """

        .. _pymetamodels_doeY:

        **Synopsis:**
            * Returns a data structure object correspoding to the DOEY output variables arrays

        **Args:**
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * Pointer to data structure object correspoding to the DOEY

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A2 <ex_coupled_function_data_struct>`
            * The data object structure can be seen using the function :ref:`print_dict() <pymetamodels_print_dict>`

        |

        """

        ## Returns the doeY outputs variable arrays per case

        return self.case[case][self.doe_out]

    def SiY_key_vals(self, case):

        ## Returns the dict keys names of interest for the sensitivity object

        # Grab the sensitivity method
        sensitivity_type = self.sensitivity_type(case)

        # Run sensitivity anlysis
        if sensitivity_type == "Sobol":
            (s1,cov) = ("S1","S1_conf")
        elif sensitivity_type == "Morris":
            (s1,cov) = ("mu_star","mu_star_conf")
        elif sensitivity_type == "RBD-Fast":
            (s1,cov) = ("S1","S1_conf")
        elif sensitivity_type == "Fast":
            (s1,cov) = ("S1","S1_conf")
        elif sensitivity_type == "Delta-MIM":
            (s1,cov) = ("S1","S1_conf")
        elif sensitivity_type == "DGSM":
            (s1,cov) = ("dgsm","dgsm_conf")
        elif sensitivity_type == "Factorial":
            (s1,cov) = ("ME","ME")
        elif sensitivity_type == "PAWN":
            (s1,cov) = ("mean","CV")
        elif sensitivity_type == "HDMR":
            (s1,cov) = ("S","S_conf")
        else:
            self.error("Sensibility norm analyisis type unknown %s" % sensitivity_type)

        return [s1,cov]

    def SiY(self, case):

        """

        .. _pymetamodels_SiY:

        **Synopsis:**
            * Returns a data structure object correspoding to the results for the sensitivity analysis values

        **Args:**
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * Pointer to data structure object correspoding to the results for the sensitivity analysis values

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A2 <ex_coupled_function_data_struct>`
            * The data object structure can be seen using the function :ref:`print_dict() <pymetamodels_print_dict>`

        |

        """

        ## Returns the results for the sensitivity analysis values per output variable

        return self.case[case][self.si_out]

    def var_name_chr_clean(self, var_name):

        out = str(var_name)

        lst_chr = ["\\","/"]

        for tx in lst_chr:

            out = out.replace(tx,"")

        return out

    #### Reading / output formats
    #####################################

    def read_xls_case(self, folder_path, file_name, sheet="cases", col_start = 0,
            row_start = 1, tit_row = 0):

        """

        .. _pymetamodels_read_xls_case:

        **Synopsis:**
            * Read metamodel configuration spreadsheet in xls format

        **Args:**
            * folder_path: path to xls spreadsheet
            * file_name: xls spreadsheet file name
            * sheet: sheet name with the cases description

        **Optional parameters:**
            * col_start = 0: the columm number where the data starts
            * row_start = 1: the row number where the data starts
            * tit_row = 0: the row number where the titles start

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * Template of a metamodel configuration spreadsheet in xls format :download:`Download template </documents/inputs.xls>`

        |

        """

        ## Read excel table values
        ## First row 0 is the titles
        ## Id is the first col

        vars_sheet_field = self.vars_sheet
        vars_out_sheet_field = self.output_sheet
        
        file_path = os.path.join(folder_path, file_name + '.xls')

        wk = xlrd.open_workbook(file_path)

        sheet = wk.sheet_by_name(sheet)

        # Read Ids, empty Id end
        num_rows = -1
        self.case = obj_data.objdata()
        for ii in range(row_start, sheet.nrows):

            value = sheet.cell(ii, col_start).value
            self.case[value] = obj_data.objdata()
            num_rows = ii

            if value == "": break

        # Read table
        for ii in range(row_start, num_rows+1):

            for jj in range(col_start, sheet.ncols):

                col_name = sheet.cell(tit_row, jj).value
                Id = sheet.cell(ii, col_start).value
                value = sheet.cell(ii, jj).value

                if self.is_number(value):
                    value = float(value)
                elif self.is_none(value):
                    value = None
                else:
                    pass

                self.case[Id][col_name] = value

                if col_name == "": break
        
        # Read variable sheets
        for key in self.case.keys():

            self.read_xls_vars(wk, key, vars_sheet_field, col_start = col_start,
                row_start = row_start, vars_in = True )

            self.read_xls_vars(wk, key, vars_out_sheet_field, col_start = col_start,
                row_start = row_start, vars_in = False )

    def read_xls_vars(self, wk, key, vars_sheet_field, col_start = 0, row_start = 1, vars_in = True):

        ## Read excel vars based on cases sheet
        ## First column variables name, other columns options

        sheet_name = self.case[key][vars_sheet_field]

        #print(sheet_name)
        sheet = wk.sheet_by_name(sheet_name)

        if vars_in:
            _vars_key = self.vars_key
        else:
            _vars_key = self.vars_out_key

        self.case[key][_vars_key] = obj_data.objdata()

        # Read variables
        for ii in range(row_start, sheet.nrows):

            varible_name = sheet.cell(ii, col_start).value
            variable_value = sheet.cell(ii, col_start+1).value

            if varible_name == "": break

            if self.is_number(variable_value):
                variable_value = float(variable_value)
            elif self.is_none(variable_value):
                variable_value = None
            else:
                pass

            if varible_name in self.case[key][_vars_key].keys():
                self.error("Variable name already in use")
            else:
                self.case[key][_vars_key][varible_name] = variable_value

            for jj in range(col_start+1, sheet.ncols):

                col_name = sheet.cell(row_start-1, jj).value

                if col_name == "": break

                op_variable_value = sheet.cell(ii, jj).value

                if self.op_key + varible_name not in self.case[key][_vars_key]:
                    self.case[key][_vars_key][self.op_key + varible_name] = obj_data.objdata()

                self.case[key][_vars_key][self.op_key + varible_name][col_name] = op_variable_value

    def output_xls(self, folder_path, file_name, col_start = 0, tit_row = 0):

        """

        .. _pymetamodels_output_xls:

        **Synopsis:**
            * Output DOEX, DOEY and analysis data into a spreadsheet in xls format

        **Args:**
            * folder_path: path to xls spreadsheet
            * file_name: xls spreadsheet file name

        **Optional parameters:**
            * col_start = 0: the columm number where the data starts
            * row_start = 1: the row number where the data starts

        **Returns:**
            * The path to the xls spreadsheet

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * It is limit to DOE with a maximun size of 65500 samples. For larger values save directly to csv

        |

        """

        ## Output to excel file values
        ## OutPut DoeX, DoeY, Sensitivities analysis

        file_path = os.path.join(folder_path, file_name + '.xls')

        wb = xlwt.Workbook() # create empty workbook object


        def _save_doe(wb,type_doe,j_col_start,i_row_start):

            type_doe = type_doe

            for key_case in self.case.keys():

                if type_doe in  self.case[key_case].keys():

                    sheet_name = "%s#%s" % (self.case[key_case][self.case_tit],self.doe_name)
                    try:
                        sheet = wb.get_sheet(sheet_name)
                    except:
                        sheet = wb.add_sheet(sheet_name)

                    j = j_col_start
                    if type_doe == self.doe_in:
                        doe = self.doeX(key_case)
                    elif type_doe == self.doe_out:
                        doe = self.doeY(key_case)

                    for var_name in doe.keys():

                        i = i_row_start
                        val = var_name #col_name
                        sheet.write(i,j,val)

                        i = i + 1
                        if type_doe == self.doe_in:
                            val = self.var_param(key_case, var_name)["ud"] #ud
                        elif type_doe == self.doe_out:
                            val = self.var_param_out(key_case, var_name)["ud"] #ud
                        sheet.write(i,j,val)

                        i = i + 1
                        if type_doe == self.doe_in:
                            val = self.doe_in
                        elif type_doe == self.doe_out:
                            val = self.doe_out
                        sheet.write(i,j,val)

                        i = i + 1
                        for ii in range(0,len(doe[var_name])):

                            val = doe[var_name][ii]
                            sheet.write(i,j,val)
                            i = i + 1

                        j = j + 1

            return j

        ## Save DoeX
        jj_end = _save_doe(wb,self.doe_in,col_start,tit_row)
        ## Save DoeY
        jj_end = _save_doe(wb,self.doe_out,jj_end+1,tit_row)
        ## Save sensitivity analysis
        self.output_xls_sensitivity(folder_path, file_name, col_start = col_start, tit_row = tit_row, wb = wb)

        ## Save
        wb.save(file_path)

        return file_path

    def _save_doe(self, wb, type_doe, j_col_start, i_row_start):

        type_doe = type_doe

        for key_case in self.case.keys():

            if type_doe in  self.case[key_case].keys():

                sheet_name = "%s" % (self.case[key_case][self.case_tit])
                try:
                    sheet = wb.get_sheet(sheet_name)
                except:
                    sheet = wb.add_sheet(sheet_name)

                j = j_col_start
                if type_doe == self.doe_in:
                    doe = self.doeX(key_case)
                elif type_doe == self.doe_out:
                    doe = self.doeY(key_case)

                for var_name in doe.keys():

                    i = i_row_start
                    val = var_name #col_name
                    sheet.write(i,j,val)

                    i = i + 1
                    if type_doe == self.doe_in:
                        val = self.var_param(key_case, var_name)["ud"] #ud
                    elif type_doe == self.doe_out:
                        val = self.var_param_out(key_case, var_name)["ud"] #ud
                    sheet.write(i,j,val)

                    i = i + 1
                    if type_doe == self.doe_in:
                        val = self.doe_in
                    elif type_doe == self.doe_out:
                        val = self.doe_out
                    sheet.write(i,j,val)

                    i = i + 1
                    for ii in range(0,len(doe[var_name])):

                        val = doe[var_name][ii]
                        sheet.write(i,j,val)
                        i = i + 1

                    j = j + 1

        return j        

    def _read_doe(self, wb, type_doe, j_col_start, i_row_start):   
        
        type_doe = type_doe

        for key_case in self.case.keys():

            if type_doe in  self.case[key_case].keys():

                sheet_name = "%s" % (self.case[key_case][self.case_tit])
                
                sheet = wb.sheet_by_name(sheet_name)

                j = j_col_start
                if type_doe == self.doe_in:
                    doe = self.doeX(key_case)
                elif type_doe == self.doe_out:
                    doe = self.doeY(key_case)

                for var_name in doe.keys():

                    i = i_row_start
                    val = var_name #col_name
                    #sheet.write(i,j,val)

                    i = i + 1
                    if type_doe == self.doe_in:
                        val = self.var_param(key_case, var_name)["ud"] #ud
                    elif type_doe == self.doe_out:
                        val = self.var_param_out(key_case, var_name)["ud"] #ud
                    #sheet.write(i,j,val)
                    #sheet.cell(i,j).value

                    i = i + 1
                    if type_doe == self.doe_in:
                        val = self.doe_in
                    elif type_doe == self.doe_out:
                        val = self.doe_out
                    #sheet.write(i,j,val)

                    i = i + 1
                    for ii in range(0,len(doe[var_name])):
                        
                        doe[var_name][ii] = sheet.cell(i,j).value
                        i = i + 1

                    j = j + 1

        return j           

    def save_tofile_DOEX(self, folder_path, file_name):

        """

        .. _pymetamodels_save_tofile_DOEX:

        **Synopsis:**
            * Save DOEX into a spreadsheet in xls format 

        **Args:**
            * folder_path: path to xls spreadsheet
            * file_name: xls spreadsheet file name

        **Optional parameters:**
            * None

        **Returns:**
            * The path to the xls spreadsheet

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`. See :ref:`tutorial A09 <ex_coupled_function_load_DOEs>`
            * It is limit to DOE with a maximun size of 65500 samples. For larger values save directly to csv
            * It can be read with the function :ref:`read_fromfile_DOEX() <pymetamodels_read_fromfile_DOEX>`

        |

        """

        ## Output to excel file values
        ## OutPut DoeX, DoeY, Sensitivities analysis

        file_path = os.path.join(folder_path, file_name + '.xls')

        wb = xlwt.Workbook() # create empty workbook object

        col_start = 0
        tit_row = 0

        ## Save DoeX
        jj_end = self._save_doe(wb, self.doe_in, col_start, tit_row)

        ## Save
        wb.save(file_path)

        return file_path   

    def read_fromfile_DOEX(self, folder_path, file_name):

        """

        .. _pymetamodels_read_fromfile_DOEX:

        **Synopsis:**
            * Read DOEX from a spreadsheet in xls format 

        **Args:**
            * folder_path: path to xls spreadsheet
            * file_name: xls spreadsheet file name

        **Optional parameters:**
            * None

        **Returns:**
            * The path to the xls spreadsheet

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`. See :ref:`tutorial A09 <ex_coupled_function_load_DOEs>`
            * It is limit to DOE with a maximun size of 65500 samples. For larger values save directly to csv
            * It read formats saved with :ref:`save_tofile_DOEX() <pymetamodels_save_tofile_DOEX>`

        |

        """

        ## Load to excel file values
        ## OutPut DoeX, DoeY

        for case in self.keys():

            ## Run samplimg cases
            self.run_sampling_routine(case)          
        
        file_path = os.path.join(folder_path, file_name + '.xls')  

        wk = xlrd.open_workbook(file_path)

        col_start = 0
        tit_row = 0

        ## Save DoeX
        jj_end = self._read_doe(wk, self.doe_in, col_start, tit_row)        

        return file_path  

    def save_tofile_DOEY(self, folder_path, file_name):

        """

        .. _pymetamodels_save_tofile_DOEY:

        **Synopsis:**
            * Save DOEY into a spreadsheet in xls format 

        **Args:**
            * folder_path: path to xls spreadsheet
            * file_name: xls spreadsheet file name

        **Optional parameters:**
            * None

        **Returns:**
            * The path to the xls spreadsheet

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`. See :ref:`tutorial A09 <ex_coupled_function_load_DOEs>`
            * It is limit to DOE with a maximun size of 65500 samples. For larger values save directly to csv
            * It can be read with the function :ref:`read_fromfile_DOEY() <pymetamodels_read_fromfile_DOEY>`

        |

        """

        ## Output to excel file values
        ## OutPut DoeX, DoeY, Sensitivities analysis

        file_path = os.path.join(folder_path, file_name + '.xls')

        wb = xlwt.Workbook() # create empty workbook object

        col_start = 0
        tit_row = 0
        
        ## Save DoeX
        jj_end = self._save_doe(wb, self.doe_out, col_start, tit_row)

        ## Save
        wb.save(file_path)

        return file_path   

    def read_fromfile_DOEY(self, folder_path, file_name):

        """

        .. _pymetamodels_read_fromfile_DOEY:

        **Synopsis:**
            * Read DOEY from a spreadsheet in xls format 

        **Args:**
            * folder_path: path to xls spreadsheet
            * file_name: xls spreadsheet file name

        **Optional parameters:**
            * None

        **Returns:**
            * The path to the xls spreadsheet

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`. See :ref:`tutorial A09 <ex_coupled_function_load_DOEs>`
            * It is limit to DOE with a maximun size of 65500 samples. For larger values save directly to csv
            * It read formats saved with :ref:`save_tofile_DOEY() <pymetamodels_save_tofile_DOEY>`

        |

        """

        ## Load to excel file values
        ## OutPut DoeX, DoeY
        
        file_path = os.path.join(folder_path, file_name + '.xls')

        wk = xlrd.open_workbook(file_path)
        
        col_start = 0
        tit_row = 0

        ## Save DoeX
        jj_end = self._read_doe(wk, self.doe_out, col_start, tit_row)

        return file_path

    def output_xls_sensitivity(self, folder_path, file_name, col_start = 0, tit_row = 0, wb = None):

        ## Output to excel file values
        ## OutPut Sensitivities analysis

        if wb is None:
            file_path = os.path.join(folder_path, file_name + '.xls')

            wb = xlwt.Workbook() # create empty workbook object
        else:
            file_path = None

        out_sensi = self.run_sensitivity_normalization()
        tx_case = self.si_anly_case
        tx_si = self.si_anly_si
        tx_cov = self.si_anly_cov
        tx_s1 = self.si_anly_s1
        tx_sio = self.si_anly_sio
        lst_tit = [tx_si, tx_cov, tx_s1, tx_sio]

        for out_var in out_sensi.keys():

            sheet_name = r"%s - sensitivity" % (self.var_name_chr_clean(out_var))
            try:
                sheet = wb.get_sheet(sheet_name)
            except:
                sheet = wb.add_sheet(sheet_name)

            row = tit_row + 3
            col = col_start
            in_var = list(out_sensi[out_var].keys())[0]

            sheet.write(row-1,col,"Case")
            sheet.write(row-1,col+1,self.sensitivity_method)
            for case in out_sensi[out_var][in_var][tx_case]:
                val = case
                sheet.write(row,col,val)
                val = self.sensitivity_type(case)
                sheet.write(row,col+1,val)
                row = row + 1

            col = col + 2
            for ll in lst_tit:

                for in_var in out_sensi[out_var].keys():

                    row = tit_row
                    val = in_var
                    sheet.write(row,col,val)
                    row = row+1
                    val = "ud"
                    sheet.write(row,col,val)
                    row = row+1
                    val = ll
                    sheet.write(row,col,val)

                    for jj in range(0,len(out_sensi[out_var][in_var][ll])):

                        row = row + 1
                        val = out_sensi[out_var][in_var][ll][jj]
                        sheet.write(row,col,val)

                    col = col + 1

                col = col + 1

        if file_path is not None:
            ## Save
            wb.save(file_path)

    def read_xls_inp(self, folder_path, file_name, sheet = "Inputs", col_start = 0, row_start = 1 ):

        ## Read excel inputs file values
        ## First column variables name, second values
        # Legacy

        file_path = os.path.join(folder_path, file_name + '.xls')

        wk = xlrd.open_workbook(file_path)

        sheet = wk.sheet_by_name(sheet)


        for ii in range(row_start, sheet.nrows):

            varible_name = sheet.cell(ii, col_start).value
            variable_value = sheet.cell(ii, col_start+1).value

            if self.is_number(variable_value):
                variable_value = float(variable_value)
            else:
                pass

            self.inp[varible_name] = variable_value

            if varible_name == "": break

    def read_xls_out(self, folder_path, file_name, sheet = "Outputs", col_start = 0, row_start = 1 ):

        ## Read excel outputs file values
        ## First column variables name, second values
        # Legacy

        file_path = os.path.join(folder_path, file_name + '.xls')

        wk = xlrd.open_workbook(file_path)

        sheet = wk.sheet_by_name(sheet)


        for ii in range(row_start, sheet.nrows):

            varible_name = sheet.cell(ii, col_start).value
            variable_value = sheet.cell(ii, col_start+1).value

            if self.is_number(variable_value):
                variable_value = float(variable_value)
            else:
                pass

            self.out[varible_name] = variable_value

            if varible_name == "": break

    def save_xls_out(self, folder_path, file_name, sheet = "Outputs", col_start = 0, row_start = 1 ):

        ## Read excel outputs file values
        ## First column variables name, second values
        # Legacy

        file_path = os.path.join(folder_path, file_name + '.xls')

        wk = xlrd.open_workbook(file_path)

        wkc = xlcopy(wk)

        sheet = wkc.get_sheet(sheet)

        ii = row_start
        for key, value in self.out.items():

            varible_name = key
            variable_value = value

            sheet.write(ii,col_start, key)
            sheet.write(ii,col_start+1, value)

            ii = ii + 1

        wkc.save(file_path)

    def save_txt_out(self, folder_path, file_name, token = " "):

        ## Read excel outputs file values
        ## First column variables name, second values
        # Legacy

        file_path = os.path.join(folder_path, file_name + '.txt')

        f = open(file_path, "w")

        for key, value in self.out.items():

            number = "%.5f" % value
            line = "%s%s%s\n"  % (key,token,number.zfill(25))
            f.write(line)

        f.close()

    def save_csv_channels(self, dict_row_col, file_name, path_out, units_chn = True, sort_chn = False):

        ### Saves a dict dictionary (rowsxcols)

        # output csv file
        fl = open(os.path.join(path_out,'%s.csv' % file_name), 'w')
        fo = csv.writer(fl, delimiter=';',dialect=csv.excel,lineterminator ='\n')

        # cols and rows
        rows = dict_row_col.keys()
        cols = []
        for row in rows:

            for col in dict_row_col[row].keys():
                if col in cols:
                    pass
                else:
                    cols.append(col)

        if sort_chn:
            cols = sorted(cols)

        # titles
        lst_tit = [self.case_tit]
        for col in cols:
            lst_tit.append(col)
        fo.writerow(lst_tit[:])

        # units
        lst_units = [""]
        for col in cols:
            for row in rows:
                dct = self.units_dictionary(row)
                if col in dct.keys():
                    lst_units.append(dct[col])
                    break
        fo.writerow(lst_units[:])

        # write csv
        for row in rows:

            lst = [row]
            for col in cols:

                if col in dict_row_col[row].keys():
                    lst.append(dict_row_col[row][col])
                else:
                    lst.append("")

            fo.writerow(lst[:])

        fl.close()

    def read_table_csv_channels(self, file_name, path_out, units_chn = False):

        ### Saves a dict dictionary of channels as csv (xcols), buffer file

        #Output csv file
        filess = open(os.path.join(path_out,file_name))
        fo = csv.reader(filess, delimiter=';', dialect=csv.excel, lineterminator ='\n')

        obj = {}
        units = {}

        # write titles
        ii = 0
        for row in fo:
            # Titles
            if ii == 0:
                for ele in row:
                    obj[ele] = []

            # Units
            elif ii == 1 and units_chn:
                jj = 0
                for key in obj.keys():
                    units[key] = row[jj]
                    jj = jj + 1

            # arrays
            else:
                jj = 0
                for key in obj.keys():
                    obj[key].append(row[jj])
                    jj = jj + 1

            ii = ii + 1

        return obj, units

    def save_table_csv_channels(self, vec_chn, file_name, path_out, units_chn = None, sort_chn = False):

        ### Saves a dict dictionary of channels as csv (xcols), buffer file

        #Output csv file
        filess = open(os.path.join(path_out,'%s.csv' % file_name), 'w')
        fo = csv.writer(filess, delimiter=';',dialect=csv.excel,lineterminator ='\n')

        lst_tit = []
        len_vec = []
        max_len = -1

        if sort_chn:
            vec_keys = sorted(vec_chn.keys())
        else:
            vec_keys = vec_chn.keys()

        # write titles
        for ele in vec_keys:
            ll = len(vec_chn[ele])
            len_vec.append(ll)
            if ll > max_len: max_len = ll
            lst_tit.append(ele)

        fo.writerow(lst_tit[:])

        # write units
        if units_chn is not None:
            lst_units = []
            for ele in vec_keys:
                if ele in units_chn.keys():
                    lst_units.append("["+units_chn[ele]+"]")
                else:
                    lst_units.append("[-]")

            fo.writerow(lst_units[:])

        for ii in range(0,max_len):

            lst = []
            for ele in vec_keys:

                ll = len(vec_chn[ele])

                if ii < ll:
                    lst.append(vec_chn[ele][ii])
                else:
                    lst.append("")

            fo.writerow(lst[:])

        filess.close()

    #### Plots
    #####################################

    def output_plts_sensitivity(self, folder_path, case):

        """

        .. _pymetamodels_output_plts_sensitivity:

        **Synopsis:**
            * Plots all cross DOEX variable sampling combinations and sensitivity analisys
            * Plots sensitivity histograms for each case
            * Plots are placed in a sub-folder of the output folder

        **Args:**
            * folder_path: path to the folder where to save the plots images
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * Saves in the folder_path location the plots regarding the sensitivity analysis

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` or :ref:`Tutorial A01 <ex_coupled_function>`
            * The kind of sampling routines available specified metamodel configuration spreadsheet. :ref:`See metamodel configuration spreadsheet <pymetamodels_configuration>`

        |

        """

        ## Plots cross variable sampling relation in the sensitivity analysis
        ##

        # get variables names
        vars_x = self.vars_keys(case, not_cte = True)
        vars_y = vars_x.copy()
        vars_in = self.vars_parameter_matrix(case)
        sensitivity_type = self.sensitivity_type(case)

        # new folder
        folder = os.path.join(folder_path, "cross_sampling")
        if not os.path.exists(folder): os.makedirs(folder)

        # iterate variables (one to one plots + histogram)
        for varx in vars_x:

            vars_y.remove(varx)

            for vary in vars_y:

                # one to one plots

                data = {}
                data["legend"] = r"$%s$ vs. $%s$ %s" % (varx,vary,sensitivity_type)
                data["output_folder"] = folder
                data["file name"] = r"cross_%s_%s_%s" % (self.var_name_chr_clean(varx),self.var_name_chr_clean(vary),case)
                data["xname"] = varx
                data["xdata"] = self.doeX(case)[varx]
                data["yname"] = vary
                data["ydata"] = self.doeX(case)[vary]
                data["format"] = "*"
                data["ylabel"] = r"$%s$ [%s]" % (vary,self.var_param(case, vary)[self.ud])
                data["xlabel"] = r"$%s$ [%s]" % (varx,self.var_param(case, varx)[self.ud])

                self.plt.plot_scatter_sensitivity(data)

        self.msg("Cross DOEX variable sampling combinations and sensitivity analisys plotted in %s" % folder)

        ## histogram plot
        # new folder
        folder2 = os.path.join(folder_path, "hist_sensitivity")
        if not os.path.exists(folder2): os.makedirs(folder2)

        vars_x = self.vars_keys(case, not_cte = True)
        lst_var_x = []
        for var_x in vars_x:
            lst_var_x.append("$%s$" % var_x)
        vars_o = self.vars_out_keys(case)
        s_y = self.SiY(case)
        [vi,conf] = self.SiY_key_vals(case)
        sensitivity_type = self.sensitivity_type(case)

        for vary in vars_o:

            data1 = {}
            data1["legend"] = r"%s Case: %s " % (sensitivity_type, case)
            data1["output_folder"] = folder2
            data1["file name"] = r"sensi_%s_%s" % (case,vary)
            data1["xname"] = "DOEX variables"
            data1["xdata"] = lst_var_x
            data1["yname"] = "Sensitivity index var $%s$" % vary
            data1["ydata"] = np.abs(s_y[vary][vi])
            data1["color"] = ""
            data1["ylabel"] = r"Sensitivity index var $%s$ [$%s$]" % (vary,self.var_param_out(case, vary)[self.ud])
            data1["xlabel"] = r""
            data1["conf"] = np.abs(s_y[vary][conf])

            self.plt.plot_scatter_histsensi(data1)

        self.msg("Sensitivity histograms plotted in %s" % folder2)

    def output_plts_models_XY(self, folder_path, case):

        """

        .. _pymetamodels_output_plts_models_XY:

        **Synopsis:**
            * XY 2D scatter plots of the DOEY variables versus DOEX variables
            * Plots are placed in a sub-folder of the output folder

        **Args:**
            * folder_path: path to the folder where to save the plots images
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * Saves in the folder_path location the plots regarding the XY plots of the DOEY variables versus DOEX variables

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` or :ref:`Tutorial A03 <ex_coupled_function_metamodel>`

        |

        """

        ## XY plots of the DOEY variables versus DOEX variables
        ##

        # new folder
        folder = os.path.join(folder_path, "XY_model_plot")
        if not os.path.exists(folder): os.makedirs(folder)

        vars_x = self.vars_keys(case, not_cte = True)
        vars_o = self.vars_out_keys(case)

        vars_in = self.vars_parameter_matrix(case)
        vars_out = self.vars_out_parameter_matrix(case)

        for vary in vars_o:

            for varx in vars_x:

                data = {}
                data["legend"] = r"$%s$ vs. $%s$ model" % (varx,vary)
                data["output_folder"] = folder
                data["file name"] = r"XYplt_%s_%s_%s" % (case,varx,vary)
                data["xname"] = varx
                data["xdata"] = self.doeX(case)[varx]
                data["yname"] = vary
                data["ydata"] = self.doeY(case)[vary]
                data["format"] = "*"
                data["ylabel"] = r"$%s$ [%s]" % (vary,self.var_param_out(case, vary)["ud"])
                data["xlabel"] = r"$%s$ [%s]" % (varx,self.var_param(case, varx)["ud"])

                self.plt.plot_scatterXY_model(data)

        self.msg("XY 2D scatter plots of the DOEY variables versus DOEX variables plotted in %s" % folder)                

    def output_plts_models_XYZ(self, folder_path, case, default_other_vars_level = 0.5, text_annotation = True, scatter = False):

        """

        .. _pymetamodels_output_plts_models_XYZ:

        **Synopsis:**
            * XYZ 3D scatter plots of the DOEY variables versus DOEX variables
            * Plots are placed in a sub-folder of the output folder

        **Args:**
            * folder_path: path to the folder where to save the plots images
            * case: case name to be execute

        **Optional parameters:**
            * default_other_vars_level = 0.5: range fraction for non doeX X,Y variables
            * text_annotation = True: switch on and off the text annotation regarding default_other_vars_level
            * scatter = True: show scatter plot

        **Returns:**
            * Saves in the folder_path location the plots regarding the XY plots of the DOEY variables versus DOEX variables

        .. note::

            * Non X,Y plots variables values are compute according var default_other_vars_level
            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` or :ref:`Tutorial A03 <ex_coupled_function_metamodel>`

        |

        """

        ## XY plots of the DOEY variables versus DOEX variables
        ##

        # new folder
        folder = os.path.join(folder_path, "XYZ_model_plot")
        if not os.path.exists(folder): os.makedirs(folder)

        vars_x = self.vars_keys(case, not_cte = True)
        vars_o = self.vars_out_keys(case)

        vars_in = self.vars_parameter_matrix(case)
        vars_out = self.vars_out_parameter_matrix(case)

        for vary in vars_o:

            vars_x_sec  = vars_x.copy()

            for varx in vars_x:

                vars_x_sec.remove(varx)

                for varxs in vars_x_sec:

                    data = {}
                    #data["legend"] = r"$%s$ vs. $%s$ model" % (varx,vary)
                    data["legend"] = "Scatter values"
                    data["output_folder"] = folder
                    data["file name"] = r"XYZplt_%s_%s_%s_%s" % (case,vary,varx,varxs)
                    data["xname"] = varx
                    data["xdata"] = self.doeX(case)[varx]
                    data["yname"] = varxs
                    data["ydata"] = self.doeX(case)[varxs]
                    data["zname"] = vary
                    data["zdata"] = self.doeY(case)[vary]
                    data["format"] = "*"
                    data["xlabel"] = r"$%s$ [%s]" % (varx,self.var_param(case, varx)["ud"])
                    data["ylabel"] = r"$%s$ [%s]" % (varxs,self.var_param(case, varxs)["ud"])
                    data["zlabel"] = r"$%s$ [%s]" % (vary,self.var_param_out(case, vary)["ud"])
                    data["color"] = "b"
                    data["marker"] = "*"
                    data["data_points_size"] = 5
                    data["scatter"] = scatter

                    # surface plot
                    X, Y, Z, doeX_other_vars_values = self.metamodel_surface(case, varx, varxs, vary, default_other_vars_level = default_other_vars_level, grid_size = 125)
                    data["legend_grid"] = "Metamodel surface"
                    data["xdata_grid"] = X
                    data["ydata_grid"] = Y
                    data["zdata_grid"] = Z

                    if text_annotation:
                        txt = "DoeX vars computed as $%.2E*(up_{range}-low_{range})+low_{range}$)\n" % default_other_vars_level
                        ii = 0
                        for key in doeX_other_vars_values:
                            if ii > 6:
                                txt += "..."
                                break
                            else:
                                txt += "%s=%.2E [%s]\n" % (key,doeX_other_vars_values[key],self.var_param(case, key)["ud"])
                            ii += 1

                        data["anotate"]=[txt,0,0.99,'xx-small'] #[txt,pos_x, pos_y, fontsize]

                    self.plt.plot_scatterXYZ_model(data)

        self.msg("XYZ 3D scatter plots of the DOEY variables versus DOEX variables in %s" % folder)

    def metamodel_surface(self, case, varx, vary, varz, default_other_vars_level = 0.5, grid_size = 100):

        ## Constrct a metamodel surface
        ## default_other_vars_level: default for the rest of the doeX vars

        obj_metamodel = self.obj_metamodel(case)
        vars_x = obj_metamodel.doeX_varX
        shape_doeY = obj_metamodel.doeY_np_shape

        if grid_size <= 1: grid_size = 2

        # Build meshgrid variables
        boundX = self.var_bounds(case, varx)
        X = np.linspace(start=boundX[0], stop=boundX[1], num=grid_size)

        boundY = self.var_bounds(case, vary)
        Y= np.linspace(start=boundY[0], stop=boundY[1], num=grid_size)

        X, Y = np.meshgrid(X, Y)

        Z = np.ndarray(X.shape,dtype=X.dtype)

        # Build doeX for predict Z values, vars of non interest computed with default_other_vars_level
        doeX_other_vars_values = {}
        for i_varx in vars_x:
            if i_varx == varx:
                pass
            elif i_varx == vary:
                pass
            else:
                bound = self.var_bounds(case, i_varx)
                var_value = ((bound[1] - bound[0]) * default_other_vars_level) + bound[0]
                doeX_other_vars_values[i_varx] = var_value

        # Build doeX for predict Z values
        doeX_predict = obj_metamodel.doeX_np_empty(grid_size*grid_size)
        zz = 0
        for ii in range(0,grid_size):

            for jj in range(0,grid_size):

                lst_var_features = []
                for i_varx in vars_x:

                    if i_varx == varx:
                        lst_var_features.append(X[ii,jj])
                    elif i_varx == vary:
                        lst_var_features.append(Y[ii,jj])
                    else:
                        lst_var_features.append(doeX_other_vars_values[i_varx])

                doeX_predict[zz,:] = np.asarray(lst_var_features, dtype=np.dtype('float64'))[:]
                zz = zz + 1

        # Predict Z values
        varz_ii = obj_metamodel.doeY_index(varz)
        if len(shape_doeY) == 1:
            arrZ = obj_metamodel.predict(doeX_predict)[:]
        else:
            arrZ = obj_metamodel.predict(doeX_predict)[:,varz_ii]

        # Build Z grid values
        zz = 0
        for ii in range(0,grid_size):

            for jj in range(0,grid_size):

                Z[ii,jj] = arrZ[zz]
                zz = zz + 1

        return X, Y, Z, doeX_other_vars_values

    def metamodel_surface_deprecated(self, case, varx, vary, varz, default_other_vars_level = 0.5, grid_size = 100):

        ## Constrct a metamodel surface
        ## default_other_vars_level: default for the rest of the doeX vars

        obj_metamodel = self.obj_metamodel(case)
        vars_x = obj_metamodel.doeX_varX

        # Build meshgrid variables
        boundX = self.var_bounds(case, varx)
        X = np.linspace(start=boundX[0], stop=boundX[1], num=grid_size)

        boundY = self.var_bounds(case, vary)
        Y= np.linspace(start=boundY[0], stop=boundY[1], num=grid_size)

        X, Y = np.meshgrid(X, Y)

        Z = np.ndarray(X.shape,dtype=X.dtype)

        # Predict Z values
        for ii in range(0,grid_size):

            for jj in range(0,grid_size):

                lst_var_features = []
                for i_varx in vars_x:

                    if i_varx == varx:
                        lst_var_features.append(X[ii,jj])
                    elif i_varx == vary:
                        lst_var_features.append(Y[ii,jj])
                    else:
                        bound = self.var_bounds(case, i_varx)
                        var_value = ((bound[1] - bound[0]) * default_other_vars_level) + bound[0]
                        lst_var_features.append(var_value)

                prediction = obj_metamodel.predict_1D(varz, lst_var_features)

                Z[ii,jj] = prediction

        return X, Y, Z

    def output_plts_models_residuals_plot(self, folder_path, case):

        """

        .. _pymetamodels_output_plts_models_residuals_plot:

        **Synopsis:**
            * XY 2D scatter plots of the residual values, DOEY varibles metamodel predictions ploted versus DOEY variable values
            * Plots are placed in a sub-folder of the output folder

        **Args:**
            * folder_path: path to the folder where to save the plots images
            * case: case name to be execute

        **Optional parameters:**
            * None

        **Returns:**
            * Saves in the folder_path location the residual values, DOEY varibles predictions ploted versus DOEY variable values

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` or :ref:`Tutorial A03 <ex_coupled_function_metamodel>`

        |

        """

        ## XY plots of the DOEY variables versus DOEX variables
        ##

        # new folder
        folder = os.path.join(folder_path, "XY_model_plot_residuals")
        if not os.path.exists(folder): os.makedirs(folder)

        vars_x = self.vars_keys(case, not_cte = True)
        vars_o = self.vars_out_keys(case)

        obj_metamodel = self.obj_metamodel(case)
        doeX_test, var_keysX = self.doeX_asnp(case, return_keysX = True)
        doeY_test, var_keysY = self.doeY_asnp(case, return_keysY = True)

        for vary in vars_o:

            (varY_predict, varY_values, score, varY_predict_line) = obj_metamodel.score_doeY_target(doeX_test, doeY_test, vary)

            data = {}
            data["legend"] = r"predicted vs. values of %s" % (vary)
            data["output_folder"] = folder
            data["file name"] = r"XYplt_residuals_%s_%s" % (case,vary)
            data["xname"] = "Data values %s" % vary
            data["yname"] = "Predicted values %s" % vary
            data["xdata"] = varY_values
            data["ydata"] = varY_predict
            data["format"] = "*"
            data["xlabel"] = r"Data values $%s$ [%s]" % (vary, self.var_param_out(case, vary)["ud"])
            data["ylabel"] = r"Predicted values $%s$ [%s]" % (vary, self.var_param_out(case, vary)["ud"])

            data["xdata2"] = varY_values
            data["ydata2"] = varY_predict_line
            data["legend2"] = r"regression, $R^2=%.2f$" % (score)
            data["format2"] = "-"

            self.plt.plot_scatterXY_model(data)

        self.msg("XY 2D scatter plots of the residual values in %s" % folder)

    #### Other functions
    #####################################

    def error(self, msg):

        print("Error: " + msg)
        if self.objlog:
            self.objlog.warning("ID01 - %s" % msg)
        raise ValueError(msg)

    def msg(self, msg):
        print("Msg: " + msg)
        if self.objlog:
            self.objlog.info("ID01 - %s" % msg)        

    def is_number(self, s):
        try:
            if s is None: return False
            float(s)
            return True
        except ValueError:
            return False        

    def print_dict(self, dictionary, ident = '', braces=1):
        """

        .. _pymetamodels_print_dict:

        **Synopsis:**
            * Recursively prints nested dictionaries

        **Args:**
            * dictionary: dictionary object to recursively be printed

        **Optional parameters:**
            * ident: identificator string
            * braces: number of braces

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`

        |

        """

        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.msg('  %s%s%s%s' % (ident,braces*'[',key,braces*']') )
                self.print_dict(value, ident+'  ', braces+1)
            else:
                self.msg("  "+ident+'%s = %s' % (key, value))

    def is_number(self,s):

        ## Check if is a number

        try:
            float(s) # for int, long and float
        except ValueError:
            try:
                complex(s) # for complex
            except ValueError:
                return False

        return True

    def is_none(self,s):

        ## Check if is None

        if s is None:
            return True
        elif type(s) == type(""):
            if s.lower() == "none".lower():
                return True
            else:
                return False
        else:
            return False
