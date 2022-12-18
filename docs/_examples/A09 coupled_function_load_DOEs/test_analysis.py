
## Analytical model,
##

import os, sys, random
import numpy as np
import gc

import pymetamodels
import coupled_function_model as model_obj

class analysis(object):

    """

    .. _model_coupled_function_obj_conf:

    **Synopsis:**
        * Analysis framework with pymetamodels
        * Coupled function tutorial

    """

    def __init__(self):

        ## Initial variables
        self.name = "coupled_function_load_DOEs"
        self.folder_script = os.path.dirname(os.path.realpath(__file__))

        self.folder_path_inputs = self.folder_script

        self.folder_path_outputs = os.path.join(os.path.join(os.path.join(self.folder_script, os.pardir), "_examples_raw"),self.name + "_out")
        if not os.path.exists(self.folder_path_outputs): os.makedirs(self.folder_path_outputs)

        self.file_name_inputs = r"configuration_spreadsheet"
        self.file_name_outputs = r"outputs"

        ## Initialise 
        self.mita = pymetamodels.load()
        self.mita.logging_start(self.folder_path_outputs)

    def run_model(self):

        ## Build programatically configuration sheet
        _conf = self.mita.objconf()
        _conf.start(self.folder_path_inputs, self.file_name_inputs)

        _conf.add_case("case_1","case_1_vars_sensi","case_out",1024,"DGSM")
        _conf.add_case("case_2","case_1_vars_sensi","case_out_2",256,"DGSM")

        _conf.add_vars_sheet_variable("case_1_vars_sensi","X1", 1, -3.14, 3.14, "unif", True, 5/100., "[-]", "X1")
        _conf.add_vars_sheet_variable("case_1_vars_sensi","X2", 1, -3.14, 3.14, "unif", True, 5/100., "[-]", "X2")
        _conf.add_vars_sheet_variable("case_1_vars_sensi","X3", 1, -3.14, 3.14, "unif", True, 5/100., "[-]", "X3")
        _conf.add_vars_sheet_variable("case_1_vars_sensi","X4", 1, -3.14, 3.14, "unif", True, 5/100., "[-]", "X4")
        _conf.add_vars_sheet_variable("case_1_vars_sensi","X5", 1, -3.14, 3.14, "unif", True, 5/100., "[-]", "X5")
        _conf.add_vars_sheet_variable("case_1_vars_sensi","cte", 1., 1., 1., "unif", False, 5/100., "[-]", "cte")

        _conf.add_output_sheet("case_out", "Y", None, "m^3", False, True, False, False, False, "function output")
        _conf.add_output_sheet("case_out_2", "Y", None, "m^3", False, True, False, False, False, "function output")
        _conf.add_output_sheet("case_out_2", "Y2", None, "m^3", False, False, False, False, False, "function output")

        _conf.save_conf()

        ## Inputs load
        self.mita.read_xls_case(self.folder_path_inputs, self.file_name_inputs)

        """
        ## Save DOEX and DOEY
        ## Sampling, metamodeling and optimization
        for case in self.mita.keys():

            ## Run samplimg cases
            self.mita.run_sampling_routine(case)

            ## Run / Load model
            self.model_iteration(case, model_obj)            

        self.mita.save_tofile_DOEX(self.folder_path_outputs, "DOEX")
        self.mita.save_tofile_DOEY(self.folder_path_outputs, "DOEY")
        """

        ## Load DOEX and DOEY
        ## Sampling, metamodeling and optimization

        self.mita.read_fromfile_DOEX(self.folder_path_outputs, "DOEX")
        self.mita.read_fromfile_DOEY(self.folder_path_outputs, "DOEY")

        for case in self.mita.keys():

            ## Sensitivity analysis
            self.mita.run_sensitivity_analisis(case)

            ## Metamodeling construction
            self.mita.run_metamodel_construction(case, scheme = "general_fast")

            ## Solve optimization problem
            self.mita.run_optimization_problem(case, scheme = "general_fast")                     
        
        ##
        self.mita.run_sensitivity_normalization()

        ## Ploting and others
        for case in self.mita.keys():

            ## Plots cross varible relation in the sensitivity analysis
            self.mita.output_plts_sensitivity(self.folder_path_outputs, case)

            ## Plots showing the model DOEX and DOEY variables relationship 2D XY
            self.mita.output_plts_models_XY(self.folder_path_outputs, case)

            ## Residual values plots
            self.mita.output_plts_models_residuals_plot(self.folder_path_outputs, case)

            ## Plots showing the model DOEX and DOEY variables relationship 3D XYZ
            self.mita.output_plts_models_XYZ(self.folder_path_outputs, case)

        ## Output variables save
        folder_path = self.folder_path_outputs
        file_name = self.file_name_outputs
        out_path = self.mita.output_xls(folder_path, file_name, col_start = 0, tit_row = 0)

    def model_iteration(self, case, _model_obj):

        ## Model iteration to generate Y doe output values
        doeX = self.mita.doeX(case)
        doeY = self.mita.doeY(case)
        vars_in = self.mita.vars_parameter_matrix(case)
        case_dict = self.mita.case[case]
        samples = len(doeX[list(doeX.keys())[0]])

        for ii in range(0, samples):

            obj_model = _model_obj.model_obj()

            # initialize dictionaries
            obj_model.doeX = doeX
            obj_model.doeY = doeY
            obj_model.case_dict = case_dict
            obj_model.ii = ii
            obj_model.vars_in = vars_in

            # add variables as atributes
            for key in obj_model.doeX.keys():
                setattr(obj_model, key, obj_model.doeX[key][obj_model.ii])

            # run the model for ii sample
            obj_model.run_model()

if __name__ == "__main__":

    analysis = analysis()

    analysis.run_model()
