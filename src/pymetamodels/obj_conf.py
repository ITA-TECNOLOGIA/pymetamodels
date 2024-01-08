#!/usr/bin/env python3

import os,sys
import xlwt, xlrd
from xlutils.copy import copy as xlutils_copy
import pymetamodels.obj_data as obj_data

class objconf(object):

    """Python class to build programatically the configuration spreadsheet

        :platform: Windows
        :synopsis: to build programatically the pymetamodels configuration spreadsheet

        :Dependences: |Excel|, xlrd, xlwt, numpy

        |

    """

    ## Common functions
    ###################

    def __init__(self, model):

        self.model = model

        self.folder_path = None
        self.file_name = None
        self.sheet = "cases"
        self.col_start = 0
        self.row_start = 1
        self.tit_row = 0
        self.vars_sheet_field= self.model.vars_sheet
        self.vars_out_sheet_field= self.model.output_sheet

        self.cases_keys = []
        self.cases = obj_data.objdata()
        self.vars_sheet_keys = []
        self.vars_sheet = obj_data.objdata()
        self.output_sheet_keys = []
        self.output_sheet = obj_data.objdata()   

        pass

    def start(self, folder_path, file_name):

        """

        .. _objconf_start:

        **Synopsis:**
            * Initialise the objconf with the file and path name

        **Args:**
            * folder_path: folder path
            * file_name: conf file name without extension

        **Optional parameters:**
            * None

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`

        |

        """        

        self.folder_path = folder_path
        self.file_name = file_name 

        self.cases_keys = []
        self.cases = obj_data.objdata()
        self.vars_sheet_keys = []
        self.vars_sheet = obj_data.objdata()
        self.output_sheet_keys = []
        self.output_sheet = obj_data.objdata()   

    def add_case(self, case_name, vars_sheet, output_sheet,	samples, sensitivity_method, comment = None, others = {}, force_overwrite = False):

        """

        .. _objconf_add_case:

        **Synopsis:**
            * Adds a new entry to the cases sheet

        **Args:**
            * case_name: name and id of the case, is a unique key value
            * vars_sheet: name and id of the sheet where are described the input vars for the given case
            * output_sheet: name and id of the sheet where are described the output vars for the given case
            * samples: number of samples for the sampling activities (:math:`2^N \ values`)
            * sensitivity_method: name and id of the sensitivity analysis method (see :numref:`pymetamodels_conf_sampling`)

        **Optional parameters:**
            * comment = None: comment for the given case
            * others = {}: dictionary with other variables
            * force_overwrite = False: force over writting **case_name** data

        **Returns:**
            * True if the action was possible, False if the key **case_name** already exist

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * See the configurations spread sheet description :ref:`Configuration spreadsheet <pymetamodels_configuration>`

        |

        """ 

        lst_sensitivity_method = list(self.model.samp.ini_analisis_type().keys())

        if self.sheet not in self.cases:
            self.cases[self.sheet] = obj_data.objdata()

        if case_name in self.cases[self.sheet] and not force_overwrite:
            self.msg("Already exist case_name item: %s" % case_name)
            return False

        else:
            if sensitivity_method not in lst_sensitivity_method:
                self.msg("Invalid sensitivity_method: %s" % sensitivity_method)
                return False

            self.cases[self.sheet][case_name] = obj_data.objdata()
            self._add_case(self.sheet, case_name, self.model.case_tit, case_name)
            self._add_case(self.sheet, case_name, self.model.vars_sheet, vars_sheet)
            self._add_case(self.sheet, case_name, self.model.output_sheet, output_sheet)
            self._add_case(self.sheet, case_name, self.model.samples, samples)
            self._add_case(self.sheet, case_name, self.model.sensitivity_method, sensitivity_method)
            
            self._add_case(self.sheet, case_name, self.model.comment, comment)

            for key in others.keys():
                self._add_case(self.sheet, case_name, key, others[key])

            return True

    def _add_case(self, case_tit, case_name, variable, value):

        self.cases[case_tit][case_name][variable] = value

        if variable not in self.cases_keys:
            self.cases_keys.append(variable)


    def add_vars_sheet_variable(self, vars_sheet, variable,	value,	min, max, distribution,	is_range, cov_un, ud, alias, comment = None, others = {}, force_overwrite = False):

        """

        .. _objconf_add_vars_sheet_variable:

        **Synopsis:**
            * Adds a new entry to the input variable to the vars_sheet

        **Args:**
            * vars_sheet: name and id of the sheet where are described the input vars for the given case
            * variable: name and id of the input variable, is a unique key value
            * value: nominal value of the input variable, use in case is not considered a ranged variable in the DOEX
            * min: min value of the ranged variable in the DOEX
            * max: max value of the ranged variable in the DOEX
            * distribution: type of range distribution (unif, triang, norm, lognorm)
            * is_range: TRUE or FALSE value to choose if the variable is a range or a single value in the DOEX
            * cov_un: covariance used for the generation of the norm distributions
            * ud: units name for the variable (i.e. [m])
            * alias: alias name for the variable

        **Optional parameters:**
            * comment = None: comment for the given case
            * others = {}: dictionary with other variables
            * force_overwrite = False: force over writting **variable** data

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * See the configurations spread sheet description :ref:`Configuration spreadsheet <pymetamodels_configuration>`

        |

        """          
        if vars_sheet not in self.vars_sheet:
            self.vars_sheet[vars_sheet] = obj_data.objdata()

        if variable in self.vars_sheet[vars_sheet] and not force_overwrite:
            self.msg("Already exist in vars_sheet the variable item : %s" % variable)
            return False

        else:

            self.vars_sheet[vars_sheet][variable] = obj_data.objdata()
            self._add_vars_sheet(vars_sheet, variable, self.model.variable, variable)
            self._add_vars_sheet(vars_sheet, variable, self.model.value, value)
            self._add_vars_sheet(vars_sheet, variable, self.model.min_bound, min)
            self._add_vars_sheet(vars_sheet, variable, self.model.max_bound, max)
            self._add_vars_sheet(vars_sheet, variable, self.model.distribution, distribution)
            self._add_vars_sheet(vars_sheet, variable, self.model.is_variable, is_range)
            self._add_vars_sheet(vars_sheet, variable, self.model.cov_un, cov_un)
            self._add_vars_sheet(vars_sheet, variable, self.model.ud, ud)
            self._add_vars_sheet(vars_sheet, variable, self.model.alias, alias)
            
            self._add_vars_sheet(vars_sheet, variable, self.model.comment, comment)

            for key in others.keys():
                self._add_vars_sheet(vars_sheet, variable, key, others[key])

            return True

    def _add_vars_sheet(self, vars_sheet, case_name, variable, value):

        self.vars_sheet[vars_sheet][case_name][variable] = value

        if variable not in self.vars_sheet_keys:
            self.vars_sheet_keys.append(variable)      

    def add_output_sheet(self, output_sheet, variable, value, ud, array, op_min, op_min_0, ineq_0, eq_0, alias, comment = None, others = {}, force_overwrite = False):

        """

        .. _objconf_add_output_sheet:

        **Synopsis:**
            * Adds a new entry to the output vars case sheet

        **Args:**
            * output_sheet: name and id of the sheet where are described the output vars for the given case
            * variable: name and id of the output variable, is a unique key value
            * value: nominal value of the output variable
            * ud: units name for the variable (i.e. [m])
            * array: TRUE or FALSE, is the output variable an array or single value
            * op_min: TRUE if variable is to be minimize, :math:`min(DOEY_{var})`
            * op_min_0: TRUE if variable is to be optimize to 0, :math:`objective(DOEY_{var}=0)`
            * ineq_0: TRUE if variables is consider for an inequality constrain, :math:`DOEY_{var}>=0`
            * eq_0: TRUE if variables is consider for an equality constrain =0, :math:`DOEY_{var}=0`
            * alias: alias name for the variable

        **Optional parameters:**
            * comment = None: comment for the given case
            * others = {}: dictionary with other variables
            * force_overwrite = False: force over writting **variable** data

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * See the configurations spread sheet description :ref:`Configuration spreadsheet <pymetamodels_configuration>`

        |

        """         

        if output_sheet not in self.output_sheet:
            self.output_sheet[output_sheet] = obj_data.objdata()

        if variable in self.output_sheet[output_sheet] and not force_overwrite:
            self.msg("Already exist in output_sheet the variable item : %s" % variable)
            return False

        else:

            self.output_sheet[output_sheet][variable] = obj_data.objdata()
            self._add_output_sheet(output_sheet, variable, self.model.variable, variable)
            self._add_output_sheet(output_sheet, variable, self.model.value, value)
            self._add_output_sheet(output_sheet, variable, self.model.ud, ud)
            self._add_output_sheet(output_sheet, variable, self.model.ud, ud)
            self._add_output_sheet(output_sheet, variable, self.model.as_array, array)
            self._add_output_sheet(output_sheet, variable, self.model.op_min, op_min)
            self._add_output_sheet(output_sheet, variable, self.model.op_min_eq, op_min_0)
            self._add_output_sheet(output_sheet, variable, self.model.op_ineq, ineq_0)
            self._add_output_sheet(output_sheet, variable, self.model.op_eq, eq_0)
            self._add_output_sheet(output_sheet, variable, self.model.alias, alias)
                        
            self._add_output_sheet(output_sheet, variable, self.model.comment, comment)

            for key in others.keys():
                self._add_output_sheet(output_sheet, variable, key, others[key])

            return True

    def _add_output_sheet(self, output_sheet, case_name, variable, value):

        self.output_sheet[output_sheet][case_name][variable] = value

        if variable not in self.output_sheet_keys:
            self.output_sheet_keys.append(variable)   

    def check_consistency(self):

        """

        .. _objconf_check_consistency:

        **Synopsis:**
            * Check consistency between cases, the input vars case sheet and the output vars case sheet

        **Args:**
            * None

        **Optional parameters:**
            * None

        **Returns:**
            * True if pass or list of text messages

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * See the configurations spread sheet description :ref:`Configuration spreadsheet <pymetamodels_configuration>`

        |

        """         

        ## On development

        ## If input var is a range check min<max

        ## The input vars case sheet and the output vars case sheet

        ## Check for only one declaredoptimization objective function 

        return True
        
    def save_conf(self):

        """

        .. _objconf_save_conf:

        **Synopsis:**
            * Save the configuration to a file

        **Args:**
            * None

        **Optional parameters:**
            * None

        **Returns:**
            * True or False if errors

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`
            * See the configurations spread sheet description :ref:`Configuration spreadsheet <pymetamodels_configuration>`

        |

        """
                
        file_path = os.path.join(self.folder_path, self.file_name + '.xls')

        wb = xlwt.Workbook() # create empty workbook object  

        self._add_sheets(wb, self.cases, self.cases_keys)        
        self._add_sheets(wb, self.vars_sheet, self.vars_sheet_keys)        
        self._add_sheets(wb, self.output_sheet, self.output_sheet_keys)        

        ## Save
        wb.save(file_path)
               
        pass
    
    def _add_sheets(self, wb, data_dict, data_dict_keys):

        ## Styles
        style_title = xlwt.XFStyle()
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = xlwt.Style.colour_map['yellow']
        style_title.pattern = pattern
        style_title.font.colour_index = xlwt.Style.colour_map['indigo']
        style_title.font.name = "Calibri"
        style_title.font.height = 11*20 # 11 * 20, for 11 point
        style_title.alignment.horz = xlwt.Alignment().HORZ_CENTER
        
        style_rows = xlwt.XFStyle()
        style_rows.font.colour_index = xlwt.Style.colour_map['black']
        style_rows.font.name = "Calibri"
        style_rows.font.height = 11*20 # 11 * 20, for 11 point        
        style_rows.alignment.horz = xlwt.Alignment().HORZ_CENTER        
        
        ## addd sheets
        for key_sheet, value_sheet in data_dict.items():

            sheet_name = key_sheet
            try:
                sheet = wb.get_sheet(sheet_name)
            except:
                sheet = wb.add_sheet(sheet_name)

            # add titles
            ii = self.tit_row
            jj = self.col_start
            for key in data_dict_keys:

                sheet.write(ii,jj, key, style_title)
                jj = jj + 1

            # add values
            ii = self.row_start
            jj = self.col_start
            for key, value in value_sheet.items():

                jj = self.col_start
                for key2, value2 in value.items():

                    sheet.write(ii, jj, value2, style_rows)

                    jj = jj + 1

                ii = ii + 1   
        
        return True
    
    #### I/O data
    #####################################    

    def read_xls_sheet_to_dict(self, folder_path, file_name, sheet_name, col_start = 0, row_title = 0, units = False):

        """

        .. __read_xls_sheet_to_dict:

        **Synopsis:**
            * Load excel sheet (in .xls format) into a dictionary of arrays
            * The format is each column is a data channel or array of data
            * First row is the channel names
            * Second row is the units names

        **Args:**
            * folder_path: the folder path name
            * file_name: file name of the coupons file (type xls)
            * sheet_name: sheet name with the channel list or array of data

        **Optional parameters:**
            * col_start = 0: first column
            * row_title = 0: first row
            * units = False: does the sheet con tain a second row of units

        **Returns:**
            * (dct, dct_units): (dictionary, dictionary of units)

        .. note::

            * None

        |

        """         

        file_path = os.path.join(folder_path, file_name + '.xls')  

        wk = xlrd.open_workbook(file_path)      
        
        worksheet = wk.sheet_by_name(sheet_name)        

        dct = {}
        dct_units = {}

        for col in range(col_start, worksheet.ncols):
        
            title = worksheet.cell_value(row_title,col)
            
            if title.strip() == "": break
            
            if units:
                start = row_title + 2
                    
                val = worksheet.cell_value(row_title + 1,col)
                    
                dct_units[title] = val                  
                
            else:
                start = row_title + 1
            
            arr = []
            for row in range(start, worksheet.nrows):
                
                val = worksheet.cell_value(row,col)
                
                if str(val).strip() == "": break
                
                arr.append(val)
                
            dct[title] = arr          

        return dct, dct_units

    def save_dict_to_xls_sheet(self, folder_path, file_name, sheet_name, data_dict, col_start = 0, row_title = 0, dict_units = None, add_row_under_units = None, overwrite = True):

        """

        .. __save_dict_to_xls_sheet:

        **Synopsis:**
            * Save a dictionary of arrays (see :ref:`read_xls_sheet_to_dict() <read_xls_sheet_to_dict>`) into a excel sheet (in .xls format) 
            * The format is each column is a data channel or array of data
            * First row is the channel names
            * Second row is the units names

        **Args:**
            * folder_path: the folder path name
            * file_name: file name of the coupons file (type xls)
            * sheet_name: sheet name with the channel list or array of data
            * data_dict: dictionary of units per channel

        **Optional parameters:**
            * col_start = 0: first column
            * row_title = 0: first row
            * dict_units = None: 
            * add_row_under_units = None: value of a third row under the units
            * overwrite = True: overwrites the file creating a new one with the same name or updates the file if it exists

        **Returns:**
            * None

        .. note::

            * None

        |

        """ 

        file_path = os.path.join(folder_path, file_name + '.xls')

        if overwrite:
            wb = xlwt.Workbook() # create empty workbook object 
        else:
            if os.path.exists(file_path):
                rb = xlrd.open_workbook(file_path, formatting_info=True)
                wb = xlutils_copy(rb)
            else:
                wb = xlwt.Workbook() # create empty workbook object      

        ## Styles
        style_title = xlwt.XFStyle()
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = xlwt.Style.colour_map['yellow']
        style_title.pattern = pattern
        style_title.font.colour_index = xlwt.Style.colour_map['indigo']
        style_title.font.name = "Calibri"
        style_title.font.height = 11*20 # 11 * 20, for 11 point
        style_title.alignment.horz = xlwt.Alignment().HORZ_CENTER
        
        style_rows = xlwt.XFStyle()
        style_rows.font.colour_index = xlwt.Style.colour_map['black']
        style_rows.font.name = "Calibri"
        style_rows.font.height = 11*20 # 11 * 20, for 11 point        
        style_rows.alignment.horz = xlwt.Alignment().HORZ_CENTER           

        ## addd sheets
        
        try:
            sheet = wb.get_sheet(sheet_name)
        except:
            sheet = wb.add_sheet(sheet_name)

        # add titles
        ii = row_title
        start = row_title + 1
        jj = col_start
        for key in data_dict.keys():

            sheet.write(ii,jj, key, style_title)
            jj = jj + 1

        ## add units
        if type(dict_units) == type({}):

            ii = row_title + 1
            start = ii + 1
            jj = col_start
            for key in dict_units.keys():

                sheet.write(ii,jj, dict_units[key], style_title)
                jj = jj + 1

        if add_row_under_units:

            ii = row_title + 2
            start = ii + 1
            jj = col_start
            for key in dict_units.keys():

                sheet.write(ii,jj, add_row_under_units, style_title)
                jj = jj + 1            

        # add values
        jj = col_start
        for key, value in data_dict.items():

            for ii in range(0, len(value)):

                sheet.write(start+ii, jj, value[ii], style_rows)

            jj = jj + 1            

        wb.save(file_path)

    #### Other functions
    #####################################

    def error(self, msg):
        print("Error: " + msg)
        if self.model.objlog:
            self.model.objlog.warning("ID02 - %s" % msg)        
        raise ValueError(msg)

    def msg(self, msg):
        print("Msg: " + msg)        
        if self.model.objlog:
            self.model.objlog.info("ID02 - %s" % msg)          