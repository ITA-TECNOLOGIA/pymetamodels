
.. _ex_coupled_function_data_struct:

Tutorial case A02: "Sensitivity analysis of a coupled function, structure of data objects"
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Based on the DOE design and sensitivity analysis for a coupled function described in the previous :ref:`Tutorial case <ex_coupled_function>`. In this tutorial are described the data object structure. For this purpose the different data object structure is access and printed.

The input and output variable for each of the analysis cases are define in the problem definition spreadsheet file (:download:`Download: Problem definition spreadsheet file </_examples/A02 coupled_function_data_struct/configuration_spreadsheet.xls>`). This definition spreadsheet includes the table with the main cases definition (:numref:`coupled_function_cases`), the table with model input variables (:numref:`coupled_func_case_0_vars_sensi`) and the table with output variables (:numref:`coupled_func_case_out`).

.. include:: /tables/coupled_function_data_struct.inc

.. include:: /tables/coupled_function_data_struct_0_.inc

.. include:: /tables/coupled_function_data_struct_ou.inc

The code to print the data structure objects are described in the script as follows in **test_analysis.py**,

.. literalinclude:: /_examples/A02 coupled_function_data_struct/test_analysis.py
    :linenos:
    :language: python
    :emphasize-lines: 67, 72, 77, 82, 87
    :caption: Coupled function test analysis. Structure of data objects
    :name: ex_coupled_function_test_analysis_structure

The console output of running the tutorial script is as follow,

.. include:: /_examples/A02 coupled_function_data_struct/test_analysis.testout
   :literal:
