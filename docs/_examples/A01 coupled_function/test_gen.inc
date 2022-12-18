
.. _ex_coupled_function:

Tutorial case A01: "Sensitivity analysis of a coupled function"
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Example of DOE design and sensitivity analysis for a coupled function.

The coupled function is defined as follows,

.. math::
    :nowrap:
    :label: eq_sens

    \begin{equation}
        Y = 0.5 \cdot X1+X2+0.5 \cdot X1 \cdot X2+5 \cdot sin(X3)+0.2 \cdot X4+0.1 \cdot X5
    \end{equation}

The input and output variable for each of the analysis cases are define in the problem definition spreadsheet file (:download:`Download: Problem definition spreadsheet file </_examples/A01 coupled_function/configuration_spreadsheet.xls>`). This definition spreadsheet includes the table with the main cases definition (:numref:`coupled_function_cases`), the table with model input variables (:numref:`coupled_func_case_0_vars_sensi`) and the table with output variables (:numref:`coupled_func_case_out`).

.. include:: /tables/coupled_function_cases.inc

.. include:: /tables/coupled_func_case_0_vars_sensi.inc

.. include:: /tables/coupled_func_case_out.inc

The analysis is performed with pymetamodels based on a basic analysis script as follows in **test_analysis.py**,

.. literalinclude:: /_examples/A01 coupled_function/test_analysis.py
    :linenos:
    :language: python
    :emphasize-lines: 59, 74
    :caption: Coupled function test analysis. Sensitivity analysis
    :name: ex_coupled_function_test_analysis

The analysis frameworks iterates the model defined in a secondary file as follows in **coupled_function_model.py**,

.. literalinclude:: /_examples/A01 coupled_function/coupled_function_model.py
    :linenos:
    :language: python
    :emphasize-lines: 47, 37
    :caption: Coupled function model code block. Sensitivity analysis
    :name: ex_coupled_function_model

Running the analysis script generates and output results files containing the DOEX input data, the DOEY output data and the senstitivity analysis,

:download:`Download: Output spreadsheet file </_examples/_examples_raw/coupled_function_out/outputs.xls>`

The console output of running the tutorial script is as follow,

.. include:: /_examples/A01 coupled_function/test_analysis.testout
   :literal:

The function **output_plts_sensitivity()** generates several plots showing the DOEX variables correlation and a summary of the sensitivity analysis. For example this can be seen in :numref:`multi_barplot_sensi_case_0_Y` and :numref:`multi_plot_cross_X1_X2_case_0`. This plot are located in the output folder.

.. include:: /images/multi_barplot_sensi_case_0_Y.inc

.. include:: /images/multi_plot_cross_X1_X2_case_0.inc
