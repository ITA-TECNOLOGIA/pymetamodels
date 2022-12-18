
.. _ex_coupled_function_metamodel:

Tutorial case A03: "Models and metamodels construction of a coupled function"
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

In this tutorial are described the construction of the model and metamodels, as well as the default plotting of them. For this purpose is used the DOE design and sensitivity analysis for a coupled function described in the previous :ref:`tutorial case <ex_coupled_function>`.

The code to print the data structure objects are described in the script as follows in **test_analysis.py**,

.. literalinclude:: /_examples/A03 coupled_function_metamodel/test_analysis.py
    :linenos:
    :language: python
    :emphasize-lines: 64, 79, 82
    :caption: Coupled function test analysis. Metamodels
    :name: ex_coupled_function_test_analysis_metamodels

The function :ref:`run_metamodel_construction() <pymetamodels_run_metamodel_construction>` is included in the code to execute the routines related with the metamodeling construction of the DOEY as function of the DOEX data. This functions allows to choose different metamodeling schemas. This schemas perform a robust metamodeling search and find to determine the most adequate metamodeling ML model. Once the ML metamodel is determined allows to predict any data in the ranges of the DOEX and DOEY training values.

The function :ref:`output_plts_models_residuals_plot() <pymetamodels_output_plts_models_residuals_plot>` plots the residual values of the ML metamodel DOEY predictions in comparison with the ground truth DOEY values showing the regression score between the data and the predictions. An example can be seen in :numref:`multi_plot_XYplt_residuals_case_1_Y`.

In addition, the function :ref:`output_plts_models_XYZ() <pymetamodels_output_plts_models_XYZ>` plot all the combinations of XYZ plots showing the prediction surfaces constructed using the ML metamodel. An example can be seen in :numref:`multi_3Dplot_XYZplt_case_1_Y_X1_X2` and :numref:`multi_3Dplot_XYZplt_case_1_Y_X2_X3`.

.. include:: /images/multi_3Dplot_XYZplt_case_1_Y_X1_X2.inc

.. include:: /images/multi_3Dplot_XYZplt_case_1_Y_X2_X3.inc

.. include:: /images/multi_plot_XYplt_residuals_case_1_Y.inc
