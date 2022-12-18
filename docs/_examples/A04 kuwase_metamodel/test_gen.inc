
.. _ex_kuwase_metamodel:

Tutorial case A04: "Sensitivity analysis and metamodeling of Kursawe Functions"
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

In this tutorial are described the construction of the model, sensitivity analysis and metamodels, as well as the default plotting of them. For this purpose is used the DOE design and sensitivity analysis for a coupled function described in the previous :ref:`tutorial case <ex_coupled_function>` as a base line. The Kursawe Functions are define by three variables :math:`X_1, X_2, X_3` and two output functions  :math:`f_{kursawe1}, f_{kursawe2}`,

.. math::
    :nowrap:
    :label: eq_kur

    \begin{equation}
    \begin{cases}
        f_{kursawe1}(x) = \sum_{i=1}^{2} -10 \cdot e^{-0.2 \cdot \sqrt{X_{i}^{2} + X_{i+1}^{2} }} \\
        f_{kursawe2}(x) = \sum_{i=1}^{3} \left|X_i\right|^{0.8} + 5 \cdot sin (X_{i}^3) \\
        where;\\
        -5 <= X_i <= 5, i=1,...,n
    \end{cases}
    \end{equation}

The input and output variable for each of the analysis cases are define in the problem definition spreadsheet file (:download:`Download: Problem definition spreadsheet file </_examples/A04 kuwase_metamodel/configuration_spreadsheet.xls>`).

The analysis is performed with pymetamodels based on a basic analysis script as follows in **test_analysis.py**,

.. literalinclude:: /_examples/A04 kuwase_metamodel/test_analysis.py
    :linenos:
    :language: python
    :caption: Kursawe Functions test analysis. Sensitivity analysis
    :name: ex_kuwase_test_analysis

The analysis frameworks iterates the model defined in a secondary file as follows in **kursawe_model.py**,

.. literalinclude:: /_examples/A04 kuwase_metamodel/kursawe_model.py
    :linenos:
    :language: python
    :caption: Kursawe Functions model code block. Sensitivity analysis
    :name: ex_kuwase_model

The function **output_plts_sensitivity()** generates several plots showing the DOEX variables correlation and a summary of the sensitivity analysis. For example this can be seen in :numref:`multi_barplot_sensi_case_1_kursawe1` and :numref:`multi_barplot_sensi_case_1_kursawe2`. These plots are located in the output folder.

.. include:: /images/multi_barplot_sensi_case_1_kursawe1.inc

.. include:: /images/multi_barplot_sensi_case_1_kursawe2.inc

The function :ref:`run_metamodel_construction() <pymetamodels_run_metamodel_construction>` is included in the code to execute the routines related with the metamodeling construction of the DOEY as function of the DOEX data. This functions allows to choose different metamodeling schemas. This schemas perform a robust metamodeling search and find to determine the most adequate metamodeling ML model. Once the ML metamodel is determined allows to predict any data in the ranges of the DOEX and DOEY training values.

The function :ref:`output_plts_models_residuals_plot() <pymetamodels_output_plts_models_residuals_plot>` plots the residual values of the ML metamodel DOEY predictions in comparison with the ground truth DOEY values showing the regression score between the data and the predictions. An example can be seen in :numref:`multi_plot_XYplt_residuals_case_1_kursawe1` and :numref:`multi_plot_XYplt_residuals_case_1_kursawe2`.

In addition, the function :ref:`output_plts_models_XYZ() <pymetamodels_output_plts_models_XYZ>` plot all the combinations of XYZ plots showing the prediction surfaces constructed using the ML metamodel. An example can be seen in  :numref:`multi_3Dplot_XYZplt_case_1_kursawe1_X1_X2`, :numref:`multi_3Dplot_XYZplt_case_1_kursawe1_X1_X3`, :numref:`multi_3Dplot_XYZplt_case_1_kursawe1_X2_X3`, :numref:`multi_3Dplot_XYZplt_case_1_kursawe2_X1_X2` and :numref:`multi_3Dplot_XYZplt_case_1_Y_X2_X3`.

.. include:: /images/multi_3Dplot_XYZplt_case_1_kursawe1_X1_X2.inc

.. include:: /images/multi_3Dplot_XYZplt_case_1_kursawe1_X1_X3.inc

.. include:: /images/multi_3Dplot_XYZplt_case_1_kursawe1_X2_X3.inc

.. include:: /images/multi_3Dplot_XYZplt_case_1_kursawe2_X1_X2.inc

.. include:: /images/multi_plot_XYplt_residuals_case_1_kursawe1.inc

.. include:: /images/multi_plot_XYplt_residuals_case_1_kursawe2.inc
