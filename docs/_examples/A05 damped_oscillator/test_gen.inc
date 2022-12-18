
.. _ex_damped_oscillator:

Tutorial case A05: "Sensitivity analysis and metamodeling of a damped oscillator"
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

In this tutorial are described the construction of the model, sensitivity analysis and metamodels, as well as the default plotting of them. For this purpose is used the DOE design and sensitivity analysis for a coupled function described in the previous :ref:`tutorial case <ex_coupled_function>` as a base line. The damped oscillator is define by a mass paramter :math:`m`, a spring constant :math:`k` and  a damping factor :math:`D` (see :numref:`diagram_damped`) drive by a kinetical energy :math:`E_{kin}`. The damped oscillator is driven by the following equations,

.. math::
    :nowrap:
    :label: eq_damp

    \begin{equation}
    \begin{cases}
        \ddot{x} + 2 \cdot D \cdot w_0 \cdot \dot{x} + w_0^2 \cdot x = 0 \quad \quad Differential \ equation \ of \ movement \\
        E_{kin} = \frac{m \cdot v^2_0}{2}, \quad x_0=0 \quad \quad Initial \ conditions \\
        x(t) = e^{-D \cdot w_0 \cdot t} \cdot \frac{v_0}{w} \cdot sin(w \cdot t) \quad \quad Solution \ eq. \ x(t) \ dependant \ on \ the \ time \\
        w_0 = \sqrt{k/m} \quad w = w_0 \sqrt{1-D^2} \quad \quad Natural \ frequency, \ damped \ and \ non-damped
    \end{cases}
    \end{equation}

.. include:: ./images/diagram_damped.inc

The input and output variable for each of the analysis cases are define in the problem definition spreadsheet file (:download:`Download: Problem definition spreadsheet file </_examples/A05 damped_oscillator/configuration_spreadsheet.xls>`).

The analysis is performed with pymetamodels based on a basic analysis script as follows in **test_analysis.py**,

.. literalinclude:: /_examples/A05 damped_oscillator/test_analysis.py
    :linenos:
    :language: python
    :caption: Damped oscillator test analysis. Sensitivity analysis
    :name: ex_damped_oscillator_test_analysis

The analysis frameworks iterates the model defined in a secondary file as follows in **damped_oscillator.py**,

.. literalinclude:: /_examples/A05 damped_oscillator/damped_model.py
    :linenos:
    :language: python
    :caption: Damped oscillator model code block. Sensitivity analysis
    :name: ex_damped_oscillator_model

The function **output_plts_sensitivity()** generates several plots showing the DOEX variables correlation and a summary of the sensitivity analysis. For example this can be seen in :numref:`multi_plot_XYplt_residuals_case_2_x_{max}` and :numref:`multi_plot_XYplt_residuals_case_2_w_{damped}`. These plots are located in the output folder.

.. include:: /images/multi_plot_XYplt_residuals_case_2_x_{max}.inc

.. include:: /images/multi_plot_XYplt_residuals_case_2_w_{damped}.inc

The function :ref:`run_metamodel_construction() <pymetamodels_run_metamodel_construction>` is included in the code to execute the routines related with the metamodeling construction of the DOEY as function of the DOEX data. This functions allows to choose different metamodeling schemas. This schemas perform a robust metamodeling search and find to determine the most adequate metamodeling ML model. Once the ML metamodel is determined allows to predict any data in the ranges of the DOEX and DOEY training values.

The function :ref:`output_plts_models_residuals_plot() <pymetamodels_output_plts_models_residuals_plot>` plots the residual values of the ML metamodel DOEY predictions in comparison with the ground truth DOEY values showing the regression score between the data and the predictions. An example can be seen in :numref:`multi_plot_XYplt_residuals_case_1_kursawe1` and :numref:`multi_plot_XYplt_residuals_case_1_kursawe2`.

In addition, the function :ref:`output_plts_models_XYZ() <pymetamodels_output_plts_models_XYZ>` plot all the combinations of XYZ plots showing the prediction surfaces constructed using the ML metamodel. An example can be seen from  :numref:`multi_3Dplot_XYZplt_case_2_x_{max}_m_k` to :numref:`multi_3Dplot_XYZplt_case_2_w_{damped}_k_E_{kin}`.

.. include:: /images/multi_3Dplot_XYZplt_case_2_x_{max}_m_k.inc

.. include:: /images/multi_3Dplot_XYZplt_case_2_x_{max}_m_E_{kin}.inc

.. include:: /images/multi_3Dplot_XYZplt_case_2_x_{max}_m_D.inc

.. include:: /images/multi_3Dplot_XYZplt_case_2_x_{max}_k_E_{kin}.inc

.. include:: /images/multi_3Dplot_XYZplt_case_2_x_{max}_k_D.inc

.. include:: /images/multi_3Dplot_XYZplt_case_2_x_{max}_D_E_{kin}.inc

.. include:: /images/multi_3Dplot_XYZplt_case_2_w_{damped}_m_k.inc

.. include:: /images/multi_3Dplot_XYZplt_case_2_w_{damped}_m_E_{kin}.inc

.. include:: /images/multi_3Dplot_XYZplt_case_2_w_{damped}_m_D.inc

.. include:: /images/multi_3Dplot_XYZplt_case_2_w_{damped}_k_E_{kin}.inc
