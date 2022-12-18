
.. _ex_damped_oscillator_optimization:

Tutorial case A07: "Calibration of a damped oscillator"
:::::::::::::::::::::::::::::::::::::::::::::::::::::::

In this tutorial is described the extension of tutorial A05 :numref:`ex_damped_oscillator` regarding a damped oscillator with the calibration of the oscillating system according a given reference signal. This problem is described in :eq:`eq_caliosci`.

.. math::
    :nowrap:
    :label: eq_caliosci

    \begin{equation}
    \begin{cases}
    optimization
    \begin{cases}
        calibrate[(simulation,reference)] = 0 \\
        constrains: \\
        \quad \quad w_{damped} \leq 10 \ \text{[1/s]} \\
        \quad \quad m \in [0.1, 5.0] \ \text{[kg]} \\
        \quad \quad k \in [10, 50] \ \text{[N/m]} \\
        \quad \quad D \in [0.01, 0.05] \\
        \quad \quad E_{kin} \in [10, 50] \ \text{[Nm]} \\
    \end{cases} \\
    \end{cases}
    \end{equation}

The input and output variable for each of the analysis cases are define in the problem definition spreadsheet file (:download:`Download: Problem definition spreadsheet file </_examples/A07 damped_oscillator_optimize/configuration_spreadsheet.xls>`).

The same model as in tutorial A05 (see :numref:`ex_damped_oscillator`) is used, with a small modification. The constrain function and the objective function to be minimize as equal to zero has been added (see :numref:`ex_damped_oscillator_model_min`). In addition the reference signal to be math in the calibration is loaded (:download:`Download: Reference signal spreadsheet file </_examples/A07 damped_oscillator_optimize/reference.csv>`) in order to build the objective function.

The analysis is performed with pymetamodels based on a basic analysis script as follows in **test_analysis.py** (see :numref:`ex_damped_oscillator_test_analysis_min`). Apart of the sensitivity analysis and metamodeling construction carried out in tutorial A05. The optimization routine is call with the function :ref:`run_optimization_problem() <pymetamodels_run_optimization_problem>` using a *"general_fast"* schema approach.

The calibration problem is solve resulting in :eq:`eq_caliosci_sol`. When the reference and simulation signal for the aforementioned values are compared in :numref:`damped_oscillator` showing a good agreement for the calibration solution values.

.. math::
    :nowrap:
    :label: eq_caliosci_sol

    \begin{equation}
    \begin{cases}
    optimization
    \begin{cases}
        calibrate[(simulation,reference)] = -8.7e-04  \\
        constrains: \\
        \quad \quad x_{max} = 4.45555556 \ \text{[m]} \\
        \quad \quad w_{damped} = 2.02715485 \ \text{[1/s]} \\
        \quad \quad m = 3.775 \ \text{[kg]} \\
        \quad \quad k = 15. \ \text{[N/m]} \\
        \quad \quad D = 0.04 \\
        \quad \quad E_{kin} = 10. \ \text{[Nm]} \\
    \end{cases} \\
    \end{cases}
    \end{equation}

.. include:: ./images/damped_oscillator.inc

.. literalinclude:: /_examples/A07 damped_oscillator_optimize/test_analysis.py
    :linenos:
    :language: python
    :caption: Damped oscillator test analysis. Calibration analysis
    :name: ex_damped_oscillator_test_analysis_min

.. literalinclude:: /_examples/A07 damped_oscillator_optimize/damped_model.py
    :linenos:
    :language: python
    :caption: Damped oscillator model code block. Calibration analysis
    :name: ex_damped_oscillator_model_min
