
.. _ex_kuwase_optimize:

Tutorial case A06: "Minimization of the Kursawe Functions"
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

In this tutorial is described the extension of tutorial A04 :numref:`ex_kuwase_metamodel` regarding the Kursawe Functions with the minimization of one of the Kursawe functions according a non linear constrain. This problem is described in :eq:`eq_kur`.

.. math::
    :nowrap:
    :label: eq_kur1

    \begin{equation}
    optimization
    \begin{cases}
        min[f_{2}(x)] \\
        constrains: \\
        \quad \quad f_{1}(x) \leqslant -14.5
    \end{cases}
    \end{equation}

The input and output variable for each of the analysis cases are define in the problem definition spreadsheet file (:download:`Download: Problem definition spreadsheet file </_examples/A06 kuwase_optimize/configuration_spreadsheet.xls>`).

The same model as in tutorial A04 (see :numref:`ex_kuwase_metamodel`) is used. The analysis is performed with pymetamodels based on a basic analysis script as follows in **test_analysis.py** (see :numref:`ex_kuwase_optimize_test_analysis`).

Apart of the sensitivity analysis and metamodeling construction carried out in tutorial A04. The optimization routine is call with the function :ref:`run_optimization_problem() <pymetamodels_run_optimization_problem>` using a *"general_fast"* schema approach. The optimization problem is solve resulting in :eq:`eq_kur1_result`.

.. math::
    :nowrap:
    :label: eq_kur1_result

    \begin{equation}
    \begin{cases}
    optimization\\
    result
    \end{cases}
    \begin{cases}
        min[f_{2}(x)] = -7.5074 \\
        constrains: \\
        \quad \quad f_{1}(x) \leqslant -14.5 \\
        where, \\
        f_{1}(x)=-14.5000 \\
        [X_1, X_2, X_3] = [-1.30575601, -0.87599992, -1.33713208]
    \end{cases}
    \end{equation}

.. literalinclude:: /_examples/A06 kuwase_optimize/test_analysis.py
    :linenos:
    :language: python
    :caption: Kursawe Functions test analysis. Minimization of the Kursawe Functions
    :name: ex_kuwase_optimize_test_analysis
