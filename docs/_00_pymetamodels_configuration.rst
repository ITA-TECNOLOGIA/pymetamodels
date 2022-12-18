.. master file, created by
   sphinx-quickstart
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: subtitutions.txt

.. _pymetamodels_description:

The pymetamodels description
============================

Metamodeling, sensitivity, optimization, calibration and robustness analysis have become important tools for the virtual development of industrial products. These are based on the construction of soft ML metamodels from discrete populations of hard physics-based models (FE, CFD, molecular, continuum, mesosocopic, ...) input and output data calibrated with experimental characterisations. This design input variables are defined by their lower and upper bounds or by several possible discrete values schemas. In particular, in parametric optimization variables are systematically modified by mathematical algorithms to get an improvement of an existing design or to find a global optimum :cite:`Gandomi2013`. In the case of sensitivity analysis, discrete populations of models input and output data are generated using the original model or the metamodel to identify the sensitivity indexes :cite:`Saltelli2008`. This indexes study how the uncertainty in the output of a model can be apportioned, qualitatively or quantitatively, to different sources of variation in the input of a model.

Soft metamodels allow handling a large number of design variables and real-time responses. These metamodels can be applied in optimization problems, where the number of design variables can often be vast, improving the time cost required to execute accurate analysis and optimizations. By the use of flexible tools for implementing these problems in high-demand servers.

|Pymetamodels| combines machine learning (ML) metamodeling, optimization and analysis tools for the virtual development of products within a common abstract framework implemented in an accessible and distributable Python package shared with a permissive :ref:`license <pymetamodels_license>`. Through the configuration spreadsheet or programmatically |pymetamodels| allows to define multivariate problems and implement multi-disciplinary problems in a pythonic fashion.

The development of |pymetamodels| package is oriented to support ML applications in the area of material science, material informatics and the construction of  materials, components and systems soft metamodels informed by hard physics-based modelling (continuum, mesosocopic, ... ) :cite:`EUCEN2018` and experimental characterisations.

In :numref:`pymetamodels_flowchart` is described an overview flowchart of |pymetamodels| Python package capabilities the following main areas:

    * :math:`Configuration_{case} = Definition_{\ inputs \ and \ responses}`. Configuration of the pymetamodel object and analysis per case in a usable spreadsheet (see :numref:`pymetamodels_configuration`). Configuration spreadsheet is also possible to be defined  programmatically using the objconf class (see :numref:`pymetamodels_objconf`).
    * :math:`DOEX = Sampling(Configuration_{case})`. DOE sampling schemes (see :numref:`pymetamodels_conf_sampling`).
    * :math:`DOEY = model_{iteration}(DOEX)`. Python models interaction and/or direct importing of DOEY data structures.
    * :math:`DOEY = metamodel(DOEX)`. Optimal forecasting metamodels construction (see :numref:`pymetamodels_conf_metamodel`).
    * :math:`Sensitivity_{indexes}=Sensitivity_{analysis}(DOEY)`. Sensitivity analysis capabilities (see :numref:`pymetamodels_conf_sampling`).
    * :math:`Optimization_{min \ local \ optima}=Optimization_{min}(DOEY, Constrains)`. Optimization problems solution, including calibrations (see :numref:`pymetamodels_conf_optimization`).
    * :math:`Confidence_{intervals}=Robustness_{analysis}(DOEY,COVs)`. Robustness analysis capabilities.
    * :math:`Design_{point}=Calibration_{analysis}(DOEY, Constrains, Reference)`. Calibration analysis capabilities.

.. to be done
  * DOEX = metamodel_{inv}(DOEY). Inverse metamodels construction

.. include:: /images/pymetamodels_flowchart.inc

.. _pymetamodels_configuration:

Metamodel configuration (spreadsheet)
-------------------------------------

The pymetamodel abstract object is defined according a configuration spreadsheet file (in a .xls format). A template of the configuration spreadsheet can be downloaded and edited (:download:`download conf spreadsheet file </_examples/configuration_spreadsheet/configuration_spreadsheet.xls>`), or build programmatically (see :numref:`pymetamodels_objconf`). The configuration structure is described as follows.

.. rubric:: **The cases sheet**

The cases sheet specifies the configuration of the different cases to be executed. Each case is described in a row, each column describes one var (see :numref:`conf_spreadsheet_struc`). The compulsory vars are the following (additional vars can be added if needed).

    * **case:** name and id of the case
    * **vars sheet:** name and id of the sheet where are described the input vars for the given case
    * **output sheet:** name and id of the sheet where are described the output vars for the given case
    * **samples:** number of samples for the sampling activities (:math:`2^N \ values`)
    * **sensitivity method:** name and id of the sensitivity analysis method (see :numref:`pymetamodels_conf_sampling`)
    * **comment:** comment for the given case

.. include:: /tables/conf_spreadsheet_struc.inc

|

.. rubric:: **The input vars case sheet**

The input vars case sheet describes the different input variables or DOEX variables for a given case. The name of this sheet refers to the case sheet "vars sheet" value. Each row defines an input var and each column defines different attributes for the input var (see :numref:`conf_spreadsheet_struct_vars`). The compulsory attributes are the following (additional attributes can be added if needed).

  * **variable:** name and id of the input variable
  * **value:** nominal value of the input variable, use in case is not considered a ranged variable in the DOEX
  * **min:** min value of the ranged variable in the DOEX
  * **max:** max value of the ranged variable in the DOEX
  * **distribution:** type of range distribution (unif, triang, norm, lognorm)
  * **cov_un:** covariance used for the generation of the norm distributions
  * **is_range:** TRUE or FALSE value to choose if the variable is a range or a single value in the DOEX
  * **ud:** units name for the variable (i.e. [m])
  * **alias:** alias name for the variable
  * **comment:** comment field
  * **constrain:** constrain field

.. include:: /tables/conf_spreadsheet_struct_vars.inc

|

.. rubric:: **The output vars case sheet**

The output vars case sheet describes the different output variables or DOEY variables for a given case. The name of this sheet refers to the case sheet "output sheet" value. Each row defines an input var and each column defines different attributes for the input var (see :numref:`conf_spreadsheet_struct_out`). The compulsory attributes are the following (additional attributes can be added if needed).

  * **variable:** name and id of the output variable
  * **value:** nominal value of the output variable
  * **ud:** units name for the variable (i.e. [m])
  * **comment:** comment field
  * **array:** TRUE or FALSE, is the output variable an array or single value
  * **op_min** TRUE if variable is to be minimize, :math:`min(DOEY_{var})`
  * **op_min=0** TRUE if variable is to be optimize to 0, :math:`objective(DOEY_{var}=0)`
  * **ineq_(>=0)** TRUE if variables is consider for an inequality constrain, :math:`DOEY_{var}>=0`
  * **eq_(=0)** TRUE if variables is consider for an equality constrain =0, :math:`DOEY_{var}=0`

.. include:: /tables/conf_spreadsheet_struct_out.inc

|

.. _pymetamodels_conf_sampling:

Available DOEX sampling generators and sensitivity analysis
-----------------------------------------------------------

In |pymetamodels| package the DOEX sampling schema and sensitivity analysis are defined together because exists some dependencies between both. This couple is defined in the configuration spreadsheet, in "sensitivity method" var. The available sampling and sensitivity analysis schemas are given in :numref:`sampling_sens_sch`. This methods are based in SALib package :cite:`Herman2017`.

.. include:: /tables/sampling_sens_sch.inc

|   

.. _pymetamodels_conf_metamodel:

Optimal forecasting metamodels construction
-------------------------------------------

In |pymetamodels| the metamodels construction is based on the automation, training and robust selection of ML models adapted for each DOEX and DOEY group population. The agile response of these metamodels allow to be used in further robustness, plotting and optimization routines with safety, within the variables ranges limits express in the configuration sheet (see :numref:`pymetamodels_configuration`). The robust selection of machine learning ML models is carry out by an schema selection approach from different ML libraries :cite:`Pedregosa2019,Most2008`. In each schema there are available multiple multi-variate ML learning models of different fashions. These models are train using adaptive parameter values estimator techniques, the most optimal model prognosis is chosen according a residuals scoring evaluation. These routine is carried out with the function :ref:`run_metamodel_construction() <pymetamodels_run_metamodel_construction>`. The metamodels are save as an external file **.metaita**  which can be implement lately in secondary applications or be used in the plotting, robustness, calibration and optimization routines.

The available schemas and models are shown in :numref:`meta_schemas` and :numref:`ML_models`.

.. include:: /tables/meta_schemas.inc

.. include:: /tables/ML_models.inc

|

.. _pymetamodels_conf_optimization:

Optimization problem resolution
-------------------------------

In the modelling optimization problems it can be always be found the following functions and variables :cite:`Christensen2009`,

* *Objective function* :math:`DOEY_{varY}`: A series of functions that includes all possible designs, and returns an indicator number of the goodness of the design. It is a common criterion that a :math:`f` small value is better than a large one (a minimization problem). :math:`f` can be an indicator of weight, displacement, effective stress, stiffness, environmental impact, cost of production ... In the case of a multi-objective optimization, the objective function is composed of the union of more than one sub-objective function.

* *Design variable* :math:`(DOEX)`: Variables, functions or vectors that describes the design, which is changed during optimization. It may represent geometry or a choice of material. When representing a geometry, it can be a sophisticated spline shape or a simply thickness of a bar.

* *State constrains* :math:`DOEY_{constrains}`: Is a series of functions or vector that represents the response of the structure for a given design :math:`x`. For a mechanical structure, it can be displacement, stress, strain or force.

A general structural optimization :math:`(SO)` problem takes the following form (see :eq:`genSO`):

.. math::
    :nowrap:
    :label: genSO

    \begin{equation}
    (SO)
    \begin{cases}
        min(DOEY_{varY}) \text{ } \forall \text{DOEX,DOEY} \\
        when
        \begin{cases}
            \text{bounds: design constraints or variables }(DOEX_{bounds}) \\
            \text{constrains: state constraints }DOEY_{constrains} \\
            \text{equilibrium constraints}
        \end{cases}
    \end{cases}
    \end{equation}

In the case of multi-objective optimization problems several objective functions are considered,

.. math::
    :nowrap:
    :label: genSOmult

    \begin{equation}
        min DOEY_{varY} = min (DOEY_{varY_1},DOEY_{varY_2}, \cdots ,DOEY_{varY_l})
    \end{equation}

where :math:`l` is the number of objectives functions, and each objective function satisfy a given set of constrains.

Pymetamodels allows to resolve optimization problems adapted for each DOEX and DOEY group population. The response of the metamodels constructed with :numref:`pymetamodels_conf_metamodel` routines allows to apply optimization algorithms and routines to minimize one of the DOEY variables from this metamodels. The optimization routines take into account DOEX variables bounds constrains (define in the input vars case sheet configuration spreadsheet) and equality and/or inequality constrains for DOEY variables defined in the output vars case sheet. Optimization routines are carried out with the function :ref:`run_optimization_problem() <pymetamodels_run_optimization_problem>`. Depending on the optimization schemas several optimization methods are proof, and the best is chosen. The optimization results are save in the external file **.metaita**  which can be implement lately in secondary applications or be used in the plotting, robustness, calibration and optimization routines.

The available optimization schemas and models are shown in :numref:`meta_schemas` and :numref:`ML_models`.

.. include:: /tables/opti_schemas.inc

.. include:: /tables/opti_methods.inc

|