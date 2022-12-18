
.. _opti_schemas-ach:

.. tabularcolumns:: | C | C |
.. list-table:: Optimization schemas
    :name: opti_schemas
    :header-rows: 1
    :widths: auto
    :align: center
    :width: 100%
    :class: right-align-right-col align-cols left-align-left-col parse-table
    
    * - Schema
      - Optimization types
    * - general
      - "iter_grid_method", "shgo", "shgo_slow", "diff_evol", "min_gen", "Powell", "Nelder-Mead", "TNC", "COBYLA", "SLSQP"
    * - general_fast
      - "iter_grid_method", "shgo", "diff_evol", "min_gen", "Powell", "Nelder-Mead", "TNC", "COBYLA", "SLSQP"
    * - general_with_constrains
      - "iter_grid_method", "COBYLA", "SLSQP", "shgo"
    * - global
      - "iter_grid_method", "shgo", "shgo_slow", "diff_evol"
    * - minimize
      - "min_gen", "Powell", "Nelder-Mead", "TNC", "COBYLA", "SLSQP"
    * - grid_method
      - "grid_method"
    * - iter_grid_method
      - "iter_grid_method"
