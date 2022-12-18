
.. _opti_methods-ach:

.. tabularcolumns:: | C | C | C | C | C |
.. list-table:: Optimization models available
    :name: opti_methods
    :header-rows: 1
    :widths: auto
    :align: center
    :width: 100%
    :class: right-align-right-col align-cols left-align-left-col parse-table
    
    * - ID
      - Type
      - Optimization method
      - Allow constrains
      - Refs.
    * - OP1.1
      - global
      - iter_grid_method
      - yes
      - NA
    * - OP1.2
      - global
      - grid_method
      - yes
      - NA
    * - OP1.3
      - global
      - shgo
      - yes
      - :cite:`Endres2018,Virtanen2020`
    * - OP1.4
      - global
      - shgo_slow
      - yes
      - :cite:`Endres2018,Virtanen2020`
    * - OP1.5
      - global
      - diff_evol
      - no
      - :cite:`Storn1997,Virtanen2020`
    * - OP2.1
      - local
      - min_gen
      - no
      - :cite:`Virtanen2020`
    * - OP2.2
      - local
      - Powell
      - no
      - :cite:`Priem2020,Powel1964,Virtanen2020`
    * - OP2.3
      - local
      - Nelder-Mead
      - no
      - :cite:`Gao2012a,Virtanen2020`
    * - OP2.4
      - local
      - TNC
      - no
      - :cite:`Virtanen2020`
    * - OP2.5
      - local
      - COBYLA
      - yes
      - :cite:`Priem2020,Virtanen2020`
    * - OP2.6
      - local
      - SLSQP
      - yes
      - :cite:`Bonnans2003,Virtanen2020`
