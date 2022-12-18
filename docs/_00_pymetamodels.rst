.. master file, created by
   sphinx-quickstart
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: subtitutions.txt

.. _pymetamodels_doc:

API pymetamodels (main class object)
====================================

Load main |pymetamodels| class as follow:

.. code-block:: python
   :linenos:

   import pymetamodels

   ### Load main object
   mita = pymetamodels.metamodel()

   ### Load main object (alternative)
   mita = pymetamodels.load()

and start building your metamodel and analysis

.. _pymetamodels_mainclass:

Class -> pymetamodels (functions)
---------------------------------

.. _pymetamodels_functions:
.. currentmodule:: pymetamodels
.. autofunction:: load
    
.. currentmodule:: pymetamodels    
.. autofunction:: newconf

Class -> pymetamodels
---------------------

.. _pymetamodels:
.. currentmodule:: pymetamodels
.. autoclass:: metamodel
    :members:


.. _pymetamodels_objconf:

Class -> objconf
----------------

Load the programmatically configuration spreadsheet builder as follows:

.. code-block:: python
    :linenos:

    import pymetamodels

    ### Load main object (alternative)
    mita = pymetamodels.load()

    ### Load objconf object
    conf = mita.objconf()

and start building your configuration spreadsheet

.. _obj_conf:
.. currentmodule:: obj_conf
.. autoclass:: objconf
    :members:


.. _pymetamodels_objsamp:

Class -> objsamplingsensitivity
-------------------------------

Access to the sampling and sensitivity object:

.. code-block:: python
     :linenos:

     import pymetamodels

     ### Load main object (alternative)
     mita = pymetamodels.load()

     ### Load obj_samplig_sensi object
     conf = mita.obj_samplig_sensi()

.. _obj_samp:
.. currentmodule:: obj_sampling_sensitivity
.. autoclass:: objsamplingsensitivity
    :members:


.. _pymetamodels_objmetamodel:

Class -> objmetamodel
---------------------

Represent the metamodel constructed with the DOEX and DOEY for each case and predict new values.

.. _obj_metamodel:
.. currentmodule:: obj_metamodel
.. autoclass:: objmetamodel
    :members:


.. _pymetamodels_objoptimization:

Class -> objoptimization
------------------------

Represent the optimization problem solution based on a constructed metamodel.

.. _obj_optimization:
.. currentmodule:: obj_optimization
.. autoclass:: objoptimization
   :members:
