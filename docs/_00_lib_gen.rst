.. _pymetamodels_index_doc:
.. pymetamodels documentation master file, created by
   sphinx-quickstart on 2022.

.. include:: subtitutions.txt

Welcome to |pymetamodels| documentation!
########################################

.. _pymetamodels_intro:

Intro
=====

The |pymetamodels| package combines machine learning (ML) metamodeling and analysis tools for the virtual development of modeling systems within a common abstract framework implemented in an accessible and distributable Python package. The development of |pymetamodels| package is oriented to support ML applications in  material science, material informatics and the construction of materials, components and systems soft metamodels informed by hard physics-based modelling (continuum, mesosocopic, ... ) :cite:`EUCEN2018` and experimental characterisations.

The package structure is as follows:

* Main |pymetamodels| class

    * :ref:`pymeta = pymetamodels.metamodel() <pymetamodels_doc>`
    * :ref:`conf = pymetamodels.objconf() <pymetamodels_objconf>`
    * :ref:`samp = pymetamodels.obj_samplig_sensi() <pymetamodels_objsamp>`
    * :ref:`obj = pymetamodels.obj_metamodel() <pymetamodels_objmetamodel>`
    * :ref:`obj = pymetamodels.obj_optimization() <pymetamodels_objoptimization>`

Basic turtorials and advanced examples can be found in the :ref:`tutorials section <pymetamodels_tutoriales>`.

The package has been build in `ITAINNOVA <https://www.itainnova.es/es>`__ (see :ref:`authors <0_authors>`). And is distributed with permissive :ref:`license <pymetamodels_license>`. The original repository can be found at `gitHub <https://github.com/ITAINNOVA>`__.

.. _pymetamodels_installing:

Installing pymetamodels
=======================

To install the latest stable version of |pymetamodels| via pip from `PyPI <https://pypi.org/project/pymetamodels>`__. together with all the dependencies, run the following command:

::

    pip install pymetamodels

First steps, basic turtorials an advanced examples can be found in the :ref:`tutorials section <pymetamodels_tutoriales>`. To load and test installation try,

.. code-block:: python
   :linenos:

   import pymetamodels

   ### Load main object
   mita = pymetamodels.metamodel()

   ### Load main object (alternative)
   mita = pymetamodels.load()

.. _pymetamodels_requisites:

Installing pre-requisite software
=================================

Pymetamodels requires Python >3.8 or an above of release `Python.org <https://www.python.org/>`_.

Pymetamodels requires `NumPy <http://www.numpy.org/>`_, `SciPy <http://www.scipy.org/>`_, `sklean <https://scikit-learn.org/>`_, `matplotlib <http://matplotlib.org/>`_ and `SALib <https://salib.readthedocs.io/>`_ installed on your computer.  Using `pip <https://pip.pypa.io/en/stable/installing/>`_, these libraries can be installed with the following command:

::

    pip install numpy scipy scikit-learn matplotlib SALib Pillow xlrd xlwt xlutils
    
The packages are normally included with most Python bundles, such as Anaconda. In any case, they are installed automatically when using pip to install |pymetamodels|.
