Pymetamodels package for materials, systems and component metamodeling
======================================================================

The pymetamodels package combines machine learning (ML) metamodeling and analysis tools for the virtual development of modeling systems within a common abstract framework implemented in an accessible and distributable Python package. The development of pymetamodels package is oriented to support ML applications in  material science, material informatics and the construction of materials, components and systems soft metamodels informed by hard physics-based modelling (continuum, mesosocopic, ... ) and experimental characterisations.

Basic turtorials and advanced examples can be found in the tutorials section [pymetamodels.readthedocs.io](https://pymetamodels.readthedocs.io/en/latest/).

The package has been build in [ITAINNOVA](https://www.itainnova.es/es). And is distributed with permissive MIT license.

Installing pymetamodels
-----------------------

To install the latest stable version of pymetamodels via pip from [PyPI](https://pypi.org/project/pymetamodels) together with all the dependencies, run the following command:

```
    pip install pymetamodels
```

First steps, basic turtorials an advanced examples can be found in the documentation tutorials section [pymetamodels.readthedocs.io](https://pymetamodels.readthedocs.io/en/latest/). To load and test installation try,

```
    import pymetamodels

    ### Load main object
    mita = pymetamodels.metamodel()

    ### Load main object (alternative)
    mita = pymetamodels.load()
```

Installing pre-requisite software
---------------------------------

Pymetamodels requires Python >3.7 or an above of release [Python.org](https://www.python.org).

Pymetamodels requires [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org), [sklean](https://scikit-learn.org), [matplotlib](http://matplotlib.org) and [SALib](https://salib.readthedocs.io) installed on your computer.  Using [pip](https://pip.pypa.io/en/stable/installing), these libraries can be installed with the following command:

```
    pip install numpy scipy scikit-learn matplotlib SALib Pillow xlrd xlwt xlutils
```

The packages are normally included with most Python bundles, such as Anaconda. Generally, they are installed automatically when using pip to install pymetamodels.