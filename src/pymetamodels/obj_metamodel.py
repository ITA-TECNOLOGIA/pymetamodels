#!/usr/bin/env python3

import os, sys, random, warnings
import numpy as np
import gc
import time
import pickle

import pymetamodels.obj_logging as obj_log

from sklearn import model_selection as sk_model_selection
from sklearn import datasets as sk_datasets
from sklearn import svm as sk_svm
from sklearn import linear_model as sk_linear_model
from sklearn import gaussian_process as sk_gaussian_process
from sklearn import metrics as sk_metrics
from sklearn import model_selection as sk_model_selection
from sklearn import pipeline as sk_model_pipeline
from sklearn import preprocessing as sk_preprocessing
from sklearn import multioutput as sk_multioutput
from sklearn import model_selection as sk_model_selection
from sklearn import neural_network as sk_neural

class objmetamodel(object):

    """Python class representing a metamodel object train with doeX and doeY data

        :platform: Windows
        :synopsis: object train with doeX and doeY data

        :Dependences: numpy, sklearn

        :ivar tol: (default 0.0001) metamodel training tolerance
        :ivar eps: (default 0.0001) metamodel training eps value for
        :ivar n_alphas: (default 200) metamodel training eps value for
        :ivar random_state: (default True) activate the random state generation (bool)

        .. note::

            * The training is based on kriging, polynomial and regressor methods
            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and for example :ref:`tutorial A3 <ex_coupled_function_metamodel>`

        |

    """

    ## Common functions
    ###################

    def __init__(self, logging_path = None):

        self.version = 0
        self.pickle_protocol = 4
        self.file_extension = ".metaita"

        self.logging_path = logging_path
        if logging_path:
            self.objlog = obj_log.objlogging(self.logging_path)
        else:
            self.objlog = None

        self.tol = 0.0001
        self.eps = 0.0001
        self.warnings = False
        self.fit_intercept = True
        self.n_alphas = 200

        self.random_state = True
        self.n_splits_limit = 60
        self.verbose_testing = False

        self._sk_model = None
        self._R2_predict = None
        self._fit_intercept = None
        self._dict_param_best = None
        self._doeX_np_shape = None
        self._doeY_np_shape = None
        self._doeX_np_varX = None
        self._doeY_np_varY = None
        self._ls_model = None

        self.lst_models_r1D = ["LassoCV", "LassoLarsCV", "ElasticNetCV", "LassoCV_CVgen_0", "LassoCV_CVgen_1", "LassoCV_CVgen_2", "LassoLarsIC", "RidgeCV", "OrthogonalMatchingPursuitCV", "HuberRegressor", "ARDRegression", "BayesianRidge","GaussianProcessRegressor","PolynomialLassoCV","SplineLassoCV","PolynomialLassoLarsIC","SplineLassoLarsIC","PolynomialBayesianRidge", "SplineBayesianRidge", "LinearSVRRegressor", "SVRRegressor", "MLPRegressor"]
        self.lst_models_rxD = ["MLassoCV", "MElasticNetCV", "MLassoCV_CVgen_0", "MLassoCV_CVgen_1", "MLassoCV_CVgen_2","MGaussianProcessRegressor","MPolynomialLassoCVProcessRegressor","MSplineLassoCVProcessRegressor","MPolynomialLassoLarsICProcessRegressor","MSplineLassoLarsICProcessRegressor","MPolynomialBayesianRidgeProcessRegressor", "MSplineBayesianRidgeProcessRegressor", "MLinearSVRRegressor", "MSVRRegressor", "MMLPRegressor"]

    def _data_obj(self):

        obj = {}

        obj["version"] = self.version
        obj["pickle_protocol"] = self.pickle_protocol

        obj["tol"] = self.tol
        obj["eps"] = self.eps
        obj["warnings"] = self.warnings
        obj["fit_intercept"] = self.fit_intercept
        obj["n_alphas"] = self.n_alphas

        obj["random_state"] = self.random_state
        obj["n_splits_limit"] = self.n_splits_limit
        obj["verbose_testing"] = self.verbose_testing

        obj["_ls_model"]  = self._ls_model
        obj["_R2_predict"] = self._R2_predict
        obj["_fit_intercept"] = self._fit_intercept
        obj["_dict_param_best"] = self._dict_param_best
        obj["_doeX_np_shape"] = self._doeX_np_shape
        obj["_doeY_np_shape"] = self._doeY_np_shape
        obj["_doeX_np_varX"] = self._doeX_np_varX
        obj["_doeY_np_varY"] = self._doeY_np_varY

        obj["_sk_model"] = pickle.dumps(self._sk_model, protocol=self.pickle_protocol, fix_imports=True, buffer_callback=None)

        return obj

    def _load_data_obj(self, data_obj):

        if data_obj["version"] == 0:

            self.version = data_obj["version"]
            self.pickle_protocol = data_obj["pickle_protocol"]

            self.tol = data_obj["tol"]
            self.eps = data_obj["eps"]
            self.warnings = data_obj["warnings"]
            self.fit_intercept = data_obj["fit_intercept"]
            self.n_alphas = data_obj["n_alphas"]

            self.random_state = data_obj["random_state"]
            self.n_splits_limit = data_obj["n_splits_limit"]
            self.verbose_testing = data_obj["verbose_testing"]

            self._ls_model = data_obj["_ls_model"]
            self._R2_predict = data_obj["_R2_predict"]
            self._fit_intercept = data_obj["_fit_intercept"]
            self._dict_param_best = data_obj["_dict_param_best"]
            self._doeX_np_shape = data_obj["_doeX_np_shape"]
            self._doeY_np_shape = data_obj["_doeY_np_shape"]
            self._doeX_np_varX = data_obj["_doeX_np_varX"]
            self._doeY_np_varY = data_obj["_doeY_np_varY"]

            self._sk_model = pickle.loads(data_obj["_sk_model"], fix_imports=True, encoding='ASCII', errors='strict', buffers=None)

        else:

            self.error("Version %i unknown" % data_obj["version"])


    ###########################################################
    ## Metamodel selector

    def save_to_file(self, folder, file_name):

        """

        .. _objmetamodel_save_to_file:

        **Synopsis:**
            * Save the metamodel object to a file with .metaita extension

        **Args:**
            * folder: folder path
            * file_name: file name

        **Optional parameters:**
            * None

        **Returns:**
            * The file path

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`

        |

        """

        file_path = os.path.join(folder, file_name + self.file_extension)

        with open(file_path, 'wb') as f:

            pickle.dump(self._data_obj(), f, protocol = self.pickle_protocol)

        return file_path

    def load_file(self, file_path):

        """

        .. _objmetamodel_load_file:

        **Synopsis:**
            * Loads the metamodel object from a file with .metaita extension

        **Args:**
            * file_path: path to the file

        **Optional parameters:**
            * None

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`

        |

        """

        with open(file_path, 'rb') as f:

            data_obj = pickle.load(f)

        self._load_data_obj(data_obj)


    def fit_model(self, doeX_train, doeY_train, var_keysX, var_keysY, doeX_test = None, doeY_test = None, scheme = None, with_test = True):

        """

        .. _objmetamodel_fit_model:

        **Synopsis:**
            * Execute the metamodelling regression fitting routines to generate a predictor of DOEY values choosing the best ME model

        **Args:**
            * doeX_train: numpy array representing the doeX (nsamplesxnfeatures) for performing the training
            * doeY_train: numpy array representing the doeY (ntargetsxnfeatures) for performing the training
            * var_keysX: list of doeX variable names
            * var_keysY: list of doeY variable names

        **Optional parameters:**
            * doeX_test = None: numpy array representing the doeX (nsamplesxnfeatures) for performing the evaluation
            * doeY_test = None: numpy array representing the doeY (ntargetsxnfeatures) for performing the evaluation
            * scheme: designate the type of metamodel search scheme that will be carried out to find the most optimal ML metamolde. The available schemes are: None, general, general_fast, general_fast_nonpol, linear, gaussian, polynomial (see :numref:`pymetamodels_conf_metamodel`)
            * with_test = True: use doeX_test and doeY_test as test data, if not available split the train data into split and train data (by 0.35)

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A3 <ex_coupled_function_metamodel>`
            * The kind of sampling routines available specified metamodel configuration spreadsheet. :ref:`See metamodel configuration spreadsheet <pymetamodels_configuration>`

        |

        """

        # Fit metamodel with metamodel selector
        if doeX_test is None or doeY_test is None:
            if with_test:
                _doeX_train, _doeX_test, _doeY_train, _doeY_test = sk_model_selection.train_test_split(doeX_train, doeY_train, test_size=0.35, random_state=self.f_random_state(True))
            else:
                [_doeX_train, _doeX_test, _doeY_train, _doeY_test] = [doeX_train, doeX_train, doeY_train, doeY_train]
        else:
            [_doeX_train, _doeX_test, _doeY_train, _doeY_test] = [doeX_train, doeX_test, doeY_train, doeY_test]

        data = [_doeX_train, _doeX_test, _doeY_train, _doeY_test]

        # Select if multiple doeY targets
        if len(_doeY_train.shape) == 1:
            case_r1D = True
        else:
            case_r1D = False

        (self._sk_model, self._R2_predict, self._ls_model, self._fit_intercept, self._dict_param_best) = self._fit_model_rXD(data, case_r1D, scheme = scheme)

        self._doeX_np_shape = _doeX_train.shape
        self._doeY_np_shape = _doeY_train.shape
        self._doeX_np_varX = var_keysX
        self._doeY_np_varY = var_keysY

    def predict(self, doeX_predict):

        """

        .. _objmetamodel_predict:

        **Synopsis:**
            * Execute the metamodelling regression fitting routines to generate a predictor of DOEY values choosing the best modelling strategy

        **Args:**
            * doeX_predict: numpy array representing the doeY (ntargetsxnfeatures) for performing the prediction

        **Optional parameters:**
            * None

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A3 <ex_coupled_function_metamodel>`

        |

        """

        # Predict with the selected model

        return self._sk_model.predict(doeX_predict)

    def predict_1D(self, var_target, lst_var_features):

        """

        .. _objmetamodel_predict_1D:

        **Synopsis:**
            * Execute the metamodelling regression fitting routines to generate a predictor of DOEY values choosing the best modelling strategy
            * Data is input as a 1D list of varibles values

        **Args:**
            * var_target: var name to predict the value
            * lst_var_features: list of values ordered as varX labels

        **Optional parameters:**
            * None

        **Returns:**
            * Prediction for var_target

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A3 <ex_coupled_function_metamodel>`

        |

        """

        # Predict with the selected model

        varsY = self.doeY_varY
        shape_doeY = self.doeY_np_shape
        arrX = self.doeX_np_empty(1)

        arr = np.asarray(lst_var_features, dtype=np.dtype('float64'))

        arrX[0,:] = arr[:]

        arrY = self._sk_model.predict(arrX)

        ii = 0
        for varY in varsY:

            if varY == var_target:
                if len(shape_doeY) == 1:
                    return arrY[ii]
                else:
                    return arrY[0,ii]

            ii = ii + 1

        return None

    def score_doeY_target(self, doeX_test, doeY_test, var_target):

        """

        .. _objmetamodel_score_doeY_target:

        **Synopsis:**
            * Execute the regression of the origin values versus predicted ones
            * Obtains the regression score of the var_target prediction

        **Args:**
            * doeX_test = None: numpy array representing the doeX (nsamplesxnfeatures) for performing the evaluation
            * doeY_test = None: numpy array representing the doeY (ntargetsxnfeatures) for performing the evaluation
            * var_target: var name to predict the value

        **Optional parameters:**
            * None

        **Returns:**
            * (varY_predict, varY_values, score, varY_predict_line)
            * varY_predict: var_target array predicted values
            * varY_values: var_target values
            * score: score of var_target array predicted values versus var_target values regression
            * varY_predict_line: var_target array predicted values with line regression

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>` and :ref:`tutorial A3 <ex_coupled_function_metamodel>`

        |

        """

        shape_doeY = self.doeY_np_shape

        doeY_predict = self.predict(doeX_test)

        vary_ii = self.doeY_index(var_target)

        if len(shape_doeY) == 1:
            varY_values = doeY_test[:].reshape(-1, 1)
            varY_predict = doeY_predict[:]
        else:
            varY_values = doeY_test[:,vary_ii].reshape(-1, 1)
            varY_predict = doeY_predict[:,vary_ii]

        ## Regression
        reg = sk_linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=False)

        reg.fit(varY_values, varY_predict)
        score = reg.score(varY_values, varY_predict)

        varY_values_line = np.linspace(np.min(varY_values),np.max(varY_values), num=150)
        varY_predict_line = reg.predict(varY_values)

        return varY_predict, varY_values, score, varY_predict_line

    def doeY_index(self, varY_name):

        ## Returns the index in the doeY array for a varY name

        varsY = self.doeY_varY

        ii = 0
        for varY in varsY:

            if varY == varY_name:
                return ii

            ii = ii + 1

        return None

    def doeX_index(self, varX_name):

        ## Returns the index in the doeY array for a varX name

        varsX = self.doeX_varX

        ii = 0
        for varX in varsX:

            if varX == varX_name:
                return ii

            ii = ii + 1

        return None

    def doeX_np_empty(self, nsamples):

        ## Returns an empty doeX_np array of nsamples

        doeX_shape = self.doeX_np_shape

        if doeX_shape[1] <= 1:
            arr = np.ndarray((nsamples,1),dtype=np.dtype('float64'))
        else:
            arr = np.ndarray((nsamples,doeX_shape[1]),dtype=np.dtype('float64'))

        return arr

    def doeY_np_empty(self):

        pass

    @property
    def doeX_np_shape(self):

        """
        Shape of the trained doeX

        :getter: Returns train doeX shape
        :type: tuple

        |

        """

        return self._doeX_np_shape

    @property
    def doeY_np_shape(self):

        """
        Shape of the trained doeY

        :getter: Returns train doeY shape
        :type: tuple

        |

        """

        return self._doeY_np_shape

    @property
    def doeX_varX(self):

        """
        List of var names corresponding to the trained doeX

        :getter: Returns list of var names corresponding to the trained doeX
        :type: tuple

        |

        """

        return self._doeX_np_varX

    @property
    def doeY_varY(self):

        """
        List of var names corresponding to the trained doeY

        :getter: Returns list of var names corresponding to the trained doeY
        :type: tuple

        |

        """

        return self._doeY_np_varY

    @property
    def metamodel_score(self):

        """
        Metamodel score value

        :getter: Returns the metamodel score value
        :type: float

        |

        """

        return self._R2_predict

    ###########################################################
    ## Metamodel selector schemes

    def type_model(self, scheme, case_r1D):

        if case_r1D:
            if scheme is None or scheme == "general":
                lst_models = ["LassoCV", "LassoCV_CVgen_2", "LassoLarsIC", "RidgeCV", "BayesianRidge", "HuberRegressor", "GaussianProcessRegressor", "PolynomialBayesianRidge", "SplineBayesianRidge", "PolynomialLassoLarsIC"]
                _stop = None
            elif scheme is None or scheme == "general_fast":
                lst_models = ["LassoCV", "LassoCV_CVgen_2", "LassoLarsIC", "RidgeCV", "BayesianRidge", "HuberRegressor", "GaussianProcessRegressor", "SplineBayesianRidge", "SplineLassoLarsIC", "SplineLassoCV", "PolynomialBayesianRidge", "PolynomialLassoLarsIC", "PolynomialLassoCV"]
                _stop = 0.98
            elif scheme is None or scheme == "general_fast_nonpol":
                lst_models = ["LassoCV", "LassoCV_CVgen_2", "LassoLarsIC", "RidgeCV", "BayesianRidge", "HuberRegressor", "GaussianProcessRegressor", "SplineBayesianRidge"]
                _stop = 0.98
            elif scheme == "linear":
                lst_models = ["LassoCV", "LassoCV_CVgen_2", "LassoLarsIC", "BayesianRidge", "HuberRegressor"]
                _stop = None
            elif scheme == "gaussian":
                lst_models = ["GaussianProcessRegressor"]
                _stop = None
            elif scheme == "spline":
                lst_models = ["SplineBayesianRidge", "SplineLassoLarsIC", "SplineLassoCV"]
                _stop = None
            elif scheme == "polynomial":
                lst_models = ["PolynomialBayesianRidge", "PolynomialLassoLarsIC", "PolynomialLassoCV"]
                _stop = None
            elif scheme == "svn":
                lst_models = ["LinearSVRRegressor", "SVRRegressor"]
                _stop = None
            elif scheme == "neural":
                lst_models = ["MLPRegressor"]
                _stop = None
            elif scheme == "test":
                lst_models = ["PolynomialLassoLarsIC", "SplineLassoLarsIC"]
                _stop = None
            else:
                lst_models = self.lst_models_r1D
                _stop = None
        else:
            if scheme is None or scheme == "general":
                lst_models = ["MLassoCV", "MElasticNetCV", "MLassoCV_CVgen_2", "MGaussianProcessRegressor", "MPolynomialBayesianRidgeProcessRegressor", "MSplineBayesianRidgeProcessRegressor", "MPolynomialLassoLarsICProcessRegressor"]
                _stop = None
            elif scheme is None or scheme == "general_fast":
                lst_models = ["MLassoCV", "MElasticNetCV", "MLassoCV_CVgen_2", "MGaussianProcessRegressor", "MSplineBayesianRidgeProcessRegressor", "MSplineLassoLarsICProcessRegressor", "MSplineLassoCVProcessRegressor", "MPolynomialBayesianRidgeProcessRegressor", "MPolynomialLassoLarsICProcessRegressor", "MPolynomialLassoCVProcessRegressor"]
                _stop = 0.98
            elif scheme is None or scheme == "general_fast_nonpol":
                lst_models = ["MLassoCV", "MElasticNetCV", "MLassoCV_CVgen_2", "MGaussianProcessRegressor", "MSplineBayesianRidgeProcessRegressor"]
                _stop = 0.98
            elif scheme == "linear":
                lst_models = ["MLassoCV", "MElasticNetCV", "MLassoCV_CVgen_2"]
                _stop = None
            elif scheme == "gaussian":
                lst_models = ["MGaussianProcessRegressor"]
                _stop = None
            elif scheme == "spline":
                lst_models = ["MSplineBayesianRidgeProcessRegressor", "MSplineLassoLarsICProcessRegressor", "MSplineLassoCVProcessRegressor"]
                _stop = None
            elif scheme == "polynomial":
                lst_models = ["MPolynomialBayesianRidgeProcessRegressor", "MPolynomialLassoLarsICProcessRegressor", "MPolynomialLassoCVProcessRegressor"]
                _stop = None
            elif scheme == "svn":
                lst_models = ["MLinearSVRRegressor","MSVRRegressor"]
                _stop = None
            elif scheme == "neural":
                lst_models = ["MMLPRegressor"]
                _stop = None
            elif scheme == "test":
                lst_models = ["MPolynomialLassoLarsICProcessRegressor", "MSplineLassoLarsICProcessRegressor"]
                _stop = None
            else:
                lst_models = self.lst_models_rxD
                _stop = None

        return lst_models, _stop

    def _fit_model_rXD(self, data, case_r1D, scheme = None):

        ## Fit metamodel r1D,rXD try different schemes
        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        # Choose scheme
        lst_models, _stop = self.type_model(scheme, case_r1D)

        # Iterate models
        resume = {}
        _time_ini_0 = time.process_time()

        for ls_model in lst_models:

            ii_degrees = 0
            # option A

            _time_ini = time.process_time()
            if case_r1D:
                (sk_model, R2_predict, score, dict_param_best) = self._fit_model_r1D_call(data, ls_model, fit_intercept=True, dict_param_best = None)
            else:
                (sk_model, R2_predict, score, dict_param_best) = self._fit_model_rXD_call(data, ls_model, fit_intercept=True, dict_param_best = None)
            _elapsed_time = time.process_time() - _time_ini

            key = ls_model+"#A"+"#%i" % ii_degrees
            resume[key] = {}
            resume[key]["R2_predict"] = R2_predict
            resume[key]["score"] = score
            resume[key]["fit_intercept"] = True
            resume[key]["ls_model"] = ls_model
            resume[key]["dict_param_best"] = dict_param_best
            resume[key]["elapsed_time"] = _elapsed_time
            self.msg("Time elapsed %.2f [s]" % _elapsed_time, type = 10)

            if _stop is None:
                pass
            else:
                if R2_predict > _stop: break

            # option B
            _time_ini = time.process_time()
            if case_r1D:
                (sk_model, R2_predict, score, dict_param_best) = self._fit_model_r1D_call(data, ls_model, fit_intercept=False, dict_param_best = None)
            else:
                (sk_model, R2_predict, score, dict_param_best) = self._fit_model_rXD_call(data, ls_model, fit_intercept=False, dict_param_best = None)
            _elapsed_time = time.process_time() - _time_ini

            key = ls_model+"#B"+"#%i" % ii_degrees
            resume[key] = {}
            resume[key]["R2_predict"] = R2_predict
            resume[key]["score"] = score
            resume[key]["fit_intercept"] = False
            resume[key]["ls_model"] = ls_model
            resume[key]["dict_param_best"] = dict_param_best
            resume[key]["elapsed_time"] = _elapsed_time
            self.msg("Time elapsed %.2f [s]" % _elapsed_time, type = 10)

            if _stop is None:
                pass
            else:
                if R2_predict > _stop: break

        # Find
        max_val = -1e10
        max_val_ls = None
        for ls_model in resume:
            if max_val_ls is None:
                max_val = resume[ls_model]["R2_predict"]
                max_val_ls = ls_model
            else:
                if max_val < resume[ls_model]["R2_predict"]:
                    max_val = resume[ls_model]["R2_predict"]
                    max_val_ls = ls_model

        # Return
        ls_model = resume[max_val_ls]["ls_model"]
        fit_intercept = resume[max_val_ls]["fit_intercept"]
        dict_param_best = resume[max_val_ls]["dict_param_best"]
        _elapsed_time = resume[max_val_ls]["elapsed_time"]

        _time_ini = time.process_time()
        if case_r1D:
            (sk_model, R2_predict, score, dict_param_best) = self._fit_model_r1D_call(data, ls_model, fit_intercept=fit_intercept, dict_param_best = dict_param_best)
        else:
            (sk_model, R2_predict, score, dict_param_best) = self._fit_model_rXD_call(data, ls_model, fit_intercept=fit_intercept, dict_param_best = dict_param_best)

        _elapsed_time_with_params = time.process_time() - _time_ini
        _elapsed_time_scheme = time.process_time() - _time_ini_0
        self.msg("Time elapsed %.2f" % _elapsed_time_with_params, type = 10)

        dict_param_best["msg_fit"] = "%s. With fit intercept %s selected. DOEX & DOEY shape %s & %s.Time elapsed %.2f [s] (total %.2f [s])" % (dict_param_best["msg"], str(fit_intercept), str(doeX_train.shape), str(doeY_train.shape), _elapsed_time, _elapsed_time_scheme)
        self.msg(dict_param_best["msg_fit"], type = 0)

        return sk_model, R2_predict, ls_model, fit_intercept, dict_param_best

    def _fit_model_r1D_call(self, data, ls_model, fit_intercept=True, dict_param_best = None):

        # Selector r1D models

        if ls_model == "LassoCV":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_LassoCV(data, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, tol=self.tol, random_state=self.random_state, dict_param_best = dict_param_best)

        elif ls_model == "LassoLarsCV":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_LassoLarsCV(data, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, dict_param_best = dict_param_best)

        elif ls_model == "ElasticNetCV":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_ElasticNetCV(data, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, tol=self.tol, random_state=self.random_state, dict_param_best = dict_param_best)

        elif ls_model == "LassoCV_CVgen_0":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_LassoCV_CVgen(data, 0, tol=self.tol, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, random_state=self.random_state, dict_param_best = dict_param_best)

        elif ls_model == "LassoCV_CVgen_1":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_LassoCV_CVgen(data, 1, tol=self.tol, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, random_state=self.random_state, dict_param_best = dict_param_best)

        elif ls_model == "LassoCV_CVgen_2":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_LassoCV_CVgen(data, 2, tol=self.tol, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, random_state=self.random_state, dict_param_best = dict_param_best)

        elif ls_model == "LassoLarsIC":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_LassoLarsIC(data, eps=self.eps, fit_intercept=fit_intercept, dict_param_best = dict_param_best)

        elif ls_model == "RidgeCV":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_RidgeCV(data, fit_intercept=fit_intercept, dict_param_best = dict_param_best)

        elif ls_model == "OrthogonalMatchingPursuitCV":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_OrthogonalMatchingPursuitCV(data, fit_intercept=fit_intercept, dict_param_best = dict_param_best)

        elif ls_model == "HuberRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_HuberRegressor(data, tol=self.tol, fit_intercept=fit_intercept, dict_param_best = dict_param_best)

        elif ls_model == "ARDRegression":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_ARDRegression(data, fit_intercept=fit_intercept, tol=self.tol, dict_param_best = dict_param_best)

        elif ls_model == "BayesianRidge":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_BayesianRidge(data, fit_intercept=fit_intercept, tol=self.tol, dict_param_best = dict_param_best)

        elif ls_model == "GaussianProcessRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_GaussianProcessRegressor(data, fit_intercept=fit_intercept, random_state=self.random_state, dict_param_best = dict_param_best)

        elif ls_model == "PolynomialLassoCV":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_PolynomialLassoCV(data, fit_intercept=fit_intercept, eps=self.eps, n_alphas=self.n_alphas, tol=self.tol, random_state = self.random_state, dict_param_best = dict_param_best )

        elif ls_model == "SplineLassoCV":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_SplineLassoCV(data, fit_intercept=fit_intercept, eps=self.eps, n_alphas=self.n_alphas, tol=self.tol, random_state = self.random_state, dict_param_best = dict_param_best )

        elif ls_model == "SplineLassoLarsIC":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_SplineLassoLarsIC(data, fit_intercept=fit_intercept, eps=self.eps, n_alphas=self.n_alphas, tol=self.tol, random_state = self.random_state, dict_param_best = dict_param_best )

        elif ls_model == "PolynomialLassoLarsIC":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_PolynomialLassoLarsIC(data, fit_intercept=fit_intercept, eps=self.eps, n_alphas=self.n_alphas, tol=self.tol, random_state = self.random_state, dict_param_best = dict_param_best )

        elif ls_model == "PolynomialBayesianRidge":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_PolynomialBayesianRidge(data, fit_intercept=fit_intercept, tol=self.tol, dict_param_best = dict_param_best )

        elif ls_model == "SplineBayesianRidge":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_SplineBayesianRidge(data, fit_intercept=fit_intercept, tol=self.tol, dict_param_best = dict_param_best)

        elif ls_model == "SVRRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_SVRRegressor(data, verbose=False, tol=self.tol, dict_param_best = dict_param_best )

        elif ls_model == "LinearSVRRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_LinearSVRRegressor(data, verbose=False, fit_intercept=fit_intercept, tol=self.tol, random_state=self.random_state, dict_param_best = dict_param_best)

        elif ls_model == "MLPRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.r1D_MLPRegressor(data, verbose=False, tol=self.tol, random_state=self.random_state, dict_param_best = dict_param_best)

        else:
            self.error("Unknown r1D ls_model %s" % ls_model)

        return sk_model, R2_predict, score, dict_param_best

    def _fit_model_rXD_call(self, data, ls_model, fit_intercept=True, dict_param_best = None):

        # Selector rXD models

        if ls_model == "MLassoCV":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_LassoCV(data, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, tol=self.tol, random_state=self.random_state, dict_param_best = dict_param_best)
        elif ls_model == "MElasticNetCV":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_ElasticNetCV(data, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, tol=self.tol, random_state=self.random_state, dict_param_best = dict_param_best)
        elif ls_model == "MLassoCV_CVgen_0":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_LassoCV_CVgen(data, 0, tol=self.tol, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, random_state=self.random_state, dict_param_best = dict_param_best)
        elif ls_model == "MLassoCV_CVgen_1":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_LassoCV_CVgen(data, 1, tol=self.tol, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, random_state=self.random_state, dict_param_best = dict_param_best)
        elif ls_model == "MLassoCV_CVgen_2":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_LassoCV_CVgen(data, 2, tol=self.tol, eps=self.eps, fit_intercept=fit_intercept ,n_alphas=self.n_alphas, random_state=self.random_state, dict_param_best = dict_param_best)
        elif ls_model == "MGaussianProcessRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_GaussianProcessRegressor(data, fit_intercept=fit_intercept, random_state=self.random_state, dict_param_best = dict_param_best)
        elif ls_model == "MPolynomialLassoCVProcessRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_PolynomialLassoCVProcessRegressor(data, fit_intercept=fit_intercept, eps=self.eps, n_alphas=self.n_alphas,  tol=self.tol, random_state = self.random_state, dict_param_best = dict_param_best )
        elif ls_model == "MPolynomialLassoLarsICProcessRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_PolynomialLassoLarsICProcessRegressor(data, fit_intercept=fit_intercept, eps=self.eps, n_alphas=self.n_alphas,  tol=self.tol, random_state = self.random_state, dict_param_best = dict_param_best )
        elif ls_model == "MSplineLassoLarsICProcessRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_SplineLassoLarsICProcessRegressor(data, fit_intercept=fit_intercept, eps=self.eps, n_alphas=self.n_alphas,  tol=self.tol, random_state = self.random_state, dict_param_best = dict_param_best )
        elif ls_model == "MSplineLassoCVProcessRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_SplineLassoLarsICProcessRegressor(data, fit_intercept=fit_intercept, eps=self.eps, n_alphas=self.n_alphas,  tol=self.tol, random_state = self.random_state, dict_param_best = dict_param_best )
        elif ls_model == "MPolynomialBayesianRidgeProcessRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_MPolynomialBayesianRidgeProcessRegressor(data, fit_intercept=fit_intercept, tol=self.tol, dict_param_best = dict_param_best )
        elif ls_model == "MSplineBayesianRidgeProcessRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_MSplineBayesianRidgeProcessRegressor(data, fit_intercept=fit_intercept, tol=self.tol, dict_param_best = dict_param_best )
        elif ls_model == "MSVRRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_MSVRRegressor(data, verbose=False, tol=self.tol, dict_param_best = dict_param_best)
        elif ls_model == "MLinearSVRRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_MLinearSVRRegressor(data, verbose=False, fit_intercept=fit_intercept, tol=self.tol, random_state=self.random_state, dict_param_best = dict_param_best)
        elif ls_model == "MMLPRegressor":
            (sk_model, R2_predict, score, dict_param_best) = self.rXD_MMLPRegressor(data, verbose=False, tol=self.tol, random_state=self.random_state, dict_param_best = dict_param_best)
        else:
            self.error("Unknown rXD ls_model%s" % ls_model)

        return sk_model, R2_predict, score, dict_param_best

    ###########################################################
    ## Linear models

    def f_random_state(self, random_state):

        # Generte random RandomState

        if random_state:
            rng = np.random.RandomState(1)
        else:
            rng = None

        return rng

    def f_scores(self, sk_model, doeX_train, doeX_test, doeY_train, doeY_test):

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
            score = sk_model.score(doeX_test,doeY_test)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)
            score = sk_model.score(doeX_train,doeY_train)

        return R2_predict,score

    def r1D_LassoCV(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000, tol=0.0001, random_state = False, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.LassoCV(eps=eps, n_alphas=n_alphas, alphas=None, fit_intercept=fit_intercept, precompute='auto', max_iter=max_iter, tol=tol, copy_X=True, cv=None, verbose=verbose, n_jobs=None, positive=False, random_state=self.f_random_state(random_state), selection='cyclic')

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "LassoCV R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_LassoLarsCV(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.LassoLarsCV(fit_intercept=fit_intercept, cv=None,verbose=verbose,eps=eps,max_n_alphas=n_alphas,max_iter=max_iter,precompute="auto")

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "LassoLarsCV R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_ElasticNetCV(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000, l1_ratio=0.5, tol=0.0001, random_state = False, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.ElasticNetCV(l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas, alphas=None, fit_intercept=fit_intercept,  precompute='auto', max_iter=max_iter, tol=tol, cv=None, copy_X=True, verbose=verbose, n_jobs=None, positive=False, random_state=self.f_random_state(random_state), selection='cyclic')

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "ElasticNetCV R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_LassoCV_CVgen(self, data, index_generator, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000, tol=0.0001, random_state = False, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        generators = [None, "KFold", "ShuffleSplit","StratifiedKFold","LeaveOneOut","LeavePOut:2"]
        generator = generators[index_generator]

        if doeX_train.shape[0] > self.n_splits_limit:
            n_splits = 30
        else:
            n_splits = int(doeX_train.shape[0]/3)+1

        if generator is None:
            obj_gen = None
        else:
            if generator == "KFold":
                obj_gen = sk_model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=self.f_random_state(random_state))
            elif generator == "ShuffleSplit":
                obj_gen = sk_model_selection.ShuffleSplit(n_splits=n_splits, test_size=None, train_size=None, random_state=self.f_random_state(random_state))
            elif generator == "StratifiedKFold":
                obj_gen = sk_model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.f_random_state(random_state))
            elif generator == "LeaveOneOut": #Not usable
                obj_gen = sk_model_selection.LeaveOneOut()
            elif generator == "LeavePOut:2": #Not usable
                obj_gen = sk_model_selection.LeavePOut(2)

        sk_model_sub = sk_linear_model.LassoCV(fit_intercept=fit_intercept, cv=obj_gen, verbose=verbose, eps=eps, n_alphas=n_alphas, tol=tol, max_iter=max_iter, precompute="auto")

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "LassoCV (cv generator: %s) R2 coefficient of determination %.3f, score %.3f" % (generator,R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_LassoLarsIC(self, data, fit_intercept=True, verbose=False, eps=0.0001, max_iter=8000, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.LassoLarsIC(criterion='aic', fit_intercept=fit_intercept, verbose=verbose, precompute='auto', max_iter=max_iter, eps=eps, copy_X=True, positive=False, noise_variance=None)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "LassoLarsIC R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_RidgeCV(self, data, fit_intercept=True, verbose=False, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=fit_intercept, scoring=None, cv=None, gcv_mode=None, store_cv_values=False, alpha_per_target=False)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model_pipe = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            if dict_param_best is None:
                dict_param = {}
                dict_param['ridgecv__alphas'] = np.linspace(1e-05, 1e02, num=20).tolist()

                sk_model = sk_model_selection.GridSearchCV(sk_model_pipe, dict_param, cv=None)
            else:
                sk_model = sk_model_pipe

            sk_model.fit(doeX_train,doeY_train)

            if dict_param_best is None:
                dict_param_best = sk_model.best_params_

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        dict_param_best["msg"] = "RidgeCV (alpha=%.2E) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['ridgecv__alphas'],R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_OrthogonalMatchingPursuitCV(self, data, fit_intercept=True, verbose=False, max_iter=8000, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.OrthogonalMatchingPursuitCV(copy=True, fit_intercept=fit_intercept, max_iter=None, cv=None, n_jobs=None, verbose=verbose)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "OrthogonalMatchingPursuitCV R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_PoissonRegressor(self, data, fit_intercept=True, verbose=False, max_iter=8000, tol=0.0001, alpha=1.5, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.PoissonRegressor(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, warm_start=False, verbose=verbose)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "PoissonRegressor R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_GammaRegressor(self, data, fit_intercept=True, verbose=False, max_iter=8000, tol=0.0001, alpha=1.0, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.GammaRegressor(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, warm_start=False, verbose=verbose)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "GammaRegressor R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_HuberRegressor(self, data, fit_intercept=True, max_iter=8000, tol=0.0001, epsilon=1.35, verbose=False, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.HuberRegressor(epsilon=epsilon, max_iter=max_iter, alpha=0.0001, warm_start=False, fit_intercept=fit_intercept, tol=tol)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "HuberRegressor R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_QuantileRegressor(self, data, fit_intercept=True, quantile=0.5, alpha=1.0, verbose=False, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.QuantileRegressor(quantile=quantile, alpha=alpha, fit_intercept=fit_intercept, solver='interior-point', solver_options=None)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "QuantileRegressor R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_ARDRegression(self, data, fit_intercept=True, verbose=False, max_iter=500, tol=0.0001, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.ARDRegression(n_iter=max_iter, tol=tol, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000.0, fit_intercept=fit_intercept, copy_X=True, verbose=verbose)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "ARDRegression R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_BayesianRidge(self, data, fit_intercept=True, verbose=False, max_iter=500, tol=0.0001, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.BayesianRidge(n_iter=max_iter, tol=tol, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=fit_intercept, copy_X=True, verbose=verbose)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "BayesianRidge R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_GaussianProcessRegressor(self, data, fit_intercept=True, verbose=False, random_state = False, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_gaussian_process.GaussianProcessRegressor(kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=fit_intercept, copy_X_train=True, random_state=self.f_random_state(random_state))

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "GaussianProcessRegressor R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def rXD_LassoCV(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000, tol=0.0001, random_state = False, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.MultiTaskLassoCV(eps=eps, n_alphas=n_alphas, alphas=None, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, copy_X=True, cv=None, verbose=verbose, n_jobs=None, random_state=self.f_random_state(random_state), selection='cyclic')

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "MultiTaskLassoCV R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def rXD_ElasticNetCV(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000, l1_ratio=0.5, tol=0.0001, random_state = False, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.MultiTaskElasticNetCV(l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas, alphas=None, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, cv=None, copy_X=True, verbose=verbose, n_jobs=None, random_state=self.f_random_state(random_state), selection='cyclic')

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "MultiTaskElasticNetCV R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def rXD_LassoCV_CVgen(self, data, index_generator, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000, tol=0.0001, random_state = False, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        generators = [None, "KFold", "ShuffleSplit","StratifiedKFold","LeaveOneOut","LeavePOut:2"]
        generator = generators[index_generator]

        if doeX_train.shape[0] > self.n_splits_limit:
            n_splits = 30
        else:
            n_splits = int(doeX_train.shape[0]/3)+1

        if generator is None:
            obj_gen = None
        else:
            if generator == "KFold":
                obj_gen = sk_model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=self.f_random_state(random_state))
            elif generator == "ShuffleSplit":
                obj_gen = sk_model_selection.ShuffleSplit(n_splits=n_splits, test_size=None, train_size=None, random_state=self.f_random_state(random_state))
            elif generator == "StratifiedKFold":
                obj_gen = sk_model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.f_random_state(random_state))
            elif generator == "LeaveOneOut": #Not usable
                obj_gen = sk_model_selection.LeaveOneOut()
            elif generator == "LeavePOut:2": #Not usable
                obj_gen = sk_model_selection.LeavePOut(2)

        sk_model_sub = sk_linear_model.MultiTaskLassoCV(eps=eps, n_alphas=n_alphas, alphas=None, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, copy_X=True, cv=obj_gen, verbose=verbose, n_jobs=None, random_state=self.f_random_state(random_state), selection='cyclic')

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "MultiTaskLassoCV (cv generator: %s) R2 coefficient of determination %.3f, score %.3f" % (generator,R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    def rXD_GaussianProcessRegressor(self, data, fit_intercept=True, verbose=False, random_state = False, dict_param_best = None):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_multioutput.MultiOutputRegressor(sk_gaussian_process.GaussianProcessRegressor(kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=fit_intercept, copy_X_train=True, random_state=self.f_random_state(random_state)))

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if dict_param_best is None: dict_param_best = {}
        dict_param_best["msg"] = "MultiGaussianProcessRegressor R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score)

        self.msg(dict_param_best["msg"], type = 10)

        return sk_model, R2_predict, score, dict_param_best

    #########################
    ## Polynomial

    def pre_process(self, type = "StandardScaler", degrees_pre = 5):

        self.pre_processors = ["StandardScaler","PolynomialFeatures","SplineTransformer"]

        if type == "PolynomialFeatures":
            sk_model_pre = sk_preprocessing.PolynomialFeatures(degree=degrees_pre, interaction_only=False, include_bias=True, order='C')

        elif type == "SplineTransformer":
            sk_model_pre = sk_preprocessing.SplineTransformer(n_knots=degrees_pre, degree=6, knots='uniform', extrapolation='constant', include_bias=True, order='C')
        else:
            sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        return sk_model_pre

    def r1D_SplineLassoCV(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=1000, tol=0.0001, random_state = False, dict_param_best = None ):

        return self.r1D_PolynomialLassoCV(data, fit_intercept=fit_intercept, verbose=verbose, eps=eps, n_alphas=n_alphas, max_iter=max_iter, tol=tol, random_state = random_state, type_pre = "SplineTransformer", dict_param_best = dict_param_best)

    def r1D_PolynomialLassoCV(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=1000, tol=0.0001, random_state = False, type_pre = "PolynomialFeatures", dict_param_best = None ):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.LassoCV(eps=eps, n_alphas=n_alphas, alphas=None, fit_intercept=fit_intercept, precompute='auto', max_iter=max_iter, tol=tol, copy_X=True, cv=None, verbose=verbose, n_jobs=None, positive=False, random_state=self.f_random_state(random_state), selection='cyclic')

        if dict_param_best is None:
            sk_model_pre = self.pre_process(type = type_pre, degrees_pre = 3)
        else:
            if type_pre == "PolynomialFeatures":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['polynomialfeatures__degree'])
            elif type_pre == "SplineTransformer":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['splinetransformer__n_knots'])

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model_pipe = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            if dict_param_best is None:
                dict_param = {}
                if type_pre == "PolynomialFeatures":
                    dict_param['polynomialfeatures__degree'] = list(range(3,8))
                elif type_pre == "SplineTransformer":
                    dict_param['splinetransformer__n_knots'] = np.linspace(5, 55, num=10, dtype=np.dtype(np.int16)).tolist()
                    #dict_param['splinetransformer__degree'] = np.linspace(3, 10, num=6, dtype=np.dtype(np.int16)).tolist()

                sk_model = sk_model_selection.GridSearchCV(sk_model_pipe, dict_param, cv=None)
            else:
                sk_model = sk_model_pipe

            sk_model.fit(doeX_train,doeY_train)

            if dict_param_best is None:
                dict_param_best = sk_model.best_params_

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if type_pre == "PolynomialFeatures":
            dict_param_best["msg"] = "PolynomialLassoCV (degree %i) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['polynomialfeatures__degree'], R2_predict ,score)
        elif type_pre == "SplineTransformer":
            dict_param_best["msg"] = "SplineLassoCV (degree %i) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['splinetransformer__n_knots'], R2_predict ,score)
            self.msg(dict_param_best["msg"], type = 10)
        else:
            self.error("Type predecessor not implemented %s" % type_pre)

        return sk_model, R2_predict, score, dict_param_best

    def rXD_SplineLassoCVProcessRegressor(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=1000, tol=0.0001, random_state = False, dict_param_best = None ):

        return self.rXD_PolynomialLassoCVProcessRegressor(data, fit_intercept=fit_intercept, verbose=verbose, eps=eps, n_alphas=n_alphas, max_iter=max_iter, tol=tol, random_state = random_state, type_pre = "SplineTransformer", dict_param_best = dict_param_best )

    def rXD_PolynomialLassoCVProcessRegressor(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=1000, tol=0.0001, random_state = False, type_pre = "PolynomialFeatures", dict_param_best = None ):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub_lin = sk_linear_model.LassoCV(eps=eps, n_alphas=n_alphas, alphas=None, fit_intercept=fit_intercept, precompute='auto', max_iter=max_iter, tol=tol, copy_X=True, cv=None, verbose=verbose, n_jobs=None, positive=False, random_state=self.f_random_state(random_state), selection='cyclic')

        sk_model_sub = sk_multioutput.MultiOutputRegressor(sk_model_sub_lin)

        if dict_param_best is None:
            sk_model_pre = self.pre_process(type = type_pre, degrees_pre = 3)
        else:
            if type_pre == "PolynomialFeatures":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['polynomialfeatures__degree'])
            elif type_pre == "SplineTransformer":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['splinetransformer__n_knots'])

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model_pipe = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            if dict_param_best is None:
                dict_param = {}
                if type_pre == "PolynomialFeatures":
                    dict_param['polynomialfeatures__degree'] = list(range(3,8))
                elif type_pre == "SplineTransformer":
                    dict_param['splinetransformer__n_knots'] = np.linspace(5, 55, num=10, dtype=np.dtype(np.int16)).tolist()

                sk_model = sk_model_selection.GridSearchCV(sk_model_pipe, dict_param, cv=None)
            else:
                sk_model = sk_model_pipe

            sk_model.fit(doeX_train,doeY_train)

            if dict_param_best is None:
                dict_param_best = sk_model.best_params_

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if type_pre == "PolynomialFeatures":
            dict_param_best["msg"] = "MPolynomialLassoCVProcessRegressor (degree %i) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['polynomialfeatures__degree'], R2_predict ,score)
            self.msg(dict_param_best["msg"], type = 10)
        elif type_pre == "SplineTransformer":
            dict_param_best["msg"] = "MSplineLassoCVProcessRegressor (degree %i) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['splinetransformer__n_knots'], R2_predict ,score)
            self.msg(dict_param_best["msg"], type = 10)
        else:
            self.error("Type predecessor not implemented %s" % type_pre)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_SplineLassoLarsIC(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=1000, tol=0.0001, random_state = False, dict_param_best = None ):

        return self.r1D_PolynomialLassoLarsIC(data, fit_intercept=fit_intercept, verbose=verbose, eps=eps, n_alphas=n_alphas, max_iter=max_iter, tol=tol, random_state = random_state, type_pre = "SplineTransformer", dict_param_best = dict_param_best)

    def r1D_PolynomialLassoLarsIC(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=1000, tol=0.0001, random_state = False, type_pre = "PolynomialFeatures", dict_param_best = None ):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        if dict_param_best is None:
            sk_model_sub = sk_linear_model.LassoLarsIC(criterion='bic', fit_intercept=fit_intercept, verbose=verbose, precompute='auto', max_iter=max_iter, eps=eps, copy_X=True, positive=False, noise_variance=None)
        else:
            sk_model_sub = sk_linear_model.LassoLarsIC(criterion='bic', fit_intercept=fit_intercept, verbose=verbose, precompute='auto', max_iter=max_iter, eps=eps, copy_X=True, positive=False, noise_variance=dict_param_best['lassolarsic__noise_variance'])

        if dict_param_best is None:
            sk_model_pre = self.pre_process(type = type_pre, degrees_pre = 3)
        else:
            if type_pre == "PolynomialFeatures":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['polynomialfeatures__degree'])
            elif type_pre == "SplineTransformer":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['splinetransformer__n_knots'])

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model_pipe = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            if dict_param_best is None:
                dict_param = {}
                if type_pre == "PolynomialFeatures":
                    dict_param['polynomialfeatures__degree'] = list(range(3,8))
                elif type_pre == "SplineTransformer":
                    dict_param['splinetransformer__n_knots'] = np.linspace(5, 55, num=10, dtype=np.dtype(np.int16)).tolist()
                    #dict_param['splinetransformer__degree'] = np.linspace(3, 10, num=6, dtype=np.dtype(np.int16)).tolist()

                dict_param['lassolarsic__noise_variance'] = np.linspace(0, 100, num=10).tolist()

                sk_model = sk_model_selection.GridSearchCV(sk_model_pipe, dict_param, cv=None)
            else:
                sk_model = sk_model_pipe

            sk_model.fit(doeX_train,doeY_train)

            if dict_param_best is None:
                dict_param_best = sk_model.best_params_

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if type_pre == "PolynomialFeatures":
            dict_param_best["msg"] = "PolynomialLassoLarsIC (degree %i, noise_variance %.2E) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['polynomialfeatures__degree'], dict_param_best['lassolarsic__noise_variance'], R2_predict ,score)
            self.msg(dict_param_best["msg"], type = 10)
        elif type_pre == "SplineTransformer":
            dict_param_best["msg"] = "SplineLassoLarsIC (degree %i, noise_variance %.2E) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['splinetransformer__n_knots'], dict_param_best['lassolarsic__noise_variance'], R2_predict ,score)
            self.msg(dict_param_best["msg"], type = 10)
        else:
            self.error("Type predecessor not implemented %s" % type_pre)

        return sk_model, R2_predict, score, dict_param_best

    def rXD_SplineLassoLarsICProcessRegressor(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=1000, tol=0.0001, random_state = False, dict_param_best = None ):

        return self.rXD_PolynomialLassoLarsICProcessRegressor(data, fit_intercept=fit_intercept, verbose=verbose, eps=eps, n_alphas=n_alphas, max_iter=max_iter, tol=tol, random_state = random_state, type_pre = "SplineTransformer", dict_param_best = dict_param_best )

    def rXD_PolynomialLassoLarsICProcessRegressor(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=1000, tol=0.0001, random_state = False, type_pre = "PolynomialFeatures", dict_param_best = None ):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        if dict_param_best is None:
            sk_model_sub_lin = sk_linear_model.LassoLarsIC(criterion='bic', fit_intercept=fit_intercept, verbose=verbose, precompute='auto', max_iter=max_iter, eps=eps, copy_X=True, positive=False, noise_variance=None)
        else:
            sk_model_sub_lin = sk_linear_model.LassoLarsIC(criterion='bic', fit_intercept=fit_intercept, verbose=verbose, precompute='auto', max_iter=max_iter, eps=eps, copy_X=True, positive=False, noise_variance=dict_param_best['multioutputregressor__estimator__noise_variance'])

        sk_model_sub = sk_multioutput.MultiOutputRegressor(sk_model_sub_lin)

        if dict_param_best is None:
            sk_model_pre = self.pre_process(type = type_pre, degrees_pre = 3)
        else:
            if type_pre == "PolynomialFeatures":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['polynomialfeatures__degree'])
            elif type_pre == "SplineTransformer":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['splinetransformer__n_knots'])

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model_pipe = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            if dict_param_best is None:
                dict_param = {}
                if type_pre == "PolynomialFeatures":
                    dict_param['polynomialfeatures__degree'] = list(range(3,8))
                elif type_pre == "SplineTransformer":
                    dict_param['splinetransformer__n_knots'] = np.linspace(5, 55, num=10, dtype=np.dtype(np.int16)).tolist()

                dict_param['multioutputregressor__estimator__noise_variance'] = np.linspace(0, 100, num=10).tolist()

                sk_model = sk_model_selection.GridSearchCV(sk_model_pipe, dict_param, cv=None)
            else:
                sk_model = sk_model_pipe

            sk_model.fit(doeX_train,doeY_train)

            if dict_param_best is None:
                dict_param_best = sk_model.best_params_

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if type_pre == "PolynomialFeatures":
            dict_param_best["msg"] = "PolynomialLassoLarsICProcessRegressor (degree %i, noise_variance %.2E) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['polynomialfeatures__degree'], dict_param_best['multioutputregressor__estimator__noise_variance'], R2_predict ,score)
            self.msg(dict_param_best["msg"], type = 10)
        elif type_pre == "SplineTransformer":
            dict_param_best["msg"] = "MSplineLassoLarsICProcessRegressor (degree %i, noise_variance %.2E) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['splinetransformer__n_knots'], dict_param_best['multioutputregressor__estimator__noise_variance'], R2_predict ,score)
            self.msg(dict_param_best["msg"], type = 10)
        else:
            self.error("Type predecessor not implemented %s" % type_pre)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_SplineBayesianRidge(self, data, fit_intercept=True, verbose=False, max_iter=500, tol=0.0001, dict_param_best = None  ):

        return self.r1D_PolynomialBayesianRidge(data, fit_intercept=fit_intercept, verbose=verbose, max_iter=max_iter, tol=tol, type_pre = "SplineTransformer", dict_param_best = dict_param_best)

    def r1D_PolynomialBayesianRidge(self, data, fit_intercept=True, verbose=False, max_iter=500, tol=0.0001, type_pre = "PolynomialFeatures", dict_param_best = None ):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        if dict_param_best is None:
            sk_model_sub = sk_linear_model.BayesianRidge(n_iter=max_iter, tol=tol, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=fit_intercept, copy_X=True, verbose=verbose)
        else:
            sk_model_sub = sk_linear_model.BayesianRidge(n_iter=max_iter, tol=tol, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=fit_intercept, copy_X=True, verbose=verbose)

        if dict_param_best is None:
            sk_model_pre = self.pre_process(type = type_pre, degrees_pre = 3)
        else:
            if type_pre == "PolynomialFeatures":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['polynomialfeatures__degree'])
            elif type_pre == "SplineTransformer":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['splinetransformer__n_knots'])

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model_pipe = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            if dict_param_best is None:
                dict_param = {}
                if type_pre == "PolynomialFeatures":
                    dict_param['polynomialfeatures__degree'] = list(range(3,8))
                elif type_pre == "SplineTransformer":
                    dict_param['splinetransformer__n_knots'] = np.linspace(5, 55, num=10, dtype=np.dtype(np.int16)).tolist()
                    #dict_param['splinetransformer__degree'] = np.linspace(3, 10, num=6, dtype=np.dtype(np.int16)).tolist()

                #dict_param['bayesianridge__alpha_1'] = np.linspace(1e-07, 1e-03, num=6).tolist()
                #dict_param['bayesianridge__alpha_2'] = np.linspace(1e-07, 1e-03, num=6).tolist()
                #dict_param['bayesianridge__lambda_1'] = np.linspace(1e-07, 1e-03, num=6).tolist()
                #dict_param['bayesianridge__lambda_2'] = np.linspace(1e-07, 1e-03, num=6).tolist()

                sk_model = sk_model_selection.GridSearchCV(sk_model_pipe, dict_param, cv=None)
            else:
                sk_model = sk_model_pipe

            sk_model.fit(doeX_train,doeY_train)

            if dict_param_best is None:
                dict_param_best = sk_model.best_params_

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if type_pre == "PolynomialFeatures":
            dict_param_best["msg"] = "PolynomialBayesianRidge (degree %i) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['polynomialfeatures__degree'], R2_predict ,score)
            self.msg(dict_param_best["msg"], type = 10)
        elif type_pre == "SplineTransformer":
            dict_param_best["msg"] = "SplineBayesianRidge (degree %i) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['splinetransformer__n_knots'], R2_predict ,score)
            self.msg(dict_param_best["msg"], type = 10)
        else:
            self.error("Type predecessor not implemented %s" % type_pre)

        return sk_model, R2_predict, score, dict_param_best

    def rXD_MSplineBayesianRidgeProcessRegressor(self, data, fit_intercept=True, verbose=False, max_iter=500, tol=0.0001, dict_param_best = None ):

        return self.rXD_MPolynomialBayesianRidgeProcessRegressor(data, fit_intercept=fit_intercept, verbose=verbose, max_iter=max_iter, tol=tol, type_pre = "SplineTransformer", dict_param_best = dict_param_best )

    def rXD_MPolynomialBayesianRidgeProcessRegressor(self, data, fit_intercept=True, verbose=False, max_iter=500, tol=0.0001, type_pre = "PolynomialFeatures", dict_param_best = None ):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub_lin = sk_linear_model.BayesianRidge(n_iter=max_iter, tol=tol, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=fit_intercept, copy_X=True, verbose=verbose)

        sk_model_sub = sk_multioutput.MultiOutputRegressor(sk_model_sub_lin)

        if dict_param_best is None:
            sk_model_pre = self.pre_process(type = type_pre, degrees_pre = 3)
        else:
            if type_pre == "PolynomialFeatures":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['polynomialfeatures__degree'])
            elif type_pre == "SplineTransformer":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['splinetransformer__n_knots'])

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model_pipe = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            if dict_param_best is None:
                dict_param = {}
                if type_pre == "PolynomialFeatures":
                    dict_param['polynomialfeatures__degree'] = list(range(3,8))
                elif type_pre == "SplineTransformer":
                    dict_param['splinetransformer__n_knots'] = np.linspace(5, 55, num=10, dtype=np.dtype(np.int16)).tolist()
                    #dict_param['splinetransformer__degree'] = np.linspace(3, 10, num=6, dtype=np.dtype(np.int16)).tolist()

                sk_model = sk_model_selection.GridSearchCV(sk_model_pipe, dict_param, cv=None)
            else:
                sk_model = sk_model_pipe

            sk_model.fit(doeX_train,doeY_train)

            if dict_param_best is None:
                dict_param_best = sk_model.best_params_

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if type_pre == "PolynomialFeatures":
            dict_param_best["msg"] = "MPolynomialBayesianRidgeProcessRegressor (degree %i) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['polynomialfeatures__degree'], R2_predict ,score)
            self.msg(dict_param_best["msg"], type = 10)
        elif type_pre == "SplineTransformer":
            dict_param_best["msg"] = "MSplineBayesianRidgeProcessRegressor (degree %i) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['splinetransformer__n_knots'], R2_predict ,score)
            self.msg(dict_param_best["msg"], type = 10)
        else:
            self.error("Type predecessor not implemented %s" % type_pre)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_SVRRegressor(self, data, verbose=False, tol=0.0001, dict_param_best = None ):

        return self.rXD_MSVRRegressor(data, verbose=verbose, tol=tol, dict_param_best = dict_param_best, r1D = True)

    def rXD_MSVRRegressor(self, data, verbose=False, tol=0.0001, dict_param_best = None, r1D = False):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        if dict_param_best is None:
            sk_model_sub_lin = sk_svm.SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=tol, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=verbose, max_iter=-1)
        else:
            if r1D:
                sk_model_sub_lin = sk_svm.SVR(kernel=dict_param_best['svr__kernel'], degree=5, gamma='scale', coef0=0.0, tol=tol, C=dict_param_best['svr__C'], epsilon=dict_param_best['svr__epsilon'], shrinking=True, cache_size=200, verbose=verbose, max_iter=-1)
            else:
                sk_model_sub_lin = sk_svm.SVR(kernel=dict_param_best['multioutputregressor__estimator__kernel'], degree=5, gamma='scale', coef0=0.0, tol=tol, C=dict_param_best['multioutputregressor__estimator__C'], epsilon=dict_param_best['multioutputregressor__estimator__epsilon'], shrinking=True, cache_size=200, verbose=verbose, max_iter=-1)

        if r1D:
            sk_model_sub = sk_model_sub_lin
        else:
            sk_model_sub = sk_multioutput.MultiOutputRegressor(sk_model_sub_lin)

        type_pre = "SplineTransformer" #"StandardScaler"
        if dict_param_best is None:
            sk_model_pre = self.pre_process(type = type_pre, degrees_pre = 3)
        else:
            if type_pre == "PolynomialFeatures":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['polynomialfeatures__degree'])
            elif type_pre == "SplineTransformer":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['splinetransformer__n_knots'])
            else:
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = 3)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model_pipe = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            if dict_param_best is None:
                dict_param = {}
                if type_pre == "PolynomialFeatures":
                    dict_param['polynomialfeatures__degree'] = list(range(3,8))
                elif type_pre == "SplineTransformer":
                    dict_param['splinetransformer__n_knots'] = np.linspace(5, 55, num=10, dtype=np.dtype(np.int16)).tolist()
                    #dict_param['splinetransformer__degree'] = np.linspace(3, 10, num=6, dtype=np.dtype(np.int16)).tolist()

                val_num = 5
                if r1D:
                    dict_param['svr__C'] = np.linspace(0.1, 7, num=val_num, dtype=np.dtype(np.float64)).tolist()
                    dict_param['svr__epsilon'] = np.linspace(0.05, 0.95, num=val_num, dtype=np.dtype(np.float64)).tolist()
                    dict_param['svr__kernel'] = ["poly", "rbf", "sigmoid"]
                else:
                    dict_param['multioutputregressor__estimator__C'] = np.linspace(0.1, 7, num=val_num, dtype=np.dtype(np.float64)).tolist()
                    dict_param['multioutputregressor__estimator__epsilon'] = np.linspace(0.05, 0.95, num=val_num, dtype=np.dtype(np.float64)).tolist()
                    dict_param['multioutputregressor__estimator__kernel'] = ["poly", "rbf", "sigmoid"]

                sk_model = sk_model_selection.GridSearchCV(sk_model_pipe, dict_param, cv=None)
            else:
                sk_model = sk_model_pipe

            sk_model.fit(doeX_train,doeY_train)

            if dict_param_best is None:
                dict_param_best = sk_model.best_params_

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if r1D:
            dict_param_best["msg"] = "SVRRegressor (kernel %s) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['svr__kernel'], R2_predict ,score)
        else:
            dict_param_best["msg"] = "MSVRRegressor (kernel %s) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['multioutputregressor__estimator__kernel'], R2_predict ,score)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_LinearSVRRegressor(self, data, verbose=False, fit_intercept=True, tol=0.0001, random_state=False, dict_param_best = None):

        return self.rXD_MLinearSVRRegressor(data, verbose=verbose, fit_intercept=fit_intercept, tol=tol, random_state=random_state, dict_param_best = dict_param_best, r1D = True)

    def rXD_MLinearSVRRegressor(self, data, verbose=False, fit_intercept=True, tol=0.0001, random_state=False, dict_param_best = None, r1D = False):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        if dict_param_best is None:
            sk_model_sub_lin = sk_svm.LinearSVR(epsilon=0.0, tol=tol, C=1.0, loss='epsilon_insensitive', fit_intercept=fit_intercept, intercept_scaling=1.0, dual=True, verbose=verbose, random_state=self.f_random_state(random_state), max_iter=2000)
        else:
            if r1D:
                sk_model_sub_lin = sk_svm.LinearSVR(epsilon=dict_param_best['linearsvr__epsilon'], tol=tol, C=dict_param_best['linearsvr__C'], loss='epsilon_insensitive', fit_intercept=fit_intercept, intercept_scaling=1.0, dual=True, verbose=verbose, random_state=self.f_random_state(random_state), max_iter=2000)
            else:
                sk_model_sub_lin = sk_svm.LinearSVR(epsilon=dict_param_best['multioutputregressor__estimator__epsilon'], tol=tol, C=dict_param_best['multioutputregressor__estimator__C'], loss='epsilon_insensitive', fit_intercept=fit_intercept, intercept_scaling=1.0, dual=True, verbose=verbose, random_state=self.f_random_state(random_state), max_iter=2000)

        if r1D:
            sk_model_sub = sk_model_sub_lin
        else:
            sk_model_sub = sk_multioutput.MultiOutputRegressor(sk_model_sub_lin)

        type_pre = "StandardScaler"
        if dict_param_best is None:
            sk_model_pre = self.pre_process(type = type_pre, degrees_pre = 3)
        else:
            if type_pre == "PolynomialFeatures":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['polynomialfeatures__degree'])
            elif type_pre == "SplineTransformer":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['splinetransformer__n_knots'])
            else:
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = 3)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model_pipe = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            if dict_param_best is None:
                dict_param = {}
                if type_pre == "PolynomialFeatures":
                    dict_param['polynomialfeatures__degree'] = list(range(3,8))
                elif type_pre == "SplineTransformer":
                    dict_param['splinetransformer__n_knots'] = np.linspace(5, 55, num=10, dtype=np.dtype(np.int16)).tolist()
                    #dict_param['splinetransformer__degree'] = np.linspace(3, 10, num=6, dtype=np.dtype(np.int16)).tolist()

                val_num = 14
                if r1D:
                    dict_param['linearsvr__C'] = np.linspace(0.1, 4, num=val_num, dtype=np.dtype(np.float64)).tolist()
                    dict_param['linearsvr__epsilon'] = np.linspace(0., 0.95, num=val_num, dtype=np.dtype(np.float64)).tolist()
                else:
                    dict_param['multioutputregressor__estimator__C'] = np.linspace(0.1, 4, num=val_num, dtype=np.dtype(np.float64)).tolist()
                    dict_param['multioutputregressor__estimator__epsilon'] = np.linspace(0., 0.95, num=val_num, dtype=np.dtype(np.float64)).tolist()

                sk_model = sk_model_selection.GridSearchCV(sk_model_pipe, dict_param, cv=None)
            else:
                sk_model = sk_model_pipe

            sk_model.fit(doeX_train,doeY_train)

            if dict_param_best is None:
                dict_param_best = sk_model.best_params_

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if r1D:
            dict_param_best["msg"] = "LinearSVRRegressor R2 coefficient of determination %.3f, score %.3f" % (R2_predict ,score)
        else:
            dict_param_best["msg"] = "MLinearSVRRegressor R2 coefficient of determination %.3f, score %.3f" % (R2_predict ,score)

        return sk_model, R2_predict, score, dict_param_best

    def r1D_MLPRegressor(self, data, verbose=False, tol=0.0001, random_state=False, dict_param_best = None):

        return self.rXD_MMLPRegressor(data, verbose=verbose, tol=tol, random_state=random_state, dict_param_best = dict_param_best, r1D = True)

    def rXD_MMLPRegressor(self, data, verbose=False, tol=0.0001, random_state=False, dict_param_best = None, r1D = False):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        if dict_param_best is None:
            sk_model_sub_lin = sk_neural.MLPRegressor(hidden_layer_sizes=(300,200), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True, random_state=self.f_random_state(random_state), tol=tol, verbose=verbose, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

        else:
            if r1D:
                sk_model_sub_lin = sk_neural.MLPRegressor(hidden_layer_sizes=(300,200), activation='relu', solver='adam', alpha=dict_param_best['mlpregressor__alpha'], batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True, random_state=self.f_random_state(random_state), tol=tol, verbose=verbose, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
            else:
                sk_model_sub_lin = sk_neural.MLPRegressor(hidden_layer_sizes=(300,200), activation='relu', solver='adam', alpha=dict_param_best['multioutputregressor__estimator__alpha'], batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True, random_state=self.f_random_state(random_state), tol=tol, verbose=verbose, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

        if r1D:
            sk_model_sub = sk_model_sub_lin
        else:
            sk_model_sub = sk_multioutput.MultiOutputRegressor(sk_model_sub_lin)

        type_pre = "StandardScaler"
        if dict_param_best is None:
            sk_model_pre = self.pre_process(type = type_pre, degrees_pre = 3)
        else:
            if type_pre == "PolynomialFeatures":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['polynomialfeatures__degree'])
            elif type_pre == "SplineTransformer":
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = dict_param_best['splinetransformer__n_knots'])
            else:
                sk_model_pre = self.pre_process(type = type_pre, degrees_pre = 3)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model_pipe = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            if dict_param_best is None:
                dict_param = {}
                if type_pre == "PolynomialFeatures":
                    dict_param['polynomialfeatures__degree'] = list(range(3,8))
                elif type_pre == "SplineTransformer":
                    dict_param['splinetransformer__n_knots'] = np.linspace(5, 55, num=10, dtype=np.dtype(np.int16)).tolist()
                    #dict_param['splinetransformer__degree'] = np.linspace(3, 10, num=6, dtype=np.dtype(np.int16)).tolist()

                if r1D:
                    dict_param['mlpregressor__alpha'] = [0.0001, 0.25]
                else:
                    dict_param['multioutputregressor__estimator__alpha'] = [0.0001, 0.25]

                sk_model = sk_model_selection.GridSearchCV(sk_model_pipe, dict_param, cv=None)
            else:
                sk_model = sk_model_pipe

            sk_model.fit(doeX_train,doeY_train)

            if dict_param_best is None:
                dict_param_best = sk_model.best_params_

        params = sk_model.get_params(deep=True)

        (R2_predict,score) = self.f_scores(sk_model, doeX_train, doeX_test, doeY_train, doeY_test)

        if r1D:
            dict_param_best["msg"] = "MLPRegressor (alpha=%.5f) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['mlpregressor__alpha'], R2_predict ,score)
        else:
            dict_param_best["msg"] = "MMLPRegressor (alpha=%.5f) R2 coefficient of determination %.3f, score %.3f" % (dict_param_best['multioutputregressor__estimator__alpha'], R2_predict, score)

        return sk_model, R2_predict, score, dict_param_best

    ###########################################################
    ## Other functions

    def error(self, msg):
        print("Error: " + msg)
        if self.objlog:
            self.objlog.warning("ID05 - %s" % msg)          
        raise ValueError(msg)

    def msg(self, msg, type = 0):

        if type == 0:
            print("Msg: " + msg)
        elif type == 10:
            if self.verbose_testing:
                print("Msg: " + msg)
            pass
        else:
            print("Msg: " + msg)
        if self.objlog:
            self.objlog.info("ID05 - %s" % msg)              

    ###########################################################
    ## Test functions

    def run_tests(self, script_folder):

        # Run tests

        self.verbose_testing = True

        #self.test_1D()
        #self.test_XD()
        #self.test_metamodel(scheme = "test")

        self.test_metamodel(scheme = "general_fast")
        #self.test_metamodel(scheme = "general_fast_nonpol")
        #self.test_metamodel(scheme = "spline")
        #self.test_metamodel(scheme = "polynomial")
        #self.test_metamodel(scheme = "gaussian")
        #self.test_metamodel(scheme = "linear")
        #self.test_metamodel(scheme = "general")
        #self.test_metamodel(scheme = "svn")
        #self.test_metamodel(scheme = "neural")

        #self.test_save_files(script_folder, scheme = "general_fast")

    def test_metamodel(self, scheme = None):
        # Test metamodel

        for index in range(2, 6, 1):

            [doeX_train, doeX_test, doeY_train, doeY_test] = self.data(index)

            self.fit_model(doeX_train, doeY_train, [], [], doeX_test = doeX_test, doeY_test = doeY_test, scheme = scheme, with_test = True)

        for index in range(100, 102, 1):

            [doeX_train, doeX_test, doeY_train, doeY_test] = self.data(index)

            self.fit_model(doeX_train, doeY_train, [], [], doeX_test = doeX_test, doeY_test = doeY_test, scheme = scheme, with_test = True)

    def test_XD(self):
        # Test XD DOEY targets

        for index in range(100, 103, 1):

            dta = self.data(index)

            self.rXD_LassoCV(dta,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,tol=self.tol,random_state=self.random_state)
            self.rXD_ElasticNetCV(dta,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,tol=self.tol,random_state=self.random_state)
            self.rXD_LassoCV_CVgen(dta,0,tol=self.tol,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,random_state=self.random_state)
            self.rXD_LassoCV_CVgen(dta,1,tol=self.tol,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,random_state=self.random_state)
            self.rXD_LassoCV_CVgen(dta,2,tol=self.tol,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,random_state=self.random_state)
            self.rXD_GaussianProcessRegressor(dta, fit_intercept=self.fit_intercept, verbose=False, random_state = self.random_state)

    def test_1D(self):
        # Test 1D DOEY targets

        for index in range(0, 8, 1):

            dta = self.data(index)

            self.r1D_LassoCV(dta,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,tol=self.tol,random_state=self.random_state)
            self.r1D_LassoLarsCV(dta,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas)
            self.r1D_ElasticNetCV(dta,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,tol=self.tol,random_state=self.random_state)
            self.r1D_LassoCV_CVgen(dta,0,tol=self.tol,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,random_state=self.random_state)
            self.r1D_LassoCV_CVgen(dta,1,tol=self.tol,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,random_state=self.random_state)
            self.r1D_LassoCV_CVgen(dta,2,tol=self.tol,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,random_state=self.random_state)
            #self.r1D_LassoCV_CVgen(dta,3,tol=self.tol,eps=self.eps) # not stable
            #self.r1D_LassoCV_CVgen(dta,4,tol=self.tol,eps=self.eps) # not stable
            self.r1D_LassoLarsIC(dta,eps=self.eps,fit_intercept=self.fit_intercept)
            self.r1D_OrthogonalMatchingPursuitCV(dta,fit_intercept=self.fit_intercept)
            #self.r1D_PoissonRegressor(dta,tol=self.tol,fit_intercept=self.fit_intercept)
            #self.r1D_GammaRegressor(dta,tol=self.tol,fit_intercept=self.fit_intercept) # not stable
            self.r1D_HuberRegressor(dta,tol=self.tol,fit_intercept=self.fit_intercept)
            #self.r1D_QuantileRegressor(dta,fit_intercept=self.fit_intercept) # not stable
            self.r1D_ARDRegression(dta,fit_intercept=self.fit_intercept, tol=self.tol)
            self.r1D_BayesianRidge(dta,fit_intercept=self.fit_intercept, tol=self.tol)
            self.r1D_GaussianProcessRegressor(dta, fit_intercept=self.fit_intercept,random_state=self.random_state)

    def data(self, index):

        # Generate datasets

        with_test = True

        if index == 0:
            name = "load_iris"
            doeX, doeY = sk_datasets.load_iris(return_X_y=True)
        elif index == 1:
            name = "fetch_california_housing"
            doeX, doeY = sk_datasets.fetch_california_housing(data_home=None, download_if_missing=True, return_X_y=True, as_frame=False)
        elif index == 2:
            name = "load_diabetes"
            doeX, doeY = sk_datasets.load_diabetes(return_X_y=True, as_frame=False, scaled=True)
        elif index == 3:
            name = "load_breast_cancer"
            doeX, doeY = sk_datasets.load_breast_cancer(return_X_y=True, as_frame=False)
        elif index == 4:
            name = "load_wine"
            doeX, doeY = sk_datasets.load_wine(return_X_y=True, as_frame=False)
        elif index == 5:
            name = "make_regression_1x1000"
            doeX, doeY = sk_datasets.make_regression(n_samples=1000, n_features=100, n_informative=10, n_targets=1, bias=0.05, effective_rank=None, tail_strength=0.5, noise=0.1, shuffle=True, coef=False, random_state=None)
        elif index == 6:
            name = "make_regression_1x10000"
            doeX, doeY = sk_datasets.make_regression(n_samples=10000, n_features=100, n_informative=10, n_targets=1, bias=0.5, effective_rank=None, tail_strength=0.5, noise=0.5, shuffle=True, coef=False, random_state=None)
        elif index == 7:
            name = "make_s_curve"
            doeX, doeY = sk_datasets.make_s_curve(1000, noise = 0.1, random_state=None)
        elif index == 100:
            name = "load_linnerud"
            doeX, doeY = sk_datasets.load_linnerud(return_X_y=True, as_frame=False)
        elif index == 101:
            name = "make_regression_3x1000"
            doeX, doeY = sk_datasets.make_regression(n_samples=1000, n_features=100, n_informative=10, n_targets=3, bias=0.05, effective_rank=None, tail_strength=0.5, noise=0.1, shuffle=True, coef=False, random_state=None)
        elif index == 102:
            name = "make_regression_9x10000"
            doeX, doeY = sk_datasets.make_regression(n_samples=10000, n_features=100, n_informative=10, n_targets=9, bias=0.5, effective_rank=None, tail_strength=0.5, noise=0.5, shuffle=True, coef=False, random_state=None)
        else:
            name = "load_iris"
            doeX, doeY = sk_datasets.load_iris(return_X_y=True)

        self.msg(("Data -%s- shape X , Y " % name) + str(doeX.shape) + " " + str(doeY.shape))

        if with_test:
            doeX_train, doeX_test, doeY_train, doeY_test = sk_model_selection.train_test_split(doeX, doeY, test_size=0.4, random_state=0)
        else:
            [doeX_train, doeX_test, doeY_train, doeY_test] = [doeX, None, doeY, None]

        return [doeX_train, doeX_test, doeY_train, doeY_test]

    def test_save_files(self, script_folder, scheme = None):

        self.verbose_testing = False


        def _test():

            [doeX_train, doeX_test, doeY_train, doeY_test] = self.data(index)

            self.fit_model(doeX_train, doeY_train, [], [], doeX_test = doeX_test, doeY_test = doeY_test, scheme = scheme, with_test = True)

            doeY_test_predict = self.predict(doeX_test)
            data_obj = self._data_obj()

            file_name = "model_%i" % index

            file_path = self.save_to_file(script_folder, file_name)

            self.load_file(file_path)

            doeY_test_predict2 = self.predict(doeX_test)
            data_obj2 = self._data_obj()

            print("Test ", doeY_test_predict == doeY_test_predict2)

            for key in data_obj:
                print("Test %s" % key, data_obj[key] == data_obj2[key])

        for index in range(0, 6, 1):

            _test()

        for index in range(101, 102, 1):

            _test()

if __name__ == "__main__":

    analysis = objmetamodel()

    analysis.run_tests("")
