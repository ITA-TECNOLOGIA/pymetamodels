#!/usr/bin/env python3

import os, sys, random, warnings
import numpy as np
import gc

from sklearn import model_selection as sk_model_selection
from sklearn import datasets as sk_datasets
from sklearn import svm as sk_svm
from sklearn import linear_model as sk_linear_model
from sklearn import metrics as sk_metrics
from sklearn import model_selection as sk_model_selection
from sklearn import pipeline as sk_model_pipeline
from sklearn import preprocessing as sk_preprocessing

class analysis(object):

    """

    .. _model_coupled_function_data_struct:

    **Synopsis:**
        * Metamodels trials

    """

    def __init__(self):

        self.name = "metamodels trials"
        self.folder_script = os.path.dirname(os.path.realpath(__file__))

        self.tol = 0.0001
        self.eps = 0.0001
        self.warnings = False
        self.fit_intercept = True
        self.n_alphas = 200

        self.random_state = True
        self.n_splits_limit = 60
        self.verbose_testing = False

        self.lst_models_r1D = ["LassoCV", "LassoLarsCV", "ElasticNetCV", "LassoCV_CVgen_0", "LassoCV_CVgen_1", "LassoCV_CVgen_2", "LassoLarsIC", "OrthogonalMatchingPursuitCV", "HuberRegressor", "ARDRegression", "BayesianRidge"]
        self.lst_models_rxD = ["MLassoCV", "MElasticNetCV", "MLassoCV_CVgen_0", "MLassoCV_CVgen_1", "MLassoCV_CVgen_2"]

    def run_tests(self):

        # Run tests

        #self.test_1D()
        #self.test_XD()
        self.test_metamodel()

    ###########################################################
    ## Metamodel selector

    def fit_model(self, doeX_train, doeY_train, doeX_test = None, doeY_test = None, scheme = None):

        # Fit metamodel with metamodel selector

        data = [doeX_train, doeX_test, doeY_train, doeY_test]

        # Select if multiple doeY targets
        if len(doeY_train.shape) == 1:
            case_r1D = True
        else:
            case_r1D = False

        self._fit_model_rXD(data, case_r1D, scheme = scheme)

    def predict(self, doeX_predict):

        pass

    def save_to_file(self):

        pass

    def load_file(self):

        pass

    ###########################################################
    ## Metamodel selector schemes

    def _fit_model_rXD(self, data, case_r1D, scheme = None):

        ## Fit metamodel r1D,rXD try differnt schemes

        # Choose scheme
        if case_r1D:
            if scheme is None:
                lst_models = ["LassoCV","LassoCV_CVgen_2","LassoLarsIC","BayesianRidge","HuberRegressor"]
            else:
                lst_models = self.lst_models_r1D
        else:
            if scheme is None:
                lst_models = ["MLassoCV", "MElasticNetCV", "MLassoCV_CVgen_2"]
            else:
                lst_models = self.lst_models_rxD

        # Iterate models
        resume = {}

        for ls_model in lst_models:

            # opecion A
            if case_r1D:
                (sk_model, R2_predict, score) = self._fit_model_r1D_call(data, ls_model, fit_intercept=True)
            else:
                (sk_model, R2_predict, score) = self._fit_model_rXD_call(data, ls_model, fit_intercept=True)

            resume[ls_model+"#A"] = {}
            resume[ls_model+"#A"]["R2_predict"] = R2_predict
            resume[ls_model+"#A"]["score"] = score
            resume[ls_model+"#A"]["fit_intercept"] = True
            resume[ls_model+"#A"]["ls_model"] = ls_model

            # opecion B
            if case_r1D:
                (sk_model, R2_predict, score) = self._fit_model_r1D_call(data, ls_model, fit_intercept=False)
            else:
                (sk_model, R2_predict, score) = self._fit_model_rXD_call(data, ls_model, fit_intercept=False)

            resume[ls_model+"#B"] = {}
            resume[ls_model+"#B"]["R2_predict"] = R2_predict
            resume[ls_model+"#B"]["score"] = score
            resume[ls_model+"#B"]["fit_intercept"] = False
            resume[ls_model+"#B"]["ls_model"] = ls_model

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
        if case_r1D:
            (sk_model, R2_predict, score) = self._fit_model_r1D_call(data, resume[max_val_ls]["ls_model"], fit_intercept=resume[max_val_ls]["fit_intercept"])
        else:
            (sk_model, R2_predict, score) = self._fit_model_rXD_call(data, resume[max_val_ls]["ls_model"], fit_intercept=resume[max_val_ls]["fit_intercept"])

        self.msg("Model %s R2 coef of determination %.3f with fit intercept %s selected" % (resume[max_val_ls]["ls_model"],resume[max_val_ls]["R2_predict"],str(resume[max_val_ls]["fit_intercept"])), type = 0)

    def _fit_model_r1D_call(self, data, ls_model, fit_intercept=True):

        # Selector r1D models

        if ls_model == "LassoCV":
            (sk_model, R2_predict, score) = self.r1D_LassoCV(data, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, tol=self.tol, random_state=self.random_state)

        elif ls_model == "LassoLarsCV":
            (sk_model, R2_predict, score) = self.r1D_LassoLarsCV(data, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas)

        elif ls_model == "ElasticNetCV":
            (sk_model, R2_predict, score) = self.r1D_ElasticNetCV(data, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, tol=self.tol, random_state=self.random_state)

        elif ls_model == "LassoCV_CVgen_0":
            (sk_model, R2_predict, score) = self.r1D_LassoCV_CVgen(data, 0, tol=self.tol, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, random_state=self.random_state)

        elif ls_model == "LassoCV_CVgen_1":
            (sk_model, R2_predict, score) = self.r1D_LassoCV_CVgen(data, 1, tol=self.tol, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, random_state=self.random_state)

        elif ls_model == "LassoCV_CVgen_2":
            (sk_model, R2_predict, score) = self.r1D_LassoCV_CVgen(data, 2, tol=self.tol, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, random_state=self.random_state)

        elif ls_model == "LassoLarsIC":
            (sk_model, R2_predict, score) = self.r1D_LassoLarsIC(data, eps=self.eps, fit_intercept=fit_intercept)

        elif ls_model == "OrthogonalMatchingPursuitCV":
            (sk_model, R2_predict, score) = self.r1D_OrthogonalMatchingPursuitCV(data, fit_intercept=fit_intercept)

        elif ls_model == "HuberRegressor":
            (sk_model, R2_predict, score) = self.r1D_HuberRegressor(data, tol=self.tol, fit_intercept=fit_intercept)

        elif ls_model == "ARDRegression":
            (sk_model, R2_predict, score) = self.r1D_ARDRegression(data, fit_intercept=fit_intercept, tol=self.tol)

        elif ls_model == "BayesianRidge":
            (sk_model, R2_predict, score) = self.r1D_BayesianRidge(data, fit_intercept=fit_intercept, tol=self.tol)

        else:
            self.error("Unknown r1D ls_model %s" % ls_model)

        return sk_model, R2_predict, score

    def _fit_model_rXD_call(self, data, ls_model, fit_intercept=True):

        # Selector rXD models

        if ls_model == "MLassoCV":
            (sk_model, R2_predict, score) = self.rXD_LassoCV(data, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, tol=self.tol, random_state=self.random_state)
        elif ls_model == "MElasticNetCV":
            (sk_model, R2_predict, score) = self.rXD_ElasticNetCV(data, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, tol=self.tol, random_state=self.random_state)
        elif ls_model == "MLassoCV_CVgen_0":
            (sk_model, R2_predict, score) = self.rXD_LassoCV_CVgen(data, 0, tol=self.tol, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, random_state=self.random_state)
        elif ls_model == "MLassoCV_CVgen_1":
            (sk_model, R2_predict, score) = self.rXD_LassoCV_CVgen(data, 1, tol=self.tol, eps=self.eps, fit_intercept=fit_intercept, n_alphas=self.n_alphas, random_state=self.random_state)
        elif ls_model == "MLassoCV_CVgen_2":
            (sk_model, R2_predict, score) = self.rXD_LassoCV_CVgen(data, 2, tol=self.tol, eps=self.eps, fit_intercept=fit_intercept ,n_alphas=self.n_alphas, random_state=self.random_state)
        else:
            self.error("Unknown rXD ls_model%s" % ls_model)

        return sk_model, R2_predict, score


    ###########################################################
    ## Linear models

    def f_random_state(self, random_state):

        # Generte random RandomState

        if random_state:
            rng = np.random.RandomState(1)
        else:
            rng = None

        return rng

    def r1D_LassoCV(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000, tol=0.0001, random_state = False):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.LassoCV(eps=eps, n_alphas=n_alphas, alphas=None, fit_intercept=fit_intercept, precompute='auto', max_iter=max_iter, tol=tol, copy_X=True, cv=None, verbose=verbose, n_jobs=None, positive=False, random_state=self.f_random_state(random_state), selection='cyclic')

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("LassoCV R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def r1D_LassoLarsCV(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.LassoLarsCV(fit_intercept=fit_intercept, cv=None,verbose=verbose,eps=eps,max_n_alphas=n_alphas,max_iter=max_iter,precompute="auto")

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("LassoLarsCV R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def r1D_ElasticNetCV(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000, l1_ratio=0.5, tol=0.0001, random_state = False):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.ElasticNetCV(l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas, alphas=None, fit_intercept=fit_intercept,  precompute='auto', max_iter=max_iter, tol=tol, cv=None, copy_X=True, verbose=verbose, n_jobs=None, positive=False, random_state=self.f_random_state(random_state), selection='cyclic')

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("ElasticNetCV R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def r1D_LassoCV_CVgen(self, data, index_generator, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000, tol=0.0001, random_state = False):

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

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("LassoCV (cv generator: %s) R2 coefficient of determination %.3f, score %.3f" % (generator,R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def r1D_LassoLarsIC(self, data, fit_intercept=True, verbose=False, eps=0.0001, max_iter=8000):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.LassoLarsIC(criterion='aic', fit_intercept=fit_intercept, verbose=verbose, precompute='auto', max_iter=max_iter, eps=eps, copy_X=True, positive=False, noise_variance=None)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("LassoLarsIC R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def r1D_OrthogonalMatchingPursuitCV(self, data, fit_intercept=True, verbose=False, max_iter=8000):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.OrthogonalMatchingPursuitCV(copy=True, fit_intercept=fit_intercept, max_iter=None, cv=None, n_jobs=None, verbose=verbose)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("OrthogonalMatchingPursuitCV R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def r1D_PoissonRegressor(self, data, fit_intercept=True, verbose=False, max_iter=8000, tol=0.0001, alpha=1.5):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.PoissonRegressor(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, warm_start=False, verbose=verbose)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("PoissonRegressor R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def r1D_GammaRegressor(self, data, fit_intercept=True, verbose=False, max_iter=8000, tol=0.0001, alpha=1.0):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.GammaRegressor(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, warm_start=False, verbose=verbose)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("GammaRegressor R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def r1D_HuberRegressor(self, data, fit_intercept=True, max_iter=8000, tol=0.0001, epsilon=1.35, verbose=False):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.HuberRegressor(epsilon=epsilon, max_iter=max_iter, alpha=0.0001, warm_start=False, fit_intercept=fit_intercept, tol=tol)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("HuberRegressor R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def r1D_QuantileRegressor(self, data, fit_intercept=True, quantile=0.5, alpha=1.0, verbose=False):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.QuantileRegressor(quantile=quantile, alpha=alpha, fit_intercept=fit_intercept, solver='interior-point', solver_options=None)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("QuantileRegressor R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def r1D_ARDRegression(self, data, fit_intercept=True, verbose=False, max_iter=500, tol=0.0001):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.ARDRegression(n_iter=max_iter, tol=tol, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000.0, fit_intercept=fit_intercept, copy_X=True, verbose=verbose)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("ARDRegression R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def r1D_BayesianRidge(self, data, fit_intercept=True, verbose=False, max_iter=500, tol=0.0001):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.BayesianRidge(n_iter=max_iter, tol=tol, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=fit_intercept, copy_X=True, verbose=verbose)

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("BayesianRidge R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def rXD_LassoCV(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000, tol=0.0001, random_state = False):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.MultiTaskLassoCV(eps=eps, n_alphas=n_alphas, alphas=None, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, copy_X=True, cv=None, verbose=verbose, n_jobs=None, random_state=self.f_random_state(random_state), selection='cyclic')

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("MultiTaskLassoCV R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def rXD_ElasticNetCV(self, data, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000, l1_ratio=0.5, tol=0.0001, random_state = False):

        [doeX_train, doeX_test, doeY_train, doeY_test] = data

        sk_model_sub = sk_linear_model.MultiTaskElasticNetCV(l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas, alphas=None, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, cv=None, copy_X=True, verbose=verbose, n_jobs=None, random_state=self.f_random_state(random_state), selection='cyclic')

        sk_model_pre = sk_preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

        with warnings.catch_warnings():
            # ignore all caught warnings
            if not self.warnings: warnings.filterwarnings("ignore")

            sk_model = sk_model_pipeline.make_pipeline(sk_model_pre, sk_model_sub, verbose=verbose)

            sk_model.fit(doeX_train,doeY_train)

        params = sk_model.get_params(deep=True)

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("MultiTaskElasticNetCV R2 coefficient of determination %.3f, score %.3f" % (R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    def rXD_LassoCV_CVgen(self, data, index_generator, fit_intercept=True, verbose=False, eps=0.0001, n_alphas=200, max_iter=8000, tol=0.0001, random_state = False):

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

        if doeX_test is not None:
            doeY_predict = sk_model.predict(doeX_test)
            R2_predict = sk_metrics.r2_score(doeY_test,doeY_predict)
        else:
            doeY_predict = sk_model.predict(doeX_train)
            R2_predict = sk_metrics.r2_score(doeY_train,doeY_predict)

        score = sk_model.score(doeX_train,doeY_train)

        self.msg("MultiTaskLassoCV (cv generator: %s) R2 coefficient of determination %.3f, score %.3f" % (generator,R2_predict,score), type = 10)

        return sk_model, R2_predict, score

    ###########################################################
    ## Other functions

    def error(self, msg):
        print("Error: " + msg)
        raise Error

    def msg(self, msg, type = 0):

        if type == 0:
            print("Msg: " + msg)
        elif type == 10:
            if self.verbose_testing:
                print("Msg: " + msg)
            pass
        else:
            print("Msg: " + msg)

    ###########################################################
    ## Test functions

    def test_metamodel(self):
        # Test metamodel

        for index in range(0, 8, 1):

            [doeX_train, doeX_test, doeY_train, doeY_test] = self.data(index)

            self.fit_model(doeX_train, doeY_train, doeX_test = doeX_test, doeY_test = doeY_test, scheme = 1)

        for index in range(100, 103, 1):

            [doeX_train, doeX_test, doeY_train, doeY_test] = self.data(index)

            self.fit_model(doeX_train, doeY_train, doeX_test = doeX_test, doeY_test = doeY_test, scheme = 1)

    def test_XD(self):
        # Test XD DOEY targets

        for index in range(100, 103, 1):

            dta = self.data(index)

            self.rXD_LassoCV(dta,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,tol=self.tol,random_state=self.random_state)
            self.rXD_ElasticNetCV(dta,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,tol=self.tol,random_state=self.random_state)
            self.rXD_LassoCV_CVgen(dta,0,tol=self.tol,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,random_state=self.random_state)
            self.rXD_LassoCV_CVgen(dta,1,tol=self.tol,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,random_state=self.random_state)
            self.rXD_LassoCV_CVgen(dta,2,tol=self.tol,eps=self.eps,fit_intercept=self.fit_intercept,n_alphas=self.n_alphas,random_state=self.random_state)

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

if __name__ == "__main__":

    analysis = analysis()

    analysis.run_tests()
