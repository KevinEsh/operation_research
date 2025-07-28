from typing import TypeVar

import lightgbm as lgbm
from mlflow.pyfunc import PythonModel
from polars import DataFrame

# from s3fs import S3FileSystem
from sklearn.base import BaseEstimator, RegressorMixin, clone

PolarsType = TypeVar("DataFrame")


def parse_fit_params(fit_params: dict):
    """Parse fit parameters for LightGBM."""
    if fit_params is None:
        return {}
    if isinstance(fit_params, dict):
        fit_params["callbacks"] = []
        if "early_stopping_rounds" in fit_params:
            fit_params["callbacks"].append(
                lgbm.early_stopping(fit_params.pop("early_stopping_rounds"))
            )
        if "log_evaluation" in fit_params:
            fit_params["callbacks"].append(lgbm.log_evaluation(fit_params.pop("log_evaluation")))
        return fit_params
    raise ValueError("fit_params must be a dictionary.")


class DirectMultihorizonForecaster(BaseEstimator, RegressorMixin, PythonModel):
    def __init__(self, horizons: int, params=None):
        """Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : str, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
            If setting this to a positive value, consider also changing ``num_leaves`` to ``<= 2^max_depth``.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        objective : str, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (Hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequency of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        n_jobs : int or None, optional (default=None)
            Number of parallel threads to use for training (can be changed at prediction time by
            passing it as an extra keyword argument).

            For better performance, it is recommended to set this to the number of physical cores
            in the CPU.

            Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
            scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
            threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
            to using the number of physical cores in the system (its correct detection requires
            either the ``joblib`` or the ``psutil`` util libraries to be installed).

            .. versionchanged:: 4.0.0

        importance_type : str, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        **kwargs
            Other parameters for the model.
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

            .. warning::

                \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

        Note
        ----
        A custom objective function can be provided for the ``objective`` parameter.
        In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``,
        ``objective(y_true, y_pred, weight) -> grad, hess``
        or ``objective(y_true, y_pred, weight, group) -> grad, hess``:

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of y_pred for each sample point.
            hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of y_pred for each sample point.

        For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
        and grad and hess should be returned in the same format.
        """
        # boosting_type: str = "gbdt",
        # num_leaves: int = 31,
        # max_depth: int = -1,
        # learning_rate: float = 0.1,
        # n_estimators: int = 100,
        # subsample_for_bin: int = 200000,
        # objective: Optional[Union[str, _LGBM_ScikitCustomObjectiveFunction]] = None,
        # class_weight: Optional[Union[Dict, str]] = None,
        # min_split_gain: float = 0.0,
        # min_child_weight: float = 1e-3,
        # min_child_samples: int = 20,
        # subsample: float = 1.0,
        # subsample_freq: int = 0,
        # colsample_bytree: float = 1.0,
        # reg_alpha: float = 0.0,
        # reg_lambda: float = 0.0,
        # random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
        # n_jobs: Optional[int] = None,
        # importance_type: str = "split",
        self.params = params if params is not None else {}
        self.base_regressor = lgbm.LGBMRegressor(**self.params)
        self.horizons = horizons
        self.models_ = []
        self.is_fitted = False
        self.target_names_ = None

    def fit(
        self,
        X: PolarsType,
        Y: PolarsType,
        Xv: PolarsType = None,
        Yv: PolarsType = None,
        fit_params=None,
        target_cols=None,
    ):
        # X: _LGBM_ScikitMatrixLike,
        # y: _LGBM_LabelType,
        # sample_weight: Optional[_LGBM_WeightType] = None,
        # init_score: Optional[_LGBM_InitScoreType] = None,
        # eval_set: Optional[List[_LGBM_ScikitValidSet]] = None,
        # eval_names: Optional[List[str]] = None,
        # eval_sample_weight: Optional[List[_LGBM_WeightType]] = None,
        # eval_init_score: Optional[List[_LGBM_InitScoreType]] = None,
        # eval_metric: Optional[_LGBM_ScikitEvalMetricType] = None,
        # feature_name: _LGBM_FeatureNameConfiguration = "auto",
        # categorical_feature: _LGBM_CategoricalFeatureConfiguration = "auto",
        # callbacks: Optional[List[Callable]] = None,
        # init_model: Optional[Union[str, Path, Booster, LGBMModel]] = None,

        self.fit_params = parse_fit_params(fit_params)
        self.target_cols_ = Y.columns if target_cols is None else target_cols

        # For direct approach, Y should contain multiple columns, each representing a different horizon.
        # Each column should be sorted out according to the horizon it represents, e.g., 'h1', 'h2', etc.
        if len(self.target_cols_) != self.horizons:
            raise ValueError(
                f"Expected {self.horizons} target columns, but got {len(self.target_cols_)}."
            )

        X = X.to_pandas()
        if Xv is not None:
            Xv = Xv.to_pandas()

        self.models_ = [clone(self.base_regressor) for _ in range(self.horizons)]

        # Fit each model for each horizon
        for h in range(self.horizons):
            y_h = Y.get_column(self.target_cols_[h]).to_numpy()
            if Yv is not None:
                yv_h = Yv.get_column(self.target_cols_[h]).to_numpy()
                self.models_[h].fit(X, y_h, eval_set=[(X, y_h), (Xv, yv_h)], **self.fit_params)
            else:
                self.fit_params["eval_set"] = [(X, y_h)]
                self.models_[h].fit(X, y_h, **self.fit_params)

        self.is_fitted = True
        return self

    def predict(self, X: PolarsType):
        """Predict using the fitted models for each horizon."""
        if not self.is_fitted:
            raise RuntimeError("You must fit the model before calling predict.")

        X = X.to_pandas()
        preds_df = DataFrame(
            {
                f"pred_{self.target_cols_[horizon]}": model.predict(X)
                for horizon, model in enumerate(self.models_)
            }
        )
        return preds_df

    def feature_importances(self) -> PolarsType:
        """Get feature importances from the fitted models.

        Parameters
        ----------
        importance_type : str, optional (default='split')
            The type of feature importance to be returned.
            Can be 'split' or 'gain'.
        """
        if not self.is_fitted:
            raise RuntimeError("You must fit the model before calling feature_importances_.")

        return DataFrame(
            {"feature_name": self.models_[0].feature_name_}
            | {
                f"{self.target_cols_[horizon]}": model.feature_importances_
                for horizon, model in enumerate(self.models_)
            }
        )


# def push_model_to_s3(forecaster: DirectMultihorizonForecaster, timestamp: str) -> None:
#     from joblib import dump

#     s3_models_path, s3_model_storage_config = get_s3_model_storage(timestamp)
#     s3_model_path = s3_models_path + "/demand_forecaster.pkl"

#     # Guardar el modelo en S3 usando s3fs
#     fs = S3FileSystem(**s3_model_storage_config)
#     with fs.open(s3_model_path, "wb") as f:
#         dump(forecaster, f)

#     return s3_model_path


# def pull_model_from_s3(s3_model_path: str) -> DirectMultihorizonForecaster:
#     """
#     Pull the model from S3 and return it.
#     """
#     from joblib import load

#     _, s3_model_storage_config = get_s3_model_storage("")
#     fs = S3FileSystem(**s3_model_storage_config)
#     with fs.open(s3_model_path, "rb") as f:
#         forecaster = load(f)
#     return forecaster
