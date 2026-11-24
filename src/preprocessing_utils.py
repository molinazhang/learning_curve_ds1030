"""Reusable, leakage-safe preprocessing for the AP-rate project (notebook 04).

Everything here builds *unfitted* objects. They are meant to be fit **inside each
cross-validation fold** (or on the full development set only for the final refit),
so that no validation/test observation ever influences a fitted transformer.

Feature handling
----------------
- skewed continuous : log1p -> StandardScaler        (StandardScaler ignores NaN)
- other continuous  : StandardScaler
- categorical       : OneHotEncoder  (NaN was turned into a 'missing' string category)
- ordinal           : SimpleImputer(constant=-1) -> OrdinalEncoder  (NaN = own category)

Continuous NaNs are deliberately preserved by build_preprocessor so the three
missing-value strategies can be compared:
  1. native       -> feed the NaN-carrying matrix to XGBoost
  2. imputation   -> build_imputation_preprocessor (preprocessor + IterativeImputer)
  3. reduced      -> ReducedFeaturesRegressor (one submodel per missing-value pattern)
"""
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                    StandardScaler, FunctionTransformer)
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
import xgboost

# lowest_grade_offered: -1 is the "unknown/placeholder" level, then grades 0..12
GRADE_LEVELS = [-1.0] + [float(i) for i in range(13)]


def build_preprocessor(cont_skewed, cont_other, cat_ftrs, ord_ftrs):
    """Base ColumnTransformer. Scales/encodes everything but PRESERVES continuous NaNs."""
    skewed_tf = Pipeline([
        ("log",    FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler())])
    other_tf = Pipeline([("scaler", StandardScaler())])
    # categoricals arrive as strings with a 'missing' category -> one-hot directly
    cat_tf = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ord_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ("ordinal", OrdinalEncoder(categories=[GRADE_LEVELS] * len(ord_ftrs)))])
    return ColumnTransformer([
        ("num_skew", skewed_tf, list(cont_skewed)),
        ("num_oth",  other_tf,  list(cont_other)),
        ("cat",      cat_tf,    list(cat_ftrs)),
        ("ord",      ord_tf,    list(ord_ftrs)),
    ])


def build_imputer(seed=42, n_estimators=100, max_iter=10):
    """Multivariate imputer. Non-deterministic estimator so imputation uncertainty is measurable."""
    return IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_estimators,
                                        max_depth=2, random_state=seed, n_jobs=1),
        initial_strategy="constant", fill_value=-999, skip_complete=True,
        max_iter=max_iter, random_state=seed)


def build_imputation_preprocessor(cont_skewed, cont_other, cat_ftrs, ord_ftrs,
                                  seed=42, n_estimators=100, max_iter=10):
    """Base preprocessor followed by the imputer, so the imputer can use the encoded
    categorical columns when estimating missing continuous values. Fit per fold."""
    return Pipeline([
        ("preprocessor", build_preprocessor(cont_skewed, cont_other, cat_ftrs, ord_ftrs)),
        ("imputer",      build_imputer(seed, n_estimators, max_iter)),
    ])


class ReducedFeaturesRegressor(RegressorMixin, BaseEstimator):
    """Reduced-features / pattern-submodel approach (lecture 19).

    Operates on a numeric matrix whose continuous columns may contain NaN (i.e. the
    output of ``build_preprocessor``). For every unique missing-value pattern seen at
    predict time it trains one submodel on the features that are present, using the
    complete training cases for those features. No imputation, no dropped columns of
    information. Fit this per CV fold.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def _make_base(self):
        if self.base_estimator is None:
            return xgboost.XGBRegressor(n_estimators=300, learning_rate=0.1,
                                        max_depth=6, subsample=0.8,
                                        colsample_bytree=0.8, n_jobs=-1,
                                        random_state=0)
        return clone(self.base_estimator)

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        mask = np.isnan(X)
        patterns, inv = np.unique(mask, axis=0, return_inverse=True)
        preds = np.empty(X.shape[0], dtype=float)
        tr_mask = np.isnan(self.X_train_)
        for i, pat in enumerate(patterns):
            rows = np.where(inv == i)[0]
            present = ~pat                                   # columns available here
            if not present.any():
                preds[rows] = self.y_train_.mean(); continue
            # complete training rows on the present columns
            complete = ~tr_mask[:, present].any(axis=1)
            if complete.sum() == 0:
                preds[rows] = self.y_train_.mean(); continue
            est = self._make_base()
            est.fit(self.X_train_[complete][:, present], self.y_train_[complete])
            preds[rows] = est.predict(X[rows][:, present])
        return preds
