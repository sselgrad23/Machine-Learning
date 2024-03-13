import logging
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy.stats import pearsonr
from sklearn.neighbors import LocalOutlierFactor
import statsmodels.api as sm
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression, r_regression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import RegressionResultsWrapper

"""
We first define two classes called "Data" and "PreprocessingStep". 
Data is used to represent the data with training and testing sets, 
while PreprocessingStep is a base class for various preprocessing 
steps.
"""
@dataclass
class Data:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: Union[np.ndarray, None]
    x_mask_train: np.ndarray = field(init=False)
    x_mask_test: np.ndarray = field(init=False)

    def __post_init__(self):
        self.x_mask_train: np.ndarray = np.zeros_like(self.x_train, dtype=bool)
        self.x_mask_test: np.ndarray = np.zeros_like(self.x_test, dtype=bool)

    def delete(self, indices: np.ndarray, train: bool = True):
        if train:
            self.x_train = np.delete(self.x_train, indices, axis=0)
            self.y_train = np.delete(self.y_train, indices, axis=0)
            self.x_mask_train = np.delete(self.x_mask_train, indices, axis=0)
        else:
            self.x_test = np.delete(self.x_test, indices, axis=0)
            self.y_test = np.delete(self.y_test, indices, axis=0)
            self.x_mask_test = np.delete(self.x_mask_test, indices, axis=0)

    def update_mask(self, mask: np.ndarray, train: bool = True):
        if train:
            self.x_mask_train = np.logical_or(self.x_mask_train, mask)
        else:
            self.x_mask_test = np.logical_or(self.x_mask_test, mask)


@dataclass
class PreprocessingStep:

    def __call__(self, data: Data) -> Data:
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__ + ':' + str(self.__dict__).replace(" ", "").replace("'", "")

"""
We also define a class called "PreprocessingPipeline", which allows
us to create pipelines to apply a sequence of preprocessing steps 
to the data.
"""
class PreprocessingPipeline:
    def __init__(self, *steps: PreprocessingStep):
        self.steps = steps

    def __str__(self):
        return "_".join([str(step) for step in self.steps])

    def __call__(self, data: Data):
        for step in self.steps:
            data = step(data)
        return data

"""
We remove highly correlated features from the training and testing data 
by calculating  the correlation matrix and removing features that have a 
correlation coefficient above the threshold.
"""
def drop_correlated_features(X_train, X_test, threshold=0.95):
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    upper = np.triu(corr_matrix, k=1)
    to_drop = [column for column in range(upper.shape[1]) if any(upper[:, column] > threshold)]
    return np.delete(X_train, to_drop, axis=1), np.delete(X_test, to_drop, axis=1)

"""
We define different classes for different data preprocessing and outlier 
detection steps. 
"""
@dataclass
class KnnImputer(PreprocessingStep):
    n_neighbors: int = 5
    imput_test: bool = False
    delete_previously_imputed: bool = False

    def __call__(self, data: Data):
        # Impute missing values (with outliers).
        # imp = SimpleImputer(missing_values=np.nan, strategy="median")
        # X_train_with_outliers = imp.fit_transform(X_train)
        imputer = KNNImputer(n_neighbors=self.n_neighbors, weights="distance")
        nan_mask = np.isnan(data.x_train)
        data.update_mask(nan_mask, train=True)
        if self.delete_previously_imputed:
            data.x_train[data.x_mask_train] = np.nan
            data.x_test[data.x_mask_test] = np.nan
        data.x_train = imputer.fit_transform(data.x_train)
        if self.imput_test:
            data.x_test = imputer.transform(data.x_test)
        return data


@dataclass
class PCAFeatureSelector(PreprocessingStep):
    n_components: Union[float, str, int] = 'mle'

    def __call__(self, data: Data):
        pca = PCA(n_components=self.n_components)
        pca.fit(data.x_train)
        # select 200 best original features
        logging.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        logging.info(f"PCA explained variance: {pca.explained_variance_}")
        logging.info(f"PCA n_components: {pca.n_components_}")
        data.x_train = pca.transform(data.x_train)
        data.x_test = pca.transform(data.x_test)

        return data


@dataclass
class OutlierDetector:
    plot_pca_outliers: bool = False

    @staticmethod
    def plot_pca(data: Data, outliers: np.ndarray, title: str):
        pca = PCA(n_components=3)
        pca.fit(data.x_train)
        transformed_data = pca.transform(data.x_train)
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(transformed_data[..., 0], transformed_data[..., 1], transformed_data[..., 2], c=outliers,
                   cmap='viridis')
        plt.title(title)
        # ax.colorbar()
        plt.show()

    def _typical_run(self, model, data: Data):
        outliers = model.fit_predict(data.x_train, data.y_train)
        if self.plot_pca_outliers:
            self.plot_pca(data, outliers, title=self.__class__.__name__)
        # logging.info(f"EE Number of outliers: {(outliers == -1).sum()}")
        data.delete(np.where(outliers == -1)[0], train=True)
        if data.y_test is not None:
            outliers_test = model.predict(data.x_test)
            data.delete(np.where(outliers_test == -1)[0], train=False)
        return data


@dataclass
class IsolationForestOutlierDetector(PreprocessingStep, OutlierDetector):
    contamination: Union[float, str] = 0.05

    def __call__(self, data: Data):
        iforest = IsolationForest(contamination=self.contamination, random_state=42)
        """
            I tried to remove outliers using IQR but it didn't work well.

            q25 = np.nanpercentile(X_train, 25, axis=0)
            q75 = np.nanpercentile(X_train, 75, axis=0)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            logging.info(f"X_train shape before removing outliers: {X_train.shape}")
            logging.info(f"X_train nan values before removing outliers: {np.isnan(X_train).sum()}")
            X_train[((X_train < lower_bound) | (X_train > upper_bound))] = np.nan
            #X_train_with_outliers = X_train_with_outliers[((X_train_with_outliers > lower_bound) & (X_train_with_outliers < upper_bound)).all(axis=1)]
            #X_train_with_outliers = y_train[((X_train_with_outliers > lower_bound) & (X_train_with_outliers < upper_bound)).all(axis=1)]
            logging.info(f"X_train shape after removing outliers: {X_train.shape}")
            logging.info(f"X_train nan values after removing outliers: {np.isnan(X_train).sum()}")
            """
        return self._typical_run(iforest, data)


@dataclass
class CooksDistOutlierDetector(PreprocessingStep, OutlierDetector):
    threshold: float = 2.0

    def __call__(self, data: Data):
        # print("Shape before", data.x_train.shape)
        ols: RegressionResultsWrapper = sm.OLS(data.y_train, data.x_train).fit()
        influence = ols.get_influence()
        cooks = influence.cooks_distance
        cooks_mean = np.mean(cooks[0])
        # print("Number of outliers", (cooks[0] > 3 * cooks_mean).sum())
        filter = cooks[0] < self.threshold * cooks_mean
        if self.plot_pca_outliers:
            self.plot_pca(data, filter, title=self.__class__.__name__)
        data.delete(np.where(filter == False)[0], train=True)
        logging.warning("Cooks distance outlier detection is not implemented for test set.")
        # print("Shape after", data.x_train.shape)
        return data


@dataclass
class EllipticEnvelopeOutlierDetector(PreprocessingStep, OutlierDetector):
    contamination: float = 0.01

    def __call__(self, data: Data):
        ee = EllipticEnvelope(random_state=42, contamination=self.contamination)
        return self._typical_run(ee, data)


@dataclass
class lof_outlier_detector(PreprocessingStep, OutlierDetector):
    contamination: float = 0.08

    def __call__(self, data: Data):
        lof = LocalOutlierFactor(n_neighbors=30, contamination=self.contamination)
        return self._typical_run(lof, data)


@dataclass()
class VarianceFeatureSelector(PreprocessingStep):
    threshold: float = 0.0

    def __call__(self, data: Data):
        selector = VarianceThreshold(threshold=self.threshold)
        data.x_train = selector.fit_transform(data.x_train)
        data.x_test = selector.transform(data.x_test)
        data.x_mask_train = selector.transform(data.x_mask_train)
        data.x_mask_test = selector.transform(data.x_mask_test)
        # Drop correlated features.
        # X_train, X_test = drop_correlated_features(X_train, X_test)
        return data


@dataclass
class PearsonCorrFeatureSelector(PreprocessingStep):
    p_val: float = 0.05

    def __call__(self, data: Data):
        values = []
        for i in range(data.x_train.shape[1]):
            values.append([pearsonr(data.x_train[..., i], data.y_train), i])
        values = sorted(values, key=lambda x: abs(x[0].statistic), reverse=True)

        c = 0
        indices = []
        for v in values:
            if v[0].pvalue <= self.p_val:
                indices.append(v[1])
                if len(indices) == 200:
                    break

        logging.info(f"Number of features: {len(indices)}")
        logging.info(f"Indices of features: {sorted(indices)}")

        data.x_train, data.x_test = data.x_train[..., indices], data.x_test[..., indices]
        data.x_mask_train = data.x_mask_train[..., indices]
        data.x_mask_test = data.x_mask_test[..., indices]
        return data


@dataclass
class KBestFeatureSelector(PreprocessingStep):
    k: int = 200

    def __call__(self, data: Data):
        raise NotImplementedError("There is some bug here that doesn't compute correctly some scores")
        corr_coef = r_regression(data.x_train, data.y_train)

        logging.info(corr_coef[193])
        logging.info(pearsonr(data.x_train[..., 193], data.y_train))
        # logging.info(data.x_train[193])
        # logging.info(data.y_train[193])
        selector = SelectKBest(f_regression, k=self.k)
        data.x_train = selector.fit_transform(data.x_train, data.y_train)
        data.x_test = selector.transform(data.x_test)
        data.extras['x_train_mask'] = selector.transform(data.extras['x_train_mask'])
        # data.extras['x_test_mask'] = selector.transform(data.extras['x_test_mask'])
        logging.info(f"Number of features: {data.x_train.shape[1]}")
        logging.info(f"Indices of features: {selector.get_support(indices=True)}")
        return data


@dataclass
class StdScaler(PreprocessingStep):
    def __call__(self, data: Data):
        scaler = StandardScaler()
        data.x_train = scaler.fit_transform(data.x_train)
        data.x_test = scaler.transform(data.x_test)
        return data
