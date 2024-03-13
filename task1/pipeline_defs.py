from PreprocessingPipeline import *


@dataclass
class Pipeline1:
    """
    Pipeline with Isolation Forest Outlier Detector
    """
    iso_forest_contamination: float = 0.1
    knn_n_neighbors: int = 5
    variance_threshold: float = 0.0
    pearson_p_val: float = 0.05
    delete_previously_imputed = True
    sweep_config = {
        "iso_forest_contamination": {
            "values": ["auto", 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "knn_n_neighbors": {
            "values": [5, 10, 15, 20, 25]
        },
        "variance_threshold": {
            "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "pearson_p_val": {
            "values": [0.05, 0.1, 0.15, 0.2, 0.25]
        },
        "delete_previously_imputed": {
            "values": [True, False]
        }
    }

    def __init__(self,**kwargs):
        super(Pipeline1, self).__init__()
        self.__dict__.update(kwargs)
        self.pipeline = PreprocessingPipeline(KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=False),
                                              VarianceFeatureSelector(threshold=self.variance_threshold),
                                              PearsonCorrFeatureSelector(p_val=self.pearson_p_val),
                                              KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=True,
                                                         delete_previously_imputed=self.delete_previously_imputed),
                                              IsolationForestOutlierDetector(
                                                  contamination=self.iso_forest_contamination),
                                              StdScaler(),
                                              )

    def get(self) -> PreprocessingPipeline:
        return self.pipeline


@dataclass
class Pipeline2:
    """
    Pipeline with Cooks Dist Outlier Detector
    """
    cooks_threshold: float = 2
    knn_n_neighbors: int = 5
    variance_threshold: float = 0.0
    pearson_p_val: float = 0.05
    delete_previously_imputed = True
    sweep_config = {
        "cooks_threshold": {
            "values": [1.5, 2, 2.5, 3, 3.5]
        },
        "knn_n_neighbors": {
            "values": [5, 10, 15, 20, 25]
        },
        "variance_threshold": {
            "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "pearson_p_val": {
            "values": [0.05, 0.1, 0.15, 0.2, 0.25]
        },
        "delete_previously_imputed": {
            "values": [True, False]
        }
    }
    def __init__(self,**kwargs):
        super(Pipeline2, self).__init__()
        self.__dict__.update(kwargs)
        self.pipeline = PreprocessingPipeline(KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=False),
                                              VarianceFeatureSelector(threshold=self.variance_threshold),
                                              PearsonCorrFeatureSelector(p_val=self.pearson_p_val),
                                              KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=True, delete_previously_imputed=self.delete_previously_imputed),
                                              CooksDistOutlierDetector(threshold=self.cooks_threshold),
                                              StdScaler(),
                                              )

    def get(self) -> PreprocessingPipeline:
        return self.pipeline


@dataclass
class Pipeline3:
    """
    Pipeline with Elliptic Envelope Outlier Detector
    """
    elliptic_envelope_contamination: float = 0.1
    knn_n_neighbors: int = 5
    variance_threshold: float = 0.0
    pearson_p_val: float = 0.05
    delete_previously_imputed = True
    sweep_config = {
        "elliptic_envelope_contamination": {
            "values": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
        },
        "knn_n_neighbors": {
            "values": [5, 10, 15, 20, 25]
        },
        "variance_threshold": {
            "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "pearson_p_val": {
            "values": [0.05, 0.1, 0.15, 0.2, 0.25]
        },
        "delete_previously_imputed": {
            "values": [True, False]
        }
    }
    def __init__(self,**kwargs):
        super(Pipeline3, self).__init__()
        self.__dict__.update(kwargs)
        self.pipeline = PreprocessingPipeline(KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=False),
                                              VarianceFeatureSelector(threshold=self.variance_threshold),
                                              PearsonCorrFeatureSelector(p_val=self.pearson_p_val),
                                              KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=True, delete_previously_imputed=self.delete_previously_imputed),
                                              EllipticEnvelopeOutlierDetector(contamination=self.elliptic_envelope_contamination),
                                              StdScaler(),
                                              )

    def get(self) -> PreprocessingPipeline:
        return self.pipeline


@dataclass
class Pipeline4:
    """
    Pipeline with Local Outlier Factor Outlier Detector
    """
    lof_contamination: Union[str,float] = "auto"
    knn_n_neighbors: int = 5
    variance_threshold: float = 0.0
    pearson_p_val: float = 0.05
    delete_previously_imputed = True
    sweep_config = {
        "lof_contamination": {
            "values": ["auto", 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
        },
        "knn_n_neighbors": {
            "values": [5, 10, 15, 20, 25]
        },
        "variance_threshold": {
            "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "pearson_p_val": {
            "values": [0.05, 0.1, 0.15, 0.2, 0.25]
        },
        "delete_previously_imputed": {
            "values": [True, False]
        }
    }
    def __init__(self,**kwargs):
        super(Pipeline4, self).__init__()
        self.__dict__.update(kwargs)
        self.pipeline = PreprocessingPipeline(KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=False),
                                              VarianceFeatureSelector(threshold=self.variance_threshold),
                                              PearsonCorrFeatureSelector(p_val=self.pearson_p_val),
                                              KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=True, delete_previously_imputed=self.delete_previously_imputed),
                                              lof_outlier_detector(contamination=self.lof_contamination),
                                              StdScaler(),
                                              )

    def get(self) -> PreprocessingPipeline:
        return self.pipeline


"""
We define a data preprocessing pipeline. The pipeline starts with a 
K-nearest neighbours (KNN) imputer, which replaces missing values with 
estimates based on the values from the k-nearest neighbors of each data 
point.

After that, we apply standard scaling to the data, which scales the data 
features to have a mean of 0 and a standard deviation of 1.

The next step involves feature selection based on variance. Feature 
selection is performed to retain only those features with a variance 
greater than or equal to a specified threshold. [Features with low 
variance are typically less informative and can be removed to reduce 
dimensionality.]

We also perform feature selection based on Pearson correlation, [which 
measures the linear relationship between features and the target 
variable]. Features with a Pearson correlation coefficient greater than 
or equal to a specified p-value threshold are selected.

After performing KNN imputation again, we detect outliers using the 
Isolation Forest method. It assigns an anomaly score to each data point, 
and those with high anomaly scores are considered outliers.
"""
@dataclass
class Pipeline5:
    """
    Pipeline with Isolation Forest Outlier Detector and standard scaler before outlier detection
    """
    iso_forest_contamination: Union[float, str] = "auto"
    knn_n_neighbors: int = 5
    variance_threshold: float = 0.4
    pearson_p_val: float = 0.25
    delete_previously_imputed = True
    sweep_config = {
        "iso_forest_contamination": {
            "values": ["auto", 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "knn_n_neighbors": {
            "values": [5, 10, 15, 20, 25]
        },
        "variance_threshold": {
            "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "pearson_p_val": {
            "values": [0.05, 0.1, 0.15, 0.2, 0.25]
        },
        "delete_previously_imputed": {
            "values": [True, False]
        }
    }

    def __init__(self,**kwargs):
        super(Pipeline5, self).__init__()
        self.__dict__.update(kwargs)
        self.pipeline = PreprocessingPipeline(KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=False),
                                              StdScaler(),
                                              VarianceFeatureSelector(threshold=self.variance_threshold),
                                              PearsonCorrFeatureSelector(p_val=self.pearson_p_val),
                                              KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=True, delete_previously_imputed=self.delete_previously_imputed),
                                              IsolationForestOutlierDetector(contamination=self.iso_forest_contamination),
                                              )

    def get(self) -> PreprocessingPipeline:
        return self.pipeline


@dataclass
class Pipeline6:
    """
    Pipeline with Cooks Dist Outlier Detector and standard scaler before outlier detection
    """
    cooks_threshold: float = 2
    knn_n_neighbors: int = 5
    variance_threshold: float = 0.0
    pearson_p_val: float = 0.05
    delete_previously_imputed = True
    sweep_config = {
        "cooks_threshold": {
            "values": [1.5, 2, 2.5, 3, 3.5]
        },
        "knn_n_neighbors": {
            "values": [5, 10, 15, 20, 25]
        },
        "variance_threshold": {
            "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "pearson_p_val": {
            "values": [0.05, 0.1, 0.15, 0.2, 0.25]
        },
        "delete_previously_imputed": {
            "values": [True, False]
        }
    }
    def __init__(self,**kwargs):
        super(Pipeline6, self).__init__()
        self.__dict__.update(kwargs)
        self.pipeline = PreprocessingPipeline(KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=False),
                                              StdScaler(),
                                              VarianceFeatureSelector(threshold=self.variance_threshold),
                                              PearsonCorrFeatureSelector(p_val=self.pearson_p_val),
                                              KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=True, delete_previously_imputed=self.delete_previously_imputed),
                                              CooksDistOutlierDetector(threshold=self.cooks_threshold),
                                              )

    def get(self) -> PreprocessingPipeline:
        return self.pipeline


@dataclass
class Pipeline7:
    """
    Pipeline with Elliptic Envelope Outlier Detector and standard scaler before outlier detection
    """
    elliptic_envelope_contamination: float = 0.1
    knn_n_neighbors: int = 5
    variance_threshold: float = 0.0
    pearson_p_val: float = 0.05
    delete_previously_imputed = True
    sweep_config = {
        "elliptic_envelope_contamination": {
            "values": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
        },
        "knn_n_neighbors": {
            "values": [5, 10, 15, 20, 25]
        },
        "variance_threshold": {
            "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "pearson_p_val": {
            "values": [0.05, 0.1, 0.15, 0.2, 0.25]
        },
        "delete_previously_imputed": {
            "values": [True, False]
        }
    }
    def __init__(self,**kwargs):
        super(Pipeline7, self).__init__()
        self.__dict__.update(kwargs)
        self.pipeline = PreprocessingPipeline(KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=False),
                                              StdScaler(),
                                              VarianceFeatureSelector(threshold=self.variance_threshold),
                                              PearsonCorrFeatureSelector(p_val=self.pearson_p_val),
                                              KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=True, delete_previously_imputed=self.delete_previously_imputed),
                                              EllipticEnvelopeOutlierDetector(contamination=self.elliptic_envelope_contamination),
                                              )

    def get(self) -> PreprocessingPipeline:
        return self.pipeline


@dataclass
class Pipeline8:
    """
    Pipeline with Local Outlier Factor Outlier Detector and standard scaler before outlier detection
    """
    lof_contamination: Union[str,float] = "auto"
    knn_n_neighbors: int = 5
    variance_threshold: float = 0.0
    pearson_p_val: float = 0.05
    delete_previously_imputed = True
    sweep_config = {
        "lof_contamination": {
            "values": ["auto", 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
        },
        "knn_n_neighbors": {
            "values": [5, 10, 15, 20, 25]
        },
        "variance_threshold": {
            "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "pearson_p_val": {
            "values": [0.05, 0.1, 0.15, 0.2, 0.25]
        },
        "delete_previously_imputed": {
            "values": [True, False]
        }
    }
    def __init__(self,**kwargs):
        super(Pipeline8, self).__init__()
        self.__dict__.update(kwargs)
        self.pipeline = PreprocessingPipeline(KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=False),
                                              StdScaler(),
                                              VarianceFeatureSelector(threshold=self.variance_threshold),
                                              PearsonCorrFeatureSelector(p_val=self.pearson_p_val),
                                              KnnImputer(n_neighbors=self.knn_n_neighbors, imput_test=True, delete_previously_imputed=self.delete_previously_imputed),
                                              lof_outlier_detector(contamination=self.lof_contamination),
                                              )

    def get(self) -> PreprocessingPipeline:
        return self.pipeline

