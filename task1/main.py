#!/usr/bin/env python3
import copy
import os

import pandas as pd
import sklearn.base
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, StackingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC, SVR
from tqdm import tqdm
from xgboost import XGBRegressor

import wandb
from PreprocessingPipeline import *
from pipeline_defs import Pipeline5

TRAINING_FEATURES_DATAPATH = "X_train.csv"
TRAINING_LABELS_DATAPATH = "y_train.csv"
TEST_FEATURES_DATAPATH = "X_test.csv"
GENERATE_SUBMISSION = True

"""
We read the training and testing data, excluding the "id" column, and 
store them in NumPy arrays.
"""
def read_data(dataset_type):
    if dataset_type == "train":
        X = pd.read_csv(TRAINING_FEATURES_DATAPATH).drop("id", axis=1).to_numpy()
        y = pd.read_csv(TRAINING_LABELS_DATAPATH).drop("id", axis=1).to_numpy().reshape(-1)
    elif dataset_type == "test":
        X = pd.read_csv(TEST_FEATURES_DATAPATH).drop("id", axis=1).to_numpy()
        y = None
    else:
        raise Exception(f"Unknown dataset type {dataset_type}")
    return X, y


def plot_predictions(X, y, predictions):
    import matplotlib.pyplot as plt
    n_samples = X.shape[0]

    predictions = np.array([prediction for _, prediction in sorted(zip(y, predictions))])
    y = np.array(sorted(y))

    # Plot ground truth against predictions
    plt.subplot(1, 2, 1)
    plt.xlabel("Sample")
    plt.ylabel("Age (in years)")
    plt.plot(range(n_samples), y, color="green", label="y")
    plt.plot(range(n_samples), predictions, color="red", label="Predictions")
    plt.legend()

    # Plot errors
    plt.subplot(1, 2, 2)
    plt.xlabel("Sample")
    plt.ylabel("Error (in years)")
    plt.bar(range(n_samples), y - predictions, label="Error (in years)")
    plt.legend()

    plt.show()

"""
We perform k-fold cross-validation. For each fold, we calculate R-squared
scores.
"""
def cross_validate_model(model, preprocessing_pipeline: PreprocessingPipeline, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    partitions = []
    for train_idx, test_idx in tqdm(kf.split(X, y), total=kf.get_n_splits(), desc="KFold"):
        data = Data(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        data = preprocessing_pipeline(data)
        partitions.append(data)
        # Train and predict.
        model.fit(data.x_train, data.y_train)
        predictions = model.predict(data.x_test)

        # plot_predictions(data.x_test, data.y_test, predictions)

        score = r2_score(data.y_test, predictions)
        scores.append(score)
    return scores, partitions

"""
We use stacking to learn how to best combine the predictions of 
multiple models. The models we use in our stacking ensemble are 
XGBRegressor, LGBMRegressor, CatBoostRegressor, ExtraTreesRegressor, 
HistGradientBoostingRegressor, and GradientBoostingRegressor. 

We then have a final estimator that uses the ExtraTreesRegressor. 

[Preprocessing pipeline, cross-validation]

Finally, we use the trained model to make brain age predictions on the 
test data and save the predictions to a csv file. 
"""
def main():
    os.environ["WANDB_SILENT"] = "true"
    wandb.login(key="820247dff99e9746cef87610049343c51a36d123")
    X, y = read_data("train")
    X_upload, _ = read_data("test")

    best_stacking_estimators = [
        ("XGBRegressor", XGBRegressor(booster="gbtree", gamma=0.9219898934941124, learning_rate=0.08870341810971061, max_depth=9, min_child_weight=37, n_estimators=351, n_jobs=-1, random_state=42)),
        ("LGBMRegressor", LGBMRegressor(boosting_type="dart", learning_rate=0.3134607002333142, max_depth=20, n_estimators=500, num_leaves=15, n_jobs=-1, random_state=42)),
        ("CatBoostRegressor", CatBoostRegressor(depth=None, iterations=None, learning_rate=0.05, random_state=42, silent=True)),
        ("ExtraTreesRegressor", ExtraTreesRegressor(criterion="squared_error", max_depth=None, n_estimators=127, n_jobs=-1, random_state=42)),
        ("HistGradientBoostingRegressor", HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05500248357517347, max_iter=500, random_state=42)),
        ("GradientBoostingRegressor", GradientBoostingRegressor(criterion="squared_error", learning_rate=0.11176223728492556, random_state=42)),
    ]

    best_stacking_final_estimator = ExtraTreesRegressor(criterion="absolute_error", max_depth=5, n_estimators=205, n_jobs=-1, random_state=42)
#("StackingRegressor", StackingRegressor(estimators=best_stacking_estimators, n_jobs=-1)),
        # ("LGBMRegressor",
        #  LGBMRegressor(random_state=42, boosting_type="dart", n_estimators=500, learning_rate=0.3134607002333142,
        #                max_depth=20, num_leaves=15)),
        # ("XGBRegressor", XGBRegressor(random_state=42, learning_rate=0.05, n_estimators=500, max_depth=4)),
        # ("CatBoostRegressor", CatBoostRegressor(random_state=42, silent=True)),
        # ("ExtraTreesRegressor", ExtraTreesRegressor(max_depth=None, n_estimators=180, n_jobs=-1, random_state=42)),
        # ("RandomForestRegressor", RandomForestRegressor(n_jobs=-1, random_state=42)),
        # ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=42)),
        # ("HistGradientBoostingRegressor", HistGradientBoostingRegressor(random_state=42)),
        # ("AdaBoostRegressor", AdaBoostRegressor(random_state=42)),
        # ("SVC", SVC(random_state=42)),
    models = [
        ("StackingRegressor_v2", StackingRegressor(estimators=best_stacking_estimators, final_estimator=best_stacking_final_estimator, n_jobs=-1))
    ]
 
    for model_name, model in models:
        config = {
            "model": model_name,
            "n_splits": 5,
        }
        mode = "disabled" if os.environ["USER"] in ["prada", "francescmartiescofet"] else None
        run = wandb.init(project="aml-task1", entity="aml-group", name=model_name, reinit=True, config=config,
                         mode=mode)
        config = run.config
        print(f"Trying {model_name}")
        # preprocessing_pipeline = PreprocessingPipeline(KnnImputer(imput_test=True, n_neighbors=5),
        #                                                VarianceFeatureSelector(threshold=0.0),
        #                                                PearsonCorrFeatureSelector(p_val=0.05),
        #                                                StdScaler(),
        #                                                IsolationForestOutlierDetector(plot_pca_outliers=False,
        #                                                                               contamination="auto"))

        preprocessing_pipeline = Pipeline5().get()
        scores, partitions = cross_validate_model(model, preprocessing_pipeline, X, y, n_splits=config["n_splits"])
        if sklearn.base.is_regressor(model):
            pass
            # wandb.sklearn.plot_regressor(model, X_train, X_test, Y_train, Y_test, model_name=model_name)
        print(f"\tScores {scores}")
        print(f"\tMean: {np.mean(scores)}")
        print(f"\tStdev: {np.std(scores)}")
        run.log({"scores": scores, "mean_r2score": np.mean(scores), "std_r2score": np.std(scores)})
        if GENERATE_SUBMISSION:
            data = Data(X, y, X_upload, None)
            data = preprocessing_pipeline(data)
            model.fit(data.x_train, data.y_train)
            predictions = model.predict(data.x_test)
            submission = pd.DataFrame({"id": np.arange(len(predictions)), "y": predictions})
            submission.to_csv(f"submission_{model_name}.csv", index=False)
        run.finish()


if __name__ == "__main__":
    np.random.seed(42)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.WARN,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Uncomment to just do preprocessing pipeline to check results like the outliers
    preprocessing_pipeline_0 = PreprocessingPipeline(KnnImputer(),
                                                      VarianceFeatureSelector(threshold=0.0),
                                                      PearsonCorrFeatureSelector(p_val=0.05),
                                                      StdScaler())
    preprocessing_pipeline_1 = PreprocessingPipeline(KnnImputer(),
                                                      VarianceFeatureSelector(threshold=0.0),
                                                      PearsonCorrFeatureSelector(p_val=0.05),
                                                      StdScaler(),
                                                      lof_outlier_detector(plot_pca_outliers=True, contamination=0.08))
    preprocessing_pipeline_2 = PreprocessingPipeline(KnnImputer(),
                                                      VarianceFeatureSelector(threshold=0.0),
                                                      PearsonCorrFeatureSelector(p_val=0.05),
                                                      StdScaler(),
                                                      IsolationForestOutlierDetector(plot_pca_outliers=True,
                                                                                     contamination=0.063829787))
    preprocessing_pipeline_3 = PreprocessingPipeline(KnnImputer(),
                                                      VarianceFeatureSelector(threshold=0.0),
                                                      PearsonCorrFeatureSelector(p_val=0.05),
                                                      StdScaler(),
                                                      CooksDistOutlierDetector(plot_pca_outliers=True, threshold=1.5))
    preprocessing_pipeline_4 = PreprocessingPipeline(KnnImputer(),
                                                      VarianceFeatureSelector(threshold=0.0),
                                                      PearsonCorrFeatureSelector(p_val=0.05),
                                                      StdScaler(),
                                                      EllipticEnvelopeOutlierDetector(plot_pca_outliers=True))
    X, y = read_data("train")
    data0 = Data(copy.deepcopy(X), copy.deepcopy(y), copy.deepcopy(X), copy.deepcopy(y))
    data1 = Data(copy.deepcopy(X), copy.deepcopy(y), copy.deepcopy(X), copy.deepcopy(y))
    data2 = Data(copy.deepcopy(X), copy.deepcopy(y), copy.deepcopy(X), copy.deepcopy(y))
    data3 = Data(copy.deepcopy(X), copy.deepcopy(y), copy.deepcopy(X), copy.deepcopy(y))
    data4 = Data(copy.deepcopy(X), copy.deepcopy(y), copy.deepcopy(X), copy.deepcopy(y))
    
    data0 = preprocessing_pipeline_0(data0)
    OutlierDetector.plot_pca(data0, data0.y_train, "y~x")
    
    data1 = preprocessing_pipeline_1(data1)
    data2 = preprocessing_pipeline_2(data2)
    data3 = preprocessing_pipeline_3(data3)
    data4 = preprocessing_pipeline_4(data4)
    #exit()

    main()
