import time
import numpy as np
import lightgbm as lgbm
import dask.dataframe as dd
import matplotlib.pyplot as plt
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from dask.diagnostics import ProgressBar
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


def load_and_preprocess_data(file_path, chunk_size='5MB'):
    dtypes = {
        "ts": "float64",
        "uid": "string",
        "id.orig_h": "string",
        "id.orig_p": "int64",
        "id.resp_h": "string",
        "id.resp_p": "int64",
        "proto": "string",
        "service": "string",
        "duration": "float64",
        "orig_bytes": "float64",
        "resp_bytes": "float64",
        "conn_state": "string",
        "local_orig": "string",
        "local_resp": "string",
        "missed_bytes": "float64",
        "history": "string",
        "orig_pkts": "int64",
        "orig_ip_bytes": "int64",
        "resp_pkts": "int64",
        "resp_ip_bytes": "int64",
        "tunnel_parents": "string",
        "label": "string",
        "detailed-label": "string"
    }

    print("Loading dataset...")

    with ProgressBar():
        df = dd.read_csv(file_path,
                         delimiter="|",
                         blocksize=chunk_size,
                         dtype=dtypes,
                         assume_missing=True,
                         na_values=["-"],
                         keep_default_na=True,
                         engine='c')

    drop_columns = ["uid", "id.orig_h", "id.resp_h",
                    "proto", "conn_state", "history"]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    df = df.fillna(0)

    df["label"] = df["label"].map(
        {"Benign": 0, "Malicious": 1}, meta=("label", "int64"))

    with ProgressBar():
        df = df.compute()

    return df


def get_models(selected_models):
    all_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42),
        "Light GBM": lgbm.LGBMClassifier(random_state=42, verbose=-1),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=75, random_state=42)
    }
    return {name: model for name, model in all_models.items() if selected_models.get(name, False)}


def train_and_evaluate_models(X_train, X_test, y_train, y_test, selected_models):
    models = get_models(selected_models)
    results = {}
    for name, model in models.items():
        print(f"Training model: {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        cm = confusion_matrix(y_test, y_test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Матриця помилок: {name}")
        plt.tight_layout()
        plt.savefig(f"confusions/confusion_{name.replace(' ', '_')}.png")
        plt.close()

        results[name] = {
            "Train Accuracy": train_acc * 100,
            "Test Accuracy": test_acc * 100,
            "Train F1": train_f1 * 100,
            "Test F1": test_f1 * 100,
            "Train Time (sec)": train_time
        }

        print(
            f"{name} done! Train Acc: {train_acc*100:.2f}%, Test Acc: {test_acc*100:.2f}%")

    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X_train.columns
            sorted_idx = np.argsort(importances)

            plt.figure(figsize=(10, 6))
            plt.barh(feature_names[sorted_idx], importances[sorted_idx])
            plt.xlabel("Feature Importance")
            plt.title(f"Feature Importance for {name}")
            plt.tight_layout()
            plt.savefig(
                f"confusions/feature_importance_{name.replace(' ', '_')}.png")
            plt.close()

    return results


def train_and_evaluate_models_cv(X, y, selected_models):
    models = get_models(selected_models)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\nStarting model training with cross-validation...")
    for name, model in models.items():
        print(f"\nTraining model: {name}...")
        start_time = time.time()

        with tqdm(total=5, desc=f"  Обробка {name}", unit=" fold") as pbar:
            accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            pbar.update(1)
            precision = cross_val_score(
                model, X, y, cv=skf, scoring='precision')
            pbar.update(1)
            recall = cross_val_score(model, X, y, cv=skf, scoring='recall')
            pbar.update(1)
            f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
            pbar.update(1)
            roc_auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
            pbar.update(1)
            mcc = cross_val_score(model, X, y, cv=skf,
                                  scoring='matthews_corrcoef')

        train_time = time.time() - start_time

        results[name] = {
            "Accuracy": np.mean(accuracy) * 100,
            "Precision": np.mean(precision) * 100,
            "Recall": np.mean(recall) * 100,
            "F1-score": np.mean(f1) * 100,
            "ROC-AUC": np.mean(roc_auc) * 100,
            "MCC": np.mean(mcc) * 100,
            "Train Time (sec)": train_time
        }

        print(f"{name} done! Training time: {train_time:.2f} sec")

    return results


def main():
    use_cv = True

    selected_models = {
        "Logistic Regression": True,
        "XGBoost": True,
        "AdaBoost": True,
        "Light GBM": True,
        "Naive Bayes": True,
        "Random Forest": True
    }

    file_path = "data_upd.csv"
    df = load_and_preprocess_data(file_path)

    X = df.drop(columns=["label"])
    y = df["label"]

    if use_cv:
        results = train_and_evaluate_models_cv(X, y, selected_models)
        print("\nModel comparison (with cross-validation):")
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"    - Accuracy: {metrics['Accuracy']:.2f}%")
            print(f"    - Precision: {metrics['Precision']:.2f}%")
            print(f"    - Recall: {metrics['Recall']:.2f}%")
            print(f"    - F1-score: {metrics['F1-score']:.2f}%")
            print(f"    - ROC-AUC: {metrics['ROC-AUC']:.2f}%")
            print(f"    - MCC: {metrics['MCC']:.2f}%")
            print(f"    - Train Time: {metrics['Train Time (sec)']:.2f} sec")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        results = train_and_evaluate_models(
            X_train, X_test, y_train, y_test, selected_models)
        print("\nModel comparison:")
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"    - Train Accuracy: {metrics['Train Accuracy']:.2f}%")
            print(f"    - Test Accuracy: {metrics['Test Accuracy']:.2f}%")
            print(f"    - Train F1-score: {metrics['Train F1']:.2f}%")
            print(f"    - Test F1-score: {metrics['Test F1']:.2f}%")
            print(f"    - Train Time: {metrics['Train Time (sec)']:.2f} sec")


if __name__ == "__main__":
    main()
