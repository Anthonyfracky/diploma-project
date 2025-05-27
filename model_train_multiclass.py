import time
import warnings
import numpy as np
import seaborn as sns
import lightgbm as lgb
import dask.dataframe as dd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from dask.diagnostics import ProgressBar
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef


warnings.filterwarnings("ignore")


def load_and_preprocess_multiclass(file_path, chunk_size='5MB'):
    print("Loading multiclass dataset...")

    dtypes = {
        "ts": "float64",
        "id.orig_p": "int64",
        "id.resp_p": "int64",
        "duration": "float64",
        "orig_bytes": "float64",
        "resp_bytes": "float64",
        "missed_bytes": "float64",
        "history": "string",
        "orig_pkts": "int64",
        "orig_ip_bytes": "int64",
        "resp_pkts": "int64",
        "resp_ip_bytes": "int64",
        "detailed-label": "int64"
    }

    with ProgressBar():
        df = dd.read_csv(file_path,
                         delimiter="|",
                         blocksize=chunk_size,
                         dtype=dtypes,
                         assume_missing=True,
                         na_values=["-"],
                         keep_default_na=True,
                         engine='c')

    df = df.fillna(0)

    with ProgressBar():
        df = df.compute()
    return df


def inverse_class_map_from_data(df):
    class_map = {
        0: "Torii",
        1: "C&C",
        2: "HeartBeat",
        3: "PartOfAHorizontalPortScan",
        4: "FileDownload",
        5: "Attack"
    }

    print("\nClass mapping:")
    for k, v in class_map.items():
        print(f"  [{k}] → {v}")

    return class_map


def plot_confusion_matrix(y_true, y_pred, labels, filename, model_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def get_metrics(y_true, y_pred, train_time, average="weighted"):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "train_time": train_time
    }


def save_metrics_report(metrics_dict, filename="confusions_multiclass/all_models_report.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(
            "Model comparison (accuracy, precision, recall, f1, mcc, train_time sec)\n")
        f.write("="*80 + "\n")
        header = "{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<12}\n".format(
            "Model", "Accuracy", "Precision", "Recall", "F1", "MCC", "TrainTime")
        f.write(header)
        f.write("-"*80 + "\n")
        for model, metrics in metrics_dict.items():
            f.write("{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<12.2f}\n".format(
                model,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["mcc"],
                metrics["train_time"]
            ))


def train_lightgbm_multiclass(X_train, X_test, y_train, y_test, class_map):
    print("Training LightGBM...")

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(np.unique(y_train)),
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)

    print(f"Training completed in {train_time:.2f} sec")
    print("\nClassification Report:")
    target_names = [class_map.get(i, str(i))
                    for i in sorted(np.unique(y_test))]
    print(classification_report(y_test, y_pred, target_names=target_names))

    plot_confusion_matrix(y_test, y_pred, labels=target_names,
                          filename="confusions_multiclass/confusion_lgbm_multiclass_normalized.png",
                          model_name="LightGBM")
    return get_metrics(y_test, y_pred, train_time)


def train_xgboost_multiclass(X_train, X_test, y_train, y_test, class_map):
    print("Training XGBoost...")
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = model.predict(X_test)
    print(f"Training completed in {train_time:.2f} sec")
    print("\nClassification Report (XGBoost):")
    target_names = [class_map.get(i, str(i))
                    for i in sorted(np.unique(y_test))]
    print(classification_report(y_test, y_pred, target_names=target_names))
    plot_confusion_matrix(y_test, y_pred, labels=target_names,
                          filename="confusions_multiclass/confusion_xgboost_multiclass_normalized.png",
                          model_name="XGBoost")
    return get_metrics(y_test, y_pred, train_time)


def train_naive_bayes_multiclass(X_train, X_test, y_train, y_test, class_map):
    print("Training Naive Bayes...")
    model = GaussianNB()
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = model.predict(X_test)
    print(f"Training completed in {train_time:.2f} sec")
    print("\nClassification Report (Naive Bayes):")
    target_names = [class_map.get(i, str(i))
                    for i in sorted(np.unique(y_test))]
    print(classification_report(y_test, y_pred, target_names=target_names))
    plot_confusion_matrix(y_test, y_pred, labels=target_names,
                          filename="confusions_multiclass/confusion_nb_multiclass_normalized.png",
                          model_name="Naive Bayes")
    return get_metrics(y_test, y_pred, train_time)


def train_adaboost_multiclass(X_train, X_test, y_train, y_test, class_map):
    print("Training AdaBoost...")
    model = AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    )
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = model.predict(X_test)
    print(f"Training completed in {train_time:.2f} sec")
    print("\nClassification Report (AdaBoost):")
    target_names = [class_map.get(i, str(i))
                    for i in sorted(np.unique(y_test))]
    print(classification_report(y_test, y_pred, target_names=target_names))
    plot_confusion_matrix(y_test, y_pred, labels=target_names,
                          filename="confusions_multiclass/confusion_adaboost_multiclass_normalized.png",
                          model_name="AdaBoost")
    return get_metrics(y_test, y_pred, train_time)


def train_random_forest_multiclass(X_train, X_test, y_train, y_test, class_map):
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = model.predict(X_test)
    print(f"Training completed in {train_time:.2f} sec")
    print("\nClassification Report (Random Forest):")
    target_names = [class_map.get(i, str(i))
                    for i in sorted(np.unique(y_test))]
    print(classification_report(y_test, y_pred, target_names=target_names))
    plot_confusion_matrix(y_test, y_pred, labels=target_names,
                          filename="confusions_multiclass/confusion_rf_multiclass_normalized.png",
                          model_name="Random Forest")
    return get_metrics(y_test, y_pred, train_time)


def plot_all_model_metrics(metrics_dict, filename="confusions_multiclass/all_models_metrics.png"):
    """
    metrics_dict: {
        "ModelName": {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
        ...
    }
    """
    models = list(metrics_dict.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    values = {m: [metrics_dict[model][m] for model in models] for m in metrics}

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(12, 6))
    for idx, metric in enumerate(metrics):
        plt.bar(x + idx*width - 1.5*width,
                values[metric], width, label=metric.capitalize())
    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model metrics comparison (accuracy, precision, recall, f1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    file_path = "data/dataset_upd_multiclass.csv"

    df = load_and_preprocess_multiclass(file_path)

    X = df.drop(columns=["detailed-label"])
    y = df["detailed-label"]

    class_map = inverse_class_map_from_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)

    metrics_nb = train_naive_bayes_multiclass(
        X_train, X_test, y_train, y_test, class_map)
    metrics_rf = train_random_forest_multiclass(
        X_train, X_test, y_train, y_test, class_map)
    metrics_adaboost = train_adaboost_multiclass(
        X_train, X_test, y_train, y_test, class_map)
    metrics_xgb = train_xgboost_multiclass(
        X_train, X_test, y_train, y_test, class_map)
    metrics_lgbm = train_lightgbm_multiclass(
        X_train, X_test, y_train, y_test, class_map)

    metrics_dict = {
        "NaiveBayes": metrics_nb,
        "RandomForest": metrics_rf,
        "AdaBoost": metrics_adaboost,
        "XGBoost": metrics_xgb,
        "LightGBM": metrics_lgbm
    }
    save_metrics_report(metrics_dict)


if __name__ == "__main__":
    main()
