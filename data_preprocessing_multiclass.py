import os
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


def preprocess_multiclass_dataset(input_file, output_file, chunk_size='5MB'):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found")
    print("Loading dataset...")
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
    with ProgressBar():
        df = dd.read_csv(
            input_file,
            delimiter="|",
            blocksize=chunk_size,
            dtype=dtypes,
            na_values=["-"],
            assume_missing=True,
            keep_default_na=True,
            engine="c"
        )
    print("Filtering empty values...")
    critical_columns = ["duration", "orig_bytes",
                        "resp_bytes", "history", "detailed-label"]
    mask = ~df[critical_columns].isnull().any(axis=1)
    for col in critical_columns:
        mask = mask & (df[col] != '-')
    df = df[mask]

    print("Dropping columns...")
    columns_to_remove = [
        "uid", "id.orig_h", "id.resp_h", "proto",
        "conn_state", "local_orig", "local_resp",
        "service", "tunnel_parents", "label"
    ]
    df = df.drop(
        columns=[col for col in columns_to_remove if col in df.columns])

    print("Filling missing values...")
    df = df.fillna(0)

    print("Dropping 'history' column...")
    if "history" in df.columns:
        df = df.drop(columns=["history"])

    print("Encoding categorical label `detailed-label`...")
    with ProgressBar():
        class_mapping = df["detailed-label"].value_counts().compute().index.tolist()
        class_dict = {label: idx for idx, label in enumerate(class_mapping)}

    df["detailed-label"] = df["detailed-label"].map(
        class_dict, meta=("detailed-label", "int64"))

    print("Saving dataset to file...")
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ProgressBar():
        df.to_csv(output_file, single_file=True, sep="|", index=False)

    print(f"Ready multiclass dataset saved to: {output_file}")
    print(f"Detected {len(class_dict)} classes:")
    for label, idx in class_dict.items():
        print(f"  [{idx}] — {label}")


if __name__ == "__main__":
    input_path = "data/dataset.csv"
    output_path = "data/dataset_upd_multiclass.csv"
    preprocess_multiclass_dataset(input_path, output_path)
