import os
import time
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


def preprocess_dataset(input_file, output_file, chunk_size='5MB'):
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File {input_file} not found")
        print("Loading dataset...")
        start_time = time.time()

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
            df = dd.read_csv(input_file,
                             delimiter="|",
                             blocksize=chunk_size,
                             dtype=dtypes,
                             assume_missing=True,
                             na_values=["-"],
                             keep_default_na=True,
                             engine='c')

            rows_before = df.shape[0].compute()
            cols_before = len(df.columns)
            print(
                f"Input dataset shape: ({rows_before}, {cols_before})")

            print("Removing rows with missing values in critical columns...")
            critical_columns = ['duration',
                                'orig_bytes', 'resp_bytes', 'history']

            mask = ~df[critical_columns].isnull().any(axis=1)

            for col in critical_columns:
                mask = mask & (df[col] != '-')

            filtered_df = df[mask]

            rows_after_filtering = filtered_df.shape[0].compute()
            rows_removed = rows_before - rows_after_filtering
            print(f"Rows removed due to missing values: {rows_removed} "
                  f"({rows_removed/rows_before*100:.2f}%)")

            print("Consolidating classes (merging all Malicious subtypes)...")
            if 'label' in filtered_df.columns:
                def simplify_labels(label):
                    if label.startswith('Malicious'):
                        return 'Malicious'
                    return label

                filtered_df['label'] = filtered_df['label'].map(
                    simplify_labels, meta=('label', 'string'))

                with ProgressBar():
                    new_label_counts = filtered_df['label'].value_counts(
                    ).compute()
                    print("New class distribution:")
                    total = new_label_counts.sum()
                    for label, count in new_label_counts.items():
                        percentage = float(
                            np.round((count / total) * 100, decimals=2))
                        print(f"   {label}: {int(count)} ({percentage}%)")

            print("Removing unnecessary columns...")
            columns_to_remove = ['service', 'local_orig',
                                 'local_resp', 'detailed-label', 'tunnel_parents']
            processed_df = filtered_df.drop(columns=columns_to_remove)

            cols_after = len(processed_df.columns)
            print(
                f"Shape after processing: ({rows_after_filtering}, {cols_after})")
            print(f"Removed columns: {', '.join(columns_to_remove)}")

            print(f"Saving processed dataset...")
            output_dir = os.path.dirname(output_file)
            if (output_dir and not os.path.exists(output_dir)):
                os.makedirs(output_dir, exist_ok=True)

            processed_df.to_csv(output_file,
                                single_file=True,
                                sep='|',
                                index=False)

            end_time = time.time()
            print(
                f"Preprocessing completed in {end_time - start_time:.2f} seconds")
            print(f"Result saved to: {output_file}")
            print(f"Processing summary:")
            print(
                f"   - Input dataset: {rows_before} rows, {cols_before} columns")
            print(f"   - Rows removed: {rows_removed}")
            print(f"   - Columns removed: {len(columns_to_remove)}")
            print(f"   - Classes consolidated: all 'Malicious' subtypes → 'Malicious'")
            print(
                f"   - Output dataset: {rows_after_filtering} rows, {cols_after} columns")

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise


def get_dataset_info(file_path, chunk_size='5MB'):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        print(
            f"Getting info for dataset {os.path.basename(file_path)}...")

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
            df = dd.read_csv(file_path,
                             delimiter="|",
                             blocksize=chunk_size,
                             dtype=dtypes,
                             assume_missing=True,
                             na_values=["-"],
                             keep_default_na=True,
                             engine='c')

            info = {
                "rows": df.shape[0].compute(),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024)
            }

            print(f"Info retrieved successfully")
            return info

    except Exception as e:
        print(f"Error getting info: {str(e)}")
        raise


if __name__ == "__main__":
    input_file = "data/8_CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv"
    output_file = "data/data_upd_small.csv"

    preprocess_dataset(input_file, output_file)

    original_info = get_dataset_info(input_file)
    processed_info = get_dataset_info(output_file)

    print("\nDataset comparison:")
    print(
        f"Original: {original_info['rows']} rows, {original_info['columns']} columns")
    print(
        f"Processed: {processed_info['rows']} rows, {processed_info['columns']} columns")
    print(
        f"Removed {original_info['columns'] - processed_info['columns']} columns")
