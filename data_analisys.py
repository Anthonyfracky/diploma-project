import os
import numpy as np
import seaborn as sns
import dask.dataframe as dd
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar


def safe_write_analysis(df, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        try:
            print("Writing basic information...")
            npartitions = df.npartitions
            with ProgressBar():
                rows = df.shape[0].compute()
                cols = len(df.columns)

            f.write(f"Number of partitions: {npartitions}\n")
            f.write(f"Dataset shape: ({rows}, {cols})\n\n")

            print("Writing data sample...")
            f.write("Data sample (preview):\n")
            with ProgressBar():
                sample = df.head(n=3, compute=True)
                f.write(sample.to_string())
            f.write("\n\n")

            f.write("Data types:\n")
            f.write(df.dtypes.to_string())
            f.write("\n\n")

            print("Analyzing missing values...")
            with ProgressBar():
                missing = (df.isnull() | (df == '-')).sum().compute()
                missing_pct = np.round((missing / rows * 100), 2)
                f.write("Missing values (count | %):\n")
                if missing.sum() == 0:
                    f.write(
                        "Dataset is complete (no missing values)\n\n")
                else:
                    for col in df.columns:
                        if missing[col] > 0:
                            f.write(
                                f"{col}: {missing[col]} | {missing_pct[col]}%\n")
                    f.write("\n")
        except Exception as e:
            f.write(f"Error during data analysis: {str(e)}\n")
            raise

        if 'label' in df.columns:
            print("Analyzing class distribution...")
            try:
                with ProgressBar():
                    label_counts = df['label'].value_counts().compute()
                    f.write("Class distribution:\n")
                    total = label_counts.sum()
                    for label, count in label_counts.items():
                        percentage = float(
                            np.round((count / total) * 100, decimals=2))
                        f.write(f"{label}: {int(count)} ({percentage}%)\n")
            except Exception as e:
                f.write(
                    f"Error during class distribution analysis: {str(e)}\n")

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            print("Analyzing numeric variables...")
            f.write("\nNumeric variable statistics:\n")
            for col in numeric_cols:
                try:
                    with ProgressBar():
                        stats = df[col].describe().compute()
                        f.write(f"\nStatistics for {col}:\n")
                        f.write(stats.to_string())
                        f.write("\n")
                except Exception as e:
                    f.write(f"Error analyzing column {col}: {str(e)}\n")


def analyze_dataset(file_path, output_file, chunk_size='5MB'):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

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

        df = dd.read_csv(file_path,
                         delimiter="|",
                         blocksize=chunk_size,
                         dtype=dtypes,
                         assume_missing=True,
                         na_values=["-"],
                         keep_default_na=True,
                         engine='c')

        safe_write_analysis(df, output_file)

        if 'label' in df.columns:
            print("Creating visualization...")
            os.makedirs("visualizations", exist_ok=True)

            with ProgressBar():
                label_dist = df['label'].value_counts().compute()
                plt.figure(figsize=(10, 6))
                sns.barplot(x=label_dist.index, y=label_dist.values)
                plt.title("Class distribution")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig("visualizations/class_distribution.png")
                plt.close()

        print("Analysis completed successfully!")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    input_file = "data/dataset.csv"
    output_file = "analysis_results_full.txt"
    analyze_dataset(input_file, output_file)
