import os
import pandas as pd


def main():
    input_folder = 'data'
    output_file = 'dataset.csv'

    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    dataframes = []

    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)
        print(f"Reading {file_path}...")
        df = pd.read_csv(file_path)
        dataframes.append(df)
        print(f"Finished reading {file_path}")

    print("Concatenating dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    print("Finished concatenating dataframes")

    print(f"Writing combined dataframe to {output_file}...")
    combined_df.to_csv(output_file, index=False)

    print(f"All CSV files have been combined into {output_file}")


if __name__ == "__main__":
    main()
