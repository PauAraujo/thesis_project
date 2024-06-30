import csv

def save_dataset_to_file(language_code: str):
    """
    Saves the dataset for a given language code into a CSV file named '<language_code>.csv'.
    These CSV files are later used to calculate the tokenizer fertility of the SIB200 dataset on Databricks.
    Note: this data is produced locally and then uploaded to Databricks.
    The reason behind this is that Databricks is unable to directly load this particular dataset from HuggingFace.
    """
    try:
        # load dataset for the specified language code
        data = load_dataset_by_task('Davlan/sib200', language_code)
        data = data['text']
        filename = f'{language_code}.csv'

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            for item in data:
                csvwriter.writerow([item])
        print(f"Data saved to {filename}")

    except Exception as e:
        raise ValueError(f"Failed to save dataset for language code '{language_code}': {str(e)}")

if __name__ == "__main__":
    from analysis.data_loader import load_dataset_by_task

    # load list of language codes to process
    from analysis.config import SIB_LANGS_SUBSET

    # Process each language code
    for lang in SIB_LANGS_SUBSET:
        save_dataset_to_file(lang)

    # load data from CSV file
    file_path = 'kor_Hang.csv'
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        loaded_data = [row[0] for row in csvreader]
    print(loaded_data[:10])  # print first 10 lines to verify
