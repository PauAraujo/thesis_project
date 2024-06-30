from typing import Optional
from datasets import load_dataset, DatasetDict, concatenate_datasets
import logging
from config import XNLI_TEXT_FIELD, PAWSX_TEXT_FIELD, SIB_TEXT_FIELD

def load_dataset_by_task(task_name: str, lang: str) -> Optional[DatasetDict]:
    """
    Loads a dataset for a specific task and language.
    """

    try:
        dataset = load_dataset(task_name, lang)
        # Concatenate train, test, and validation
        full_dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['validation']])

        # Create task-specific column
        if task_name == 'xnli':
            full_dataset = full_dataset.map(lambda x: {XNLI_TEXT_FIELD : f"{x['premise']} {x['hypothesis']}"})
        elif task_name == 'paws-x':
            full_dataset = full_dataset.map(lambda x: {PAWSX_TEXT_FIELD: f"{x['sentence1']} {x['sentence2']}"})
        elif task_name == 'Davlan/sib200':
            #full_dataset = full_dataset[SIB_TEXT_FIELD]
            pass
        else:
            logging.warning(f"Task '{task_name}' not supported for custom column creation.")

        logging.info(f"Successfully loaded {task_name} dataset for language: {lang}")
        return full_dataset
    except Exception as e:
        logging.error(f"Error loading {task_name} dataset for language {lang}: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load PAWS-X dataset
    pawsx_dataset = load_dataset_by_task('paws-x', 'ja')
    print(pawsx_dataset)

