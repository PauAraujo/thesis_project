# Databricks notebook source
# !pip install stanza
# !pip install datasets
# dbutils.library.restartPython()

# COMMAND ----------

import json
import stanza
import logging 
import csv
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union
from datasets import load_dataset, DatasetDict, concatenate_datasets

from subword_fertility.data_loader import load_dataset_by_task
from subword_fertility.config import (
    MODEL_MT5SMALL, MODEL_BLOOM560M,
    XNLI_LANGS, PAWSX_LANGS, SIB_LANGS_SUBSET, SIB_LANG_CODE_MAP,
    PAWSX_TEXT_FIELD, XNLI_TEXT_FIELD, SIB_TEXT_FIELD
    
    )

# COMMAND ----------

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Stanza pipelines
print("Building Stanza pipelines...")
try:
    ar_nlp = stanza.Pipeline('ar', processors='tokenize', verbose=False, use_gpu=True)
    bg_nlp = stanza.Pipeline('bg', processors='tokenize', verbose=False, use_gpu=True)
    de_nlp = stanza.Pipeline('de', processors='tokenize', verbose=False, use_gpu=True)
    el_nlp = stanza.Pipeline('el', processors='tokenize', verbose=False, use_gpu=True)
    en_nlp = stanza.Pipeline('en', processors='tokenize', verbose=False, use_gpu=True)
    es_nlp = stanza.Pipeline('es', processors='tokenize', verbose=False, use_gpu=True)
    fr_nlp = stanza.Pipeline('fr', processors='tokenize', verbose=False, use_gpu=True)
    hi_nlp = stanza.Pipeline('hi', processors='tokenize', verbose=False, use_gpu=True)
    ru_nlp = stanza.Pipeline('ru', processors='tokenize', verbose=False, use_gpu=True)
    th_nlp = stanza.Pipeline('th', processors='tokenize', verbose=False, use_gpu=True)
    tr_nlp = stanza.Pipeline('tr', processors='tokenize', verbose=False, use_gpu=True)
    ur_nlp = stanza.Pipeline('ur', processors='tokenize', verbose=False, use_gpu=True)
    vi_nlp = stanza.Pipeline('vi', processors='tokenize', verbose=False, use_gpu=True)
    zh_nlp = stanza.Pipeline('zh', processors='tokenize', verbose=False, use_gpu=True)
    ja_nlp = stanza.Pipeline('ja', processors='tokenize', verbose=False, use_gpu=True)
    ko_nlp = stanza.Pipeline('ko', processors='tokenize', verbose=False, use_gpu=True)
except Exception as e:
    logging.error(f"Error initializing Stanza pipelines: {e}")

# COMMAND ----------

def load_tokenizer(model_name: str) -> Optional[AutoTokenizer]:
    """
    loads a tokenizer from the Hugging Face model hub.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info(f"Successfully loaded tokenizer: {model_name}")
        return tokenizer
    except (OSError, ConnectionError) as e:
        logging.error(f"Error loading tokenizer: {model_name}, {e}")
        return None

def calculate_subword_fertility(
        lang_dataset: Union[List, Dict], tokenizer: AutoTokenizer, text_column: str, lang: str, invalid_entries: List[Dict]) -> float:
    """
    Calculates the average subword fertility for a dataset in a specific language.
    Args:
        lang_dataset: dataset for the language (has to be compatible with Transformers datasets).
        tokenizer: tokenizer to use for subword tokenization.
        text_column: name of the column in the dataset containing the text.
        lang: anguage code for specific handling of tokenization.
        invalid_entries: list to collect invalid entries (samples with 0 words or 0 subwords).
    Returns:
        float: avg. subword fertility for the language in the dataset.
    """

    words_count: List[int] = []
    subwords_count: List[int] = []

    for i, sample in enumerate(lang_dataset):
        # lang_dataset is a Dataset object for xpaws and xnli, and a List[str] for SIB200
        if text_column != SIB_TEXT_FIELD:
            text = sample[text_column]
        else:
            text = sample 
        try:
            # Stanza for word segmentation
            if lang == 'ar':
                words = [word.text for sentence in ar_nlp(text).sentences for word in sentence.words]
            elif lang == 'bg':
                words = [word.text for sentence in bg_nlp(text).sentences for word in sentence.words]
            elif lang == 'de':
                words = [word.text for sentence in de_nlp(text).sentences for word in sentence.words]
            elif lang == 'el':
                words = [word.text for sentence in el_nlp(text).sentences for word in sentence.words]
            elif lang == 'en':
                words = [word.text for sentence in en_nlp(text).sentences for word in sentence.words]
            elif lang == 'es':
                words = [word.text for sentence in es_nlp(text).sentences for word in sentence.words]
            elif lang == 'fr':
                words = [word.text for sentence in fr_nlp(text).sentences for word in sentence.words]
            elif lang == 'hi':
                words = [word.text for sentence in hi_nlp(text).sentences for word in sentence.words]
            elif lang == 'ja':
                words = [word.text for sentence in ja_nlp(text).sentences for word in sentence.words]
            elif lang == 'ko':
                words = [word.text for sentence in ko_nlp(text).sentences for word in sentence.words]
            elif lang == 'ru':
                words = [word.text for sentence in ru_nlp(text).sentences for word in sentence.words]
            elif lang == 'th':
                words = [word.text for sentence in th_nlp(text).sentences for word in sentence.words]
            elif lang == 'tr':
                words = [word.text for sentence in tr_nlp(text).sentences for word in sentence.words]
            elif lang == 'ur':
                words = [word.text for sentence in ur_nlp(text).sentences for word in sentence.words]
            elif lang == 'vi':
                words = [word.text for sentence in vi_nlp(text).sentences for word in sentence.words]
            elif lang == 'zh':
                words = [word.text for sentence in zh_nlp(text).sentences for word in sentence.words]
            else:  # for sw
                words = text.split()  # use simple whitespace split, since sw not available in Stanza

            subwords = tokenizer.tokenize(text)

            if len(words) > 0 and len(subwords) > 0:
                words_count.append(len(words))
                subwords_count.append(len(subwords))
            else:
                invalid_entries.append({
                    "id": i,
                    "sample": text,
                    "language": lang,
                    "words_count": len(words),
                    "subwords_count": len(subwords)
                })
                logging.warning(f"Sample {i} produced {len(words)} words and {len(subwords)} subwords")

        except Exception as e:
            logging.error(f"Error processing sample {i}: {text}")
            logging.exception(e)

    if not words_count or not subwords_count:
        raise ValueError("Empty dataset or text column not found or no valid entries to calculate fertility.")

    # filter out any zero counts to avoid division by zero
    valid_entries = [(w, s) for w, s in zip(words_count, subwords_count) if w != 0 and s != 0]
    if not valid_entries:
        raise ValueError("No valid entries with non-zero words and subwords count to calculate fertility.")

    words_count, subwords_count = zip(*valid_entries)

    # calculate avg. fertility
    language_avg_fertility = round(np.mean(np.array(subwords_count) / np.array(words_count)), 4)

    return language_avg_fertility

def get_fertility_for_all_languages(model: str, task_name: str) -> Dict[str, float]:
    """
    Calculates subword fertility for all languages in a dataset for a given task.
    """

    tokenizer = load_tokenizer(model)

    logging.info(f"Gathering subword fertility for {task_name} per language.")

    if task_name == "xnli":
        languages = XNLI_LANGS
        text_field = XNLI_TEXT_FIELD
    elif task_name == "paws-x":
        languages = PAWSX_LANGS
        text_field = PAWSX_TEXT_FIELD
    elif task_name == "sib":
        languages = SIB_LANGS_SUBSET 
        text_field = SIB_TEXT_FIELD
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    invalid_entries = []

    # Calculate fertility for each language
    fertility = {}
    for lang in languages:
        # If SIB, process from files (directly loading from HF does not work for this dataset on DB)
        if task_name == 'sib':
            file_path = f'subword_fertility/sib_data/{lang}.csv'
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                lang_dataset = [row[0] for row in csvreader] 
            # Turn 8 char SIB lang code into 2 char lang code, , e.g., 'eng_Latn' into 'en'
            lang = SIB_LANG_CODE_MAP[lang]
        else:
            lang_dataset = load_dataset_by_task(task_name, lang)
        if lang_dataset:
            logging.info(f"Calculating subword fertility")
            fertility[lang] = calculate_subword_fertility(lang_dataset, tokenizer, text_field, lang, invalid_entries)
        else:
            logging.warning(f"Failed to load dataset for language: {lang}")

    # dave invalid entries to CSV for further analysis
    if invalid_entries:
        invalid_df = pd.DataFrame(invalid_entries)
        invalid_df.to_csv(f"{base_path}invalid_entries.csv", index=False)
        logging.info(f"Saved invalid entries to {base_path}invalid_entries.csv")

    return fertility

def save_fertility_to_json(fertility_data: Dict[str, float], filename: str):
    """Saves  fertility dict to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(fertility_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved fertility to {filename}")

base_path = "/Workspace/Users/paraujorabinovich@deloitte.nl/LLMResearch/subword_fertility/fertilities/"


# COMMAND ----------

mt5_pawsx_fertility = get_fertility_for_all_languages(MODEL_MT5SMALL, "paws-x")
save_fertility_to_json(mt5_pawsx_fertility, base_path + "mt5_pawsx_fertility.json")
print('MT5 and PAWS-X fertility:')
print(mt5_pawsx_fertility)

# COMMAND ----------

mt5_xnli_fertility = get_fertility_for_all_languages(MODEL_MT5SMALL, "xnli")
save_fertility_to_json(mt5_xnli_fertility, base_path + "mt5_xnli_fertility.json")
print('MT5 and SIB fertility:')
print(mt5_xnli_fertility)

# COMMAND ----------

mt5_sib_fertility = get_fertility_for_all_languages(MODEL_MT5SMALL, "sib")
save_fertility_to_json(mt5_sib_fertility, base_path + "mt5_sib_fertility.json")
print('MT5 and SIB fertility:')
print(mt5_sib_fertility)

# COMMAND ----------

bloom_pawsx_fertility = get_fertility_for_all_languages(MODEL_BLOOM560M, "paws-x")
save_fertility_to_json(bloom_pawsx_fertility, base_path + "bloom_pawsx_fertility.json")
print('BLOOM and PAWS-X fertility:')
print(bloom_pawsx_fertility)

# COMMAND ----------

bloom_xnli_fertility = get_fertility_for_all_languages(MODEL_BLOOM560M, "xnli")
save_fertility_to_json(bloom_pawsx_fertility, base_path + "bloom_xnli_fertility.json")
print('BLOOM and XNLI fertility:')
print(bloom_xnli_fertility)

# COMMAND ----------

bloom_sib_fertility = get_fertility_for_all_languages(MODEL_BLOOM560M, "xnli")
save_fertility_to_json(bloom_sib_fertility, base_path + "bloom_sib_fertility.json")
print('BLOOM and SIB200 fertility:')
print(bloom_sib_fertility)



