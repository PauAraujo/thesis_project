import numpy as np


from config import (
    SIB_LANG_CODE_MAP,
    # MT5 models with LORA finetuning
    METRICS_MT5SMALL_PAWSX_LORA_r64, METRICS_MT5SMALL_XNLI_LORA_r64,
    METRICS_MT5BASE_PAWSX_LORA_r64, METRICS_MT5BASE_XNLI_LORA_r64,
    METRICS_MT5LARGE_PAWSX_LORA_r64, METRICS_MT5LARGE_XNLI_LORA_r64,
    # MT5 models with QLORA finetuning
    METRICS_MT5SMALL_PAWSX_QLORA_r64, METRICS_MT5SMALL_XNLI_QLORA_r64,
    METRICS_MT5BASE_PAWSX_QLORA_r64, METRICS_MT5BASE_XNLI_QLORA_r64,
    METRICS_MT5LARGE_PAWSX_QLORA_r64, METRICS_MT5LARGE_XNLI_QLORA_r64,
    # MT5 models with FULL finetuning
    METRICS_MT5SMALL_PAWSX_FULLFINETUNE, METRICS_MT5SMALL_XNLI_FULLFINETUNE,
    METRICS_MT5BASE_PAWSX_FULLFINETUNE, METRICS_MT5BASE_XNLI_FULLFINETUNE,
    METRICS_MT5LARGE_PAWSX_FULLFINETUNE, METRICS_MT5LARGE_XNLI_FULLFINETUNE,
    # BLOOM models with LORA finetuning
    METRICS_BLOOM560M_PAWSX_LORA_r64, METRICS_BLOOM560M_XNLI_LORA_r64,
    METRICS_BLOOM1B1_PAWSX_LORA_r64, METRICS_BLOOM1B1_XNLI_LORA_r64,
    METRICS_BLOOM1B7_PAWSX_LORA_r64, METRICS_BLOOM1B7_XNLI_LORA_r64,
    # BLOOM models with QLORA finetuning
    METRICS_BLOOM560M_PAWSX_QLORA_r64, METRICS_BLOOM560M_XNLI_QLORA_r64,
    METRICS_BLOOM1B1_PAWSX_QLORA_r64, METRICS_BLOOM1B1_XNLI_QLORA_r64,
    METRICS_BLOOM1B7_PAWSX_QLORA_r64, METRICS_BLOOM1B7_XNLI_QLORA_r64,
    # BLOOM models with FULL finetuning
    METRICS_BLOOM560M_PAWSX_FULLFINETUNE, METRICS_BLOOM560M_XNLI_FULLFINETUNE,
    METRICS_BLOOM1B1_PAWSX_FULLFINETUNE, METRICS_BLOOM1B1_XNLI_FULLFINETUNE,
    METRICS_BLOOM1B7_PAWSX_FULLFINETUNE, METRICS_BLOOM1B7_XNLI_FULLFINETUNE,
)

# import SIB200 metrics subset with relevant languages
from analysis.data_processing.process_sib_metrics import (
    METRICS_MT5SMALL_SIB_LORA_r64, METRICS_MT5BASE_SIB_LORA_r64, METRICS_MT5LARGE_SIB_LORA_r64,
    METRICS_MT5SMALL_SIB_QLORA_r64, METRICS_MT5BASE_SIB_QLORA_r64, METRICS_MT5LARGE_SIB_QLORA_r64,
    METRICS_MT5SMALL_SIB_FULLFINETUNE, METRICS_MT5BASE_SIB_FULLFINETUNE, METRICS_MT5LARGE_SIB_FULLFINETUNE,
    METRICS_BLOOM560M_SIB_LORA_r64, METRICS_BLOOM1B1_SIB_LORA_r64, METRICS_BLOOM1B7_SIB_LORA_r64,
    METRICS_BLOOM560M_SIB_QLORA_r64, METRICS_BLOOM1B1_SIB_QLORA_r64, METRICS_BLOOM1B7_SIB_QLORA_r64,
    METRICS_BLOOM560M_SIB_FULLFINETUNE, METRICS_BLOOM1B1_SIB_FULLFINETUNE, METRICS_BLOOM1B7_SIB_FULLFINETUNE,

    METRICS_MT5SMALL_SIB_LORA_r64_MORE_LANGS, METRICS_MT5BASE_SIB_LORA_r64_MORE_LANGS, METRICS_MT5LARGE_SIB_LORA_r64_MORE_LANGS,
    METRICS_MT5SMALL_SIB_QLORA_r64_MORE_LANGS, METRICS_MT5BASE_SIB_QLORA_r64_MORE_LANGS, METRICS_MT5LARGE_SIB_QLORA_r64_MORE_LANGS,
    METRICS_MT5SMALL_SIB_FULLFINETUNE_MORE_LANGS, METRICS_MT5BASE_SIB_FULLFINETUNE_MORE_LANGS, METRICS_MT5LARGE_SIB_FULLFINETUNE_MORE_LANGS,
    METRICS_BLOOM560M_SIB_LORA_r64_MORE_LANGS, METRICS_BLOOM1B1_SIB_LORA_r64_MORE_LANGS, METRICS_BLOOM1B7_SIB_LORA_r64_MORE_LANGS,
    METRICS_BLOOM560M_SIB_QLORA_r64_MORE_LANGS, METRICS_BLOOM1B1_SIB_QLORA_r64_MORE_LANGS, METRICS_BLOOM1B7_SIB_QLORA_r64_MORE_LANGS,
    METRICS_BLOOM560M_SIB_FULLFINETUNE_MORE_LANGS, METRICS_BLOOM1B1_SIB_FULLFINETUNE_MORE_LANGS, METRICS_BLOOM1B7_SIB_FULLFINETUNE_MORE_LANGS
)

# List of languages: all PAWS-X and XNLI languages
languages = list(SIB_LANG_CODE_MAP.values())

def average_performance(all_metrics):
    accumulated_accuracies = {lang: [] for lang in languages}

    for metrics in all_metrics:
        for lang, acc in zip(metrics['language'], metrics['accuracy']):
            accumulated_accuracies[lang].append(acc)

    # compute the average accuracies
    average_accuracies = {lang: round(np.mean(accs),4) if accs else None for lang, accs in accumulated_accuracies.items()}

    # create the final dictionary with score averages
    metrics_avg = {
        'language': languages,
        'accuracy': [average_accuracies[lang] for lang in languages]
    }

    return metrics_avg


def format_metrics(metrics, description):
    lines = []
    lines.append(description)
    lines.append("Language | Accuracy")
    lines.append("---------|---------")
    for lang, acc in zip(metrics['language'], metrics['accuracy']):
        lines.append(f"{lang:<9} | {acc:<8}")
    lines.append("\n")
    return "\n".join(lines)


def calculate_percentage_change(new_value, old_value):
    """Function to calculate percentage change"""
    return round((((new_value - old_value) / old_value) * 100) ,2)

def create_percentage_change_dict(lora_metrics, qlora_metrics, full_finetune_metrics): #OLD used for family ranks plot
    """Function to create percentage change dictionary"""
    languages = full_finetune_metrics['language']
    # LORA accuracy percentage % change over full finetuning
    perc_change_lora = {
        'language': languages,
        'accuracy': [calculate_percentage_change(lora, full) for lora, full in
                     zip(lora_metrics['accuracy'], full_finetune_metrics['accuracy'])]
    }
    # QLORA accuracy percentage % change over full finetuning
    perc_change_qlora = {
        'language': languages,
        'accuracy': [calculate_percentage_change(qlora, full) for qlora, full in
                     zip(qlora_metrics['accuracy'], full_finetune_metrics['accuracy'])]
    }
    return perc_change_lora, perc_change_qlora

def create_qlora_percentage_change_dict(qlora_metrics, full_finetune_metrics):
    """Function to create percentage change dictionary"""
    languages = full_finetune_metrics['language']
    # QLORA accuracy percentage % change over full finetuning
    perc_change_qlora = {
        'language': languages,
        'accuracy': [calculate_percentage_change(qlora, full) for qlora, full in
                     zip(qlora_metrics['accuracy'], full_finetune_metrics['accuracy'])]
    }
    return perc_change_qlora

def calculate_average_percentage_change(*percentage_changes):
    """Calculate the average percentage change for each model and dataset"""
    average_changes = []
    for change in percentage_changes:
        avg_change = np.mean(change['accuracy'])
        average_changes.append(round(avg_change,4))
    return average_changes

def calculate_average_metrics(metrics_list):
    if not metrics_list:
        return {}

    languages = metrics_list[0]['language']
    num_languages = len(languages)
    total_accuracy = [0.0] * num_languages

    for metrics in metrics_list:
        for i, acc in enumerate(metrics['accuracy']):
            total_accuracy[i] += acc

    average_accuracy = [acc / len(metrics_list) for acc in total_accuracy]

    return {'language': languages, 'accuracy': average_accuracy}

###############################################################################################################
# Aggregate accuracies for all datasets of each MT5 model
###############################################################################################################
# LORA Metrics -----------------------------------------------------------------------------------------------
# Average LORA performance scores per language for all MT5-Small Models (All Datasets)
all_mt5small_lora_metrics = [
    METRICS_MT5SMALL_PAWSX_LORA_r64,
    METRICS_MT5SMALL_XNLI_LORA_r64,
    METRICS_MT5SMALL_SIB_LORA_r64_MORE_LANGS,
]
METRICS_MT5SMALL_AVG_LORA = average_performance(all_mt5small_lora_metrics)

# Average LORA performance scores per language for all MT5-Base Models (All Datasets)
all_mt5base_lora_metrics = [
    METRICS_MT5BASE_PAWSX_LORA_r64,
    METRICS_MT5BASE_XNLI_LORA_r64,
    METRICS_MT5BASE_SIB_LORA_r64_MORE_LANGS,
]
METRICS_MT5BASE_AVG_LORA = average_performance(all_mt5base_lora_metrics)

# Average LORA performance scores per language for all MT5-Large Models (All Datasets)
all_mt5large_lora_metrics = [
    METRICS_MT5LARGE_PAWSX_LORA_r64,
    METRICS_MT5LARGE_XNLI_LORA_r64,
    METRICS_MT5LARGE_SIB_LORA_r64_MORE_LANGS,
]
METRICS_MT5LARGE_AVG_LORA = average_performance(all_mt5large_lora_metrics)

# QLORA Metrics -----------------------------------------------------------------------------------------------
# Average QLORA performance scores per language for all MT5-Small Models (All Datasets)
all_mt5small_qlora_metrics = [
    METRICS_MT5SMALL_PAWSX_QLORA_r64,
    METRICS_MT5SMALL_XNLI_QLORA_r64,
    METRICS_MT5SMALL_SIB_QLORA_r64_MORE_LANGS,
]
METRICS_MT5SMALL_AVG_QLORA = average_performance(all_mt5small_qlora_metrics)

# Average QLORA performance scores per language for all MT5-Base Models (All Datasets)
all_mt5base_qlora_metrics = [
    METRICS_MT5BASE_PAWSX_QLORA_r64,
    METRICS_MT5BASE_XNLI_QLORA_r64,
    METRICS_MT5BASE_SIB_QLORA_r64_MORE_LANGS,
]
METRICS_MT5BASE_AVG_QLORA = average_performance(all_mt5base_qlora_metrics)

# Average QLORA performance scores per language for all MT5-Large Models (All Datasets)
all_mt5large_qlora_metrics = [
    METRICS_MT5LARGE_PAWSX_QLORA_r64,
    METRICS_MT5LARGE_XNLI_QLORA_r64,
    METRICS_MT5LARGE_SIB_QLORA_r64_MORE_LANGS,
]
METRICS_MT5LARGE_AVG_QLORA = average_performance(all_mt5large_qlora_metrics)

# FULLFINETUNE Metrics -----------------------------------------------------------------------------------------------
# Average FULLFINETUNE performance scores per language for all MT5-Small Models (All Datasets)
all_mt5small_fullfinetune_metrics = [
    METRICS_MT5SMALL_PAWSX_FULLFINETUNE,
    METRICS_MT5SMALL_XNLI_FULLFINETUNE,
    METRICS_MT5SMALL_SIB_FULLFINETUNE_MORE_LANGS,
]
METRICS_MT5SMALL_AVG_FULLFINETUNE = average_performance(all_mt5small_fullfinetune_metrics)

# Average FULLFINETUNE performance scores per language for all MT5-Base Models (All Datasets)
all_mt5base_fullfinetune_metrics = [
    METRICS_MT5BASE_PAWSX_FULLFINETUNE,
    METRICS_MT5BASE_XNLI_FULLFINETUNE,
    METRICS_MT5BASE_SIB_FULLFINETUNE_MORE_LANGS,
]
METRICS_MT5BASE_AVG_FULLFINETUNE = average_performance(all_mt5base_fullfinetune_metrics)

# Average FULLFINETUNE performance scores per language for all MT5-Large Models (All Datasets)
all_mt5large_fullfinetune_metrics = [
    METRICS_MT5LARGE_PAWSX_FULLFINETUNE,
    METRICS_MT5LARGE_XNLI_FULLFINETUNE,
    METRICS_MT5LARGE_SIB_FULLFINETUNE_MORE_LANGS
]
METRICS_MT5LARGE_AVG_FULLFINETUNE = average_performance(all_mt5large_fullfinetune_metrics)


###############################################################################################################
# Aggregate accuracies for all datasets of each BLOOM model
###############################################################################################################
# LORA Metrics -----------------------------------------------------------------------------------------------
# Average LORA performance scores per language for all BLOOM-560M Models (All Datasets)
all_bloom560m_lora_metrics = [
    METRICS_BLOOM560M_PAWSX_LORA_r64,
    METRICS_BLOOM560M_XNLI_LORA_r64,
    METRICS_BLOOM560M_SIB_LORA_r64_MORE_LANGS,
]
METRICS_BLOOM560M_AVG_LORA = average_performance(all_bloom560m_lora_metrics)

# Average LORA performance scores per language for all BLOOM-1B1 Models (All Datasets)
all_bloom1b1_lora_metrics = [
    METRICS_BLOOM1B1_PAWSX_LORA_r64,
    METRICS_BLOOM1B1_XNLI_LORA_r64,
    METRICS_BLOOM1B1_SIB_LORA_r64_MORE_LANGS,
]
METRICS_BLOOM1B1_AVG_LORA = average_performance(all_bloom1b1_lora_metrics)

# Average LORA performance scores per language for all BLOOM-1B7 Models (All Datasets)
all_bloom1b7_lora_metrics = [
    METRICS_BLOOM1B7_PAWSX_LORA_r64,
    METRICS_BLOOM1B7_XNLI_LORA_r64,
    METRICS_BLOOM1B7_SIB_LORA_r64_MORE_LANGS,
]
METRICS_BLOOM1B7_AVG_LORA = average_performance(all_bloom1b7_lora_metrics)

# QLORA Metrics -----------------------------------------------------------------------------------------------
# Average LORA performance scores per language for all BLOOM-560M Models (All Datasets)
all_bloom560m_qlora_metrics = [
    METRICS_BLOOM560M_PAWSX_QLORA_r64,
    METRICS_BLOOM560M_XNLI_QLORA_r64,
    METRICS_BLOOM560M_SIB_QLORA_r64_MORE_LANGS,
]
METRICS_BLOOM560M_AVG_QLORA = average_performance(all_bloom560m_qlora_metrics)

# Average LORA performance scores per language for all BLOOM-1B1 Models (All Datasets)
all_bloom1b1_qlora_metrics = [
    METRICS_BLOOM1B1_PAWSX_QLORA_r64,
    METRICS_BLOOM1B1_XNLI_QLORA_r64,
    METRICS_BLOOM1B1_SIB_QLORA_r64_MORE_LANGS,
]
METRICS_BLOOM1B1_AVG_QLORA = average_performance(all_bloom1b1_qlora_metrics)

# Average LORA performance scores per language for all BLOOM-1B7 Models (All Datasets)
all_bloom1b7_qlora_metrics = [
    METRICS_BLOOM1B7_PAWSX_QLORA_r64,
    METRICS_BLOOM1B7_XNLI_QLORA_r64,
    METRICS_BLOOM1B7_SIB_QLORA_r64_MORE_LANGS,
]
METRICS_BLOOM1B7_AVG_QLORA = average_performance(all_bloom1b7_qlora_metrics)

# FULLFINETUNE Metrics -----------------------------------------------------------------------------------------------
# Average FULLFINETUNE performance scores per language for all BLOOM-560M Models (All Datasets)
all_bloom560m_fullfinetune_metrics = [
    METRICS_BLOOM560M_PAWSX_FULLFINETUNE,
    METRICS_BLOOM560M_XNLI_FULLFINETUNE,
    METRICS_BLOOM560M_SIB_FULLFINETUNE_MORE_LANGS,
]
METRICS_BLOOM560M_AVG_FULLFINETUNE = average_performance(all_bloom560m_fullfinetune_metrics)

# Average FULLFINETUNE performance scores per language for all BLOOM-1B1 Models (All Datasets)
all_bloom1b1_fullfinetune_metrics = [
    METRICS_BLOOM1B1_PAWSX_FULLFINETUNE,
    METRICS_BLOOM1B1_XNLI_FULLFINETUNE,
    METRICS_BLOOM1B1_SIB_FULLFINETUNE_MORE_LANGS,
]
METRICS_BLOOM1B1_AVG_FULLFINETUNE = average_performance(all_bloom1b1_fullfinetune_metrics)

# Average FULLFINETUNE performance scores per language for all BLOOM-1B7 Models (All Datasets)
all_bloom1b7_fullfinetune_metrics = [
    METRICS_BLOOM1B7_PAWSX_FULLFINETUNE,
    METRICS_BLOOM1B7_XNLI_FULLFINETUNE,
    METRICS_BLOOM1B7_SIB_FULLFINETUNE_MORE_LANGS,
]
METRICS_BLOOM1B7_AVG_FULLFINETUNE = average_performance(all_bloom1b7_fullfinetune_metrics)

###############################################################################################################
# Aggregate accuracies for all MT5 models
###############################################################################################################
# lists for all MT5 metrics per category
all_mt5_lora_metrics = [
    *all_mt5small_lora_metrics, # list within a list is unpacked and added to the list
    *all_mt5base_lora_metrics,
    *all_mt5large_lora_metrics
]

all_mt5_qlora_metrics = [
    *all_mt5small_qlora_metrics,
    *all_mt5base_qlora_metrics,
    *all_mt5large_qlora_metrics
]

all_mt5_fullfinetune_metrics = [
    *all_mt5small_fullfinetune_metrics,
    *all_mt5base_fullfinetune_metrics,
    *all_mt5large_fullfinetune_metrics
]
# Calculate average performance for each category
METRICS_MT5_AVG_LORA = average_performance(all_mt5_lora_metrics)
METRICS_MT5_AVG_QLORA = average_performance(all_mt5_qlora_metrics)
METRICS_MT5_AVG_FULLFINETUNE = average_performance(all_mt5_fullfinetune_metrics)


###############################################################################################################
# Aggregate accuracies for all BLOOM models
###############################################################################################################
# lists for all MT5 metrics per category
all_bloom_lora_metrics = [
    *all_bloom560m_lora_metrics,
    *all_bloom1b1_lora_metrics,
    *all_bloom1b7_lora_metrics
]

all_bloom_qlora_metrics = [
    *all_bloom560m_qlora_metrics,
    *all_bloom1b1_qlora_metrics,
    *all_bloom1b7_qlora_metrics
]

all_bloom_fullfinetune_metrics = [
    *all_bloom560m_fullfinetune_metrics,
    *all_bloom1b1_fullfinetune_metrics,
    *all_bloom1b7_fullfinetune_metrics
]
METRICS_BLOOM_AVG_LORA = average_performance(all_bloom_lora_metrics)
METRICS_BLOOM_AVG_QLORA = average_performance(all_bloom_qlora_metrics)
METRICS_BLOOM_AVG_FULLFINETUNE = average_performance(all_bloom_fullfinetune_metrics)

# Calculate the % change for each language's accuracy from the LORA and QLORA finetuning methods relative to FULLFINETUNE
# Where % change = [( (new value - old value) / old value) * 100] - 100
# Create percentage change dictionaries for MT5SMALL
PERC_CHANGE_MT5SMALL_LORA, PERC_CHANGE_MT5SMALL_QLORA = create_percentage_change_dict(
    METRICS_MT5SMALL_AVG_LORA,
    METRICS_MT5SMALL_AVG_QLORA,
    METRICS_MT5SMALL_AVG_FULLFINETUNE
)
# Create percentage change dictionaries for MT5BASE
PERC_CHANGE_MT5BASE_LORA, PERC_CHANGE_MT5BASE_QLORA = create_percentage_change_dict(
    METRICS_MT5BASE_AVG_LORA,
    METRICS_MT5BASE_AVG_QLORA,
    METRICS_MT5BASE_AVG_FULLFINETUNE
)

# Create percentage change dictionaries for MT5LARGE
PERC_CHANGE_MT5LARGE_LORA, PERC_CHANGE_MT5LARGE_QLORA = create_percentage_change_dict(
    METRICS_MT5LARGE_AVG_LORA,
    METRICS_MT5LARGE_AVG_QLORA,
    METRICS_MT5LARGE_AVG_FULLFINETUNE
)

# Create percentage change dictionaries for BLOOM560M
PERC_CHANGE_BLOOM560M_LORA, PERC_CHANGE_BLOOM560M_QLORA = create_percentage_change_dict(
    METRICS_BLOOM560M_AVG_LORA,
    METRICS_BLOOM560M_AVG_QLORA,
    METRICS_BLOOM560M_AVG_FULLFINETUNE
)

# Create percentage change dictionaries for BLOOM1B1
PERC_CHANGE_BLOOM1B1_LORA, PERC_CHANGE_BLOOM1B1_QLORA = create_percentage_change_dict(
    METRICS_BLOOM1B1_AVG_LORA,
    METRICS_BLOOM1B1_AVG_QLORA,
    METRICS_BLOOM1B1_AVG_FULLFINETUNE
)

# Create percentage change dictionaries for BLOOM1B7
PERC_CHANGE_BLOOM1B7_LORA, PERC_CHANGE_BLOOM1B7_QLORA = create_percentage_change_dict(
    METRICS_BLOOM1B7_AVG_LORA,
    METRICS_BLOOM1B7_AVG_QLORA,
    METRICS_BLOOM1B7_AVG_FULLFINETUNE
)

#### PERCENTAGE CHANGE PER DATASET
# Calculate percentage changes for PAWSX dataset
PERC_CHANGE_MT5SMALL_PAWSX_LORA, PERC_CHANGE_MT5SMALL_PAWSX_QLORA = create_percentage_change_dict(
    METRICS_MT5SMALL_PAWSX_LORA_r64, METRICS_MT5SMALL_PAWSX_QLORA_r64, METRICS_MT5SMALL_PAWSX_FULLFINETUNE
)

PERC_CHANGE_MT5BASE_PAWSX_LORA, PERC_CHANGE_MT5BASE_PAWSX_QLORA = create_percentage_change_dict(
    METRICS_MT5BASE_PAWSX_LORA_r64, METRICS_MT5BASE_PAWSX_QLORA_r64, METRICS_MT5BASE_PAWSX_FULLFINETUNE
)

PERC_CHANGE_MT5LARGE_PAWSX_LORA, PERC_CHANGE_MT5LARGE_PAWSX_QLORA = create_percentage_change_dict(
    METRICS_MT5LARGE_PAWSX_LORA_r64, METRICS_MT5LARGE_PAWSX_QLORA_r64, METRICS_MT5LARGE_PAWSX_FULLFINETUNE
)

# Calculate percentage changes for XNLI dataset
PERC_CHANGE_MT5SMALL_XNLI_LORA, PERC_CHANGE_MT5SMALL_XNLI_QLORA = create_percentage_change_dict(
    METRICS_MT5SMALL_XNLI_LORA_r64, METRICS_MT5SMALL_XNLI_QLORA_r64, METRICS_MT5SMALL_XNLI_FULLFINETUNE
)

PERC_CHANGE_MT5BASE_XNLI_LORA, PERC_CHANGE_MT5BASE_XNLI_QLORA = create_percentage_change_dict(
    METRICS_MT5BASE_XNLI_LORA_r64, METRICS_MT5BASE_XNLI_QLORA_r64, METRICS_MT5BASE_XNLI_FULLFINETUNE
)

PERC_CHANGE_MT5LARGE_XNLI_LORA, PERC_CHANGE_MT5LARGE_XNLI_QLORA = create_percentage_change_dict(
    METRICS_MT5LARGE_XNLI_LORA_r64, METRICS_MT5LARGE_XNLI_QLORA_r64, METRICS_MT5LARGE_XNLI_FULLFINETUNE
)

# Calculate percentage changes for SIB dataset
PERC_CHANGE_MT5SMALL_SIB_LORA, PERC_CHANGE_MT5SMALL_SIB_QLORA = create_percentage_change_dict(
    METRICS_MT5SMALL_SIB_LORA_r64_MORE_LANGS, METRICS_MT5SMALL_SIB_QLORA_r64_MORE_LANGS, METRICS_MT5SMALL_SIB_FULLFINETUNE_MORE_LANGS
)

PERC_CHANGE_MT5BASE_SIB_LORA, PERC_CHANGE_MT5BASE_SIB_QLORA = create_percentage_change_dict(
    METRICS_MT5BASE_SIB_LORA_r64_MORE_LANGS, METRICS_MT5BASE_SIB_QLORA_r64_MORE_LANGS, METRICS_MT5BASE_SIB_FULLFINETUNE_MORE_LANGS
)

PERC_CHANGE_MT5LARGE_SIB_LORA, PERC_CHANGE_MT5LARGE_SIB_QLORA = create_percentage_change_dict(
    METRICS_MT5LARGE_SIB_LORA_r64_MORE_LANGS, METRICS_MT5LARGE_SIB_QLORA_r64_MORE_LANGS, METRICS_MT5LARGE_SIB_FULLFINETUNE_MORE_LANGS
)

# Calculate percentage changes for BLOOM560M dataset
PERC_CHANGE_BLOOM560M_PAWSX_LORA, PERC_CHANGE_BLOOM560M_PAWSX_QLORA = create_percentage_change_dict(
    METRICS_BLOOM560M_PAWSX_LORA_r64, METRICS_BLOOM560M_PAWSX_QLORA_r64, METRICS_BLOOM560M_PAWSX_FULLFINETUNE
)

PERC_CHANGE_BLOOM1B1_PAWSX_LORA, PERC_CHANGE_BLOOM1B1_PAWSX_QLORA = create_percentage_change_dict(
    METRICS_BLOOM1B1_PAWSX_LORA_r64, METRICS_BLOOM1B1_PAWSX_QLORA_r64, METRICS_BLOOM1B1_PAWSX_FULLFINETUNE
)

PERC_CHANGE_BLOOM1B7_PAWSX_LORA, PERC_CHANGE_BLOOM1B7_PAWSX_QLORA = create_percentage_change_dict(
    METRICS_BLOOM1B7_PAWSX_LORA_r64, METRICS_BLOOM1B7_PAWSX_QLORA_r64, METRICS_BLOOM1B7_PAWSX_FULLFINETUNE
)

# Calculate percentage changes for BLOOM1B1 dataset
PERC_CHANGE_BLOOM560M_XNLI_LORA, PERC_CHANGE_BLOOM560M_XNLI_QLORA = create_percentage_change_dict(
    METRICS_BLOOM560M_XNLI_LORA_r64, METRICS_BLOOM560M_XNLI_QLORA_r64, METRICS_BLOOM560M_XNLI_FULLFINETUNE
)

PERC_CHANGE_BLOOM1B1_XNLI_LORA, PERC_CHANGE_BLOOM1B1_XNLI_QLORA = create_percentage_change_dict(
    METRICS_BLOOM1B1_XNLI_LORA_r64, METRICS_BLOOM1B1_XNLI_QLORA_r64, METRICS_BLOOM1B1_XNLI_FULLFINETUNE
)

PERC_CHANGE_BLOOM1B7_XNLI_LORA, PERC_CHANGE_BLOOM1B7_XNLI_QLORA = create_percentage_change_dict(
    METRICS_BLOOM1B7_XNLI_LORA_r64, METRICS_BLOOM1B7_XNLI_QLORA_r64, METRICS_BLOOM1B7_XNLI_FULLFINETUNE
)

# Calculate percentage changes for BLOOM1B7 dataset
PERC_CHANGE_BLOOM560M_SIB_LORA, PERC_CHANGE_BLOOM560M_SIB_QLORA = create_percentage_change_dict(
    METRICS_BLOOM560M_SIB_LORA_r64_MORE_LANGS, METRICS_BLOOM560M_SIB_QLORA_r64_MORE_LANGS, METRICS_BLOOM560M_SIB_FULLFINETUNE_MORE_LANGS
)

PERC_CHANGE_BLOOM1B1_SIB_LORA, PERC_CHANGE_BLOOM1B1_SIB_QLORA = create_percentage_change_dict(
    METRICS_BLOOM1B1_SIB_LORA_r64_MORE_LANGS, METRICS_BLOOM1B1_SIB_QLORA_r64_MORE_LANGS, METRICS_BLOOM1B1_SIB_FULLFINETUNE_MORE_LANGS
)

PERC_CHANGE_BLOOM1B7_SIB_LORA, PERC_CHANGE_BLOOM1B7_SIB_QLORA = create_percentage_change_dict(
    METRICS_BLOOM1B7_SIB_LORA_r64_MORE_LANGS, METRICS_BLOOM1B7_SIB_QLORA_r64_MORE_LANGS, METRICS_BLOOM1B7_SIB_FULLFINETUNE_MORE_LANGS
)

#### Aggregate PERCENTAGE% CHANGES by DATASET
# Calculate average percentage changes for each dataset LORA
AVG_CHANGES_MT5_PAWSX_LORA = calculate_average_percentage_change(
    PERC_CHANGE_MT5SMALL_PAWSX_LORA,
    PERC_CHANGE_MT5SMALL_PAWSX_LORA,
    PERC_CHANGE_MT5LARGE_PAWSX_LORA,
    PERC_CHANGE_MT5LARGE_PAWSX_LORA,
)

AVG_CHANGES_MT5_XNLI_LORA = calculate_average_percentage_change(
    PERC_CHANGE_MT5SMALL_XNLI_LORA,
    PERC_CHANGE_MT5SMALL_XNLI_LORA,
    PERC_CHANGE_MT5BASE_XNLI_LORA,
    PERC_CHANGE_MT5BASE_XNLI_LORA,
)

AVG_CHANGES_MT5_SIB_LORA = calculate_average_percentage_change(
    PERC_CHANGE_MT5SMALL_SIB_LORA,
    PERC_CHANGE_MT5SMALL_SIB_LORA,
    PERC_CHANGE_MT5BASE_SIB_LORA,
    PERC_CHANGE_MT5BASE_SIB_LORA,
)
AVG_CHANGES_BLOOM_PAWSX_LORA = calculate_average_percentage_change(
    PERC_CHANGE_MT5SMALL_PAWSX_LORA,
    PERC_CHANGE_BLOOM560M_PAWSX_LORA,
    PERC_CHANGE_BLOOM1B1_PAWSX_LORA,
    PERC_CHANGE_BLOOM1B7_PAWSX_LORA,
)

AVG_CHANGES_BLOOM_XNLI_LORA = calculate_average_percentage_change(
    PERC_CHANGE_MT5SMALL_XNLI_LORA,
    PERC_CHANGE_BLOOM560M_XNLI_LORA,
    PERC_CHANGE_BLOOM1B1_XNLI_LORA,
    PERC_CHANGE_BLOOM1B7_XNLI_LORA,
)

AVG_CHANGES_BLOOM_SIB_LORA = calculate_average_percentage_change(
    PERC_CHANGE_MT5SMALL_SIB_LORA,
    PERC_CHANGE_BLOOM560M_SIB_LORA,
    PERC_CHANGE_BLOOM1B1_SIB_LORA,
    PERC_CHANGE_BLOOM1B7_SIB_LORA,
)

# Calculate average percentage changes for each dataset QLORA
AVG_CHANGES_MT5_PAWSX_QLORA = calculate_average_percentage_change(
    PERC_CHANGE_MT5SMALL_PAWSX_QLORA,
    PERC_CHANGE_MT5SMALL_PAWSX_QLORA,
    PERC_CHANGE_MT5LARGE_PAWSX_QLORA,
    PERC_CHANGE_MT5LARGE_PAWSX_QLORA,
)

AVG_CHANGES_MT5_XNLI_QLORA = calculate_average_percentage_change(
    PERC_CHANGE_MT5SMALL_XNLI_QLORA,
    PERC_CHANGE_MT5SMALL_XNLI_QLORA,
    PERC_CHANGE_MT5BASE_XNLI_QLORA,
    PERC_CHANGE_MT5BASE_XNLI_QLORA,
)

AVG_CHANGES_MT5_SIB_QLORA = calculate_average_percentage_change(
    PERC_CHANGE_MT5SMALL_SIB_QLORA,
    PERC_CHANGE_MT5SMALL_SIB_QLORA,
    PERC_CHANGE_MT5BASE_SIB_QLORA,
    PERC_CHANGE_MT5BASE_SIB_QLORA,
)
AVG_CHANGES_BLOOM_PAWSX_QLORA = calculate_average_percentage_change(
    PERC_CHANGE_MT5SMALL_PAWSX_QLORA,
    PERC_CHANGE_BLOOM560M_PAWSX_QLORA,
    PERC_CHANGE_BLOOM1B1_PAWSX_QLORA,
    PERC_CHANGE_BLOOM1B7_PAWSX_QLORA,
)

AVG_CHANGES_BLOOM_XNLI_QLORA = calculate_average_percentage_change(
    PERC_CHANGE_MT5SMALL_XNLI_QLORA,
    PERC_CHANGE_BLOOM560M_XNLI_QLORA,
    PERC_CHANGE_BLOOM1B1_XNLI_QLORA,
    PERC_CHANGE_BLOOM1B7_XNLI_QLORA,
)

AVG_CHANGES_BLOOM_SIB_QLORA = calculate_average_percentage_change(
    PERC_CHANGE_MT5SMALL_SIB_QLORA,
    PERC_CHANGE_BLOOM560M_SIB_QLORA,
    PERC_CHANGE_BLOOM1B1_SIB_QLORA,
    PERC_CHANGE_BLOOM1B7_SIB_QLORA,
)

########################## AVG PER DATASET ######################################
# Aggregate for all MT5 per datase - PAWS-X
mt5_pawsx_lora = [
    METRICS_MT5SMALL_PAWSX_LORA_r64,
    METRICS_MT5BASE_PAWSX_LORA_r64,
    METRICS_MT5LARGE_PAWSX_LORA_r64,
]
mt5_pawsx_qlora = [
    METRICS_MT5SMALL_PAWSX_QLORA_r64,
    METRICS_MT5BASE_PAWSX_QLORA_r64,
    METRICS_MT5LARGE_PAWSX_QLORA_r64,
]
mt5_pawsx_fullfinetune = [
    METRICS_MT5SMALL_PAWSX_FULLFINETUNE,
    METRICS_MT5BASE_PAWSX_FULLFINETUNE,
    METRICS_MT5LARGE_PAWSX_FULLFINETUNE,
]
MT5_PAWSX_LORA_AVG = calculate_average_metrics(mt5_pawsx_lora)
MT5_PAWSX_QLORA_AVG = calculate_average_metrics(mt5_pawsx_qlora)
MT5_PAWSX_FULLFINETUNE_AVG = calculate_average_metrics(mt5_pawsx_fullfinetune)

# Aggregate for all MT5 per dataset - XNLI
mt5_xnli_lora = [
    METRICS_MT5SMALL_XNLI_LORA_r64,
    METRICS_MT5BASE_XNLI_LORA_r64,
    METRICS_MT5LARGE_XNLI_LORA_r64,
]
mt5_xnli_qlora = [
    METRICS_MT5SMALL_XNLI_QLORA_r64,
    METRICS_MT5BASE_XNLI_QLORA_r64,
    METRICS_MT5LARGE_XNLI_QLORA_r64,
]
mt5_xnli_fullfinetune = [
    METRICS_MT5SMALL_XNLI_FULLFINETUNE,
    METRICS_MT5BASE_XNLI_FULLFINETUNE,
    METRICS_MT5LARGE_XNLI_FULLFINETUNE,
]
MT5_XNLI_LORA_AVG = calculate_average_metrics(mt5_xnli_lora)
MT5_XNLI_QLORA_AVG = calculate_average_metrics(mt5_xnli_qlora)
MT5_XNLI_FULLFINETUNE_AVG = calculate_average_metrics(mt5_xnli_fullfinetune)

# Aggregate for all MT5 per dataset - SIB
mt5_sib_lora = [
    METRICS_MT5SMALL_SIB_LORA_r64,
    METRICS_MT5BASE_SIB_LORA_r64,
    METRICS_MT5LARGE_SIB_LORA_r64,
]
mt5_sib_qlora = [
    METRICS_MT5SMALL_SIB_QLORA_r64,
    METRICS_MT5BASE_SIB_QLORA_r64,
    METRICS_MT5LARGE_SIB_QLORA_r64,
]
mt5_sib_fullfinetune = [
    METRICS_MT5SMALL_SIB_FULLFINETUNE,
    METRICS_MT5BASE_SIB_FULLFINETUNE,
    METRICS_MT5LARGE_SIB_FULLFINETUNE,
]
MT5_SIB_LORA_AVG = calculate_average_metrics(mt5_sib_lora)
MT5_SIB_QLORA_AVG = calculate_average_metrics(mt5_sib_qlora)
MT5_SIB_FULLFINETUNE_AVG = calculate_average_metrics(mt5_sib_fullfinetune)

# Aggregate for all BLOOM per dataset - PAWS-X
bloom_pawsx_lora = [
    METRICS_BLOOM560M_PAWSX_LORA_r64,
    METRICS_BLOOM1B1_PAWSX_LORA_r64,
    METRICS_BLOOM1B7_PAWSX_LORA_r64,
]
bloom_pawsx_qlora = [
    METRICS_BLOOM560M_PAWSX_QLORA_r64,
    METRICS_BLOOM1B1_PAWSX_QLORA_r64,
    METRICS_BLOOM1B7_PAWSX_QLORA_r64,
]
bloom_pawsx_fullfinetune = [
    METRICS_BLOOM560M_PAWSX_FULLFINETUNE,
    METRICS_BLOOM1B1_PAWSX_FULLFINETUNE,
    METRICS_BLOOM1B7_PAWSX_FULLFINETUNE,
]
BLOOM_PAWSX_LORA_AVG = calculate_average_metrics(bloom_pawsx_lora)
BLOOM_PAWSX_QLORA_AVG = calculate_average_metrics(bloom_pawsx_qlora)
BLOOM_PAWSX_FULLFINETUNE_AVG = calculate_average_metrics(bloom_pawsx_fullfinetune)

# Aggregate for all BLOOM per dataset - XNLI
bloom_xnli_lora = [
    METRICS_BLOOM560M_XNLI_LORA_r64,
    METRICS_BLOOM1B1_XNLI_LORA_r64,
    METRICS_BLOOM1B7_XNLI_LORA_r64,
]
bloom_xnli_qlora = [
    METRICS_BLOOM560M_XNLI_QLORA_r64,
    METRICS_BLOOM1B1_XNLI_QLORA_r64,
    METRICS_BLOOM1B7_XNLI_QLORA_r64,
]
bloom_xnli_fullfinetune = [
    METRICS_BLOOM560M_XNLI_FULLFINETUNE,
    METRICS_BLOOM1B1_XNLI_FULLFINETUNE,
    METRICS_BLOOM1B7_XNLI_FULLFINETUNE,
]
BLOOM_XNLI_LORA_AVG = calculate_average_metrics(bloom_xnli_lora)
BLOOM_XNLI_QLORA_AVG = calculate_average_metrics(bloom_xnli_qlora)
BLOOM_XNLI_FULLFINETUNE_AVG = calculate_average_metrics(bloom_xnli_fullfinetune)

# Aggregate for all BLOOM per dataset - SIB
bloom_sib_lora = [
    METRICS_BLOOM560M_SIB_LORA_r64,
    METRICS_BLOOM1B1_SIB_LORA_r64,
    METRICS_BLOOM1B7_SIB_LORA_r64,
]
bloom_sib_qlora = [
    METRICS_BLOOM560M_SIB_QLORA_r64,
    METRICS_BLOOM1B1_SIB_QLORA_r64,
    METRICS_BLOOM1B7_SIB_QLORA_r64,
]
bloom_sib_fullfinetune = [
    METRICS_BLOOM560M_SIB_FULLFINETUNE,
    METRICS_BLOOM1B1_SIB_FULLFINETUNE,
    METRICS_BLOOM1B7_SIB_FULLFINETUNE,
]
BLOOM_SIB_LORA_AVG = calculate_average_metrics(bloom_sib_lora)
BLOOM_SIB_QLORA_AVG = calculate_average_metrics(bloom_sib_qlora)
BLOOM_SIB_FULLFINETUNE_AVG = calculate_average_metrics(bloom_sib_fullfinetune)

# Prepare descriptions to format the results
descriptions = {
    "METRICS_MT5SMALL_AVG_LORA": "MT5-Small Average LORA Performance:",
    "METRICS_MT5BASE_AVG_LORA": "MT5-Base Average LORA Performance:",
    "METRICS_MT5LARGE_AVG_LORA": "MT5-Large Average LORA Performance:",

    "METRICS_MT5SMALL_AVG_QLORA": "MT5-Small Average QLORA Performance:",
    "METRICS_MT5BASE_AVG_QLORA": "MT5-Base Average QLORA Performance:",
    "METRICS_MT5LARGE_AVG_QLORA": "MT5-Large Average QLORA Performance:",

    "METRICS_MT5SMALL_AVG_FULLFINETUNE": "MT5-Small Average Full Finetune Performance:",
    "METRICS_MT5BASE_AVG_FULLFINETUNE": "MT5-Base Average Full Finetune Performance:",
    "METRICS_MT5LARGE_AVG_FULLFINETUNE": "MT5-Large Average Full Finetune Performance:",

    "METRICS_BLOOM560M_AVG_LORA": "BLOOM-560M Average LORA Performance:",
    "METRICS_BLOOM1B1_AVG_LORA": "BLOOM-1B1 Average LORA Performance:",
    "METRICS_BLOOM1B7_AVG_LORA": "BLOOM-1B7 Average LORA Performance:",

    "METRICS_BLOOM560M_AVG_QLORA": "BLOOM-560M Average QLORA Performance:",
    "METRICS_BLOOM1B1_AVG_QLORA": "BLOOM-1B1 Average QLORA Performance:",
    "METRICS_BLOOM1B7_AVG_QLORA": "BLOOM-1B7 Average QLORA Performance:",

    "METRICS_BLOOM560M_AVG_FULLFINETUNE": "BLOOM-560M Average Full Finetune Performance:",
    "METRICS_BLOOM1B1_AVG_FULLFINETUNE": "BLOOM-1B1 Average Full Finetune Performance:",
    "METRICS_BLOOM1B7_AVG_FULLFINETUNE": "BLOOM-1B7 Average Full Finetune Performance:",

    "METRICS_MT5_AVG_LORA": "MT5 Average LORA Performance:",
    "METRICS_MT5_AVG_QLORA": "MT5 Average QLORA Performance:",
    "METRICS_MT5_AVG_FULLFINETUNE": "MT5 Average Full Finetune Performance:",

    "METRICS_BLOOM_AVG_LORA": "BLOOM Average LORA Performance:",
    "METRICS_BLOOM_AVG_QLORA": "BLOOM Average QLORA Performance:",
    "METRICS_BLOOM_AVG_FULLFINETUNE": "BLOOM Average Full Finetune Performance:"
}

results = {
    "METRICS_MT5SMALL_AVG_LORA": METRICS_MT5SMALL_AVG_LORA,
    "METRICS_MT5BASE_AVG_LORA": METRICS_MT5BASE_AVG_LORA,
    "METRICS_MT5LARGE_AVG_LORA": METRICS_MT5LARGE_AVG_LORA,

    "METRICS_MT5SMALL_AVG_QLORA": METRICS_MT5SMALL_AVG_QLORA,
    "METRICS_MT5BASE_AVG_QLORA": METRICS_MT5BASE_AVG_QLORA,
    "METRICS_MT5LARGE_AVG_QLORA": METRICS_MT5LARGE_AVG_QLORA,

    "METRICS_MT5SMALL_AVG_FULLFINETUNE": METRICS_MT5SMALL_AVG_FULLFINETUNE,
    "METRICS_MT5BASE_AVG_FULLFINETUNE": METRICS_MT5BASE_AVG_FULLFINETUNE,
    "METRICS_MT5LARGE_AVG_FULLFINETUNE": METRICS_MT5LARGE_AVG_FULLFINETUNE,

    "METRICS_BLOOM560M_AVG_LORA": METRICS_BLOOM560M_AVG_LORA,
    "METRICS_BLOOM1B1_AVG_LORA": METRICS_BLOOM1B1_AVG_LORA,
    "METRICS_BLOOM1B7_AVG_LORA": METRICS_BLOOM1B7_AVG_LORA,

    "METRICS_BLOOM560M_AVG_QLORA": METRICS_BLOOM560M_AVG_QLORA,
    "METRICS_BLOOM1B1_AVG_QLORA": METRICS_BLOOM1B1_AVG_QLORA,
    "METRICS_BLOOM1B7_AVG_QLORA": METRICS_BLOOM1B7_AVG_QLORA,

    "METRICS_BLOOM560M_AVG_FULLFINETUNE": METRICS_BLOOM560M_AVG_FULLFINETUNE,
    "METRICS_BLOOM1B1_AVG_FULLFINETUNE": METRICS_BLOOM1B1_AVG_FULLFINETUNE,
    "METRICS_BLOOM1B7_AVG_FULLFINETUNE": METRICS_BLOOM1B7_AVG_FULLFINETUNE,

    "METRICS_MT5_AVG_LORA": METRICS_MT5_AVG_LORA,
    "METRICS_MT5_AVG_QLORA": METRICS_MT5_AVG_QLORA,
    "METRICS_MT5_AVG_FULLFINETUNE": METRICS_MT5_AVG_FULLFINETUNE,

    "METRICS_BLOOM_AVG_LORA": METRICS_BLOOM_AVG_LORA,
    "METRICS_BLOOM_AVG_QLORA": METRICS_BLOOM_AVG_QLORA,
    "METRICS_BLOOM_AVG_FULLFINETUNE": METRICS_BLOOM_AVG_FULLFINETUNE
}

with open("model_performance_metrics.txt", "w") as file:
    for key, metrics in results.items():
        file.write(format_metrics(metrics, descriptions[key]))
        file.write("\n")

if __name__ == "__main__":
    # print(f"METRICS_MT5SMALL_AVG_LORA: \n{METRICS_MT5SMALL_AVG_LORA}")
    # print(f"METRICS_MT5BASE_AVG_LORA: \n{METRICS_MT5BASE_AVG_LORA}")
    # print(f"METRICS_MT5LARGE_AVG_LORA: \n{METRICS_MT5LARGE_AVG_LORA}")
    # print(f"METRICS_MT5SMALL_AVG_QLORA: \n{METRICS_MT5SMALL_AVG_QLORA}")
    # print(f"METRICS_MT5BASE_AVG_QLORA: \n{METRICS_MT5BASE_AVG_QLORA}")
    # print(f"METRICS_MT5LARGE_AVG_QLORA: \n{METRICS_MT5LARGE_AVG_QLORA}")
    # print(f"METRICS_MT5SMALL_AVG_FULLFINETUNE: \n{METRICS_MT5SMALL_AVG_FULLFINETUNE}")
    # print(f"METRICS_MT5BASE_AVG_FULLFINETUNE: \n{METRICS_MT5BASE_AVG_FULLFINETUNE}")
    # print(f"METRICS_MT5LARGE_AVG_FULLFINETUNE: \n{METRICS_MT5LARGE_AVG_FULLFINETUNE}")
    # print(f"METRICS_BLOOM560M_AVG_LORA: \n{METRICS_BLOOM560M_AVG_LORA}")
    # print(f"METRICS_BLOOM1B1_AVG_LORA: \n{METRICS_BLOOM1B1_AVG_LORA}")
    # print(f"METRICS_BLOOM1B7_AVG_LORA: \n{METRICS_BLOOM1B7_AVG_LORA}")
    # print(f"METRICS_BLOOM560M_AVG_QLORA: \n{METRICS_BLOOM560M_AVG_QLORA}")
    # print(f"METRICS_BLOOM1B1_AVG_QLORA: \n{METRICS_BLOOM1B1_AVG_QLORA}")
    # print(f"METRICS_BLOOM1B7_AVG_QLORA: \n{METRICS_BLOOM1B7_AVG_QLORA}")
    # print(f"METRICS_BLOOM560M_AVG_FULLFINETUNE: \n{METRICS_BLOOM560M_AVG_FULLFINETUNE}")
    # print(f"METRICS_BLOOM1B1_AVG_FULLFINETUNE: \n{METRICS_BLOOM1B1_AVG_FULLFINETUNE}")
    # print(f"METRICS_BLOOM1B7_AVG_FULLFINETUNE: \n{METRICS_BLOOM1B7_AVG_FULLFINETUNE}")
    #
    # # Print average results (17 langs, XNLI + PAWS-X)
    # print("MT5 Average LORA Performance:\n", METRICS_MT5_AVG_LORA)
    # print("MT5 Average QLORA Performance:\n", METRICS_MT5_AVG_QLORA)
    # print("MT5 Average Full Finetune Performance:\n", METRICS_MT5_AVG_FULLFINETUNE)
    #
    # print("BLOOM Average LORA Performance:\n", METRICS_BLOOM_AVG_LORA)
    # print("BLOOM Average QLORA Performance:\n", METRICS_BLOOM_AVG_QLORA)
    # print("BLOOM Average Full Finetune Performance:\n", METRICS_BLOOM_AVG_FULLFINETUNE)
    #
    # print("PERC_CHANGE_MT5SMALL_LORA:", PERC_CHANGE_MT5SMALL_LORA)
    # print("PERC_CHANGE_MT5SMALL_QLORA:", PERC_CHANGE_MT5SMALL_QLORA)
    #
    # print("PERC_CHANGE_MT5BASE_LORA:", PERC_CHANGE_MT5BASE_LORA)
    # print("PERC_CHANGE_MT5BASE_QLORA:", PERC_CHANGE_MT5BASE_QLORA)
    #
    # print("PERC_CHANGE_MT5LARGE_LORA:", PERC_CHANGE_MT5LARGE_LORA)
    # print("PERC_CHANGE_MT5LARGE_QLORA:", PERC_CHANGE_MT5LARGE_QLORA)
    #
    # print("PERC_CHANGE_BLOOM560M_LORA:", PERC_CHANGE_BLOOM560M_LORA)
    # print("PERC_CHANGE_BLOOM560M_QLORA:", PERC_CHANGE_BLOOM560M_QLORA)
    #
    # print("PERC_CHANGE_BLOOM1B1_LORA:", PERC_CHANGE_BLOOM1B1_LORA)
    # print("PERC_CHANGE_BLOOM1B1_QLORA:", PERC_CHANGE_BLOOM1B1_QLORA)
    #
    # print("PERC_CHANGE_BLOOM1B7_LORA:", PERC_CHANGE_BLOOM1B7_LORA)
    # print("PERC_CHANGE_BLOOM1B7_QLORA:", PERC_CHANGE_BLOOM1B7_QLORA)
    #
    # print("PERC_CHANGE_MT5SMALL_PAWSX_LORA:", PERC_CHANGE_MT5SMALL_PAWSX_LORA)
    # print("PERC_CHANGE_MT5SMALL_PAWSX_QLORA:", PERC_CHANGE_MT5SMALL_PAWSX_QLORA)
    #
    # print("PERC_CHANGE_MT5BASE_PAWSX_LORA:", PERC_CHANGE_MT5BASE_PAWSX_LORA)
    # print("PERC_CHANGE_MT5BASE_PAWSX_QLORA:", PERC_CHANGE_MT5BASE_PAWSX_QLORA)
    #
    # print("PERC_CHANGE_MT5LARGE_PAWSX_LORA:", PERC_CHANGE_MT5LARGE_PAWSX_LORA)
    # print("PERC_CHANGE_MT5LARGE_PAWSX_QLORA:", PERC_CHANGE_MT5LARGE_PAWSX_QLORA)
    #
    # print("PERC_CHANGE_MT5SMALL_XNLI_LORA:", PERC_CHANGE_MT5SMALL_XNLI_LORA)
    # print("PERC_CHANGE_MT5SMALL_XNLI_QLORA:", PERC_CHANGE_MT5SMALL_XNLI_QLORA)
    #
    # print("PERC_CHANGE_MT5BASE_XNLI_LORA:", PERC_CHANGE_MT5SMALL_XNLI_LORA)
    # print("PERC_CHANGE_MT5BASE_XNLI_QLORA:", PERC_CHANGE_MT5BASE_XNLI_QLORA)
    #
    # print("PERC_CHANGE_MT5LARGE_XNLI_LORA:", PERC_CHANGE_MT5LARGE_XNLI_LORA)
    # print("PERC_CHANGE_MT5LARGE_XNLI_QLORA:", PERC_CHANGE_MT5LARGE_XNLI_QLORA)
    #
    # print("PERC_CHANGE_MT5SMALL_SIB_LORA:", PERC_CHANGE_MT5SMALL_SIB_LORA)
    # print("PERC_CHANGE_MT5SMALL_SIB_QLORA:", PERC_CHANGE_MT5SMALL_SIB_QLORA)
    #
    # print("PERC_CHANGE_MT5BASE_SIB_LORA:", PERC_CHANGE_MT5BASE_SIB_LORA)
    # print("PERC_CHANGE_MT5BASE_SIB_QLORA:", PERC_CHANGE_MT5BASE_SIB_QLORA)
    #
    # print("PERC_CHANGE_MT5LARGE_SIB_LORA:", PERC_CHANGE_MT5LARGE_SIB_LORA)
    # print("PERC_CHANGE_MT5LARGE_SIB_QLORA:", PERC_CHANGE_MT5LARGE_SIB_QLORA)
    #
    # print("PERC_CHANGE_BLOOM560M_PAWSX_LORA:", PERC_CHANGE_BLOOM560M_PAWSX_LORA)
    # print("PERC_CHANGE_BLOOM560M_PAWSX_QLORA:", PERC_CHANGE_BLOOM560M_PAWSX_QLORA)
    #
    # print("PERC_CHANGE_BLOOM1B1_PAWSX_LORA:", PERC_CHANGE_BLOOM1B1_PAWSX_LORA)
    # print("PERC_CHANGE_BLOOM1B1_PAWSX_QLORA:", PERC_CHANGE_BLOOM1B1_PAWSX_QLORA)
    #
    # print("PERC_CHANGE_BLOOM1B7_PAWSX_LORA:", PERC_CHANGE_BLOOM1B7_PAWSX_LORA)
    # print("PERC_CHANGE_BLOOM1B7_PAWSX_QLORA:", PERC_CHANGE_BLOOM1B7_PAWSX_QLORA)
    #
    # print("PERC_CHANGE_BLOOM560M_XNLI_LORA:", PERC_CHANGE_BLOOM560M_XNLI_LORA)
    # print("PERC_CHANGE_BLOOM560M_XNLI_QLORA:", PERC_CHANGE_BLOOM560M_XNLI_QLORA)
    #
    # print("PERC_CHANGE_BLOOM1B1_XNLI_LORA:", PERC_CHANGE_BLOOM1B1_XNLI_LORA)
    # print("PERC_CHANGE_BLOOM1B1_XNLI_QLORA:", PERC_CHANGE_BLOOM1B1_XNLI_QLORA)
    #
    # print("PERC_CHANGE_BLOOM1B7_XNLI_LORA:", PERC_CHANGE_BLOOM1B7_XNLI_LORA)
    # print("PERC_CHANGE_BLOOM1B7_XNLI_QLORA:", PERC_CHANGE_BLOOM1B7_XNLI_QLORA)
    #
    # print("PERC_CHANGE_BLOOM560M_SIB_LORA:", PERC_CHANGE_BLOOM560M_SIB_LORA)
    # print("PERC_CHANGE_BLOOM560M_SIB_QLORA:", PERC_CHANGE_BLOOM560M_SIB_QLORA)
    #
    # print("PERC_CHANGE_BLOOM1B1_SIB_LORA:", PERC_CHANGE_BLOOM1B1_SIB_LORA)
    # print("PERC_CHANGE_BLOOM1B1_SIB_QLORA:", PERC_CHANGE_BLOOM1B1_SIB_QLORA)
    #
    # print("PERC_CHANGE_BLOOM1B7_SIB_LORA:", PERC_CHANGE_BLOOM1B7_SIB_LORA)
    # print("PERC_CHANGE_BLOOM1B7_SIB_QLORA:", PERC_CHANGE_BLOOM1B7_SIB_QLORA)

    print(PERC_CHANGE_MT5SMALL_LORA)
    print(f"all_bloom1b1_qlora_metrics: {all_bloom1b1_qlora_metrics}")
    qlora = METRICS_BLOOM1B1_AVG_QLORA['accuracy']
    print(f"METRICS_BLOOM1B1_AVG_QLORA: {qlora}")
    fullfinetune = METRICS_BLOOM1B1_AVG_FULLFINETUNE['accuracy']
    print(f"METRICS_BLOOM1B1_AVG_FULLFINETUNE:{fullfinetune}")


    difference = [a - b for a, b in zip(qlora, fullfinetune)]
    print(f"Difference: {np.mean(difference)}")

    print("PERC_CHANGE_BLOOM1B1_QLORA:", PERC_CHANGE_BLOOM1B1_QLORA)
    print(np.mean(PERC_CHANGE_BLOOM1B1_QLORA['accuracy']))

    print("PERC_CHANGE_BLOOM1B1_LORA:", PERC_CHANGE_BLOOM1B1_LORA)

    print(f"BLOOM_SIB_LORA_AVG: {BLOOM_SIB_LORA_AVG}")

    for dict in mt5_pawsx_lora:
        print(dict['accuracy'])

    print(MT5_PAWSX_LORA_AVG)

    print(MT5_PAWSX_FULLFINETUNE_AVG)


