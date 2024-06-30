from statistics import mean

from config import (
    SIB_LANGS,
    METRICS_MT5SMALL_SIB_LORA_r16, METRICS_MT5SMALL_SIB_LORA_r32, METRICS_MT5SMALL_SIB_LORA_r64, METRICS_MT5SMALL_SIB_LORA_r128,
    METRICS_MT5SMALL_SIB_QLORA_r16, METRICS_MT5SMALL_SIB_QLORA_r32, METRICS_MT5SMALL_SIB_QLORA_r64, METRICS_MT5SMALL_SIB_QLORA_r128,

    METRICS_BLOOM560M_SIB_LORA_r16, METRICS_BLOOM560M_SIB_LORA_r32, METRICS_BLOOM560M_SIB_LORA_r64, METRICS_BLOOM560M_SIB_LORA_r128,
    METRICS_BLOOM560M_SIB_QLORA_r16, METRICS_BLOOM560M_SIB_QLORA_r32, METRICS_BLOOM560M_SIB_QLORA_r64, METRICS_BLOOM560M_SIB_QLORA_r128,

    METRICS_MT5BASE_PAWSX_QLORA_r16, METRICS_MT5BASE_PAWSX_QLORA_r32, METRICS_MT5BASE_PAWSX_QLORA_r64, METRICS_MT5BASE_PAWSX_QLORA_r128,
    METRICS_MT5BASE_XNLI_QLORA_r16, METRICS_MT5BASE_XNLI_QLORA_r32, METRICS_MT5BASE_XNLI_QLORA_r64, METRICS_MT5BASE_XNLI_QLORA_r128,
    METRICS_MT5BASE_SIB_QLORA_r16, METRICS_MT5BASE_SIB_QLORA_r32, METRICS_MT5BASE_SIB_QLORA_r64, METRICS_MT5BASE_SIB_QLORA_r128,

    METRICS_BLOOM1B1_PAWSX_QLORA_r16, METRICS_BLOOM1B1_PAWSX_QLORA_r32, METRICS_BLOOM1B1_PAWSX_QLORA_r64, METRICS_BLOOM1B1_PAWSX_QLORA_r128,
    METRICS_BLOOM1B1_XNLI_QLORA_r16, METRICS_BLOOM1B1_XNLI_QLORA_r32, METRICS_BLOOM1B1_XNLI_QLORA_r64, METRICS_BLOOM1B1_XNLI_QLORA_r128,
    METRICS_BLOOM1B1_SIB_QLORA_r16, METRICS_BLOOM1B1_SIB_QLORA_r32, METRICS_BLOOM1B1_SIB_QLORA_r64, METRICS_BLOOM1B1_SIB_QLORA_r128,

    METRICS_MT5LARGE_PAWSX_QLORA_r64, METRICS_MT5LARGE_XNLI_QLORA_r64, METRICS_MT5LARGE_SIB_QLORA_r64,
    METRICS_BLOOM1B7_PAWSX_QLORA_r64, METRICS_BLOOM1B7_XNLI_QLORA_r64, METRICS_BLOOM1B7_SIB_QLORA_r64,

    METRICS_MT5SMALL_PAWSX_FULLFINETUNE, METRICS_MT5SMALL_XNLI_FULLFINETUNE, METRICS_MT5SMALL_SIB_FULLFINETUNE,
    METRICS_MT5BASE_PAWSX_FULLFINETUNE, METRICS_MT5BASE_XNLI_FULLFINETUNE, METRICS_MT5BASE_SIB_FULLFINETUNE,
    METRICS_MT5LARGE_PAWSX_FULLFINETUNE, METRICS_MT5LARGE_XNLI_FULLFINETUNE, METRICS_MT5LARGE_SIB_FULLFINETUNE,
    METRICS_BLOOM560M_PAWSX_FULLFINETUNE, METRICS_BLOOM560M_XNLI_FULLFINETUNE, METRICS_BLOOM560M_SIB_FULLFINETUNE,
    METRICS_BLOOM1B1_PAWSX_FULLFINETUNE, METRICS_BLOOM1B1_XNLI_FULLFINETUNE, METRICS_BLOOM1B1_SIB_FULLFINETUNE,
    METRICS_BLOOM1B7_PAWSX_FULLFINETUNE, METRICS_BLOOM1B7_XNLI_FULLFINETUNE, METRICS_BLOOM1B7_SIB_FULLFINETUNE,
    SIB_LANG_FAMILY_NAMES,
)

from avg_performance_by_language import create_qlora_percentage_change_dict
from utils import average_accuracies

# Percentage change of QLORA accuracies compared to FULLFINETUNE
FAMILY_PERC_CHANGE_SIB_MT5SMALL = create_qlora_percentage_change_dict(METRICS_MT5SMALL_SIB_QLORA_r64,
                                                            METRICS_MT5SMALL_SIB_FULLFINETUNE)
FAMILY_PERC_CHANGE_SIB_MT5BASE = create_qlora_percentage_change_dict(METRICS_MT5BASE_SIB_QLORA_r64,
                                                            METRICS_MT5BASE_SIB_FULLFINETUNE)
FAMILY_PERC_CHANGE_SIB_MT5LARGE = create_qlora_percentage_change_dict(METRICS_MT5LARGE_SIB_QLORA_r64,
                                                            METRICS_MT5LARGE_SIB_FULLFINETUNE)
FAMILY_PERC_CHANGE_SIB_BLOOM560M = create_qlora_percentage_change_dict(METRICS_BLOOM560M_SIB_QLORA_r64,
                                                            METRICS_BLOOM560M_SIB_FULLFINETUNE)
FAMILY_PERC_CHANGE_SIB_BLOOM1B1 = create_qlora_percentage_change_dict(METRICS_BLOOM1B1_SIB_QLORA_r64,
                                                            METRICS_BLOOM1B1_SIB_FULLFINETUNE)
FAMILY_PERC_CHANGE_SIB_BLOOM1B7 = create_qlora_percentage_change_dict(METRICS_BLOOM1B7_SIB_QLORA_r64,
                                                            METRICS_BLOOM1B7_SIB_FULLFINETUNE)
FAMILY_AVG_PERC_CHANGE_SIB_MT5 = average_accuracies(FAMILY_PERC_CHANGE_SIB_MT5SMALL,FAMILY_PERC_CHANGE_SIB_MT5BASE,FAMILY_PERC_CHANGE_SIB_MT5LARGE)
FAMILY_AVG_PERC_CHANGE_SIB_BLOOM = average_accuracies(FAMILY_PERC_CHANGE_SIB_BLOOM560M,FAMILY_PERC_CHANGE_SIB_BLOOM1B1,FAMILY_PERC_CHANGE_SIB_BLOOM1B7)



# Function to get the metrics dictionary based on model and finetuning type
def get_metrics_dict(model):
    key = f'FAMILY_PERC_CHANGE_SIB_{model}'
    return globals().get(key)

# Function to calculate the average accuracy per language family
def calculate_average_accuracy(metrics_dict):
    average_accuracies = {}
    family_accuracies = {}

    for lang_code, lang_info in SIB_LANG_FAMILY_NAMES.items():
        family = lang_info['family']
        if lang_code in metrics_dict['language']:
            index = metrics_dict['language'].index(lang_code)
            if index < len(metrics_dict['accuracy']):
                accuracy = metrics_dict['accuracy'][index]
                if family not in family_accuracies:
                    family_accuracies[family] = []
                family_accuracies[family].append(accuracy)
            else:
                print(f"Index {index} out of range for accuracy list for language {lang_code}")
        else:
            print(f"Language {lang_code} not found in metrics_dict['language']")

    for family, accuracies in family_accuracies.items():
        if accuracies:
            average_accuracies[family] = mean(accuracies)
        else:
            average_accuracies[family] = None  # default value

    return average_accuracies


# Function to get the METRICS_SIB_BY_CATEGORY dictionary
def get_metrics_by_category(model):
    metrics_dict = get_metrics_dict(model)
    average_accuracies = calculate_average_accuracy(metrics_dict)

    return {
        'language': list(average_accuracies.keys()),
        'accuracy': list(average_accuracies.values())
    }

if __name__ == "__main__":

    from config import SIB_LANG_FAMILY_NAMES
    from accuracy_by_rank import plot_average_accuracies_per_family

    # Calculate average accuracies per family for each model
    family_perc_change_bloom560m = get_metrics_by_category("BLOOM560M")
    family_perc_change_bloom1b1 = get_metrics_by_category("BLOOM1B1")
    family_perc_change_bloom1b7 = get_metrics_by_category("BLOOM1B7")

    # Prepare the data for plotting
    family_accuracies = [
        family_perc_change_bloom560m,
        family_perc_change_bloom1b1,
        family_perc_change_bloom1b7
    ]
    family_accuracies = [FAMILY_AVG_PERC_CHANGE_SIB_MT5, FAMILY_AVG_PERC_CHANGE_SIB_BLOOM]
    model_type = 'MT5 and BLOOM'
    finetuning_type = "QLORA"
    labels = ["MT5", "BLOOM"]
    family_names = list(set(entry['family'] for entry in SIB_LANG_FAMILY_NAMES.values()))

    # Plot the results
    plot_average_accuracies_per_family('plots/test_bloom_qlora_average_accuracies_per_family_comparison.png',
                                       model_type,
                                       finetuning_type,
                                       labels,
                                       family_names,
                                       *family_accuracies)

    # Count total number of languages
    total_languages = sum(len(langs) for langs in SIB_LANG_CATEGORIES.values())

    # Print total number of languages
    print(f"Total number of languages: {total_languages}")

    # Print categories and languages per category
    for category, languages in SIB_LANG_CATEGORIES.items():
        print(f"\nCategory: {category}")
        print(f"Number of languages: {len(languages)}")
        print("Languages:", ", ".join(languages))

    model = "BLOOM1B1"
    finetuning_type = "QLORA"
    rank = 128  # can be None if not applicable
    metrics_by_family_qlora = get_metrics_by_category(model, finetuning_type, rank)
    print(metrics_by_family_qlora)

    model = "MT5SMALL"
    finetuning_type = "FULLFINETUNE"
    rank = None  # full finetutning
    metrics_by_family_fullfinetuning = get_metrics_by_category(model, finetuning_type, rank)
    print(metrics_by_family_fullfinetuning)