import json
import logging
from typing import Dict, List, Union

from analysis.config import PAWSX_LANGS, XNLI_LANGS, SIB_LANG_CODE_MAP
def save_to_json(accuracies: Dict[str, float], filename: str):
    """Saves the fertility dictionary to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(accuracies, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved accuracies to {filename}")


def load_from_json(filename: str) -> Dict[str, float]:
    """Loads a dictionary from a JSON file.
    """
    with open(filename, 'r') as f:
        dictionary = json.load(f)
        logging.info(f"Loaded  {dictionary}")
    return dictionary

def filter_dict_by_languages(input_dict: Union[Dict[str, float], Dict[str, List[Union[str, float]]]],
                             languages: List[str]) -> Dict[str, Union[float, List[Union[str, float]]]]:
    """
    Filters a dictionary by a list of languages.

    Args:
    - input_dict: Input dict, can be either a dictionary where keys are language codes and values are floats,
                  or a dictionary where 'language' is a list of language codes and 'accuracy' is a list of floats.
    - languages: list of languages to filter by.

    Returns:
    - Dict: filtered dictionary containing only the specified languages.
    """
    if isinstance(next(iter(input_dict.values())), list):
        # Case for input_dict with lists
        filtered_languages = []
        filtered_accuracies = []

        for lang, acc in zip(input_dict['language'], input_dict['accuracy']):
            if lang in languages:
                filtered_languages.append(lang)
                filtered_accuracies.append(acc)

        return {'language': filtered_languages, 'accuracy': filtered_accuracies}
    else:
        # case for standard dictionary
        return {lang: input_dict[lang] for lang in languages if lang in input_dict}

def filter_sib_metrics_dict(metrics_dict, lang_code_map, dataset_langs_to_keep=None):
    """Filters out SIB200 results to only include the languages present in PAWS-X and XNLI dataset"""
    # extract language list and accuracy list from the metrics dictionary
    languages = metrics_dict['language']
    accuracies = metrics_dict['accuracy']

    filtered_metrics = {'language': [], 'accuracy': []} # dict to store the filtered and renamed metrics

    # go over all 205 SIB languages and their corresponding accuracies
    for lang, acc in zip(languages, accuracies):
        # Checking if the language is in the SIB_LANG_CODE_MAP
        if lang in lang_code_map: # where lang_code_map contains all languages of XNLI+PAWSX
            # append the 2-letter language code and the corresponding accuracy to the new dictionary
            filtered_metrics['language'].append(lang_code_map[lang])
            filtered_metrics['accuracy'].append(acc)

    # If we are only interested in keeping XNLI langs, then
    if dataset_langs_to_keep == 'xnli':
        # exclude: ja and ko (PAWSX langs not in XNLI)
        filtered_metrics = filter_dict_by_languages(filtered_metrics, XNLI_LANGS)
    # If we are only interested in PAWSX keeping langs, then
    elif dataset_langs_to_keep == 'pawsx':
        # exclude: 'ar', 'bg', 'el', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi' (XNLI langs not in PAWSX)
        filtered_metrics = filter_dict_by_languages(filtered_metrics, PAWSX_LANGS)

    return filtered_metrics

def calculate_average_fertilities(all_fertilities: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Create a dictionary with unique languages as keys and their average fertility values as values.
    """
    combined_fertilities = {}
    count_fertilities = {}

    for fertility in all_fertilities:
        for lang, value in fertility.items():
            if lang in combined_fertilities:
                combined_fertilities[lang] += value
                count_fertilities[lang] += 1
            else:
                combined_fertilities[lang] = value
                count_fertilities[lang] = 1

    # calculate average and round to 4 d.p.
    average_fertilities = {}
    for lang, total_sum in combined_fertilities.items():
        average_fertilities[lang] = round(total_sum / count_fertilities[lang], 4)

    return average_fertilities

def average_accuracies(metrics_small, metrics_medium, metrics_large):
    """
    Create a dictionary with unique languages as keys and their average accuracies values as values.
    """
    from collections import defaultdict

    # this dict stores the cumulative accuracies and counts
    cumulative_accuracies = defaultdict(lambda: {'sum': 0, 'count': 0})

    dicts = [metrics_small, metrics_medium, metrics_large]

    for d in dicts:
        for lang, acc in zip(d['language'], d['accuracy']):
            cumulative_accuracies[lang]['sum'] += acc
            cumulative_accuracies[lang]['count'] += 1

    # calculate the avg. accuracies
    average_accuracies = {'language': [], 'accuracy': []}
    for lang, data in cumulative_accuracies.items():
        average_accuracies['language'].append(lang)
        average_accuracies['accuracy'].append(data['sum'] / data['count'])

    return average_accuracies



if __name__ == "__main__":
    # example use case
    from analysis.config import METRICS_BLOOM560M_SIB_FULLFINETUNE

    print(f"Old dict: \n{METRICS_BLOOM560M_SIB_FULLFINETUNE}")

    # filtering and renaming the SIB metrics dictionary
    filtered_metrics = filter_sib_metrics_dict(METRICS_BLOOM560M_SIB_FULLFINETUNE, SIB_LANG_CODE_MAP, dataset_langs_to_keep='xnli')
    print(f"Filtered SIB metrics dict: \n{filtered_metrics}")
    print(f"Num of languages: {len(filtered_metrics['language'])}")


