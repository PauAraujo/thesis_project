import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import all PEFT metrics r=64 and Fullfinetuning metrics
# Import aggregate measures
from analysis.avg_performance_by_language import (
    # Aggregate performance on all datasets for all MT5 and BLOOM model sizes
    METRICS_MT5_AVG_LORA, METRICS_MT5_AVG_QLORA, METRICS_MT5_AVG_FULLFINETUNE,
    METRICS_BLOOM_AVG_LORA,METRICS_BLOOM_AVG_QLORA, METRICS_BLOOM_AVG_FULLFINETUNE ,
    # Percentage change per model compared to full finetuning baseline
    PERC_CHANGE_MT5SMALL_LORA, PERC_CHANGE_MT5SMALL_QLORA,
    PERC_CHANGE_MT5BASE_LORA, PERC_CHANGE_MT5BASE_QLORA,
    PERC_CHANGE_MT5LARGE_LORA, PERC_CHANGE_MT5LARGE_QLORA,
    PERC_CHANGE_BLOOM560M_LORA, PERC_CHANGE_BLOOM560M_QLORA,
    PERC_CHANGE_BLOOM1B1_LORA, PERC_CHANGE_BLOOM1B1_QLORA,
    PERC_CHANGE_BLOOM1B7_LORA, PERC_CHANGE_BLOOM1B7_QLORA,

)

from analysis.avg_performance_by_language import (
    METRICS_MT5SMALL_AVG_LORA, METRICS_MT5SMALL_AVG_QLORA,
    METRICS_MT5BASE_AVG_LORA, METRICS_MT5BASE_AVG_QLORA,
    METRICS_MT5LARGE_AVG_LORA, METRICS_MT5LARGE_AVG_QLORA,
    METRICS_BLOOM560M_AVG_LORA, METRICS_BLOOM560M_AVG_QLORA,
    METRICS_BLOOM1B1_AVG_QLORA,
    METRICS_BLOOM1B7_AVG_LORA, METRICS_BLOOM1B7_AVG_QLORA,
    METRICS_MT5SMALL_AVG_FULLFINETUNE, METRICS_MT5BASE_AVG_FULLFINETUNE, METRICS_MT5LARGE_AVG_FULLFINETUNE,
    METRICS_BLOOM560M_AVG_FULLFINETUNE, METRICS_BLOOM1B1_AVG_FULLFINETUNE, METRICS_BLOOM1B7_AVG_FULLFINETUNE
)

def plot_aggregate_by_language(output_file=None):
    # define data for plotting
    labels = METRICS_MT5_AVG_LORA['language']
    mt5_lora = METRICS_MT5_AVG_LORA['accuracy']
    mt5_qlora = METRICS_MT5_AVG_QLORA['accuracy']
    mt5_fullfinetune = METRICS_MT5_AVG_FULLFINETUNE['accuracy']
    bloom_lora = METRICS_BLOOM_AVG_LORA['accuracy']
    bloom_qlora = METRICS_BLOOM_AVG_QLORA['accuracy']
    bloom_fullfinetune = METRICS_BLOOM_AVG_FULLFINETUNE['accuracy']

    x = np.arange(len(labels))  # locations label
    width = 0.15  # width of the bars

    fig, ax = plt.subplots(figsize=(15, 10))

    rects1 = ax.bar(x - 2 * width, mt5_lora, width, label='MT5 LORA', color='#E9C46A')
    rects2 = ax.bar(x - width, mt5_qlora, width, label='MT5 QLORA', color='#F4A261')
    rects3 = ax.bar(x, mt5_fullfinetune, width, label='MT5 Full Finetune', color='#E76F51')
    rects4 = ax.bar(x + width, bloom_lora, width, label='BLOOM LORA', color='#EFC3E6')
    rects5 = ax.bar(x + 2 * width, bloom_qlora, width, label='BLOOM QLORA', color='#F0A6CA')
    rects6 = ax.bar(x + 3 * width, bloom_fullfinetune, width, label='BLOOM Full Finetune', color='#9C89B8')

    # labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Languages')
    ax.set_ylabel('Aggregate Accuracy')
    ax.set_title('Aggregate Performance Metrics for MT5 and BLOOM Models per Language')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()

def plot_aggregate_mt5(output_file=None):
    labels = METRICS_MT5_AVG_LORA['language']
    mt5_lora = METRICS_MT5_AVG_LORA['accuracy']
    mt5_qlora = METRICS_MT5_AVG_QLORA['accuracy']
    mt5_fullfinetune = METRICS_MT5_AVG_FULLFINETUNE['accuracy']

    x = np.arange(len(labels))  # label locations
    width = 0.25  # bar width
    fig, ax = plt.subplots(figsize=(15, 10))

    rects1 = ax.bar(x - width, mt5_lora, width, label='MT5 LORA', color='#E9C46A')
    rects2 = ax.bar(x, mt5_qlora, width, label='MT5 QLORA', color='#F4A261')
    rects3 = ax.bar(x + width, mt5_fullfinetune, width, label='MT5 Full Finetune', color='#E76F51')

    # labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Languages')
    ax.set_ylabel('Aggregate Accuracy')
    ax.set_title('Aggregate Performance Metrics for MT5 Models per Language')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()

def plot_aggregate_bloom(output_file=None):
    # define data for BLOOM plotting
    labels = METRICS_BLOOM_AVG_LORA['language']
    bloom_lora = METRICS_BLOOM_AVG_LORA['accuracy']
    bloom_qlora = METRICS_BLOOM_AVG_QLORA['accuracy']
    bloom_fullfinetune = METRICS_BLOOM_AVG_FULLFINETUNE['accuracy']

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15, 10))

    rects1 = ax.bar(x - width, bloom_lora, width, label='BLOOM LORA', color='#EFC3E6')
    rects2 = ax.bar(x, bloom_qlora, width, label='BLOOM QLORA', color='#F0A6CA')
    rects3 = ax.bar(x + width, bloom_fullfinetune, width, label='BLOOM Full Finetune', color='#9C89B8')

    # labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Languages')
    ax.set_ylabel('Aggregate Accuracy')
    ax.set_title('Aggregate Performance Metrics for BLOOM Models per Language')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()


def plot_accuracy_change_per_lang(lora_data, qlora_data, model_name, output_file=None):
    languages = lora_data['language']
    lora_accuracy = lora_data['accuracy']
    qlora_accuracy = qlora_data['accuracy']
    x = np.arange(len(languages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width / 2, lora_accuracy, width,
                   label=f'{model_name} LoRA Fine-tuning',
                   edgecolor='#6C8E5A', fill=False, hatch='//', linewidth=1.5)
    bars2 = ax.bar(x + width / 2, qlora_accuracy, width,
                   label=f'{model_name} QLoRA Fine-tuning',
                   color='#6C8E5A')

    ax.set_ylabel('Accuracy Percentage Change', fontsize=11)
    #ax.set_title(f'Percentage Change in Accuracy for {model_name} compared to Full Finetuning Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(languages, rotation=45, fontsize=11)
    ax.grid(True)
    ax.legend(fontsize=14)

    # Highlight negative bars in red
    for bar in bars1:
        if bar.get_height() < 0:
            bar.set_edgecolor('#D58367')
            bar.set_hatch('xx')
    for bar in bars2:
        if bar.get_height() < 0:
            bar.set_color('#D58367')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()


def calculate_avg_differences(metrics_peft, metrics_fullfinetune):
    peft = metrics_peft['accuracy']
    fullfinetune = metrics_fullfinetune['accuracy']
    metrics_diff = [a - b for a, b in zip(peft, fullfinetune)]
    return round(np.mean(metrics_diff),4)

def calculate_accuracy_change_by(by = 'size'):
    if by == 'size':
        percentage_changes = {
            "MT5-Small + LORA": calculate_avg_differences(METRICS_MT5SMALL_AVG_LORA, METRICS_MT5SMALL_AVG_FULLFINETUNE),
            "MT5-Small + QLORA": calculate_avg_differences(METRICS_MT5SMALL_AVG_QLORA, METRICS_MT5SMALL_AVG_FULLFINETUNE),
            "MT5-Base + LORA": calculate_avg_differences(METRICS_MT5BASE_AVG_LORA, METRICS_MT5BASE_AVG_FULLFINETUNE),
            "MT5-Base + QLORA": calculate_avg_differences(METRICS_MT5BASE_AVG_QLORA,METRICS_MT5BASE_AVG_FULLFINETUNE) ,
            "MT5-Large + LORA": calculate_avg_differences(METRICS_MT5LARGE_AVG_LORA, METRICS_MT5LARGE_AVG_FULLFINETUNE),
            "MT5-Large + QLORA": calculate_avg_differences(METRICS_MT5LARGE_AVG_QLORA, METRICS_MT5LARGE_AVG_FULLFINETUNE),
            "BLOOM-560M + LORA": calculate_avg_differences(METRICS_BLOOM560M_AVG_LORA, METRICS_BLOOM560M_AVG_FULLFINETUNE),
            "BLOOM-560M + QLORA": calculate_avg_differences(METRICS_BLOOM560M_AVG_QLORA, METRICS_BLOOM560M_AVG_FULLFINETUNE),
            "BLOOM-1B1 + LORA": calculate_avg_differences(METRICS_BLOOM1B1_AVG_QLORA, METRICS_BLOOM1B1_AVG_FULLFINETUNE),
            "BLOOM1B1 + QLORA": calculate_avg_differences(METRICS_BLOOM1B1_AVG_QLORA, METRICS_BLOOM1B1_AVG_FULLFINETUNE),
            "BLOOM-1B7 + LORA": calculate_avg_differences(METRICS_BLOOM1B7_AVG_LORA, METRICS_BLOOM1B7_AVG_FULLFINETUNE),
            "BLOOM-1B7 + QLORA": calculate_avg_differences(METRICS_BLOOM1B7_AVG_QLORA, METRICS_BLOOM1B7_AVG_FULLFINETUNE),
        }
    elif by == 'dataset':
        percentage_changes = {
            "MT5 + PAWS-X + LORA": 10.50,
            "MT5 + PAWS-X + QLORA": 11.14,
            "MT5 + XNLI + LORA": 18.66,
            "MT5 + XNLI + QLORA": 22.36,
            "MT5 + SIB200 + LORA": 60.05,
            "MT5 + SIB200 + QLORA": 61.28,
            "BLOOM + PAWS-X + LORA": 12.22,
            "BLOOM + PAWS-X + QLORA": 10.28,
            "BLOOM + XNLI + LORA": 2.48,
            "BLOOM + XNLI + QLORA": 0.51,
            "BLOOM + SIB200 + LORA": 11.23,
            "BLOOM + SIB200 + QLORA": 14.61,
        }

    return percentage_changes

def plot_accuracy_change_by_model_size(percentage_changes, output_file=None):
    labels = list(percentage_changes.keys())
    values = list(percentage_changes.values())
    colors = ['#D58367' if val < 0 else '#9CB68E' for val in values]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(labels, values, color=colors)

    ax.set_ylabel('Average Percentage Change in Accuracy')
    #ax.set_title('Aggregate percentage change in accuracy for MT5 and BLOOM models fine-tuned using LORA and QLORA')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, axis='y')

    # adding percentage % labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()


def plot_aggregate_change_by_size(output_file=None):
    percentage_changes = {
        "MT5-Small + LORA": PERC_CHANGE_MT5SMALL_LORA,
        "MT5-Small + QLORA": PERC_CHANGE_MT5SMALL_QLORA,
        "MT5-Base + LORA": PERC_CHANGE_MT5BASE_LORA,
        "MT5-Base + QLORA": PERC_CHANGE_MT5BASE_QLORA,
        "MT5-Large + LORA": PERC_CHANGE_MT5LARGE_LORA,
        "MT5-Large + QLORA": PERC_CHANGE_MT5LARGE_QLORA,
        "BLOOM-560M + LORA": PERC_CHANGE_BLOOM560M_LORA,
        "BLOOM-560M + QLORA": PERC_CHANGE_BLOOM560M_QLORA,
        "BLOOM-1B1 + LORA": PERC_CHANGE_BLOOM1B1_LORA,
        "BLOOM1B1 + QLORA": PERC_CHANGE_BLOOM1B1_QLORA,
        "BLOOM-1B7 + LORA": PERC_CHANGE_BLOOM1B7_LORA,
        "BLOOM-1B7 + QLORA": PERC_CHANGE_BLOOM1B7_QLORA,
    }

    # get average percentage change for each model and finetuning type
    average_changes = {}
    for key, value in percentage_changes.items():
        average_changes[key] = np.mean(value['accuracy'])
    # prepare data for plotting
    labels = list(average_changes.keys())
    average_percentages = list(average_changes.values())

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(labels))
    width = 0.6
    # colors based on positive or negative values
    colors = ['#D58367' if avg >= 0 else '#9CB68E' for avg in average_percentages]
    bars = ax.bar(x, average_percentages, width, color=colors)
    ax.set_xlabel('Model and Finetuning Type')
    ax.set_ylabel('Average Percentage Change in Accuracy')
    #ax.set_title('Aggregate Percentage Change in Accuracy Compared to Full Finetuning Baseline for Different Models and Finetuning Types')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()


def plot_aggregate_change_by_dataset(percentage_changes, output_file_path=None):
    models_datasets = list(percentage_changes.keys())
    accuracy_changes = list(percentage_changes.values())

    # Define colors for MT5 and BLOOM models
    mt5_color = '#f5dd91'
    bloom_color = '#b9bad5'

    # Define the bar colors based on the model type
    bar_colors = [mt5_color if 'MT5' in model else bloom_color for model in models_datasets]

    plt.figure(figsize=(16, 10))
    bars = plt.bar(models_datasets, accuracy_changes, color=bar_colors, edgecolor='black')

    plt.xlabel('Models and Datasets', fontsize=14)
    plt.ylabel('Average Change in Accuracy (%)', fontsize=14)
    #plt.title('Average Percentage Change in Model Performance', fontsize=16)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.002, f'{yval}%',
                 va='bottom', ha='center', fontsize=10, color='black')

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust the top margin to add space between the highest bars and the end of the graph
    plt.ylim(0, max(accuracy_changes) * 1.2)

    # Add legend
    handles = [
        plt.Line2D([0], [0], color=mt5_color, lw=4),
        plt.Line2D([0], [0], color=bloom_color, lw=4)
    ]
    labels = ['MT5 models', 'BLOOM models']
    plt.legend(handles, labels, fontsize=14)

    plt.tight_layout()

    if output_file_path:
        plt.savefig(output_file_path, format='png', dpi=300)

    plt.show()


# def plot_metrics_per_dataset(metrics, languages, title, output_file=None):
#     fig, axes = plt.subplots(3, 2, figsize=(15, 18), sharey=True)
#     fig.suptitle(title, fontsize=16)
#
#     model_keys = list(metrics.keys())
#     finetuning_keys = ['LORA', 'QLORA', 'FULLFINETUNE']
#
#     for i, model_key in enumerate(model_keys):
#         ax = axes[i // 2, i % 2]
#         bar_width = 0.25
#         index = np.arange(len(languages))
#
#         if 'MT5' in model_key:
#             colors = ['#E9C46A', '#F4A261', '#E76F51']
#         elif 'BLOOM' in model_key:
#             colors = ['#EFC3E6', '#F0A6CA', '#9C89B8']
#
#         for j, finetuning_key in enumerate(finetuning_keys):
#             ax.bar(index + j * bar_width, metrics[model_key][finetuning_key], bar_width, label=finetuning_key,
#                    color=colors[j])
#
#         ax.set_title(model_key)
#         ax.set_xlabel('Language')
#         ax.set_ylabel('Accuracy')
#         ax.set_xticks(index + bar_width)
#         ax.set_xticklabels(languages)
#         if output_file:
#             plt.savefig(output_file)
#         ax.legend()
#
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()


if __name__ == "__main__":
    output_dir = Path('plots/aggregate performance')
    # plot_aggregate_by_language(output_dir / 'barplot_performance_by_language_MT5_BLOOM.png')
    # plot_aggregate_mt5(output_dir / 'barplot_performance_by_language_MT5.png')
    # plot_aggregate_bloom(output_dir / 'barplot_performance_by_language_BLOOM.png')

    # plot_accuracy_change_per_lang(PERC_CHANGE_MT5SMALL_LORA,
    #                               PERC_CHANGE_MT5SMALL_QLORA,
    #                               'MT5-Small',
    #                               output_dir / 'mt5small_accuracy_perc_change.png')
    # plot_accuracy_change_per_lang(PERC_CHANGE_MT5BASE_LORA,
    #                               PERC_CHANGE_MT5BASE_QLORA,
    #                               'MT5-Base',
    #                               output_dir / 'mt5base_accuracy_perc_change.png')
    # plot_accuracy_change_per_lang(PERC_CHANGE_MT5LARGE_LORA,
    #                               PERC_CHANGE_MT5LARGE_QLORA,
    #                               'MT5-Large',
    #                               output_dir / 'mt5large_accuracy_perc_change.png')
    # plot_accuracy_change_per_lang(PERC_CHANGE_BLOOM560M_LORA,
    #                               PERC_CHANGE_BLOOM560M_QLORA,
    #                               'BLOOM-560M',
    #                               output_dir / 'bloom560m_accuracy_perc_change.png')
    # plot_accuracy_change_per_lang(PERC_CHANGE_BLOOM1B1_LORA,
    #                               PERC_CHANGE_BLOOM1B1_QLORA,
    #                               'BLOOM-1B1',
    #                               output_dir / 'bloom1b1_accuracy_perc_change.png')
    # plot_accuracy_change_per_lang(PERC_CHANGE_BLOOM1B7_LORA,
    #                               PERC_CHANGE_BLOOM1B7_QLORA,
    #                               'BLOOM-1B7',
    #                               output_dir / 'bloom1b7_accuracy_perc_change.png')

    # Percentage change by model size
    percentage_changes=calculate_accuracy_change_by(by='size')
    plot_accuracy_change_by_model_size(percentage_changes,
                                       output_dir / 'performance_perc_change_by_model_size.png')

    # # Percentage change by dataset
    percentage_changes=calculate_accuracy_change_by(by='dataset')
    print(percentage_changes)
    plot_aggregate_change_by_dataset(percentage_changes, output_dir / 'by dataset' / 'performance_perc_change_by_dataset.png') # OLD



    # plot_metrics_per_dataset(ALL_PAWSX_METRICS, PAWSX_LANGS,
    #                          "Performance Metrics of MT5 and BLOOM Models on PAWS-X Dataset",
    #                          output_dir / 'by dataset' / 'performance_per_model_xpaws.png')
    # plot_metrics_per_dataset(ALL_XNLI_METRICS, XNLI_LANGS,
    #                          "Performance Metrics of MT5 and BLOOM Models on XNLI Dataset",
    #                          output_dir / 'by dataset' / 'performance_per_model_xnli.png')
    # plot_metrics_per_dataset(ALL_SIB200_METRICS_ALL_LANGS, SIB_LANGS_SUBSET_TWO_CHAR,
    #                          "Performance Metrics of MT5 and BLOOM Models on SIB200 Dataset",
    #                          output_dir / 'by dataset' / 'performance_per_model_sib.png')
    #
