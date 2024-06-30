import matplotlib.pyplot as plt
from statistics import mean

# all metrics
from config import (
    METRICS_MT5SMALL_PAWSX_QLORA_r2, METRICS_MT5SMALL_PAWSX_QLORA_r4, METRICS_MT5SMALL_PAWSX_QLORA_r8,
    METRICS_MT5SMALL_PAWSX_QLORA_r16, METRICS_MT5SMALL_PAWSX_QLORA_r32, METRICS_MT5SMALL_PAWSX_QLORA_r64,
    METRICS_MT5SMALL_PAWSX_QLORA_r128,

    METRICS_MT5SMALL_XNLI_QLORA_r2, METRICS_MT5SMALL_XNLI_QLORA_r4, METRICS_MT5SMALL_XNLI_QLORA_r8,
    METRICS_MT5SMALL_XNLI_QLORA_r16, METRICS_MT5SMALL_XNLI_QLORA_r32, METRICS_MT5SMALL_XNLI_QLORA_r64,
    METRICS_MT5SMALL_XNLI_QLORA_r128,

    METRICS_MT5SMALL_SIB_QLORA_r2, METRICS_MT5SMALL_SIB_QLORA_r4, METRICS_MT5SMALL_SIB_QLORA_r8,
    METRICS_MT5SMALL_SIB_QLORA_r16, METRICS_MT5SMALL_SIB_QLORA_r32, METRICS_MT5SMALL_SIB_QLORA_r64,
    METRICS_MT5SMALL_SIB_QLORA_r128,

    METRICS_BLOOM560M_PAWSX_QLORA_r2, METRICS_BLOOM560M_PAWSX_QLORA_r4, METRICS_BLOOM560M_PAWSX_QLORA_r8,
    METRICS_BLOOM560M_PAWSX_QLORA_r16, METRICS_BLOOM560M_PAWSX_QLORA_r32, METRICS_BLOOM560M_PAWSX_QLORA_r64,
    METRICS_BLOOM560M_PAWSX_QLORA_r128,

    METRICS_BLOOM560M_XNLI_QLORA_r2, METRICS_BLOOM560M_XNLI_QLORA_r4, METRICS_BLOOM560M_XNLI_QLORA_r8,
    METRICS_BLOOM560M_XNLI_QLORA_r16, METRICS_BLOOM560M_XNLI_QLORA_r32, METRICS_BLOOM560M_XNLI_QLORA_r64,
    METRICS_BLOOM560M_XNLI_QLORA_r128,

    METRICS_BLOOM560M_SIB_QLORA_r2, METRICS_BLOOM560M_SIB_QLORA_r4, METRICS_BLOOM560M_SIB_QLORA_r8,
    METRICS_BLOOM560M_SIB_QLORA_r16, METRICS_BLOOM560M_SIB_QLORA_r32, METRICS_BLOOM560M_SIB_QLORA_r64,
    METRICS_BLOOM560M_SIB_QLORA_r128
)


# calculate average accuracies for a given set of metrics
def calculate_average_accuracies_per_rank(*metrics):
    avg_accuracies = [mean(metric['accuracy']) for metric in metrics]
    return avg_accuracies



def plot_average_accuracies_per_rank_qlora(save_path, model_name, ranks, pawsx_avg_accuracies, xnli_avg_accuracies,
                                           sib_avg_accuracies):
    """Plotting function for ranks"""
    x = range(len(ranks))
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, pawsx_avg_accuracies, width, label='PAWS-X', color='#6A994E')
    ax.bar([p + width for p in x], xnli_avg_accuracies, width, label='XNLI', color='#A7C957')
    ax.bar([p + 2 * width for p in x], sib_avg_accuracies, width, label='SIB200', color='#E7DDC4')  # tempo to change

    ax.set_xlabel('Rank', fontsize=14)
    ax.set_ylabel('Average Accuracy', fontsize=14)
    ax.set_title(f'Average Accuracy of {model_name} per Rank using QLoRA finetuning', fontsize=16)
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(ranks, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# plot average accuracies per language family
def plot_average_accuracies_per_family(save_path, model, tuning_type, x_labels, families, *family_accuracies):
    width = 0.04  # the width of the bars
    x = range(len(x_labels))

    fig, ax = plt.subplots(figsize=(14, 10))  # Adjusted the figsize to make the plot wider and taller
    for i, family in enumerate(families):
        y = [acc['accuracy'][i] for acc in family_accuracies]
        ax.bar([p + i * width for p in x], y, width, label=family)

    ax.set_xlabel('Model')
    plt.yscale('log')
    ax.set_ylabel('Average Change in Accuracy')
    ax.set_title(f'Average Change in Accuracy per Language Family using {tuning_type} finetuning on {model}')
    ax.set_xticks([p + (len(families) / 2) * width for p in x])
    ax.set_xticklabels(x_labels)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)  # Moved the legend below the plot
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()




if __name__ == "__main__":
    from pathlib import Path

    output_dir = Path('plots/rank barplots')

    # Define ranks
    ranks = ['r=2', 'r=4', 'r=8','r=16', 'r=32', 'r=64', 'r=128']

    # Calculate average accuracies per rank for each dataset for MT5-Small
    mt5small_pawsx_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_MT5SMALL_PAWSX_QLORA_r2, METRICS_MT5SMALL_PAWSX_QLORA_r4, METRICS_MT5SMALL_PAWSX_QLORA_r8, METRICS_MT5SMALL_PAWSX_QLORA_r16, METRICS_MT5SMALL_PAWSX_QLORA_r32,
        METRICS_MT5SMALL_PAWSX_QLORA_r64, METRICS_MT5SMALL_PAWSX_QLORA_r128)

    mt5small_xnli_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_MT5SMALL_XNLI_QLORA_r2, METRICS_MT5SMALL_XNLI_QLORA_r4, METRICS_MT5SMALL_XNLI_QLORA_r8,
        METRICS_MT5SMALL_XNLI_QLORA_r16, METRICS_MT5SMALL_XNLI_QLORA_r32,
        METRICS_MT5SMALL_XNLI_QLORA_r64, METRICS_MT5SMALL_XNLI_QLORA_r128)

    mt5small_sib_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_MT5SMALL_SIB_QLORA_r2, METRICS_MT5SMALL_SIB_QLORA_r4, METRICS_MT5SMALL_SIB_QLORA_r8,
        METRICS_MT5SMALL_SIB_QLORA_r16, METRICS_MT5SMALL_SIB_QLORA_r32,
        METRICS_MT5SMALL_SIB_QLORA_r64, METRICS_MT5SMALL_SIB_QLORA_r128)

    # Calculate average accuracies per rank for each dataset for BLOOM560M
    bloom560_pawsx_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_BLOOM560M_PAWSX_QLORA_r2, METRICS_BLOOM560M_PAWSX_QLORA_r4, METRICS_BLOOM560M_PAWSX_QLORA_r8,
        METRICS_BLOOM560M_PAWSX_QLORA_r16, METRICS_BLOOM560M_PAWSX_QLORA_r32,
        METRICS_BLOOM560M_PAWSX_QLORA_r64, METRICS_BLOOM560M_PAWSX_QLORA_r128)

    bloom560_xnli_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_BLOOM560M_XNLI_QLORA_r2, METRICS_BLOOM560M_XNLI_QLORA_r4, METRICS_BLOOM560M_XNLI_QLORA_r8,
        METRICS_BLOOM560M_XNLI_QLORA_r16, METRICS_BLOOM560M_XNLI_QLORA_r32,
        METRICS_BLOOM560M_XNLI_QLORA_r64, METRICS_BLOOM560M_XNLI_QLORA_r128)

    bloom560_sib_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_BLOOM560M_SIB_QLORA_r2, METRICS_BLOOM560M_SIB_QLORA_r4, METRICS_BLOOM560M_SIB_QLORA_r8,
        METRICS_BLOOM560M_SIB_QLORA_r16, METRICS_BLOOM560M_SIB_QLORA_r32,
        METRICS_BLOOM560M_SIB_QLORA_r64, METRICS_BLOOM560M_SIB_QLORA_r128)

    # Plot the results
    plot_average_accuracies_per_rank_qlora(output_dir / 'mt5-small'/ 'mt5small_qlora_average_accuracies_per_rank.png','MT5-Small', ranks, mt5small_pawsx_avg_accuracies,
                                           mt5small_xnli_avg_accuracies, mt5small_sib_avg_accuracies)
    plot_average_accuracies_per_rank_qlora(output_dir / 'bloom-560m' / 'bloom560m_qlora_average_accuracies_per_rank.png','BLOOM560', ranks, bloom560_pawsx_avg_accuracies,
                                           bloom560_xnli_avg_accuracies,
                                           bloom560_sib_avg_accuracies)


    ###############################################################
    # Plots for MEDIUM models with all available ranks (16 to 128)
    ################################################################
    ranks = ['r=2', 'r=4', 'r=8','r=16', 'r=32', 'r=64', 'r=128']
    output_dir = Path('plots/rank barplots')

    from config import (
        METRICS_MT5BASE_PAWSX_QLORA_r16, METRICS_MT5BASE_PAWSX_QLORA_r32,
        METRICS_MT5BASE_PAWSX_QLORA_r64, METRICS_MT5BASE_PAWSX_QLORA_r128,
        METRICS_MT5BASE_XNLI_QLORA_r16, METRICS_MT5BASE_XNLI_QLORA_r32,
        METRICS_MT5BASE_XNLI_QLORA_r64, METRICS_MT5BASE_XNLI_QLORA_r128,
        METRICS_MT5BASE_SIB_QLORA_r16, METRICS_MT5BASE_SIB_QLORA_r32,
        METRICS_MT5BASE_SIB_QLORA_r64, METRICS_MT5BASE_SIB_QLORA_r128,
        METRICS_BLOOM1B1_PAWSX_QLORA_r16, METRICS_BLOOM1B1_PAWSX_QLORA_r32,
        METRICS_BLOOM1B1_PAWSX_QLORA_r64, METRICS_BLOOM1B1_PAWSX_QLORA_r128,
        METRICS_BLOOM1B1_XNLI_QLORA_r16, METRICS_BLOOM1B1_XNLI_QLORA_r32,
        METRICS_BLOOM1B1_XNLI_QLORA_r64, METRICS_BLOOM1B1_XNLI_QLORA_r128,
        METRICS_BLOOM1B1_SIB_QLORA_r16, METRICS_BLOOM1B1_SIB_QLORA_r32,
        METRICS_BLOOM1B1_SIB_QLORA_r64, METRICS_BLOOM1B1_SIB_QLORA_r128
    )
    # Calculate average accuracies per rank for each dataset for MT5-Small
    mt5base_pawsx_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_MT5BASE_PAWSX_QLORA_r16, METRICS_MT5BASE_PAWSX_QLORA_r32,
        METRICS_MT5BASE_PAWSX_QLORA_r64, METRICS_MT5BASE_PAWSX_QLORA_r128)

    mt5base_xnli_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_MT5BASE_XNLI_QLORA_r16, METRICS_MT5BASE_XNLI_QLORA_r32,
        METRICS_MT5BASE_XNLI_QLORA_r64, METRICS_MT5BASE_XNLI_QLORA_r128)

    mt5base_sib_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_MT5BASE_SIB_QLORA_r16, METRICS_MT5BASE_SIB_QLORA_r32,
        METRICS_MT5BASE_SIB_QLORA_r64, METRICS_MT5BASE_SIB_QLORA_r128)

    # Calculate average accuracies per rank for each dataset for BLOOM560M
    bloom1b1_pawsx_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_BLOOM1B1_PAWSX_QLORA_r16, METRICS_BLOOM1B1_PAWSX_QLORA_r32,
        METRICS_BLOOM1B1_PAWSX_QLORA_r64, METRICS_BLOOM1B1_PAWSX_QLORA_r128)

    bloom1bq_xnli_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_BLOOM1B1_XNLI_QLORA_r16, METRICS_BLOOM1B1_XNLI_QLORA_r32,
        METRICS_BLOOM1B1_XNLI_QLORA_r64, METRICS_BLOOM1B1_XNLI_QLORA_r128)

    bloom1b1_sib_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_BLOOM1B1_SIB_QLORA_r16, METRICS_BLOOM1B1_SIB_QLORA_r32,
        METRICS_BLOOM1B1_SIB_QLORA_r64, METRICS_BLOOM1B1_SIB_QLORA_r128)

    # # Plot the results
    plot_average_accuracies_per_rank_qlora(output_dir / 'mt5-base'/ 'mt5base_qlora_average_accuracies_per_rank.png', 'MT5-Base', ranks,
                                           mt5base_pawsx_avg_accuracies,
                                           mt5base_xnli_avg_accuracies, mt5base_sib_avg_accuracies)
    plot_average_accuracies_per_rank_qlora(output_dir / 'bloom-1b1'/ 'bloom1b1_qlora_average_accuracies_per_rank.png', 'BLOOM-1B1', ranks,
                                           bloom1b1_pawsx_avg_accuracies,
                                           bloom1bq_xnli_avg_accuracies,
                                           bloom1b1_sib_avg_accuracies)

    ################################################################
    # Plots for LARGE models
    #################################################################

    from config import (
        METRICS_MT5LARGE_PAWSX_QLORA_r16, METRICS_MT5LARGE_PAWSX_QLORA_r32,
        METRICS_MT5LARGE_PAWSX_QLORA_r64, METRICS_MT5LARGE_PAWSX_QLORA_r128,
        METRICS_MT5LARGE_XNLI_QLORA_r16, METRICS_MT5LARGE_XNLI_QLORA_r32,
        METRICS_MT5LARGE_XNLI_QLORA_r64, METRICS_MT5LARGE_XNLI_QLORA_r128,
        METRICS_MT5LARGE_SIB_QLORA_r16, METRICS_MT5LARGE_SIB_QLORA_r32,
        METRICS_MT5LARGE_SIB_QLORA_r64, METRICS_MT5LARGE_SIB_QLORA_r128,
        METRICS_BLOOM1B7_PAWSX_QLORA_r16, METRICS_BLOOM1B7_PAWSX_QLORA_r32,
        METRICS_BLOOM1B7_PAWSX_QLORA_r64, METRICS_BLOOM1B7_PAWSX_QLORA_r128,
        METRICS_BLOOM1B7_XNLI_QLORA_r16, METRICS_BLOOM1B7_XNLI_QLORA_r32,
        METRICS_BLOOM1B7_XNLI_QLORA_r64, METRICS_BLOOM1B7_XNLI_QLORA_r128,
        METRICS_BLOOM1B7_SIB_QLORA_r16, METRICS_BLOOM1B7_SIB_QLORA_r32,
        METRICS_BLOOM1B7_SIB_QLORA_r64, METRICS_BLOOM1B7_SIB_QLORA_r128
    )

    # Calculate average accuracies per rank for each dataset for MT5-Small
    mt5large_pawsx_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_MT5LARGE_PAWSX_QLORA_r16, METRICS_MT5LARGE_PAWSX_QLORA_r32,
        METRICS_MT5LARGE_PAWSX_QLORA_r64, METRICS_MT5LARGE_PAWSX_QLORA_r128)

    mt5large_xnli_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_MT5LARGE_XNLI_QLORA_r16, METRICS_MT5LARGE_XNLI_QLORA_r32,
        METRICS_MT5LARGE_XNLI_QLORA_r64, METRICS_MT5LARGE_XNLI_QLORA_r128)

    mt5large_sib_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_MT5LARGE_SIB_QLORA_r16, METRICS_MT5LARGE_SIB_QLORA_r32,
        METRICS_MT5LARGE_SIB_QLORA_r64, METRICS_MT5LARGE_SIB_QLORA_r128)

    # Calculate average accuracies per rank for each dataset for BLOOM560M
    bloom1b7_pawsx_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_BLOOM1B7_PAWSX_QLORA_r16, METRICS_BLOOM1B7_PAWSX_QLORA_r32,
        METRICS_BLOOM1B7_PAWSX_QLORA_r64, METRICS_BLOOM1B7_PAWSX_QLORA_r128)

    bloom1b7_xnli_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_BLOOM1B7_XNLI_QLORA_r16, METRICS_BLOOM1B7_XNLI_QLORA_r32,
        METRICS_BLOOM1B7_XNLI_QLORA_r64, METRICS_BLOOM1B7_XNLI_QLORA_r128)

    bloom1b7_sib_avg_accuracies = calculate_average_accuracies_per_rank(
        METRICS_BLOOM1B7_SIB_QLORA_r16, METRICS_BLOOM1B7_SIB_QLORA_r32,
        METRICS_BLOOM1B7_SIB_QLORA_r64, METRICS_BLOOM1B7_SIB_QLORA_r128)

    # Plot the results
    plot_average_accuracies_per_rank_qlora(output_dir / 'mt5-large' / 'mt5large_qlora_average_accuracies_per_rank.png',
                                           'MT5-Large', ranks,
                                           mt5large_pawsx_avg_accuracies,
                                           mt5large_xnli_avg_accuracies,
                                           mt5large_sib_avg_accuracies)
    plot_average_accuracies_per_rank_qlora(output_dir / 'bloom-1b7' / 'bloom1b7_qlora_average_accuracies_per_rank.png',
                                           'BLOOM-1B7', ranks,
                                           bloom1b7_pawsx_avg_accuracies,
                                           bloom1b7_xnli_avg_accuracies,
                                           bloom1b7_sib_avg_accuracies)

    family_names = list(set(entry['family'] for entry in SIB_LANG_FAMILY_NAMES.values()))

    # Calculate average accuracies per family for each rank
    model = "BLOOM560M"
    finetuning_type = "QLORA"
    family_accuracies_r16 = get_metrics_by_category(model, finetuning_type, 16)
    family_accuracies_r32 = get_metrics_by_category(model, finetuning_type, 32)
    family_accuracies_r64 = get_metrics_by_category(model, finetuning_type, 64)
    family_accuracies_r128 = get_metrics_by_category(model, finetuning_type, 128)

    # Prepare the data for plotting
    family_accuracies = [
        family_accuracies_r16,
        family_accuracies_r32,
        family_accuracies_r64,
        family_accuracies_r128
    ]

    # Plot the results
    plot_average_accuracies_per_family('plots/bloom560m_qlora_average_accuracies_per_family_per_rank.png',
                                       model,
                                       finetuning_type,
                                       ranks,
                                       family_names,
                                       *family_accuracies)

    # Calculate average accuracies per family for each rank
    model = "MT5SMALL"
    finetuning_type = "QLORA"
    family_accuracies_r16 = get_metrics_by_category(model, finetuning_type, 16)
    family_accuracies_r32 = get_metrics_by_category(model, finetuning_type, 32)
    family_accuracies_r64 = get_metrics_by_category(model, finetuning_type, 64)
    family_accuracies_r128 = get_metrics_by_category(model, finetuning_type, 128)

    # Prepare the data for plotting
    family_accuracies = [
        family_accuracies_r16,
        family_accuracies_r32,
        family_accuracies_r64,
        family_accuracies_r128
    ]

    # Plot the results
    plot_average_accuracies_per_family('plots/mt5small_lora_average_accuracies_per_family_per_rank.png', model, finetuning_type, ranks, family_names, *family_accuracies)

    # Calculate average accuracies per family for each rank
    model = "BLOOM1B1"
    finetuning_type = "QLORA"
    # do for r = 1,2, 4, 8
    family_accuracies_r16 = get_metrics_by_category(model, finetuning_type, 16)
    family_accuracies_r32 = get_metrics_by_category(model, finetuning_type, 32)
    family_accuracies_r64 = get_metrics_by_category(model, finetuning_type, 64)
    family_accuracies_r128 = get_metrics_by_category(model, finetuning_type, 128)

    # Prepare the data for plotting
    family_accuracies = [
        family_accuracies_r16,
        family_accuracies_r32,
        family_accuracies_r64,
        family_accuracies_r128
    ]

    # Plot the results
    plot_average_accuracies_per_family(output_dir / 'bloom-1b1'/ 'bloom1b1_qlora_average_accuracies_per_family_per_rank.png', model,
                                       finetuning_type, ranks, family_names, *family_accuracies)

    # Calculate average accuracies per family for each rank
    model = "MT5BASE"
    finetuning_type = "QLORA"
    family_accuracies_r16 = get_metrics_by_category(model, finetuning_type, 16)
    family_accuracies_r32 = get_metrics_by_category(model, finetuning_type, 32)
    family_accuracies_r64 = get_metrics_by_category(model, finetuning_type, 64)
    family_accuracies_r128 = get_metrics_by_category(model, finetuning_type, 128)

    # Prepare the data for plotting
    family_accuracies = [
        family_accuracies_r16,
        family_accuracies_r32,
        family_accuracies_r64,
        family_accuracies_r128
    ]

    # Plot the results
    plot_average_accuracies_per_family(output_dir / 'mt5-base'/ 'mt5base_lora_average_accuracies_per_family_per_rank.png', model, finetuning_type, ranks, family_names, *family_accuracies)