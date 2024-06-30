import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_subword_fertility(fertility_data, title, output_file=None):
    # set up the figure with a suitable proportion for double column paper
    fig, ax = plt.subplots(figsize=(10, 8))
    languages = list(fertility_data.keys())
    fertility_values = list(fertility_data.values())
    # plot
    bars = ax.bar(languages, fertility_values, color='#5072A7', edgecolor='black')
    ax.set_xlabel('Language', fontsize=14, weight='bold')
    ax.set_ylabel('Average Subword Fertility', fontsize=14, weight='bold')
    ax.set_title(title, fontsize=16, weight='bold')

    # this adds num values on top of the bars for better readability
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')

    # customize ticks & adjust y-axis
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xticks(np.arange(len(languages)))
    ax.set_xticklabels(languages, rotation=45, ha='right', fontsize=16)
    max_yval = max(fertility_values)
    ax.set_ylim(0, max_yval + 0.3)
    # view and save
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()

def plot_conjoined_subword_fertility(fertility_data_xnli,
                                     fertility_data_pawsx,
                                     fertility_data_sib,
                                     title1,
                                     title2,
                                     title3,
                                     output_file1,
                                     output_file2,
                                     output_file3,
                                     colors=None):
    def plot_bars(data, common_langs, bar_labels, title, filename, colors):
        # sort languages alphabetically
        sorted_langs = sorted(common_langs)
        sorted_data = {label: [data[label][common_langs.index(lang)] for lang in sorted_langs] for label in bar_labels}

        df = pd.DataFrame(sorted_data)
        df['language'] = sorted_langs

        fig, ax = plt.subplots(figsize=(14, 8))
        bar_width = 0.25
        index = np.arange(len(sorted_langs))

        bars = []
        for i, label in enumerate(bar_labels):
            color = colors[i] if colors and i < len(colors) else None
            bars.append(ax.bar(index + i * bar_width, df[label], bar_width, label=label, edgecolor='black', color=color))

        ax.set_xlabel('Language', fontsize=16, weight='bold')
        ax.set_ylabel('Subword Fertility', fontsize=16, weight='bold')
        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_xticks(index + bar_width * (len(bar_labels) - 1) / 2)
        ax.set_xticklabels(sorted_langs, rotation=45, ha='right', fontsize=16)
        ax.legend(fontsize='16')

        for bar_group in bars:
            for bar in bar_group:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')

        max_yval = max((df[label].max() for label in bar_labels if pd.notna(df[label].max())), default=0)
        ax.set_ylim(0, max_yval + 0.3)

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        plt.show()

    # plot for XPAWS, XNLI and SIB200
    fertility_data_sib_subset_all = filter_dict_by_languages(fertility_data_sib, PAWSX_XNLI_LANGS)
    data_all = {
        'PAWS-X': [fertility_data_pawsx[lang] for lang in PAWSX_XNLI_LANGS],
        'XNLI': [fertility_data_xnli[lang] for lang in PAWSX_XNLI_LANGS],
        'SIB200': [fertility_data_sib_subset_all[lang] for lang in PAWSX_XNLI_LANGS]
    }
    plot_bars(data_all, PAWSX_XNLI_LANGS, ['XNLI', 'PAWS-X', 'SIB200'], title1, output_file1, colors)

    # plot for only XNLI and SIB200
    fertility_data_sib_subset_xnli = filter_dict_by_languages(fertility_data_sib, SIB_XNLI_LANGS)
    data_xnli_sib = {
        'XNLI': [fertility_data_xnli[lang] for lang in SIB_XNLI_LANGS],
        'SIB200': [fertility_data_sib_subset_xnli[lang] for lang in SIB_XNLI_LANGS]
    }
    plot_bars(data_xnli_sib, SIB_XNLI_LANGS, ['XNLI', 'SIB200'], title2, output_file2, colors)

    # plot for only PAWSX and SIB200
    fertility_data_sib_subset_pawsx = filter_dict_by_languages(fertility_data_sib, SIB_PAWSX_LANGS)
    data_pawx_sib = {
        'PAWS-X': [fertility_data_pawsx[lang] for lang in SIB_PAWSX_LANGS],
        'SIB200': [fertility_data_sib_subset_pawsx[lang] for lang in SIB_PAWSX_LANGS]
    }
    plot_bars(data_pawx_sib, SIB_PAWSX_LANGS, ['PAWS-X', 'SIB200'], title3, output_file3, colors)

def plot_trend(performance_df, model_name, output_file=None):
    plt.figure(figsize=(10, 8))
    plt.scatter(performance_df['subword_fertility'], performance_df['accuracy'], color='#5072A7')

    # fit linear trend line
    z = np.polyfit(performance_df['subword_fertility'], performance_df['accuracy'], 1)
    p = np.poly1d(z)
    plt.plot(performance_df['subword_fertility'], p(performance_df['subword_fertility']), "r--")

    plt.title(f'Subword Fertility vs. Accuracy for {model_name}')
    plt.xlabel('Subword Fertility')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # add language annotations
    for i, row in performance_df.iterrows():
        plt.annotate(row['language'], (row['subword_fertility'], row['accuracy']),
                     textcoords="offset points", xytext=(14, -4), ha='center', fontsize=14)  # Increased font size
    if output_file:
        plt.savefig(output_file)
    plt.show()


def plot_trend_with_baseline(qlora_df, fullfinetune_df, title, grey_lines = True, output_file=None):

    plt.figure(figsize=(10, 8))

    # experimental condition
    plt.scatter(qlora_df['subword_fertility'], qlora_df['accuracy'], color='#1f77b4', label='QLORA')

    # baseline condition
    plt.scatter(fullfinetune_df['subword_fertility'], fullfinetune_df['accuracy'], color='#ff7f0e', label='Full Finetune')

    # fit linear trend line for QLoRA experimental condition
    z_exp = np.polyfit(qlora_df['subword_fertility'], qlora_df['accuracy'], 1)
    p_exp = np.poly1d(z_exp)
    plt.plot(qlora_df['subword_fertility'], p_exp(qlora_df['subword_fertility']), "#1f77b4",
             label='QLORA Trend')

    # fit a linear trend line for Full Finetune baseline condition
    z_base = np.polyfit(fullfinetune_df['subword_fertility'], fullfinetune_df['accuracy'], 1)
    p_base = np.poly1d(z_base)
    plt.plot(fullfinetune_df['subword_fertility'], p_base(fullfinetune_df['subword_fertility']), "#ff7f0e",
             label='Full Finetune Trend')

    # draw grey lines between the QLORA and Full Finetune dots
    if grey_lines:
        for i, row in qlora_df.iterrows():
            qlora_x = row['subword_fertility']
            qlora_y = row['accuracy']
            fullfinetune_y = fullfinetune_df[fullfinetune_df['language'] == row['language']]['accuracy'].values[0]
            plt.plot([qlora_x, qlora_x], [qlora_y, fullfinetune_y], color='grey', linestyle='--')

    plt.title(f'Subword Fertility vs. Accuracy for {title}')
    plt.xlabel('Subword Fertility')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # language annotations for experimental & baseline conditions
    for i, row in qlora_df.iterrows():
        plt.annotate(row['language'], (row['subword_fertility'], row['accuracy']),
                     textcoords="offset points", xytext=(14, -4), ha='center', fontsize=14)
    for i, row in fullfinetune_df.iterrows():
        plt.annotate(row['language'], (row['subword_fertility'], row['accuracy']),
                     textcoords="offset points", xytext=(14, -4), ha='center', fontsize=14, color='brown')
    plt.legend()
    if output_file:
        plt.savefig(output_file)
    plt.show()

def plot_subword_fertility_by_dataset(fertility_data_mt5, fertility_data_bloom, title, output_file=None):
    # set up figure with suitable proportion for double column paper
    fig, ax = plt.subplots(figsize=(10, 8))
    languages = list(fertility_data_mt5.keys())
    mt5_values = list(fertility_data_mt5.values())
    bloom_values = list(fertility_data_bloom.values())

    bar_width = 0.35
    index = np.arange(len(languages))

    # bar chart for MT5
    bars_mt5 = ax.bar(index, mt5_values, bar_width, color='#FFE197', edgecolor='black', label='MT5')
    # bar chart for BLOOM
    bars_bloom = ax.bar(index + bar_width, bloom_values, bar_width, color='#B8BEDD', edgecolor='black', label='BLOOM')

    # labels & title
    ax.set_xlabel('Language', fontsize=18)
    ax.set_ylabel('Average Subword Fertility', fontsize=18)
    #ax.set_title(title, fontsize=20)

    # add values on top of the bars for better readability
    for bar in bars_mt5:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')

    for bar in bars_bloom:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')

    # customize tick & y-axis limits
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(languages, rotation=45, ha='right', fontsize=14)
    max_yval = max(max(mt5_values), max(bloom_values))
    ax.set_ylim(0, max_yval + 0.3)

    # view and save
    plt.tight_layout()
    ax.legend(fontsize=18)
    if output_file:
        plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    from pathlib import Path
    from utils import load_from_json, filter_dict_by_languages
    from analysis.config import PAWSX_XNLI_LANGS, SIB_XNLI_LANGS, SIB_PAWSX_LANGS

    barplots_dir = Path('plots/fertility barplots/individual barplots')
    comparative_barplots_dir = Path('plots/fertility barplots/comparative barplots')
    custom_colors = ['#5072A7', '#BFCC94', '#950952'] #  colors for the bar plots
    #
    # #################################################################
    # # Load data for MT5-Small models
    # #################################################################
    mt5_pawsx_fertility = load_from_json("fertilities/mt5_pawsx_fertility.json")
    print('Loaded MT5-Small and PAWS-X fertility:')
    print(mt5_pawsx_fertility)

    mt5_xnli_fertility = load_from_json("fertilities/mt5_xnli_fertility.json")
    print('Loaded MT5-Small and XNLI fertility:')
    print(mt5_xnli_fertility)

    mt5_sib_fertility = load_from_json("fertilities/mt5_sib_fertility.json")
    print('Loaded MT5 and SIB fertility:')
    print(mt5_sib_fertility)

    # Create individual fertility plots for MT5-Small models finetuned on each dataset
    plot_subword_fertility(mt5_pawsx_fertility, 'PAWS-X Subword Fertility for MT5 Models', barplots_dir / 'mt5_pawsx_fertility_barplot.png')
    plot_subword_fertility(mt5_xnli_fertility, 'XNLI Subword Fertility for MT5 Models', barplots_dir / 'mt5_xnli_fertility_barplot.png')
    plot_subword_fertility(mt5_sib_fertility, 'SIB200 Subword Fertility for MT5 Models', barplots_dir / 'mt5_xnli_fertility_barplot.png')


    # Create collective fertility plots to compare all MT5-Small models trained on each dataset
    plot_conjoined_subword_fertility(
        mt5_xnli_fertility,
        mt5_pawsx_fertility,
        mt5_sib_fertility,
        'Comparison of Subword Fertility between XNLI, PAWS-X, and SIB200 for MT5 Models',
        'Comparison of Subword Fertility between XNLI and SIB200 for MT5 Models',
        'Comparison of Subword Fertility between PAWS-X and SIB200 for MT5 Models',
        comparative_barplots_dir / "mt5_conjoined_xnli_pawsx_sib_fertility.png",
        comparative_barplots_dir / "mt5_conjoined_xnli_sib_fertility.png",
        comparative_barplots_dir / "mt5_conjoined_pawsx_sib_fertility.png",
        custom_colors
    )

    #################################################################
    # Load data for Bloom560M models
    #################################################################
    bloom_pawsx_fertility = load_from_json("fertilities/bloom_pawsx_fertility.json")
    print('Loaded BLOOM-560M and PAWS-X fertility:')
    print(bloom_pawsx_fertility)

    bloom_xnli_fertility = load_from_json("fertilities/bloom_xnli_fertility.json") # change
    print('Loaded BLOOM-560M and XNLI fertility:')
    print(bloom_xnli_fertility)

    bloom_sib_fertility = load_from_json("fertilities/bloom_sib_fertility.json")
    print('Loaded BLOOM-560M and SIB fertility:')
    print(bloom_sib_fertility)

    # Create individual fertility plots for MT5-Small models finetuned on each dataset
    plot_subword_fertility(bloom_pawsx_fertility, 'PAWS-X Subword Fertility for BLOOM Models', barplots_dir / 'bloom_pawsx_fertility_barplot.png')
    plot_subword_fertility(bloom_xnli_fertility, 'XNLI Subword Fertility for BLOOM Models', barplots_dir / 'bloom_xnli_fertility_barplot.png')
    plot_subword_fertility(bloom_sib_fertility, 'SIB200 Subword Fertility for BLOOM Models', barplots_dir / 'bloom_sib_fertility_barplot.png')

    # Create collective fertility plots to compare all MT5-Small models trained on each dataset
    plot_conjoined_subword_fertility(
        bloom_xnli_fertility,
        bloom_pawsx_fertility,
        bloom_sib_fertility,
        'Comparison of Subword Fertility between XNLI, PAWS-X, and SIB200',
        'Comparison of Subword Fertility between XNLI and SIB200 for BLOOM Models',
        'Comparison of Subword Fertility between PAWS-X and SIB200 for BLOOM Models',
        comparative_barplots_dir / "bloom_conjoined_xnli_pawsx_sib_fertility.png",
        comparative_barplots_dir / "bloom_conjoined_xnli_sib_fertility.png",
        comparative_barplots_dir / "bloom_conjoined_pawsx_sib_fertility.png",
        custom_colors
    )

    # Create plots per dataset containing fertility metrics for both MT5 and BLOOM models
    # Plot PAWS-X fertility for MT5 and BLOOM models
    plot_subword_fertility_by_dataset(mt5_pawsx_fertility, bloom_pawsx_fertility,
                                      'PAWS-X Subword Fertility for BLOOM and MT5 Models',
                                      comparative_barplots_dir / "pawsx_mt5_bloom_fertility.png")
    # Plot XNLI fertility for MT5 and BLOOM models
    plot_subword_fertility_by_dataset(mt5_xnli_fertility, bloom_xnli_fertility,
                                      'XNLI Subword Fertility for BLOOM and MT5 Models',
                                      comparative_barplots_dir / "xnli_mt5_bloom_fertility.png")
    # Plot SIB200 fertility for MT5 and BLOOM models
    plot_subword_fertility_by_dataset(mt5_sib_fertility, bloom_sib_fertility,
                                      'SIB200 Subword Fertility for BLOOM and MT5 Models',
                                      comparative_barplots_dir / "sib_mt5_bloom_fertility.png")