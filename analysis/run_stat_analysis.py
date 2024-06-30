import pandas as pd
import logging
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from plot_subword_fertility import plot_trend, plot_trend_with_baseline
from utils import load_from_json, calculate_average_fertilities

def stat_analysis(fertility_data, metrics_qlora, metrics_fullfinetune, model, dataset, title, output_dir=None):
    """
    Perform analysis by calculating the Spearman correlation between subword fertility
    and accuracy, and return the combined DataFrame.
    Args:
        fertility_data (dict): A dictionary mapping languages to their subword fertility values.
        metrics_qlora (dict): A dictionary containing 'language' and 'accuracy' lists for QLORA finetuning condition
        metrics_fullfinetune (dict): A dictionary containing 'language' and 'accuracy' for Baseline finetuning condition
        title (str): A mention of the name of the model and task being analyzed.
        output_dir (str, optional): File path to save the plot. If None, plot is not saved.

    Returns:
        pd.DataFrame: DataFrame containing the combined performance metrics and subword fertility.
    """
    if dataset == 'sib': # if dataset is SIB, and if we are only interested in XNLI languages
        langs_to_drop = ['ja', 'ko'] # drop languages that are in PAWS-X but are not in XNLI
        fertility_data = {key: value for key, value in fertility_data.items() if key not in langs_to_drop}

    output_plot_file = output_dir / model / dataset / f'{model}_{dataset}_fertility_trend'

    qlora_df = pd.DataFrame(metrics_qlora)
    fullfinetune_df = pd.DataFrame(metrics_fullfinetune)

    # sort by language
    qlora_df = qlora_df.sort_values(by='language').reset_index(drop=True)
    sorted_fertility_data = {k: fertility_data[k] for k in sorted(fertility_data)}

    # add subword fertility to performance dataframes
    qlora_df['subword_fertility'] = qlora_df['language'].map(sorted_fertility_data)
    fullfinetune_df['subword_fertility'] = fullfinetune_df['language'].map(fertility_data)

    # Plot relationship between subword fertility and model performance
    plot_trend(qlora_df, title, f"{output_plot_file}.png")
    plot_trend_with_baseline(qlora_df, fullfinetune_df, title, grey_lines=False, output_file=f"{output_plot_file}_comparison.png")
    plot_trend_with_baseline(qlora_df, fullfinetune_df, title, grey_lines=True,
                             output_file=f"{output_plot_file}_comparison_greylines.png")

    # correlations
    pearson_corr, pearson_p_value = pearsonr(qlora_df['subword_fertility'], qlora_df['accuracy'])
    spearman_corr, spearman_p_value = spearmanr(qlora_df['subword_fertility'], qlora_df['accuracy'])

    logging.info(title)
    logging.info(f"Pearson correlation: {pearson_corr:.3f}, p-value: {pearson_p_value:.3f}")
    logging.info(f"Spearman correlation: {spearman_corr:.3f}, p-value: {spearman_p_value:.3f}")

    # save results to a text file
    all_results_file = output_dir.parent / 'all_correlation_results.txt'
    output_dir = Path(output_plot_file).parent if output_plot_file else Path('..')
    results_file = output_dir / f"{title.replace(' ', '_')}_results.txt"

    with open(results_file, 'w') as file:
        file.write(f"{title}\n")
        file.write(f"Pearson correlation: {pearson_corr:.3f}, p-value: {pearson_p_value:.3f}\n")
        file.write(f"Spearman correlation: {spearman_corr:.3f}, p-value: {spearman_p_value:.3f}\n")
        file.write(qlora_df.to_string(index=False))

    # append results
    with open(all_results_file, 'a') as file:
        file.write(f"{title}\n")
        file.write(f"Pearson correlation: {pearson_corr:.3f}, p-value: {pearson_p_value:.3f}\n")
        file.write(f"Spearman correlation: {spearman_corr:.3f}, p-value: {spearman_p_value:.3f}\n")
        file.write(qlora_df.to_string(index=False))
        file.write("\n\n")

    return qlora_df

if __name__ == "__main__":
    from config import (
        PAWSX_LANGS, PAWSX_XNLI_LANGS, SIB_LANG_CODE_MAP, SIB_LANGS_SUBSET_TWO_CHAR,
        METRICS_MT5SMALL_XNLI_QLORA_r64,
        METRICS_BLOOM560M_PAWSX_QLORA_r64, METRICS_BLOOM560M_XNLI_QLORA_r64,
        METRICS_MT5BASE_PAWSX_QLORA_r64, METRICS_MT5BASE_XNLI_QLORA_r64,
        METRICS_BLOOM1B1_PAWSX_QLORA_r64, METRICS_BLOOM1B1_XNLI_QLORA_r64,
        METRICS_MT5LARGE_PAWSX_QLORA_r64, METRICS_MT5LARGE_XNLI_QLORA_r64,
        METRICS_BLOOM1B7_PAWSX_QLORA_r64, METRICS_MT5SMALL_PAWSX_FULLFINETUNE, METRICS_MT5SMALL_XNLI_FULLFINETUNE,
        METRICS_MT5BASE_PAWSX_FULLFINETUNE, METRICS_MT5BASE_XNLI_FULLFINETUNE,
        METRICS_MT5LARGE_PAWSX_FULLFINETUNE, METRICS_MT5LARGE_XNLI_FULLFINETUNE,
        METRICS_BLOOM560M_PAWSX_FULLFINETUNE, METRICS_BLOOM560M_XNLI_FULLFINETUNE,
        METRICS_BLOOM1B1_PAWSX_FULLFINETUNE, METRICS_BLOOM1B1_XNLI_FULLFINETUNE,
        METRICS_BLOOM1B7_PAWSX_FULLFINETUNE, )
    from avg_performance_by_language import (
        METRICS_MT5_AVG_QLORA, METRICS_MT5_AVG_FULLFINETUNE,
        METRICS_BLOOM_AVG_QLORA, METRICS_BLOOM_AVG_FULLFINETUNE,
    )
    # import SIB200 metrics subset with relevant languages
    from utils import load_from_json, filter_dict_by_languages, filter_sib_metrics_dict

    # output directory for logs
    log_dir = Path('plots/trend')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / 'correlation_results.txt'

    # write to a file in the trend subfolder
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename, mode='w'),
                            logging.StreamHandler()
                        ])

    ###############################################################################################################
    # Plots for MT5 models
    ###############################################################################################################
    output_dir = log_dir / 'mt5'

    # Load data for MT5-Small models
    mt5_pawsx_fertility = load_from_json("fertilities/mt5_pawsx_fertility.json")
    mt5_xnli_fertility = load_from_json("fertilities/mt5_xnli_fertility.json")
    mt5_sib_fertility = load_from_json("fertilities/mt5_sib_fertility.json")
    mt5_sib_fertility = filter_dict_by_languages(mt5_sib_fertility, SIB_LANGS_SUBSET_TWO_CHAR)

    # MT5-Small models ------------------------------------------------------------------------------------------
    model = 'mt5-small'
    # analysis for MT5-Small with PAWS-X
    dataset = 'pawsx'
    stat_analysis(mt5_pawsx_fertility, METRICS_MT5SMALL_PAWSX_QLORA_r64, METRICS_MT5SMALL_PAWSX_FULLFINETUNE,
                  model, dataset,'MT5-Small QLORA finetuning with PAWS-X Dataset', output_dir)

    # analysis for MT5-Small with XNLI
    dataset = 'xnli'
    stat_analysis(mt5_xnli_fertility, METRICS_MT5SMALL_XNLI_QLORA_r64, METRICS_MT5SMALL_XNLI_FULLFINETUNE,
                  model, dataset,'MT5-Small QLORA finetuning with XNLI Dataset', output_dir)

    # analysis for MT5-Small with SIB200
    dataset = 'sib'
    # using only XNLI langs
    stat_analysis(mt5_sib_fertility, METRICS_MT5SMALL_SIB_QLORA_r64, METRICS_MT5SMALL_SIB_FULLFINETUNE,
                  model, dataset,'MT5-Small QLORA finetuning with SIB200 Dataset', output_dir)
    # using only XNLI+PAWSX langs
    stat_analysis(mt5_sib_fertility, METRICS_MT5SMALL_SIB_QLORA_r64_MORE_LANGS, METRICS_MT5SMALL_SIB_FULLFINETUNE_MORE_LANGS,
                  model, f"{dataset}_more_langs",'MT5-Small QLORA finetuning with SIB200 Dataset', output_dir)

    # MT5-Base models ------------------------------------------------------------------------------------------
    model = 'mt5-base'
    # analysis for MT5-Base with PAWS-X
    dataset = 'pawsx'
    stat_analysis(mt5_pawsx_fertility, METRICS_MT5BASE_PAWSX_QLORA_r64, METRICS_MT5BASE_PAWSX_FULLFINETUNE,
                  model, dataset,'MT5-Base QLORA finetuning with PAWS-X Dataset', output_dir)

    # analysis for MT5-Base with XNLI
    dataset = 'xnli'
    stat_analysis(mt5_xnli_fertility, METRICS_MT5BASE_XNLI_QLORA_r64, METRICS_MT5BASE_XNLI_FULLFINETUNE,
                  model, dataset,'MT5-Base QLORA finetuning with XNLI Dataset', output_dir)

    # analysis for MT5-Base with SIB200
    dataset = 'sib'
    # using only XNLI langs
    stat_analysis(mt5_sib_fertility, METRICS_MT5BASE_SIB_QLORA_r64, METRICS_MT5BASE_SIB_FULLFINETUNE,
                  model, dataset,'MT5-Base QLORA finetuning with SIB200 Dataset', output_dir)
    # using only XNLI+PAWSX langs
    stat_analysis(mt5_sib_fertility, METRICS_MT5BASE_SIB_QLORA_r64_MORE_LANGS, METRICS_MT5BASE_SIB_FULLFINETUNE_MORE_LANGS,
                  model, f"{dataset}_more_langs",'MT5-Base QLORA finetuning with SIB200 Dataset', output_dir)


    MT5-Large models ------------------------------------------------------------------------------------------
    model = 'mt5-large'
    # analysis for MT5-Large with PAWS-X
    dataset = 'pawsx'
    stat_analysis(mt5_pawsx_fertility, METRICS_MT5LARGE_PAWSX_QLORA_r64, METRICS_MT5LARGE_PAWSX_FULLFINETUNE,
                  model, dataset,'MT5-Large QLORA finetuning with PAWS-X Dataset', output_dir)

    # analysis for MT5-Large with XNLI
    dataset = 'xnli'
    stat_analysis(mt5_xnli_fertility, METRICS_MT5LARGE_XNLI_QLORA_r64, METRICS_MT5LARGE_XNLI_FULLFINETUNE,
                  model, dataset,'MT5-Large QLORA finetuning with XNLI Dataset', output_dir)

    # analysis for MT5-Large with SIB200
    dataset = 'sib'
    # using only XNLI langs
    stat_analysis(mt5_sib_fertility, METRICS_MT5LARGE_SIB_QLORA_r64, METRICS_MT5LARGE_SIB_FULLFINETUNE,
                  model, dataset,'MT5-Large QLORA finetuning with SIB200 Dataset', output_dir)
    # using only XNLI+PAWSX langs
    stat_analysis(mt5_sib_fertility, METRICS_MT5LARGE_SIB_QLORA_r64_MORE_LANGS, METRICS_MT5LARGE_SIB_FULLFINETUNE_MORE_LANGS,
                  model, f"{dataset}_more_langs",'MT5-Large QLORA finetuning with SIB200 Dataset', output_dir)



    ###############################################################################################################
    # Plots for BLOOM models
    ###############################################################################################################
    output_dir = log_dir / 'bloom'

    # Load fertility data for Bloom models
    bloom_pawsx_fertility = load_from_json("fertilities/bloom_pawsx_fertility.json")
    bloom_xnli_fertility = load_from_json("fertilities/bloom_xnli_fertility.json")
    bloom_sib_fertility = load_from_json("fertilities/bloom_sib_fertility.json") # 17 langs

    # BLOOM-560M models ------------------------------------------------------------------------------------------
    model = 'bloom-560m'
    # analysis for BLOOM-560M with PAWS-X
    dataset = 'pawsx'
    stat_analysis(bloom_pawsx_fertility, METRICS_BLOOM560M_PAWSX_QLORA_r64, METRICS_BLOOM560M_PAWSX_FULLFINETUNE,
                  model, dataset,'BLOOM-560M QLORA finetuning with PAWS-X Dataset', output_dir)

    # analysis for BLOOM-560M with XNLI
    dataset = 'xnli'
    stat_analysis(bloom_xnli_fertility, METRICS_BLOOM560M_XNLI_QLORA_r64, METRICS_BLOOM560M_XNLI_FULLFINETUNE,
                  model, dataset,'BLOOM-560M QLORA finetuning with XNLI Dataset', output_dir)

    # analysis for BLOOM-560M with SIB200
    dataset = 'sib'
    # using only XNLI langs
    stat_analysis(bloom_sib_fertility, METRICS_BLOOM560M_SIB_QLORA_r64, METRICS_BLOOM560M_SIB_FULLFINETUNE,
                  model, dataset,'BLOOM-560M QLORA finetuning with SIB200 Dataset', output_dir) # 15 langs
    # using only XNLI+PAWSX langs
    stat_analysis(bloom_sib_fertility, METRICS_BLOOM560M_SIB_QLORA_r64_MORE_LANGS, METRICS_BLOOM560M_SIB_FULLFINETUNE_MORE_LANGS,
                  model, f"{dataset}_more_langs",'BLOOM-560M QLORA finetuning with SIB200 Dataset', output_dir) # 17 langs

    # BLOOM-1B1 models ------------------------------------------------------------------------------------------
    model = 'bloom-1b1'
    # analysis for BLOOM-1B1 with PAWS-X
    dataset = 'pawsx'
    stat_analysis(bloom_pawsx_fertility, METRICS_BLOOM1B1_PAWSX_QLORA_r64, METRICS_BLOOM1B1_PAWSX_FULLFINETUNE,
                  model, dataset,'BLOOM-1B1 QLORA finetuning with PAWS-X Dataset', output_dir)

    # analysis for BLOOM-560M with XNLI
    dataset = 'xnli'
    stat_analysis(bloom_xnli_fertility, METRICS_BLOOM1B1_XNLI_QLORA_r64, METRICS_BLOOM1B1_XNLI_FULLFINETUNE,
                  model, dataset,'BLOOM-1B1 QLORA finetuning with XNLI Dataset', output_dir)

    # analysis for BLOOM-560M with SIB200
    dataset = 'sib'
    # using only XNLI langs
    stat_analysis(bloom_sib_fertility, METRICS_BLOOM1B1_SIB_QLORA_r64, METRICS_BLOOM1B1_SIB_FULLFINETUNE,
                  model, dataset,'BLOOM-1B1 QLORA finetuning with SIB200 Dataset', output_dir) # 15 langs
    # using only XNLI+PAWSX langs
    stat_analysis(bloom_sib_fertility, METRICS_BLOOM1B1_SIB_QLORA_r64_MORE_LANGS, METRICS_BLOOM1B1_SIB_FULLFINETUNE_MORE_LANGS,
                  model, f"{dataset}_more_langs",'BLOOM-1B1 QLORA finetuning with SIB200 Dataset', output_dir) # 17 langs

    # BLOOM-1B7 models ------------------------------------------------------------------------------------------
    model = 'bloom-1b7'
    # analysis for BLOOM-1B7 with PAWS-X
    dataset = 'pawsx'
    stat_analysis(bloom_pawsx_fertility, METRICS_BLOOM1B7_PAWSX_QLORA_r64, METRICS_BLOOM1B7_PAWSX_FULLFINETUNE,
                  model, dataset, 'BLOOM-1B7 QLORA finetuning with PAWS-X Dataset', output_dir)

    # analysis for BLOOM-1B7 with XNLI
    dataset = 'xnli'
    stat_analysis(bloom_xnli_fertility, METRICS_BLOOM1B7_XNLI_QLORA_r64, METRICS_BLOOM1B7_XNLI_FULLFINETUNE,
                  model, dataset, 'BLOOM-1B7 QLORA finetuning with XNLI Dataset', output_dir)

    # analysis for BLOOM-1B7 with SIB200
    dataset = 'sib'
    # using only XNLI langs
    stat_analysis(bloom_sib_fertility, METRICS_BLOOM1B7_SIB_QLORA_r64, METRICS_BLOOM1B7_SIB_FULLFINETUNE,
                  model, dataset, 'BLOOM-1B7 QLORA finetuning with SIB200 Dataset', output_dir)  # 15 langs
    # using only XNLI+PAWSX langs
    stat_analysis(bloom_sib_fertility, METRICS_BLOOM1B7_SIB_QLORA_r64_MORE_LANGS, METRICS_BLOOM1B7_SIB_FULLFINETUNE_MORE_LANGS,
                  model, f"{dataset}_more_langs", 'BLOOM-1B7 QLORA finetuning with SIB200 Dataset',
                  output_dir)  # 17 langs

    ###############################################################################################################
    # Plots Average Accuracies vs Average Fertility for MT5 and BLOOM models
    ###############################################################################################################
    output_dir = log_dir
    # For all BLOOM models
    all_dataset_fertilities_on_bloom = [
        bloom_pawsx_fertility,
        bloom_xnli_fertility,
        bloom_sib_fertility,
    ]
    avg_dataset_fertilities_on_bloom = calculate_average_fertilities(all_dataset_fertilities_on_bloom)
    # analysis for BLOOM-1B7 with XNLI
    model = 'bloom'
    dataset = 'all'
    stat_analysis(avg_dataset_fertilities_on_bloom, METRICS_MT5_AVG_QLORA, METRICS_MT5_AVG_FULLFINETUNE,
                  model, dataset, 'BLOOM QLORA finetuning, Average for PAWS-X, XNLI and SIB00 Datasets', output_dir)

    # For all MT5 models
    all_dataset_fertilities_on_mt5 = [
        mt5_pawsx_fertility,
        mt5_xnli_fertility,
        mt5_sib_fertility,
    ]
    avg_dataset_fertilities_on_mt5 = calculate_average_fertilities(all_dataset_fertilities_on_mt5)
    # analysis for BLOOM-1B7 with XNLI
    model = 'mt5'
    dataset = 'all'
    stat_analysis(avg_dataset_fertilities_on_mt5, METRICS_BLOOM_AVG_QLORA, METRICS_BLOOM_AVG_FULLFINETUNE,
                  model, dataset, 'MT5 QLORA finetuning, Average for PAWS-X, XNLI and SIB00 Datasets', output_dir)

