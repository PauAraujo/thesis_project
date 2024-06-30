# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# Install packages
!pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 
!pip install -e .

!pip install pydantic-settings
!pip install peft
!pip install trl
!pip install bitsandbytes
# Restart the Python process to use the updated packages
dbutils.library.restartPython()

# COMMAND ----------

# Imports
import os
import sys
from pathlib import Path
from src.llm_research.experiment_runner import ExperimentArgs, run_experiment
from src.llm_research.train import TrainingType
from typing import List

# COMMAND ----------

# Verify the Files
display(dbutils.fs.ls("dbfs:/Paula/data"))

# COMMAND ----------

# Define each dataset languages in 
XPAWS_LANGS = ["en", "de", "es", "fr", "ja", "ko", "zh"]
XNLI_LANGS = ["en", "ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
SIB_LANGS = [
            'aceArab', 'aceLatn', 'acmArab', 'acqArab', 'aebArab', 'afrLatn', 'ajpArab',
            'akaLatn', 'alsLatn', 'amhEthi', 'apcArab', 'arbArab', 'arbLatn', 'arsArab',
            'aryArab', 'arzArab', 'asmBeng', 'astLatn', 'awaDeva', 'ayrLatn', 'azbArab',
            'azjLatn', 'bakCyrl', 'bamLatn', 'banLatn', 'belCyrl', 'bemLatn', 'benBeng',
            'bhoDeva', 'bjnArab', 'bjnLatn', 'bodTibt', 'bosLatn', 'bugLatn', 'bulCyrl',
            'catLatn', 'cebLatn', 'cesLatn', 'cjkLatn', 'ckbArab', 'crhLatn', 'cymLatn',
            'danLatn', 'deuLatn', 'dikLatn', 'dyuLatn', 'dzoTibt', 'ellGrek', 'engLatn',
            'epoLatn', 'estLatn', 'eusLatn', 'eweLatn', 'faoLatn', 'fijLatn', 'finLatn',
            'fonLatn', 'fraLatn', 'furLatn', 'fuvLatn', 'gazLatn', 'glaLatn', 'gleLatn',
            'glgLatn', 'grnLatn', 'gujGujr', 'hatLatn', 'hauLatn', 'hebHebr', 'hinDeva',
            'hneDeva', 'hrvLatn', 'hunLatn', 'hyeArmn', 'iboLatn', 'iloLatn', 'indLatn',
            'islLatn', 'itaLatn', 'javLatn', 'jpnJpan', 'kabLatn', 'kacLatn', 'kamLatn',
            'kanKnda', 'kasArab', 'kasDeva', 'katGeor', 'kazCyrl', 'kbpLatn', 'keaLatn',
            'khkCyrl', 'khmKhmr', 'kikLatn', 'kinLatn', 'kirCyrl', 'kmbLatn', 'kmrLatn',
            'kncArab', 'kncLatn', 'konLatn', 'korHang', 'laoLaoo', 'lijLatn', 'limLatn',
            'linLatn', 'litLatn', 'lmoLatn', 'ltgLatn', 'ltzLatn', 'luaLatn', 'lugLatn',
            'luoLatn', 'lusLatn', 'lvsLatn', 'magDeva', 'maiDeva', 'malMlym', 'marDeva',
            'minArab', 'minLatn', 'mkdCyrl', 'mltLatn', 'mniBeng', 'mosLatn', 'mriLatn',
            'myaMymr', 'nldLatn', 'nnoLatn', 'nobLatn', 'npiDeva', 'nqoNkoo', 'nsoLatn',
            'nusLatn', 'nyaLatn', 'ociLatn', 'oryOrya', 'pagLatn', 'panGuru', 'papLatn',
            'pbtArab', 'pesArab', 'pltLatn', 'polLatn', 'porLatn', 'prsArab', 'quyLatn',
            'ronLatn', 'runLatn', 'rusCyrl', 'sagLatn', 'sanDeva', 'satOlck', 'scnLatn',
            'shnMymr', 'sinSinh', 'slkLatn', 'slvLatn', 'smoLatn', 'snaLatn', 'sndArab',
            'somLatn', 'sotLatn', 'spaLatn', 'srdLatn', 'srpCyrl', 'sswLatn', 'sunLatn',
            'sweLatn', 'swhLatn', 'szlLatn', 'tamTaml', 'taqLatn', 'taqTfng', 'tatCyrl',
            'telTelu', 'tgkCyrl', 'tglLatn', 'thaThai', 'tirEthi', 'tpiLatn', 'tsnLatn',
            'tsoLatn', 'tukLatn', 'tumLatn', 'turLatn', 'twiLatn', 'tzmTfng', 'uigArab',
            'ukrCyrl', 'umbLatn', 'urdArab', 'uznLatn', 'vecLatn', 'vieLatn', 'warLatn',
            'wolLatn', 'xhoLatn', 'yddHebr', 'yorLatn', 'yueHant', 'zhoHans', 'zhoHant',
            'zsmLatn', 'zulLatn'
        ]

# COMMAND ----------

def setup_and_run_experiment(
    pretrained_model_id: str,
    task_name: str,
    dataset_train_path: str,
    dataset_val_path: str,
    output_dir_base: str,
    lora_rs: List[int],
    training_types: List[TrainingType],
    epochs: int,
    learning_rate: float
) -> None:
    """
    Set up and run the experiment with the specified parameters.

    Args:
        pretrained_model_id (str): The ID of the pretrained model to use.
        task_name (str): The name of the task (e.g., 'xpaws', 'xnli', 'sib').
        dataset_train_path (str): The path to the training dataset.
        dataset_val_path (str): The path to the validation dataset.
        output_dir_base (str): The base directory for output files.
        lora_rs (List[int]): A list of LoRA ranks to use in the experiment.
        training_types (List[TrainingType]): A list of training types to use.
        epochs (int): Number of training epochs.
        learning_rate (float): The learning rate for training.

    Returns:
        None
    """
    for lora_r in lora_rs:
        args = ExperimentArgs(
            pretrained_model_id=pretrained_model_id,
            task_name=task_name,
            dataset_train_path=Path(dataset_train_path),
            dataset_val_path=Path(dataset_val_path),
            output_dir=output_dir_base,
            num_train_epochs=epochs,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=16,
            inference_batch_size=16,
            learning_rate=learning_rate,
            max_seq_length=2048,
            save_steps=500,
            eval_steps=500,
            filter_too_long_samples=True,
            pad_token_to_eos=False,
            debug=False,
            debug_sample_size=100,
            lora_r=lora_r
        )

        for training_type in training_types:
            if lora_r == 0:
                args.output_dir = f"{output_dir_base}_{training_type}"
            else:
                args.output_dir = f"{output_dir_base}_r{lora_r}_{training_type}"

            if task_name == "xpaws":
                langs = XPAWS_LANGS
            elif task_name == "xnli":
                langs = XNLI_LANGS
            elif task_name == "sib":
                langs = SIB_LANGS
            else:
                raise ValueError(f"Invalid task name {task_name}")

            test_paths = [Path(f"/dbfs/Paula/data/{task_name}_{lang}_validation.json") for lang in langs]
            args.dataset_test_paths = test_paths
            args.training_type = training_type

            run_experiment(args)


# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **MT5-Small** LoRA and QLoRA finetuning with **PAWS-X**

# COMMAND ----------

pretrained_model_id = "google/mt5-small"
task_name = "xpaws"
dataset_train_path = "/dbfs/Paula/data/xpaws_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xpaws_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5small_xpaws_PEFT"
epochs = 14
lora_rs = [2, 4, 8, 16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)
# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **MT5-Small** LoRA and QLoRA finetuning with **XNLI**

# COMMAND ----------

pretrained_model_id = "google/mt5-small"
task_name = "xnli"
dataset_train_path = "/dbfs/Paula/data/xnli_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xnli_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5small_xnli_PEFT"
epochs = 2
lora_rs = [2, 4, 8, 16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **MT5-Small** LoRA and QLoRA finetuning with **SIB200**

# COMMAND ----------

pretrained_model_id = "google/mt5-small"
task_name = "sib"
dataset_train_path = "/dbfs/Paula/data/sib_engLatn_train.json"
dataset_val_path = "/dbfs/Paula/data/sib_engLatn_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5small_sib_PEFT"
epochs = 80
lora_rs = [2, 4, 8, 16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **BLOOM-560M** LoRA and QLoRA finetuning with **PAWS-X**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-560m"
task_name = "xpaws"
dataset_train_path = "/dbfs/Paula/data/xpaws_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xpaws_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom560m_xpaws_PEFT"
epochs = 2
lora_rs = [2, 4, 8, 16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)
# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **BLOOM-560M** LoRA and QLoRA finetuning with **XNLI**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-560m"
task_name = "xnli"
dataset_train_path = "/dbfs/Paula/data/xnli_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xnli_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom560m_xnli_PEFT"
epochs = 2
lora_rs = [2, 4, 8, 16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)
# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **BLOOM-560M** LoRA and QLoRA finetuning with **SIB200**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-560m"
task_name = "sib"
dataset_train_path = "/dbfs/Paula/data/sib_engLatn_train.json"
dataset_val_path = "/dbfs/Paula/data/sib_engLatn_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom560m_sib_PEFT"
epochs = 80
lora_rs = [2, 4, 8, 16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)


# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **MT5-Base** LoRA and QLoRA finetuning with **PAWS-X**

# COMMAND ----------

pretrained_model_id = "google/mt5-base"
task_name = "xpaws"
dataset_train_path = "/dbfs/Paula/data/xpaws_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xpaws_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5base_xpaws_PEFT"
epochs = 2
lora_rs = [16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)
# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **MT5-Base** LoRA and QLoRA finetuning with **XNLI**

# COMMAND ----------

pretrained_model_id = "google/mt5-base"
task_name = "xnli"
dataset_train_path = "/dbfs/Paula/data/xnli_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xnli_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5base_xnli_PEFT"
epochs = 2
lora_rs = [16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **MT5-Base** LoRA and QLoRA finetuning with **SIB200**

# COMMAND ----------

pretrained_model_id = "google/mt5-base"
task_name = "sib"
dataset_train_path = "/dbfs/Paula/data/sib_engLatn_train.json"
dataset_val_path = "/dbfs/Paula/data/sib_engLatn_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5base_sib_PEFT"
epochs = 80
lora_rs = [16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **BLOOM-1B1** LoRA and QLoRA finetuning with **PAWS-X**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-1b1"
task_name = "xpaws"
dataset_train_path = "/dbfs/Paula/data/xpaws_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xpaws_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom1b1_xpaws_PEFT"
epochs = 2
lora_rs = [16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)
# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **BLOOM-1B1** LoRA and QLoRA finetuning with **XNLI**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-1b1"
task_name = "xnli"
dataset_train_path = "/dbfs/Paula/data/xnli_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xnli_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom1b1_xnli_PEFT"
epochs = 2
lora_rs = [16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)
# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **BLOOM-1B1** LoRA and QLoRA finetuning with **SIB200**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-1b1"
task_name = "sib"
dataset_train_path = "/dbfs/Paula/data/sib_engLatn_train.json"
dataset_val_path = "/dbfs/Paula/data/sib_engLatn_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom1b1_sib_PEFT"
epochs = 80
lora_rs = [16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)


# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **MT5-Large** LoRA and QLoRA finetuning with **PAWS-X**

# COMMAND ----------

pretrained_model_id = "google/mt5-large"
task_name = "xpaws"
dataset_train_path = "/dbfs/Paula/data/xpaws_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xpaws_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5large_xpaws_PEFT"
epochs = 2
lora_rs = [16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)
# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **MT5-Large** LoRA and QLoRA finetuning with **XNLI**

# COMMAND ----------

pretrained_model_id = "google/mt5-large"
task_name = "xnli"
dataset_train_path = "/dbfs/Paula/data/xnli_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xnli_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5large_xnli_PEFT"
epochs = 2
lora_rs = [16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **MT5-Large** LoRA and QLoRA finetuning with **SIB200**

# COMMAND ----------

pretrained_model_id = "google/mt5-large"
task_name = "sib"
dataset_train_path = "/dbfs/Paula/data/sib_engLatn_train.json"
dataset_val_path = "/dbfs/Paula/data/sib_engLatn_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5large_sib_PEFT"
epochs = 80
lora_rs = [16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **BLOOM-1B7** LoRA and QLoRA finetuning with **PAWS-X**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-1b7"
task_name = "xpaws"
dataset_train_path = "/dbfs/Paula/data/xpaws_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xpaws_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom1b7_xpaws_PEFT"
epochs = 2
lora_rs = [16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)
# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **BLOOM-1B7** LoRA and QLoRA finetuning with **XNLI**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-1b7"
task_name = "xnli"
dataset_train_path = "/dbfs/Paula/data/xnli_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xnli_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom1b7_xnli_PEFT"
epochs = 2
lora_rs = [16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)
# COMMAND ----------

# MAGIC %md
# MAGIC # PEFT
# MAGIC ### **BLOOM-1B7** LoRA and QLoRA finetuning with **SIB200**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-1b7"
task_name = "sib"
dataset_train_path = "/dbfs/Paula/data/sib_engLatn_train.json"
dataset_val_path = "/dbfs/Paula/data/sib_engLatn_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom1b7_sib_PEFT"
epochs = 80
lora_rs = [16, 32, 64, 128]
training_types = [TrainingType.LORA, TrainingType.QLORA]

learning_rate = 2e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Full finetuning
# MAGIC ### **MT5-Small** full model finetuning with **PAWS-X**

# COMMAND ----------

pretrained_model_id = "google/mt5-small"
task_name = "xpaws"
dataset_train_path = "/dbfs/Paula/data/xpaws_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xpaws_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5small_xpaws"
epochs = 14
lora_rs = [0] # placeholder
training_types = [TrainingType.FULL_FINETUNE] 

learning_rate = 1e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
) 


# COMMAND ----------

# MAGIC %md
# MAGIC ### **MT5-Small** full finetuning with **XNLI**

# COMMAND ----------

pretrained_model_id = "google/mt5-small"
task_name = "xnli"
dataset_train_path = "/dbfs/Paula/data/xnli_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xnli_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5small_xnli"
epochs = 2
lora_rs = [0] # placeholder
training_types = [TrainingType.FULL_FINETUNE] 

learning_rate = 1e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### **MT5-Small** full finetuning with **SIB**

# COMMAND ----------

pretrained_model_id = "google/mt5-small"
task_name = "sib"
dataset_train_path = "/dbfs/Paula/data/sib_engLatn_train.json"
dataset_val_path = "/dbfs/Paula/data/sib_engLatn_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5small_sib"
epochs = 80
lora_rs = [0] # placeholder
training_types = [TrainingType.FULL_FINETUNE] 

learning_rate = 1e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### **BLOOM560** full model finetuning with **PAWS-X**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-560m"
task_name = "xpaws"
dataset_train_path = "/dbfs/Paula/data/xpaws_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xpaws_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom560_xpaws"
epochs = 2
lora_rs = [0] # placeholder
training_types = [TrainingType.FULL_FINETUNE] 

learning_rate = 1e-5

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### **BLOOM560** full model finetuning with **XNLI**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-560m"
task_name = "xnli"
dataset_train_path = "/dbfs/Paula/data/xnli_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xnli_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom560_xnli"
epochs = 2
lora_rs = [0] # placeholder
training_types = [TrainingType.FULL_FINETUNE] 

learning_rate = 1e-5

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Bloom560M** full finetuning with **SIB**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-560m"
task_name = "sib"
dataset_train_path = "/dbfs/Paula/data/sib_engLatn_train.json"
dataset_val_path = "/dbfs/Paula/data/sib_engLatn_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom560_sib"
epochs = 80
lora_rs = [0] # placeholder
training_types = [TrainingType.FULL_FINETUNE] 

learning_rate = 1e-5

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### **MT5-Base** full model finetuning with **PAWS-X**

# COMMAND ----------

pretrained_model_id = "google/mt5-base"
task_name = "xpaws"
dataset_train_path = "/dbfs/Paula/data/xpaws_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xpaws_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5base_xpaws"
epochs = 2
lora_rs = [0]  # placeholder
training_types = [TrainingType.FULL_FINETUNE]

learning_rate = 1e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **MT5-Base** full finetuning with **XNLI**

# COMMAND ----------

pretrained_model_id = "google/mt5-base"
task_name = "xnli"
dataset_train_path = "/dbfs/Paula/data/xnli_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xnli_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5base_xnli"
epochs = 2
lora_rs = [0]  # placeholder
training_types = [TrainingType.FULL_FINETUNE]

learning_rate = 1e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **MT5-Base** full finetuning with **SIB**

# COMMAND ----------

pretrained_model_id = "google/mt5-base"
task_name = "sib"
dataset_train_path = "/dbfs/Paula/data/sib_engLatn_train.json"
dataset_val_path = "/dbfs/Paula/data/sib_engLatn_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5base_sib"
epochs = 80
lora_rs = [0]  # placeholder
training_types = [TrainingType.FULL_FINETUNE]

learning_rate = 1e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **BLOOM-1B1** full model finetuning with **PAWS-X**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-1b1"
task_name = "xpaws"
dataset_train_path = "/dbfs/Paula/data/xpaws_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xpaws_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom1b1_xpaws"
epochs = 2
lora_rs = [0]  # placeholder
training_types = [TrainingType.FULL_FINETUNE]

learning_rate = 1e-5

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **BLOOM-1B1** full model finetuning with **XNLI**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-1b1"
task_name = "xnli"
dataset_train_path = "/dbfs/Paula/data/xnli_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xnli_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom1b1_xnli"
epochs = 2
lora_rs = [0]  # placeholder
training_types = [TrainingType.FULL_FINETUNE]

learning_rate = 1e-5

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **BLOOM-1B1** full finetuning with **SIB200**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-1b1"
task_name = "sib"
dataset_train_path = "/dbfs/Paula/data/sib_engLatn_train.json"
dataset_val_path = "/dbfs/Paula/data/sib_engLatn_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom1b1_sib"
epochs = 80
lora_rs = [0]  # placeholder
training_types = [TrainingType.FULL_FINETUNE]

learning_rate = 1e-5

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **MT5-Large** full model finetuning with **PAWS-X**

# COMMAND ----------

pretrained_model_id = "google/mt5-large"
task_name = "xpaws"
dataset_train_path = "/dbfs/Paula/data/xpaws_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xpaws_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5large_xpaws"
epochs = 2
lora_rs = [0]  # placeholder
training_types = [TrainingType.FULL_FINETUNE]

learning_rate = 1e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **MT5-Large** full finetuning with **XNLI**

# COMMAND ----------

pretrained_model_id = "google/mt5-large"
task_name = "xnli"
dataset_train_path = "/dbfs/Paula/data/xnli_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xnli_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5large_xnli"
epochs = 2
lora_rs = [0]  # placeholder
training_types = [TrainingType.FULL_FINETUNE]

learning_rate = 1e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **MT5-Large** full finetuning with **SIB**

# COMMAND ----------

pretrained_model_id = "google/mt5-large"
task_name = "sib"
dataset_train_path = "/dbfs/Paula/data/sib_engLatn_train.json"
dataset_val_path = "/dbfs/Paula/data/sib_engLatn_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_mt5large_sib"
epochs = 80
lora_rs = [0]  # placeholder
training_types = [TrainingType.FULL_FINETUNE]

learning_rate = 1e-4

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **BLOOM-1B7** full model finetuning with **PAWS-X**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-1b7"
task_name = "xpaws"
dataset_train_path = "/dbfs/Paula/data/xpaws_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xpaws_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom1b7_xpaws"
epochs = 2
lora_rs = [0]  # placeholder
training_types = [TrainingType.FULL_FINETUNE]

learning_rate = 1e-5

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **BLOOM-1B7** full model finetuning with **XNLI**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-1b7"
task_name = "xnli"
dataset_train_path = "/dbfs/Paula/data/xnli_en_train.json"
dataset_val_path = "/dbfs/Paula/data/xnli_en_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom1b7_xnli"
epochs = 2
lora_rs = [0]  # placeholder
training_types = [TrainingType.FULL_FINETUNE]

learning_rate = 1e-5

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **BLOOM-1B7** full finetuning with **SIB200**

# COMMAND ----------

pretrained_model_id = "bigscience/bloom-1b7"
task_name = "sib"
dataset_train_path = "/dbfs/Paula/data/sib_engLatn_train.json"
dataset_val_path = "/dbfs/Paula/data/sib_engLatn_validation.json"
output_dir_base = "/dbfs/Paula/output/exp_bloom1b7_sib"
epochs = 80
lora_rs = [0]  # placeholder
training_types = [TrainingType.FULL_FINETUNE]

learning_rate = 1e-5

setup_and_run_experiment(
    pretrained_model_id,
    task_name,
    dataset_train_path,
    dataset_val_path,
    output_dir_base,
    lora_rs,
    training_types,
    epochs,
    learning_rate
)
