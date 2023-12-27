import os
from pathlib import Path

from datasets import Dataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer

from openrlhf.utils import DeepspeedStrategy

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_sp_tokens(args):
    sp_tokens = dict()
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        sp_token = getattr(args, key, None)
        if sp_token is not None:
            sp_tokens[key] = sp_token
    return sp_tokens


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    sp_tokens = get_sp_tokens(strategy.args)

    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, **sp_tokens)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    model.resize_token_embeddings(len(tokenizer))

    return tokenizer


def get_strategy(args):
    # default args for deepspeed
    if "seed" not in args:
        args.seed = 42
    if "max_norm" not in args:
        args.max_norm = 1.0
    if "micro_train_batch_size" not in args:
        args.micro_train_batch_size = 1
    if "train_batch_size" not in args:
        args.train_batch_size = 8
    if "local_rank" not in args:
        args.local_rank = -1
    if "bf16" not in args:
        args.bf16 = True
    if "inference_tp_size" not in args:
        args.inference_tp_size = 1
    if "adam_offload" not in args:
        args.adam_offload = False
    if "zpg" not in args:
        args.zpg = 1
    # max_out_tokens for DS inference
    if "max_len" in args and args.max_len is not None:
        args.max_out_tokens = args.max_len
    elif "generate_max_len" in args and "prompt_max_len" in args:
        args.max_out_tokens = args.prompt_max_len + args.generate_max_len
    else:
        args.max_out_tokens = 2048

    strategy = DeepspeedStrategy(
        seed=args.seed,
        max_norm=args.max_norm,
        micro_train_batch_size=args.micro_train_batch_size,
        train_batch_size=args.train_batch_size,
        zero_stage=args.zero_stage,
        max_out_tokens=args.max_out_tokens,
        inference_tp_size=args.inference_tp_size,
        args=args,
    )

    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=2000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        dataset_subfold_list = dataset.split("@")
        strategy.print(f"dataset: {dataset}")
        # local dir with python script or common local file
        if os.path.isdir(os.path.join(os.getcwd(), dataset)) or dataset.endswith(
            (".json", ".jsonl", ".csv", ".parquet", ".txt")
        ):
            if dataset.endswith((".json", ".jsonl", ".csv", ".parquet", ".txt")):
                files = dataset
                data_type = os.path.splitext(files)[1][1:]
            else:
                path = Path(dataset)
                script = [str(file.resolve()) for file in Path(path).rglob("*.py")]
                extensions = ("*.json", "*.jsonl", "*.csv", "*.parquet", "*.txt")
                files = [str(file) for ext in extensions for file in Path(path).rglob(ext)]
                strategy.print(f"script: {script}")
                strategy.print(f"files: {files}")
                # For dir, follow python script or first file type
                data_type = script[0] if len(script) == 1 else os.path.splitext(files[0])[1][1:]
            # reformat data type
            if data_type in ["json", "jsonl"]:
                data_type = "json"
            elif data_type == "txt":
                data_type = "text"
            elif data_type.endswith(".py"):
                # load local dir with python script
                files = None
            if data_type.endswith(".py"):
                strategy.print(f"load {dataset} with script {data_type}")
            else:
                strategy.print(f"load {files} from {dataset}")
            data = load_dataset(data_type, data_files=files)
        elif len(dataset_subfold_list) == 2:
            dataset = dataset_subfold_list[0]
            subfold = dataset_subfold_list[1]
            data = load_dataset(dataset, data_dir=subfold.strip())
        elif len(dataset_subfold_list) == 1:
            dataset = dataset_subfold_list[0]
            data = load_dataset(dataset)
        else:
            raise Exception(f"Dataset Name {dataset}: Format error")

        if "train" in data:
            train_data_list.append(data["train"].select(range(min(max_count, len(data["train"])))))
        else:
            train_data_list.append(data.select(range(min(max_count, len(data)))))  # train will contains eval? TODO

        if return_eval:
            if "test" in data:
                eval_data = data["test"].select(range(min(int(max_count * 0.1), len(data["test"]))))
            elif "validation" in data:
                eval_data = data["validation"].select(range(min(int(max_count * 0.1), len(data["validation"]))))
            elif "train" in data:
                eval_data = data["train"].select(range(min(int(max_count * 0.1), int(len(data["train"]) * 0.01))))
            else:
                eval_data = data.select(range(min(int(max_count * 0.1), int(len(data) * 0.001))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset
