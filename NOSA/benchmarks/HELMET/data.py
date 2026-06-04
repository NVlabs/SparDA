# This file is copied/adapted from HELMET (https://github.com/princeton-nlp/HELMET).
# Copyright (c) 2024 Princeton Natural Language Processing.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

import json
import copy
import math
import random
import numpy as np
import hashlib
import os
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any

import datasets
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import re
from utils import calculate_metrics, parse_output, parse_rankings, calculate_retrieval_metrics

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_module_cache_cleared = False


def _hf_lock_root():
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    lock_dir = os.path.join(hf_home, "dataset_load_locks")
    os.makedirs(lock_dir, exist_ok=True)
    return lock_dir


@contextmanager
def _acquire_dataset_load_lock(dataset_id):
    import fcntl

    lock_name = hashlib.sha1(dataset_id.encode("utf-8")).hexdigest() + ".lock"
    lock_path = os.path.join(_hf_lock_root(), lock_name)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

def _clear_hf_module_cache():
    """Remove corrupted HF datasets module cache.

    When switching ``datasets`` library versions (e.g. from >=3.0 to 2.x), the
    cached loading-script modules may be gzip-compressed in a format the current
    version cannot read, causing ``UnicodeDecodeError: can't decode byte 0x8b``.
    Deleting the cache forces a clean re-download.
    """
    import shutil
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_dir = os.path.join(hf_home, "modules", "datasets_modules")
    if os.path.isdir(cache_dir):
        logger.info("Clearing corrupted datasets module cache: %s", cache_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)


def _clear_hub_dataset_cache(dataset_id):
    """Remove the Hub cache entry for a specific dataset."""
    import shutil
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
    cache_name = "datasets--" + dataset_id.replace("/", "--")
    cache_path = os.path.join(hub_dir, cache_name)
    if os.path.isdir(cache_path):
        logger.info("Clearing corrupted hub cache for %s: %s", dataset_id, cache_path)
        shutil.rmtree(cache_path, ignore_errors=True)


def _strip_unsupported_kwargs(kwargs):
    """Remove kwargs that newer datasets versions reject."""
    stripped = dict(kwargs)
    stripped.pop("trust_remote_code", None)
    return stripped


def _config_name_from_args(args, kwargs):
    config_name = kwargs.get("name")
    if config_name is None and args and isinstance(args[0], str):
        config_name = args[0]
    return config_name


def _infer_parquet_split(path):
    parts = path.split("/")
    for part in parts[:-1]:
        if part in {"train", "validation", "test"}:
            return part

    filename = parts[-1]
    for split_name in ("train", "validation", "test"):
        if filename.startswith(split_name):
            return split_name
    return None


def _load_hf_dataset_from_parquet(dataset_id, *args, **kwargs):
    """Fallback loader for Hub datasets whose loading scripts are disabled."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise RuntimeError(
            f"Cannot load '{dataset_id}' from parquet fallback because huggingface_hub "
            "is unavailable."
        ) from e

    config_name = _config_name_from_args(args, kwargs)
    requested_split = kwargs.get("split")
    features = kwargs.get("features")
    last_error = None

    for revision in ("refs/convert/parquet", None):
        try:
            repo_files = list_repo_files(
                dataset_id,
                repo_type="dataset",
                revision=revision,
            )
        except Exception as e:
            last_error = e
            continue

        split_files = {}
        for repo_file in repo_files:
            if not repo_file.endswith(".parquet"):
                continue

            parts = repo_file.split("/")
            if config_name is not None and config_name not in parts:
                continue

            split_name = _infer_parquet_split(repo_file)
            if split_name is None:
                continue
            if requested_split is not None and split_name != requested_split:
                continue

            split_files.setdefault(split_name, []).append(repo_file)

        if requested_split is not None:
            expected_splits = [requested_split]
        else:
            expected_splits = [name for name in ("train", "validation", "test") if split_files.get(name)]

        if not expected_splits:
            last_error = RuntimeError(
                f"No parquet files found for dataset '{dataset_id}'"
                + (f" config '{config_name}'" if config_name is not None else "")
            )
            continue

        data_files = {}
        try:
            for split_name in expected_splits:
                files = sorted(split_files.get(split_name, []))
                if not files:
                    raise RuntimeError(
                        f"Missing parquet files for split '{split_name}' in dataset '{dataset_id}'"
                    )
                data_files[split_name] = [
                    hf_hub_download(
                        repo_id=dataset_id,
                        filename=repo_file,
                        repo_type="dataset",
                        revision=revision,
                    )
                    for repo_file in files
                ]
        except Exception as e:
            last_error = e
            continue

        load_kwargs = {"data_files": data_files}
        if features is not None:
            load_kwargs["features"] = features
        if requested_split is not None:
            load_kwargs["split"] = requested_split

        logger.info(
            "Loading '%s' via parquet fallback%s%s",
            dataset_id,
            f" (config={config_name})" if config_name is not None else "",
            f" from revision '{revision}'" if revision is not None else "",
        )
        return load_dataset("parquet", **load_kwargs)

    raise RuntimeError(
        f"Unable to resolve a parquet fallback for dataset '{dataset_id}'"
        + (f" config '{config_name}'" if config_name is not None else "")
    ) from last_error


def _load_hf_dataset(dataset_id, *args, **kwargs):
    """Load a HF dataset, with defensive fallbacks for HF cache/script issues."""
    global _module_cache_cleared
    def _load_with_relaxed_default_config(**local_kwargs):
        """Bypass fragile DEFAULT_CONFIG_NAME probe in some datasets versions."""
        split = local_kwargs.pop("split", None)
        download_mode = local_kwargs.pop("download_mode", None)

        try:
            builder = datasets.load_dataset_builder(
                dataset_id,
                *args,
                download_mode=download_mode,
                _require_default_config_name=False,
                **local_kwargs,
            )
        except TypeError:
            if download_mode is not None:
                local_kwargs["download_mode"] = download_mode
            return load_dataset(dataset_id, *args, **local_kwargs)

        builder.download_and_prepare(download_mode=download_mode)
        return builder.as_dataset(split=split)

    with _acquire_dataset_load_lock(dataset_id):
        try:
            return load_dataset(dataset_id, *args, **kwargs)
        except (ValueError, TypeError) as e:
            if "trust_remote_code" in str(e):
                logger.info("Retrying '%s' without trust_remote_code", dataset_id)
                kwargs = _strip_unsupported_kwargs(kwargs)
                return load_dataset(dataset_id, *args, **kwargs)
            raise
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" in str(e):
                logger.warning(
                    "Dataset script loading is disabled for '%s'; retrying via parquet mirror",
                    dataset_id,
                )
                return _load_hf_dataset_from_parquet(dataset_id, *args, **kwargs)
            raise
        except UnicodeDecodeError:
            logger.warning("UnicodeDecodeError loading '%s' – clearing caches and retrying", dataset_id)
            if not _module_cache_cleared:
                _clear_hf_module_cache()
                _module_cache_cleared = True
            _clear_hub_dataset_cache(dataset_id)
            retry_kwargs = _strip_unsupported_kwargs(kwargs)
            retry_kwargs["download_mode"] = "force_redownload"
            try:
                return load_dataset(dataset_id, *args, **retry_kwargs)
            except UnicodeDecodeError:
                logger.warning(
                    "UnicodeDecodeError persists for '%s' after force redownload; "
                    "retrying with relaxed default-config loading path",
                    dataset_id,
                )
                return _load_with_relaxed_default_config(**retry_kwargs)


def filter_contexts(data):
    # filter the contexts and only keep the ones that contain the answer
    new_data = []
    for d in data:
        d = copy.deepcopy(d)
        d["ctxs"] = [ctx for ctx in d["ctxs"] if ctx["has_answer"]]
        if len(d["ctxs"]) > 0:
            d["gold_doc"] = d["ctxs"][0]["text"]
            d["gold_title"] = d["ctxs"][0]["title"]
            new_data.append(d)
    return new_data


def drop_duplicates(data, key="id"):
    indices_to_keep = []
    keys = set()
    for i, d in enumerate(data):
        if d[key] in keys:
            continue
        indices_to_keep.append(i)
        keys.add(d[key])
    data = data.select(indices_to_keep)
    return data


def load_qa(dataset, path, demo_path, max_test_samples=None, popularity_threshold=None, shots=0):
    """
    Load the data for QA tasks
    """
    if "nq_bad" in dataset:
        user_template = "Use the given documents to write a concise and short answer to the question. Only use the information presented in the documents, and output 'unanswerable' if the question is not valid or cannot be answered with the given document. Write your answer in the following format:\nAnswer: [answer]\n\n{demos}{context}\n\nQuestion: {question}"
    else:
        user_template = "Use the given documents to write a concise and short answer to the question. Write your answer in the following format:\nAnswer: [answer]\n\n{demos}{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    if path.endswith(".json"):
        data = load_dataset("json", data_files=path, field="data")["train"]
    elif path.endswith(".jsonl"):
        data = load_dataset("json", data_files=path)["train"]
    else:
        data = load_from_disk(path)
        return {"data": data, "prompt_template": prompt_template, "user_template": user_template, "system_template": system_template}

    if demo_path.endswith(".json"):
        if "nq_bad" in dataset:
            with open(demo_path) as f:
                demo_data = json.load(f)
        else:
            demo_data = load_dataset("json", data_files=demo_path, field="data")["train"]
    else:
        demo_data = load_dataset("json", data_files=demo_path)["train"]

    # popularity filtering for popqa
    if "popqa" in dataset and popularity_threshold is not None:
        data = data.filter(lambda x: math.log10(x['s_pop']) < popularity_threshold)
        demo_data = demo_data.filter(lambda x: math.log10(x['s_pop']) < popularity_threshold)

    key = "id" if "id" in data.column_names else "question"
    if max_test_samples is not None:
        # some datasets do not have id (e.g., nq), so we assume unique questions
        keys = set(data[key])
        keys = random.sample(sorted(keys), min(max_test_samples, len(keys)))
        data = data.filter(lambda x: x[key] in keys)

    # demo_template = "Document (Title: {gold_title}): {gold_doc}\n\nQuestion: {question}\nAnswer: {answer}"
    demo_template = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"
    passage_template = "Document (Title: {title}): {text}"
    def update(sample):
        demos = demo_data
        demo_text = ""
        if shots > 0:
            if 'popqa' in dataset:
                # popqa only has one split
                demos = demo_data.filter(lambda x: x[key] != sample[key])

            # seed ensures that we get the same demos for the same question
            # hashlib is deterministic while hash() is not in Python>=3.3, the seed has to be a positive integer
            h = int(hashlib.sha256(str(sample[key]).encode("utf-8")).hexdigest(), 16) % 2**31
            demos = demos.shuffle(seed=h)
            demos = drop_duplicates(demos, key).select(range(shots))
            demo_text = "\n\n".join([demo_template.format(**d, documents="\n\n".join([passage_template.format(**c) for c in d["ctxs"]]), answer=d["answers"][0]) for d in demos]) + "\n\n"
        passage_text = ""
        if len(sample['ctxs']) > 0:
            passage_text = "\n\n".join([passage_template.format(**c) for c in sample['ctxs']])
        return {"demos": demo_text, "context": passage_text, "answer": sample["answers"]}
    data = data.map(update)

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
    }


def load_json_kv(path, shots, max_test_samples=None, seed=42):
    # prompt from https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/prompts/kv_retrieval.prompt
    user_template = "{context}\n\nExtract the value corresponding to the specified key in the JSON object below.\n\n{demos}Key: {question}"
    system_template = "Corresponding value:"
    prompt_template = user_template + "\n" + system_template

    if path.endswith(".json"):
        data = load_dataset("json", data_files=path, field="data")["train"]
    elif path.endswith(".jsonl"):
        data = load_dataset("json", data_files=path)["train"]
    else:
        data = load_from_disk(path)
        return {"data": data, "prompt_template": prompt_template, "user_template": user_template, "system_template": system_template}

    demo_template = "Key: {key}\nCorresponding value:{value}"
    data = data.map(lambda x: {
        "demos": "\n\n".join([demo_template.format(key=key, value=" "+value) for key, value in x["demos"][:shots]]) + ("\n\n" if shots > 0 else ""),
        "k": x["num_kvs"],
    })

    if max_test_samples is not None:
        data = data.shuffle(seed=seed).select(range(min(max_test_samples, len(data))))

    def post_process(output, example):
        prediction = output["output"]
        answer = example["answer"]
        mets = calculate_metrics(prediction, answer)
        # we don't really need to parse because we ues substring em, but could be nice to see how precise the model is
        parsed_pred = parse_output(prediction, "corresponding value:")
        new_mets = calculate_metrics(parsed_pred, answer)
        mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
        return mets, {"parsed_output": parsed_pred}

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


def _load_length_tokenizer(tokenizer_path):
    if not tokenizer_path:
        raise ValueError("A tokenizer path is required for HELMET length filtering.")
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def truncate_llama2(dataset, data, tokenizer_path, postfix_text=" ... [the rest of the text is omitted]"):
    # Truncate only the main document (context), excluding instructions and demos.
    # Use the evaluated model tokenizer so release runs do not require gated Llama 2 access.
    max_length = int(dataset.split("_")[-1])
    tokenizer = _load_length_tokenizer(tokenizer_path)
    separator_length = len(tokenizer(postfix_text)["input_ids"])

    def truncate(sample):
        tokens = tokenizer(sample["context"], return_offsets_mapping=True)
        if len(tokens["input_ids"]) > max_length:
            # truncate
            sample["context"] = sample["context"][:tokens["offset_mapping"][max_length-separator_length][1]] + postfix_text
        return sample
    return data.map(truncate, num_proc=16)


def filter_length(data, min_length, key, tokenizer_path):
    tokenizer = _load_length_tokenizer(tokenizer_path)
    data = data.filter(lambda x: len(tokenizer(x[key])['input_ids']) >= min_length, num_proc=32)
    return data


def load_narrativeqa(dataset, shots=0, max_samples=None, seed=42, tokenizer_path=None):
    user_template = "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\n{demo}{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    all_data = _load_hf_dataset("narrativeqa")
    data = all_data["test"].shuffle(seed=seed)
    
    # filter for a specific length
    tokenizer = _load_length_tokenizer(tokenizer_path)
    data = data.map(lambda x: {'input_length': len(tokenizer(x['document']['text'])['input_ids'])})
    data = data.filter(lambda x: x['input_length'] > 131072) # this should yield 1330 samples
    data = data.remove_columns("input_length")
    
    data = data.map(lambda example: {
        "context": example["document"]["text"],
        "question": example["question"]["text"],
        "answer": [ex["text"] for ex in example["answers"]],
        "demo": "" if shots == 0 else "For example:\n\n" + "\n\n".join([f"Question: {ex['question']['text']}\nAnswer: {ex['answers'][0]['text']}" for ex in all_data["train"].shuffle().select(range(shots))]) + "\n\nNow, use the following story to answer the question:\n\n"
    }, remove_columns=["document", "answers"])

    data = filter_length(data, 131072, "context", tokenizer_path)
    data = truncate_llama2(dataset, data, tokenizer_path)
    if max_samples is not None:
        data = data.select(range(min(max_samples, len(data))))

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
    }


def load_multi_lexsum(dataset, shots=0, max_samples=None, seed=42, tokenizer_path=None):
    all_data = _load_hf_dataset("allenai/multi_lexsum", name="v20230518")
    all_data = all_data.filter(lambda x: x["summary/short"] is not None)

    user_template = "You are given the legal documents in a civil rights lawsuit, and you are tasked to summarize the case. Write a concise summary of one paragraph (200 to 250 words). The summary should contain a short description of the background, the parties involved, and the outcomes of the case.\n\n{demo}Legal documents:\n{context}\n\nNow please summarize the case."
    system_template = "Summary:"
    prompt_template = user_template + "\n\n" + system_template
    train_data = all_data["train"]

    all_data = all_data.map(lambda x: {
        "context": '\n\n'.join(x["sources"]),
        "demo": "" if shots == 0 else "Example summaries:\n\n" + "\n\n".join(["Summary: {}".format(ex["summary/short"]) for ex in train_data.shuffle().select(range(shots))]) + "\n\nNow, write a summary of the following legal documents.\n",
        "answer": x["summary/short"],
        "question": "",
    })

    test_data = all_data["validation"]
    test_data = filter_length(test_data, 65536, "context", tokenizer_path)
    test_data = truncate_llama2(dataset, test_data, tokenizer_path)

    if max_samples is not None and len(test_data) > max_samples:
        test_data = test_data.shuffle(seed=seed).select(range(max_samples))

    def post_process(output, example):
        prediction = output["output"]
        answer = example["answer"]
        mets = calculate_metrics(prediction, answer)
        # we don't really need to parse because we ues substring em, but could be nice to see how precise the model is
        parsed_pred = parse_output(prediction, system_template)
        if parsed_pred is not None:
            new_mets = calculate_metrics(parsed_pred, answer)
            mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
        return mets, {"parsed_output": parsed_pred}
    
    return {
        "data": test_data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


def load_msmarco_rerank(path, demo_path=None, max_test_samples=None, shots=0, seed=42):
    random.seed(seed)
    user_template = "You are provided with a list of documents, each indicated by their ID. Rank each document based on their relevance to the question in descending order from most relelvant to least relevant texts. Include all documents in the rankings. Write your answer using the unique IDs, with the following format:\nRanking: ID3 > ID1 > ID2\n\n{demos}{context}\n\nQuery: {question}"
    system_template = "Ranking:"
    prompt_template = user_template + "\n" + system_template

    if path.endswith(".jsonl"):
        # we have preprocessed it into a jsonl file
        data = load_dataset("json", data_files=path)["train"]
    else:
        data = load_from_disk(path)

    demos = load_dataset("json", data_files=demo_path)["train"]

    if max_test_samples is not None:
        key = "qid" if "qid" in data.column_names else "query"
        keys = set(data[key])
        keys = random.sample(sorted(keys), min(max_test_samples, len(keys)))
        data = data.filter(lambda x: x[key] in keys)

    # the k values are used to calculate metrics later
    k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    k_values = [k for k in k_values if k <= len(data[0]["ctxs"])]

    # could also do this question by question, but not necessary if we are sampling
    demo_filtered = False
    if len(demos) > 2*len(data):
        qids = set(data["qid"])
        demos = demos.filter(lambda x: x["qid"] not in qids)
        demo_filtered = True

    def update(sample, demos):
        passage_text = ""

        passage_template = "[ID: {id}] Document (Title: {title}): {text}"  if "title" in sample["ctxs"][0] else "[ID: {id}] Document: {text}"
        passage_text = "\n\n".join([passage_template.format(**c) for c in sample['ctxs']])
        gold_ranking = " > ".join([x['id'] for x in sorted(sample["ctxs"], key=lambda x: x["label"], reverse=True)])
        demo_text = ""

        if shots > 0:
            # need to make sure we don't pick the same question as the demos
            if not demo_filtered:
                demos = demos.filter(lambda x: x["qid"] != sample["qid"])
            # hashlib is deterministic while hash() is not in Python>=3.3, the seed has to be a positive integer
            h = abs(int(hashlib.sha256(sample["qid"].encode("utf-8")).hexdigest(), 16) % 2**31)
            demo = demos.shuffle(seed=h)
            demo = drop_duplicates(demo, 'qid').select(range(shots))

            demo_ids = set()
            for d in demo:
                if d["qid"] in demo_ids or len(demo_ids) >= shots:
                    continue
                demo_ids.add(d["qid"])
                # sort ids by label
                ids = sorted(d["ctxs"], key=lambda x: x["label"], reverse=True)
                ranking = " > ".join([x['id'] for x in ids])
                demo_text += "\n\n".join([passage_template.format(**c) for c in d['ctxs']]) + f"\n\nQuery: {d['query']}\nRanking: {ranking}" + "\n\n"

        qrel = [[c['id'], str(c['label'])] for c in sample["ctxs"]]
        return {"context": passage_text, "question": sample["query"], "demos": demo_text, "answer": gold_ranking, "qrel": qrel}

    data = data.map(lambda x: update(x, demos), remove_columns=["query", "ctxs"])

    def post_process(output, example):
        parsed_pred = parse_rankings(output["output"])
        o = {"parsed_output": parsed_pred}
        qrels = {example["qid"]: {c[0]: int(c[1]) for c in example["qrel"]}}
        mets = calculate_retrieval_metrics(results={example['qid']: parsed_pred}, qrels=qrels, k_values=k_values)
        mets = {**mets, "num_preds": len(parsed_pred)}
        return mets, o

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "k_values": k_values,
        "post_process": post_process,
    }


def load_icl(dataset, max_test_sample=None, seed=42):
    shot = int(dataset.split("shot")[0].split("_")[-1])

    if "trec_fine" in dataset.lower():
        train_data = _load_hf_dataset("CogComp/trec")["train"]
        test_data = _load_hf_dataset("CogComp/trec")["test"]
        id2label = train_data.features['fine_label'].names
        text_field = "text"
        label_field = "fine_label"
        num_labels = 50
    elif "trec_coarse" in dataset.lower():
        train_data = _load_hf_dataset("CogComp/trec")["train"]
        test_data = _load_hf_dataset("CogComp/trec")["test"]
        id2label = train_data.features['coarse_label'].names
        text_field = "text"
        label_field = "coarse_label"
        num_labels = 6
    elif "banking77" in dataset.lower():
        train_data = _load_hf_dataset("PolyAI/banking77")["train"]
        test_data = _load_hf_dataset("PolyAI/banking77")["test"]
        id2label = train_data.features["label"].names
        id2label = {i: id2label[i] for i in range(len(id2label))}
        text_field = "text"
        label_field = "label"
        num_labels = 77
    elif "clinic150" in dataset.lower():
        train_data = _load_hf_dataset("clinc_oos", "plus")["train"]
        test_data = _load_hf_dataset("clinc_oos", "plus")["validation"]
        id2label = train_data.features["intent"].names
        text_field = "text"
        label_field = "intent"
        num_labels = 151
    elif "nlu" in dataset.lower():
        data = _load_hf_dataset("xingkunliuxtracta/nlu_evaluation_data")["train"]
        id2label = data.features["label"].names
        data = data.train_test_split(test_size=0.1, seed=seed)
        train_data = data["train"]
        test_data = data["test"]
        text_field = "text"
        label_field = "label"
        num_labels = 68
    else:
        raise NotImplementedError(f"Unknown ICL dataset")
   
    def build_label_buckets(data, sample_factory=None):
        sample_factory = sample_factory or (lambda x: x)
        label_buckets = {}
        for sample in data:
            label = int(sample[label_field])
            label_buckets.setdefault(label, []).append(sample_factory(sample))
        return label_buckets

    def balance_labels_from_buckets(label_buckets, shots, seed):
        # for each data point, sample a roughly label-balanced demo set.
        rand = random.Random(seed)
        if shots <= 0 or not label_buckets:
            return []

        num_rounds = math.ceil(shots / len(label_buckets))
        new_data = [[] for _ in range(num_rounds)]
        for label in sorted(label_buckets):
            samples = label_buckets[label]
            if not samples:
                continue

            indices = []
            while len(indices) < num_rounds:
                indices += rand.sample(range(len(samples)), min(num_rounds - len(indices), len(samples)))

            for i, idx in enumerate(indices):
                new_data[i].append(samples[idx])

        for i in range(len(new_data)):
            # this shuffles the order of the labels within each set
            rand.shuffle(new_data[i])
        new_data = [item for sublist in new_data for item in sublist][:shots]
        return new_data

    def sample_examples(examples, shots, seed):
        rand = random.Random(seed)
        selected = []
        if shots <= 0 or not examples:
            return selected

        while len(selected) < shots:
            selected += rand.sample(examples, min(len(examples), shots - len(selected)))
        return selected

    train_examples = [
        (sample[text_field], int(sample[label_field]))
        for sample in train_data
    ]
    train_label_buckets = build_label_buckets(
        train_data,
        sample_factory=lambda sample: (sample[text_field], int(sample[label_field])),
    )

    if max_test_sample is not None and len(test_data) > max_test_sample:
        test_data = test_data.shuffle(seed=seed)
        # we also balance the output labels
        test_label_buckets = build_label_buckets(test_data)
        test_data = balance_labels_from_buckets(test_label_buckets, max_test_sample, seed)
        test_data = datasets.Dataset.from_list(test_data)

    item_template = "{text}\nlabel: {label}"
    user_template = "Use the provided mapping from the text to label to assign a label to the text. Only output \"label: {{label}}\" and nothing else. \n\n{context}\n\n{question}"
    system_template = "label:"
    prompt_template = user_template + "\n" + system_template

    def preprocess(sample):
        # use a different seed for every sample, but is also deterministic and affected by the set seed
        local_seed = (int(hashlib.sha256(sample[text_field].encode("utf-8")).hexdigest(), 16) + seed) % 2**31
        if "balance" in dataset:
            demos = balance_labels_from_buckets(train_label_buckets, shot, local_seed)
        else:
            demos = sample_examples(train_examples, shot, local_seed)

        if "natural_label" in dataset:
            label_mapping = [id2label[i] for i in range(num_labels)]
        else:
            # we map the labels to a random integer
            label_mapping = list(range(num_labels))
            random.Random(local_seed).shuffle(label_mapping)

        context = "\n\n".join([
            item_template.format(text=selected_text, label=str(label_mapping[selected_label]))
            for selected_text, selected_label in demos]
        )
        return {"context": context, "question": sample[text_field], "answer": str(label_mapping[int(sample[label_field])])}

    final_data = test_data.map(preprocess, num_proc=40)

    def post_process(output, example):
        prediction = output["output"]
        answer = example["answer"]
        prediction = parse_output(prediction, system_template)
        mets = calculate_metrics(prediction, answer)
        return mets, {"parsed_output": prediction}

    return {
        "data": final_data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


def load_ruler(dataset, path, max_test_samples=None, seed=42):
    data = load_dataset("json", data_files=path)["train"]
    user_template = "{context}\n\n{question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    # https://github.com/hsiehjackson/RULER/blob/main/scripts/data/synthetic/constants.py
    if "niah_mv" in dataset or "niah_mq" in dataset:
        user_template = "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
        system_template = "The special magic {type_needle_v} for {query} mentioned in the provided text are"
    elif "niah" in dataset:
        user_template = "A special magic {type_needle_v} is hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat is the special magic {type_needle_v} for {query} mentioned in the provided text?"
        system_template = "The special magic {type_needle_v} for {query} mentioned in the provided text is"
    elif "vt" in dataset:
        user_template = "{example}Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above."
        system_template = "Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assigned the value {query}, they are:"
    elif "cwe" in dataset:
        user_template = "{example}Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?"
        system_template = "Answer: The top 10 words that appear most often in the list are:"
    elif "fwe" in dataset:
        user_template = "Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words.\n{context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?"
        system_template = "Answer: According to the coded text above, the three most frequently appeared words are:"
    elif "qa" in dataset:
        # note that for qa, instead of calculating the recall, we simply check for substring exact match
        user_template = "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {question}"
        system_template = "Answer:"
    else:
        raise NotImplementedError(f"Unknown ruler dataset {dataset}")
    prompt_template = user_template + "\n" + system_template

    def process_example(example):
        return {
            "question": example["query"] if "query" in example else example["question"] if "question" in example else "",
            "example": example["example"] + "\n\n" if "example" in example and example["example"] != "" else "",
            "answer": example["answer"] if "answer" in example else example['outputs'],
        }
    data = data.map(process_example)

    def post_process(output, example):
        # we don't do any parsing since we are only checking for substring exact match
        prediction = output["output"]
        answer = example["answer"]
        recall = sum([a.lower() in prediction.lower() for a in answer]) / len(answer)
        mets = {"ruler_recall": recall}
        return mets, {"parsed_output": prediction}

    if max_test_samples is not None:
        data = data.shuffle(seed).select(range(min(len(data), max_test_samples)))

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process if "qa" not in dataset else default_post_process,
    }


def load_alce(dataset, path, demo_path, shots=0):
    # demo path is the prompt file
    with open(demo_path, "r") as f:
        demos = json.load(f)
    instruction = demos["instruction"]
    demo_prompt = demos["demo_prompt"]
    doc_prompt = demos["doc_prompt"]
    # there are 5 docs for each demo, and we use all of them

    user_template = "{demo_text}{instruction}\n\nQuestion: {question}\n\n{context}"
    system_template = "Answer:"
    prompt_template = user_template + "\n\n" + system_template

    data = load_dataset("json", data_files=path)["train"]

    num_docs = int(dataset.split("_")[-1])

    def preprocess_example(example):
        context = "\n\n".join([doc_prompt.format(**d, ID=idx+1) for idx, d in enumerate(example["docs"][:num_docs])])
        demo_text = "\n\n\n".join([
            demo_prompt.format(**demo, instruction=instruction, context = "\n\n".join([doc_prompt.format(**d, ID=idx+1) for idx, d in enumerate(demo["docs"])]))
            for demo in random.sample(demos["demos"], shots)
        ])
        if shots > 0:
            demo_text += "\n\n\n"
        return {"context": context, "demo_text": demo_text, "instruction": instruction}
    data = data.map(preprocess_example)

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
    }


def load_infbench(dataset, shots=0, max_test_samples=None, seed=42, tokenizer_path=None):
    from datasets import Value, Sequence, Features
    ft = Features({"id": Value("int64"), "context": Value("string"), "input": Value("string"), "answer": Sequence(Value("string")), "options": Sequence(Value("string"))})
    data = _load_hf_dataset("xinrongzhang2022/infinitebench", features=ft)

    # https://github.com/OpenBMB/InfiniteBench/blob/main/src/prompt.py
    # slightly modified to be consistent with other datasets, shouldn't affect performance
    post_process = default_post_process
    if "qa_eng" in dataset:
        user_template = "You are given a story and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\n{demo}{context}\n\nQuestion: {question}"
        system_template = "Answer:"
        data = data["longbook_qa_eng"]
    elif "choice_eng" in dataset:
        user_template = "You are given a story and a question with multiple choices. Choose the best answer from the options provided. Only one of the following options is correct, output the answer using one single letter (A, B, C, or D). Don't say anything else.\n\n{demo}{context}\n\nQuestion: {question}\nOptions:\n{options}"
        system_template = "Answer:"
        data = data["longbook_choice_eng"]

        def choice_post_process(output, example):
            prediction = output["output"]
            answer = example["answer"]
            mets = calculate_metrics(prediction, answer)
            mets.pop("substring_exact_match")

            parsed_pred = parse_output(prediction)
            if parsed_pred is not None:
                new_mets = calculate_metrics(parsed_pred, answer)
                new_mets.pop("substring_exact_match")
                mets = {k: max(v, new_mets[k]) for k, v in mets.items()}

            # we only allow for substring exact match for the second answer (A. option)
            # to make it easier to collect the results, we merge exact_match and substring_exact_match here
            mets["substring_exact_match"] = False
            if answer[1].lower() in prediction.lower():
                # we shouldn't need to do other normalization
                mets["substring_exact_match"] = True
                mets["exact_match"] = True
            return mets, {"parsed_output": parsed_pred}

        post_process = choice_post_process
        
    elif "sum_eng" in dataset:
        user_template = "You are given a book and you are tasked to summarize it. Write a summary of about 1000 to 1200 words. Only write about the plot and characters of the story. Do not discuss the themes or background of the book. Do not provide any analysis or commentary.\n\n{demo}{context}\n\nNow summarize the book."
        system_template = "Summary:"
        data = data["longbook_sum_eng"]
    prompt_template = user_template + "\n\n" + system_template

    def process_example(example):
        update = {"question": example["input"], "demo": ""}
        if "choice" in dataset:
            options = "A. {}\nB. {}\nC. {}\nD. {}".format(*example["options"])
            answer = example["options"].index(example["answer"][0])
            answer = chr(ord("A") + answer)
            update["options"] = options
            update["answer"] = [answer, f"{answer}. {example['answer'][0]}"]
        return update
    data = data.map(process_example)

    def add_demos(example):
        demos = data.filter(lambda x: x["id"] != example["id"]).shuffle(seed=seed).select(range(shots))
        if "qa_eng" in dataset:
            temp = "[story text]\nQuestion: {question}\nAnswer: {answer[0]}"
            demo = "\n\n".join([temp.format(**x) for x in demos])
        elif "choice_eng" in dataset:
            temp = "[story text]\nQuestion: {question}\nOptions:\n{options}\nAnswer: {answer[0]}"
            demo = "\n\n".join([temp.format(**x) for x in demos])
        elif "sum_eng" in dataset:
            demo = "\n\n".join([f"[story text]\nSummary: {x['answer'][0].strip()}" for x in demos])
        return {"demo": f"For example:\n\n{demo}\n\nNow, read the following story:\n\n"}
    if shots > 0:
        data = data.map(add_demos)

    # all samples are already longer than 65536 tokens, but this is just a sanity step
    data = filter_length(data, 65536, "context", tokenizer_path)
    data = truncate_llama2(dataset, data, tokenizer_path)

    if max_test_samples is not None:
        data = data.shuffle(seed=seed).select(range(min(len(data), max_test_samples)))

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


def default_post_process(output, example):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """
    prediction = output["output"]
    answer = example["answer"]
    mets = calculate_metrics(prediction, answer)
    # we check the metrics after parsing and take the max
    parsed_pred = parse_output(prediction)
    if parsed_pred is not None:
        new_mets = calculate_metrics(parsed_pred, answer)
        mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
    return mets, {"parsed_output": parsed_pred}


def load_data(args, dataset, path=None, demo_path=None):
    if "popqa" in dataset:
        popularity_threshold = float(dataset.split("_")[-1])
        data = load_qa(dataset, path, demo_path, max_test_samples=args.max_test_samples, popularity_threshold=popularity_threshold, shots=args.shots)
    elif any([x in dataset for x in ["nq", "hotpotqa", "triviaqa"]]):
        data = load_qa(dataset, path, demo_path, max_test_samples=args.max_test_samples, shots=args.shots)
    elif dataset == "json_kv":
        data = load_json_kv(path, args.shots, args.max_test_samples, args.seed)
    elif "narrativeqa" in dataset:
        data = load_narrativeqa(
            dataset, args.shots, args.max_test_samples, args.seed,
            tokenizer_path=args.model_name_or_path,
        )
    elif "msmarco" in dataset:
        data = load_msmarco_rerank(path, demo_path, args.max_test_samples, args.shots, args.seed)
    elif "alce" in dataset:
        data = load_alce(dataset, path, demo_path, args.shots)
        if args.max_test_samples is not None:
            data["data"] = data["data"].shuffle(seed=args.seed).select(range(min(args.max_test_samples, len(data["data"]))))
    elif "icl" in dataset:
        data = load_icl(dataset, max_test_sample=args.max_test_samples, seed=args.seed)
    elif "multi_lexsum" in dataset:
        data = load_multi_lexsum(
            dataset, args.shots, args.max_test_samples, seed=args.seed,
            tokenizer_path=args.model_name_or_path,
        )
    elif "ruler" in dataset:
        if args.shots != 0:
            logger.info("RULER does not support ICL demos, not using any shots")
        data = load_ruler(dataset, path, args.max_test_samples, seed=args.seed)
    elif "infbench" in dataset:
        data = load_infbench(
            dataset, args.shots, args.max_test_samples, seed=args.seed,
            tokenizer_path=args.model_name_or_path,
        )
    elif any([x in dataset for x in ["html_to_tsv", "pseudo_to_code", "path_traversal", "tom_tracking", "countdown", "travel_planning"]]):
        from longproc_addon.longproc_helmet_loader import load_longproc_data_for_helmet
        data = load_longproc_data_for_helmet(dataset, path=path, max_test_samples=args.max_test_samples, seed=args.seed)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    if "post_process" not in data:
        data["post_process"] = default_post_process

    return data


class TestItemDataset(Dataset):
    """
    data is a dictionary that should contain the "data" field, which is a list of samples
    llm is of type LLM from model_utils
    tokenizer is any callable tokenizer with decode method, but not necessary
    """
    def __init__(self, data: Dict[str, Any], llm, tokenizer=None):
        self.data = data
        self.llm = llm
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data["data"])

    def __getitem__(self, idx):
        inputs = self.llm.prepare_inputs(self.data["data"][idx], self.data)
        original_text = None
        if "input_ids" in inputs:
            original_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        return inputs, original_text
