<div align="center">

<h1>NOSA: Native and Offloadable Sparse Attention</h1>


**Boost Decoding Efficiency via High-Locality Offloading**
</div>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/thunlp/NOSA" style="margin: 2px;">
    <img alt="Code" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/collections/openbmb/nosa" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/NOSA-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2510.13602" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-2510.13602-b31b1b.svg" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huangyuxiang03.github.io/blogs_nosa" style="margin: 2px;">
    <img alt="Blog" src="https://img.shields.io/badge/Blog-000000?style=for-the-badge&logo=googlechrome&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Overview

**NOSA** is a trainable sparse attention mechanism designed for KV-cache offloading with an explicit locality constraint, paired with an inference system (**NOSI**) to realize its efficiency. It improves long-context/long-generation quality over prior offloading baselines while boosting decoding throughput by up to **5.04×** vs **FullAttn**, **1.92×** vs **InfLLMv2**, and **1.83×** vs **ShadowKV** on **1B/3B/8B** LLMs.

<img width="4212" height="1200" alt="framework_github" src="https://github.com/user-attachments/assets/7063339f-8c97-4246-bdd0-5bc6026fbe7f" />



## Models

We train 1B, 3B, and 8B models  FullAttn, InfLLMv2, DMA, and NOSA, resulting in a total of 12 models. The following models have been released on Hugging Face.

|Model|Link|
|:-:|:-:|
|NOSA-1B | [NOSA-1B](http://huggingface.co/openbmb/NOSA-1B) |
|NOSA-3B | [NOSA-3B](http://huggingface.co/openbmb/NOSA-3B) |
|NOSA-8B | [NOSA-8B](http://huggingface.co/openbmb/NOSA-8B) |

Please reach out to us if additional baseline models (FullAttn, InfLLMv2, or DMA) are needed. You may open an issue or contact us directly via email (our email addresses are provided in the paper).





## Setup

We set up our experimental environment using uv inside Docker. We do not provide
or update a prebuilt NOSA runtime image in this release; users should build
their own image from `dependencies/Dockerfile.nosa` or follow the manual flow in
`dependencies/README.md`.

After building the environment, activate the NOSA virtual environment. For
vLLM, SGLang, ShadowKV, InfLLM, and ArkVale baselines, use their official
repositories or Docker images; those third-party source trees are not bundled in
this release.
```
source /venv/nosa/bin/activate # for NOSA, FullAttn, InfLLMv2, DMA
```

LM-Harness-Eval is not bundled in this release. For general-task baselines, use
the upstream lm-evaluation-harness project with the model checkpoints released
for NOSA/InfLLMv2/FullAttn/DMA.

Also, please install NOSI as follows.
```
uv pip install ./nosi
```

### Important: `infllm_v2` Runtime

The canonical sparse-attention library is the `infllm_v2` package installed in `/venv/nosa`. Build it from `../infllmv2_cuda_impl` when you update the environment or container, for example:

```
source /venv/nosa/bin/activate
python -m pip install --no-deps --no-build-isolation ../infllmv2_cuda_impl
```

Root training (`training/run_train.sh`, `models/minicpm`, and current `models/nosa`) uses that installed package directly. It does not build or install `infllm_v2` on the fly.

Backend support is intentionally split:

- RULER uses the HF/modeling path. It does not expose `nosi` as a backend.
- LongBench, Reasoning, and HELMET also run through the HF/modeling paths and therefore use the active `/venv/nosa` install.
- `NOSA/nosi` is reserved for the efficiency/inference path only.



## Run Experiments

### Long-Input Evaluation

We run all methods on LongBench and HELMET.

- LongBench
```
cd benchmarks/LongBench

# download test data
bash download_data.sh
# activate the corresponding venv
source /venv/nosa/bin/activate
# run LongBench
python pred.py --model 8b_nosa_sft
python eval.py --model 8b_nosa_sft

cd -
```

- HELMET
```
cd benchmarks/HELMET

# download test data
bash scripts/download_data.sh
# activate the corresponding venv
source /venv/nosa/bin/activate
# run HELMET
python eval.py --output_dir output
bash collect_result.sh

cd -

```

### Decoding Efficiency Tests

Use the unified local efficiency launcher in `benchmarks/Efficiency`.

```
cd benchmarks/Efficiency

# activate the corresponding venv
source /venv/nosa/bin/activate

# examples
bash bench.sh --model-path openbmb/NOSA-8B --seq-len 128K
bash bench.sh --model-path openbmb/MiniCPM4.1-8B --dense -B 4 -L 16K

cd -
```

## Acknowledgment

Some content of this repository are adapted from [LongBench](https://github.com/THUDM/LongBench), [HELMET](https://github.com/princeton-nlp/HELMET), [RULER](https://github.com/NVIDIA/RULER), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [ShadowKV](https://github.com/ByteDance-Seed/ShadowKV), [ArkVale](https://github.com/pku-liang/ArkVale/), [InfLLM](https://github.com/thunlp/InfLLM), and [InfLLMv2](http://github.com/OpenBMB/infllmv2_cuda_impl/).

## Citation

```bibtex
@article{huang2025nosa,
  title={NOSA: Native and Offloadable Sparse Attention},
  author={Huang, Yuxiang and Wang, Pengjie and Han, Jicheng and Zhao, Weilin and Su, Zhou and Sun, Ao and Lyu, Hongya and Zhao, Hengyu and Wang, Yudong and Xiao, Chaojun and Han, Xu and Liu, Zhiyuan},
  journal={arXiv preprint arXiv:2510.13602},
  year={2025}
}
```
