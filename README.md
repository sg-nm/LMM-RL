# OpenLLaMA2

<div style="font-size: 1.5rem;">
  <a href="./README.md">English</a> |
  <a href="./README_cn.md">Chinese</a>
</div>

</br>

<h1 align="center">OpenLLaMA2</h1>
<div align="center">
  <a href="[https://github.com/openllmai/OpenLLaMA2 ↗](https://github.com/openllmai/OpenLLaMA2)">
    <img src="./docs/imgs/openllama2.png" alt="Logo" height="210">
  </a>

<p align="center">
    <h3>A Deepspeed/Ray based High-performance LLaMA2 SFT/RLHF framework!</h3>
      <a href="https://github.com/openllmai/OpenLLaMA2/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/openllmai/OpenLLaMA2" />
      </a>
      <a href="https://github.com/openllmai/OpenLLaMA2/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/openllmai/OpenLLaMA2?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenLLaMA2/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/openllmai/OpenLLaMA2?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenLLaMA2/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/openllmai/OpenLLaMA2?color=0088ff" />
      <a href="https://github.com/openllmai/OpenLLaMA2/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/openllmai/OpenLLaMA2?color=ccf" />
      </a>
      <br/>
      <em>Open-source ChatGPT / Comprehensive / Lightweight / Easy-to-use</em>
      <br/>
    </p>

</p>
</div>

> **The code is open-source, feel free to use it, contributions are welcome! Note: The license of the model depends on the provider of the model.**

- [💥Latest News](#latest-news)
- [💫OpenLLaMA2](#openllama2-project)
- [💫Features](#features)
- [📄Running Example](#running-llama2-example)
- [📄Result Display](#inference)
- [⛏️Pull Request](#pull-request)
- [💐References & Acknowledgements](#references-&-acknowledgements)
- [🌟Sponsor Us](#sponsor-us)
- [🌈Starchart](#starchart)
- [🏆Contributors](#contributors)

## Latest News

- 2023/9/7: Add pre-commit
- 2023/9/6: Support FlashAttention2
- 2023/8/26: Support wandb logs
- 2023/8/22: Support ceval
- 2023/8/20: Add some PPO vs SFT <a href="./docs/ppo_examples.md">examples</a>
- 2023/8/18: **support LLaMA2 7B PPO training on Single A100**, [add pre-trained SFT/RM checkpoints](https://huggingface.co/chuyi777/openllama2_checkpoint)
- 2023/8/13: LLaMA2 7B + SFT+ RM + PPO + DeepSpeed training features finished
- 2023/07/30: OpenLLaMA2 project officially launched

## OpenLLaMA2 Project

OpenLLaMA2 aims to develop a high-performance distributed LLaMA2 SFT/RLHF training framework.

The sister project of this project is [chinese-llama2 ↗](https://github.com/OpenLLMAI/chinese-llama2), which aims to fine-tune the Chinese LLaMA2 using SFT/RLHF.

### Features

- [✔️] A fast LLaMA2 SFT/PPO Training Framework based on DeepSpeed.
- [✔️] Multi-nodes training scripts for Slurm.
- [✔️] Wandb log.
- [✔️] Support conda env.
- [✔️] FlashAttention2.
- [WIP] Multi-nodes RLHF based on Ray.
- [WIP] Multiple RM models.
- [WIP] Rejection Sampling.
- [WIP] Support QLora.
- [WIP] Support DPO.
- [WIP] Better docs and examples
- [WIP] Develop the [RLHF datasets ↗](https://github.com/OpenLLMAI/OpenLLMData) for Multiple reward models.
- [WIP] Train a [chinese-llama2 ↗](https://github.com/OpenLLMAI/chinese-llama2) RLHF model.
- [TODO] Support Context Distillation.
- [TODO] Inference kernel fusion (such as DS inference)
- [TODO] Large-scale model (> 70B) support with ZeRO++ and FasterTransformer inference.


## Running LLaMA2 Example

* Verified envs

You can build openllama2 from **nvidia-docker(recomended)** or from conda envs.

```shell
Clone the repository: 
git clone https://github.com/openllmai/OpenLLaMA2.git

# Download the pre-trained SFT/RM checkpoints (Optional)
git lfs install
git clone https://huggingface.co/chuyi777/openllama2_checkpoint
```

* Single-node training with nvidia-docker

```shell
cd examples/scripts

# install nvidia-docker (Optional)
./nvidia_docker_install.sh

# launch nvidia container
./docker_run.sh

# cd in container
cd /openllama2/examples/scripts

# build OpenLLaMA2 (i.e, pip install)
./build_openllama2.sh

# huggingface login 
~/.local/bin/huggingface-cli login

# train SFT model
./train_sft_llama.sh

# train RM model
./train_rm_llama.sh

# train PPO model
./train_ppo_llama.sh
```

* Multi-nodes training on Slurm

```shell
cd examples/scripts

# huggingface login on Slurm 
pip install transformers
huggingface-cli login

# Moidfy the Slurm Account/Nodes ... in `train_llama_slurm.sh`

# For SFT, RM, and PPO training stage:
# Modify the variable `training_script` in `train_llama_slurm.sh` to
readonly training_script="train_sft_llama.sh"
readonly training_script="train_rm_llama.sh"
readonly training_script="train_ppo_llama.sh"

# set `GPUS_PER_NODE` in `train_llama_slurm.sh`
readonly GPUS_PER_NODE=8

# run multi-nodes training script
# train_llama_slurm.sh will load the training args from `training_script`
sbatch ./train_llama_slurm.sh
```

* build openllama2 from conda envs 

If you really don't want to use nvidia-docker, we also provide tutorials for building openllama2 from a conda environment. (We prefer nvidia-docker to avoid errors caused by the environment.)
```shell
# we need conda
conda create -n llama2 python=3.10
# so, we need install some package manualy: when installing torch, you may need to match the corresponding cuda version.
pip install packaging ninja
pip install torch --index-url https://download.pytorch.org/whl/cu118
# check ninjia
ninja --version
echo $? 
# install flash-attn: may take some time
pip install flash-attn=2.1.1 --no-build-isolation
./build_openllama2.sh
# enjoy it!
```

## Inference

After completing the training, you can evaluate your model by using the `inference` script:

```shell
./inference_llama.sh { model_path } "Please introduce the GTA5 game."
```

## Pull Request
If you want to contribute code please format the code using the following command,

```
pip install pre-commit
pre-commit install
git add .
git commit -m "xxx"
```

## References & Acknowledgements

We would like to express our gratitude to the following projects and organizations for their contributions to the field of AI and NLP:

- [Hugging Face Transformers ↗](https://github.com/huggingface/transformers)
- [OpenAI GPT ↗](https://github.com/openai/gpt-3)
- [LLaMA2 ↗](https://ai.meta.com/llama/)
- [DeepSpeed ↗](https://github.com/microsoft/DeepSpeed)
- [Ray ↗](https://github.com/ray-project/ray)


### Join Us

**How to Join?**

1. Email us at xianyuai@openllmai.top(official email) or janhu9527@gmail.com/jjgxw@outlook.com(PIC). Please include the following details:
   - Your name
   - Your GitHub username
   - Your areas of interest
   - Your skills and experience related to NLP and/or AI
1. You can also join us through the official GitHub [OpenLLaMA2 ↗](https://github.com/openllmai/OpenLLaMA2) project page. Just create an issue about your interest to contribute and we will get back to you.

**What can you do?**

1. Join the team and participate in the development of the OpenLLaMA2 project.
1. Contribute to the project by submitting pull requests.
1. Help improve documentation, fix bugs, or create new features.
1. Share the project and help us grow the community.

## Sponsor Us

Your sponsorship can help us maintain and improve OpenLLaMA2. If you find this project useful, please consider sponsoring us. You can sponsor us on [Open Collective ↗](https://opencollective.com/openllmai).

## Starchart


[![Star History Chart](https://api.star-history.com/svg?repos=openllmai/OpenLLaMA2&type=Date)](https://star-history.com/#openllmai/OpenLLaMA2&Date)

## Contributors

A big thank you to all our contributors! If you want to contribute, feel free to make a pull request or create an issue.

<a href="https://github.com/openllmai/OpenLLaMA2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openllmai/OpenLLaMA2" />
</a>

## Citation
```
@misc{openllmai23,
   author = {OpenLLMAI},
   title = {OpenLLaMA2},
   year={2023},
   howpublished = {\url{https://github.com/OpenLLMAI/OpenLLaMA2}}
}
```

______________________________________________________________________

*OpenLLaMA2 © 2023 OpenLLMAI. All Rights Reserved.*
