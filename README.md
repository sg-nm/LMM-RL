![](./docs/logo.png)
<div align="center">
<p align="center">
      <a href="https://github.com/openllmai/OpenRLHF/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/openllmai/OpenRLHF" />
      </a>
      <a href="https://github.com/openllmai/OpenRLHF/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/openllmai/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenRLHF/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/openllmai/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenRLHF/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/openllmai/OpenRLHF?color=0088ff" />
      <a href="https://github.com/openllmai/OpenRLHF/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/openllmai/OpenRLHF?color=ccf" />
      </a>
      <br>
      <em>Open-source / Comprehensive / Lightweight / Easy-to-use</em>
    </p>
</p>
</div>

<hr>

OpenRLHF is a high-performance RLHF framework built on Ray, DeepSpeed and HuggingFace Transformers:

- **Simple and easy to use**: OpenRLHF is one of the simplest high-performance RLHF libraries currently available, enabling 34B model RLHF training with just a single DGXA100 node (see the training [script](./examples/scripts/train_ppo_llama_ray_34b.sh)).
- **Distributed RLHF**: The key idea behind OpenRLHF is to distribute the Actor, Reward, Reference, and Critic models onto separate GPUs using Ray, while placing the Adam optimizer on the CPU. This enables full-scale fine-tuning of 7B models across multiple 24GB RTX 4090 GPUs (or 34B models with multiple A100 80G GPUs).
- **High performance**: Thanks to the ability to use a large inference batch size with Ray and DeepSpeed's CPUAdam, the performance of OpenRLHF with the 13B LLaMA2 model is 4x that of DeepSpeedChat.

## Features

- Distributed [PPO based on Ray](./examples/scripts/train_ppo_llama_ray.sh). 
- Support Multiple Reward models.
- Support [Rejection Sampling](./examples/scripts/train_rejection_sampling_llama.sh).
- Support [DPO (direct-preference-optimization)/IPO/cDPO](./examples/scripts/train_dpo_llama.sh).
- Support [Conditional Alignment](./examples/scripts/train_conditional_llama.sh) (https://arxiv.org/abs/2308.12050).
- Support [top chinese models](https://github.com/OpenLLMAI/OpenRLHF/issues/116).
- Multi-nodes [training scripts](./examples/scripts/train_llama_slurm.sh) for Slurm.
- Support Wandb log (--wandb).
- Support FlashAttention2 (--flash_attn).
- Support [GPT4 evaluation](./evaluation/gpt4/README.md) \& PPO vs SFT <a href="./docs/ppo_examples.md">examples</a>
- Pre-trained 7B/13B llama2 [checkpoints](https://huggingface.co/OpenLLMAI/openrlhf_checkpoint)


**TODO** 
- **RLHF compatible with models larger than 100B using vLLM**
- Allows saving and loading training checkpoints.
- Integrates with the QLora.


Support Matrix


|        | Best Hyperparameters  | Ray  | 34B Full Tuning with 4 A100   | 7B Full Tuning with 1 A100  | 7B Full Tuning with 4 RTX4090 |
|  ----  | ----  |  ----  | ----  | ----  | ----  |  
| OpenRLHF  | ✔ | ✔  | ✔ | ✔ | ✔ |
| DeepSpeedChat  | ✖️ | ✖️  | ✖️ | ✖️ | ✖️ |
| ColossalAIChat  | ✖️ | ✖️  | ✖️ |✖️ | ✖️ |
| TRL  | ✖️ | ✖️  | ✖️ | ✖️ | ✖️ |

## Performance

|        | 7B llama2 RLHF | 13B llama2 RLHF (50k samples) | 
|  ----  | ----  |  ----  |
| OpenRLHF  | - | 22 hours with 8 A100  | 
| DeepSpeedChat  | - | 48 hours with 16 A100  |

**Configs for Ray and DeepSpeed:** 

- 4 A100 80G for Actor, 2 A100 80G for Critic, 1 A100 80G for RM, and 1 A100 80G for InitPolicy
- ZeRO2 with Adam Offload
- Max Sequence Length: 2048 

**Throughput:**

- 7B llama2: 0.105 samples/gpu/secs
  - micro_batch_size = 16/8 (rollout/train), generation_length = 100~300
- 13B llama2: 0.04 samples/gpu/secs
  - micro_batch_size = 8/4 (rollout/train), generation_length = 200~400
- 34B codellama: 0.007 samples/gpu/secs
  - micro_batch_size = 2/1 (rollout/train), generation_length = 300~800

samples/gpu/secs = Number of PPO Samples / Number of A100 GPUS / Seconds

## Running Example

You can build openrlhf from **nvidia-docker(recommended)** or from conda envs.

```shell
Clone the repository: 
git clone https://github.com/openllmai/OpenRLHF.git

# Download the pre-trained SFT/RM checkpoints (Optional)
git lfs install
git clone --depth=1 https://huggingface.co/OpenLLMAI/openrlhf_checkpoint
```

* Single-node training with nvidia-docker

```shell
cd examples/scripts

# install nvidia-docker (Optional)
./nvidia_docker_install.sh

# launch nvidia container
./docker_run.sh

# cd in container
cd /openrlhf/examples/scripts

# build OpenRLHF (i.e, pip install)
./build_openrlhf.sh

# huggingface login 
~/.local/bin/huggingface-cli login

# train SFT model
./train_sft_llama.sh

# train RM model
./train_rm_llama.sh

# train PPO model
./train_ppo_llama.sh

# train DPO model
./train_dpo_llama.sh

# train Rejection Sampling model
./train_rejection_sampling_llama.sh

# train Conditional Alignment model
./train_conditional_llama.sh
```

* PPO training with Ray
> for 13B/34B models on A100/H100.. or 7B models on RTX4090

```shell
cd examples/scripts

# launch nvidia container
./docker_run.sh

# cd in container
cd /openrlhf/examples/scripts

# build OpenRLHF (i.e, pip install)
./build_openrlhf.sh

# huggingface login 
~/.local/bin/huggingface-cli login

# launch ray in container
nohup ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --block &> ray.log &

# if you want to launch ray on more nodes, use
# ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8 --block

# train ray PPO model, requires 8 gpus in default config
./train_ppo_llama_ray.sh
```

* Multi-nodes training on Slurm

```shell
cd examples/scripts

# huggingface login on Slurm 
pip install transformers
huggingface-cli login

# Moidfy the Slurm Account/Nodes ... in `train_llama_slurm.sh`

# For SFT, RM, and PPO and DPO training:
# Modify the variable `training_script` in `train_llama_slurm.sh` to
readonly training_script="train_sft_llama.sh"
readonly training_script="train_rm_llama.sh"
readonly training_script="train_ppo_llama.sh"
readonly training_script="train_dpo_llama.sh"

# set `GPUS_PER_NODE` in `train_llama_slurm.sh`
readonly GPUS_PER_NODE=8

# run multi-nodes training script
# train_llama_slurm.sh will load the training args from `training_script`
sbatch ./train_llama_slurm.sh
```

* Inference and Evaluation

After completing the training, you can evaluate your model by using the `inference` script:

```shell
# interactive_chat
./interactive_chat_llama.sh { model_path }

# batch generate
python examples/batch_inference.py {args}
```

* build openrlhf from conda envs 

If you really don't want to use nvidia-docker, we also provide tutorials for building openrlhf from a conda environment. (We prefer nvidia-docker to avoid errors caused by the environment.)
```shell
# we need conda
conda create -n openrlhf python=3.10
# so, we need install some package manually: when installing torch, you may need to match the corresponding cuda version.
pip install packaging ninja
pip3 install torch
# check ninjia
ninja --version
echo $? # output: 0
# install flash-attn: may take some time.
# For network error: you can download specified version from https://github.com/Dao-AILab/flash-attention/releases.
pip install flash-attn==2.3.6
./build_openrlhf.sh
# enjoy it!
```


## References & Acknowledgements

We would like to express our gratitude to the following projects and organizations for their contributions to the field of AI and NLP:

- [Hugging Face Transformers ↗](https://github.com/huggingface/transformers)
- [OpenAI GPT ↗](https://github.com/openai/gpt-3)
- [LLaMA2 ↗](https://ai.meta.com/llama/)
- [DeepSpeed ↗](https://github.com/microsoft/DeepSpeed)
- [Ray ↗](https://github.com/ray-project/ray)


## Join Us

**How to Join?**

1. Email us at xianyuai@openllmai.top(official email) or janhu9527@gmail.com/jjgxw@outlook.com(PIC). Please include the following details:
   - Your name
   - Your GitHub username
   - Your areas of interest
   - Your skills and experience related to NLP and/or AI
1. You can also join us through the official GitHub [OpenRLHF ↗](https://github.com/openllmai/OpenRLHF) project page. Just create an issue about your interest to contribute and we will get back to you.

**What can you do?**

1. Join the team and participate in the development of the OpenRLHF project.
1. Contribute to the project by submitting pull requests.
1. Help improve documentation, fix bugs, or create new features.
1. Share the project and help us grow the community.

## Sponsor Us

Your sponsorship can help us maintain and improve OpenRLHF. If you find this project useful, please consider sponsoring us. You can sponsor us on [Open Collective ↗](https://opencollective.com/openllmai).

## Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=openllmai/OpenRLHF&type=Date)](https://star-history.com/#openllmai/OpenRLHF&Date)

## Contributors

A big thank you to all our contributors! If you want to contribute, feel free to make a pull request or create an issue.

<a href="https://github.com/openllmai/OpenRLHF/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openllmai/OpenRLHF" />
</a>

Our project would also like to thank [ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat) and [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat). In the early stages of the project, we referred to their code design.

## Citation
```
@misc{hu23openrlhf,
   author = {Jian Hu and Xibin Wu and Xianyu and Chen Su and Leon Qiu and Daoning Jiang and Qing Wang and Weixun Wang},
   title = {OpenRLHF: A Ray-based High-performance RLHF framework},
   year={2023},
   publisher = {GitHub},
   journal = {GitHub repository},
   howpublished = {\url{https://github.com/OpenLLMAI/OpenRLHF}}
}
```

______________________________________________________________________

*OpenRLHF © 2023 OpenLLMAI. All Rights Reserved.*
