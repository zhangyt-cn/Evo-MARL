# 💪 Evo-MARL


<p align="left">
    🧐&nbsp;<a href="#-about">About</a>
    | 🔧&nbsp;<a href="#-setup">Setup</a>
    | 🚀&nbsp;<a href="#-quick-start">Quick Start</a>
    | 🙏&nbsp;<a href="#-acknowledgement">Acknowledgement</a>
    | 📝&nbsp;<a href="#-citation">Citation</a>
</p>


## 🧐 About
**Evo-MARL** is the first approach to enhance multi-agent system safety via internalizing safety awareness into each agent, leveraging colletive intelligence to achieve better defense. 

![Overview of Evo-MARL](assets/overview_ff.drawio.png)

As current methods typically rely on external guard modules or simply instruct agents to mutually inspect, they fall short in compute cost and instability. 
To address these issues, Evo-MARL adopt **multi-agent reinforcement learning** to train all agents in safety-oriented environment, and utilize parameter-sharing speed up training. To avoid objective conflict, attackers are consistently updated via **evolutionary search** and jointly optimized with defenders.

## 🔧 Setup
```python
conda create -n masrl python=3.11
conda activate masrl

git clone https://github.com/zhangyt-cn/Evo-MARL.git
cd Evo-MARL/OpenRLHF
pip install -e .
```

## 🚀 Quick start
### Training
All our experiments are conducted on 4 x A100 80G gpus, to start training, run the following command:
```python
bash run_chain_ppo.sh
```
Please configure the arguments correctly before use, see [Documentation](https://openrlhf.readthedocs.io/en/latest/) for parameters explanation.

We have also released trained model weight on [Evo-MARL-QWen2.5-1.5B](https://huggingface.co/WendyZhang21/Evo-MARL-QWen2.5-1.5B-Instruct) [Evo-MARL-QWen2.5-3B](https://huggingface.co/WendyZhang21/Evo-MARL-QWen2.5-3B-Instruct) 

The codebase is built upon [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), our core implementation is at [mas-train.py](https://github.com/zhangyt-cn/Evo-MARL/blob/main/mas-train.py)


### Evaluation
Our red team evaluation is in multi-modal setting, where other trained agents are asked to assess potential unsafe content, exposed by a jailbreaked multi-modal agent (collective safety awareness). Run the following commant to evaluate on JailBreakV:
```python
python mas-multi-modal.py
```
Evaluations on other datasets need subtle modifications. "Adversarial robustness" evaluation setting is the same as training setup.

## 🙏 Acknowledgement 
We thank [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) team for their wonderful work!

## 📝 Citation
If you find this work useful, please consider citing:

