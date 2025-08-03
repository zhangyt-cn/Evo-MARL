# 💪 Evo-MARL

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
cd Evo-MARL
pip install -r requirements.txt
```

## 🚀 Training
```python
bash run_chain_ppo.sh
```
Please configure the arguments correctly before use, see ![OpenRLHF Documentation](https://openrlhf.readthedocs.io/en/latest/) for parameters explanation.

We have also released trained model weight at ![Evo-MARL-QWen2.5-1.5B](https://huggingface.co/WendyZhang21/Evo-MARL-QWen2.5-1.5B-Instruct)

Codebase is built upon ![OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), our core implementation is at ![mas-train.py](https://github.com/zhangyt-cn/Evo-MARL/blob/main/mas-train.py)
