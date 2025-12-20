# NeuroAI-2025-小组: 基于空间结构化Hodgkin-Huxley网络的局部场电位(LFP)生成模拟
## 1. 项目概述 (Project Overview)
技术框架: 
  - 脉冲神经网络模拟器 Brian2（https://brian2.readthedocs.io/en/stable/）
  - Numpy\
本项目实现了一个符合生物学特性的脉冲神经网络 (SNN)，旨在探究局部场电位 (LFP) 的产生机制。与标准的人工神经网络 (ANN) 不同，本模型致力于连接微观的离子通道动力学与电生理学中观察到的宏观细胞外信号。
通过显式模拟容积导电 (Volume Conduction) 物理过程，本模型能够在不同的网络拓扑结构（如兴奋/抑制平衡、稀疏度及种群规模）下生成合成的 LFP 数据。
## 2. 神经科学背景 
### 2.1 LFP 的起源
局部场电位 (LFP) 代表了脑组织局部体积内神经活动产生的总电场。LFP 的主要物理源头是突触输入和动作电位引发的跨膜电流 (Transmembrane Current, $$I_m$$)。
### 2.2 容积导电理论 (Volume Conduction Theory)
细胞外介质充当容积导体。根据电磁场理论（准静态近似），特定位置的电势 $\Phi$ 是所有周围神经元电流的加权和，且与距离成反比：
$$\Phi(r, t) = \frac{1}{4 \pi \sigma} \sum_{j} \frac{I_{m, j}(t)}{|r - r_j|}$$
其中 $\sigma$ 是细胞外电导率，是神经元 $j$ 的总跨膜电流。
### 3. 模型架构 (Model Architecture)
neurol_model.py 和 neuron.ipynb的主要目的是：构建一个由 Hodgkin-Huxley (HH) 神经元组成的兴奋-抑制（E-I）平衡网络，并基于容积导电理论（Volume Conduction Theory）模拟记录到的局部场电位（LFP）
它主要模拟了以下三个层面的神经科学机制：
#### 微观层面：生物物理神经元模型
- Hodgkin-Huxley 模型： 代码没有使用简化的“积分-发放”模型，而是使用了完整的 HH 模型。
  - 离子通道： 显式模拟了 钠离子通道 ($I_{Na}$)、钾离子通道 ($I_K$) 和 漏电流 ($I_L$)。
  - 门控变量： 包含了 $m, n, h$ 三个门控变量的动态微分方程。
- 意义： 只有这种通过离子流产生动作电位的模型，才能精确计算跨膜电流 (Transmembrane Current, $I_m$)。而在神经物理学中，跨膜电流正是细胞外电场（即 LFP）的物理源头。
#### 中观层面：E-I 平衡网络与空间结构
- E/I 网络架构：
  - 网络包含兴奋性（Excitatory, ~80%）和抑制性（Inhibitory, ~20%）神经元群。
  - 突触连接采用电导模型（Conductance-based synapses），而非简单的电流注入，这更符合生物学突触的特性。
- 空间拓扑（关键点）：
  - 代码显式地给每个神经元分配了空间坐标 $(x, y, z)$。
  - 神经元被排列在一条线上（G.x = 'i * neuron_spacing'）。
  - 这对于 LFP 模拟至关重要，因为电信号在组织中的衰减与距离有关。
#### 宏观层面：LFP 的生成机制
- 正向建模 (Forward Modeling)： 代码通过“点源近似”和“欧姆定律”计算 LFP。
- 物理公式：
- 代码中的核心公式是：
  $$V_{electrode} = \frac{1}{4\pi\sigma} \sum_{i=1}^{N} \frac{I_{m, i}}{r_i}$$
- 其中 $$I_{m, i}$$ 是第 $$i$$ 个神经元的跨膜电流， $$r_i$$ 是该神经元到电极的距离。
- 模拟现象：
  - 代码在距离网络中心不同位置放置了虚拟电极（Ne=3）。
  - 由于 $1/r$ 的衰减特性，离神经元群体越远的电极，记录到的 LFP 振幅越小，高频成分衰减越严重（低通滤波效应）。
#### 数据生成与参数扫描
- if name == "main": 模块展示了这是一个数据生成器。
- 它扫描了不同的 网络规模 (N)、连接概率 (p_conn) 和 兴奋比例 (frac_exc)。
- 最终将生成的 LFP 时间序列数据保存为 lfp_dataset.npz。这通常用于后续分析，比如研究“网络结构如何影响 LFP 的频谱特征”。
## 文件结构 (File Structure)
```
NeuroAI/
├── neurol_model.py    # 主模拟脚本。定义了 HH 网络和 LFP 计算逻辑。
│                      # 包含一个参数扫描循环以生成数据集。
├── neuron.ipynb       # 原型 Notebook。用于可视化单次运行的动力学和脉冲发放。
├── lfp_dataset.npz    # (生成的输出) 包含 LFP 轨迹和元数据的数据集。
└── .gitignore         # Git 配置文件。
```
## 安装与使用 (Installation & Usage)
环境依赖
本项目需要 Python 3.x 及以下库：
```
pip install brian2 numpy matplotlib
运行模拟
运行完整的参数扫描并生成 LFP 数据集：
python NeuroAI/neurol_model.py
```
该脚本将执行以下操作：
1. 初始化 Brian2 模拟环境。
2. 遍历不同的网络配置（神经元数量 N, 连接概率 p_conn 等）。
3. 对每种配置模拟 2 秒的网络动力学过程。
4. 将生成的 LFP 数据保存为 lfp_dataset.npz。
可视化结果
在 Jupyter Notebook 中打开 NeuroAI/neuron.ipynb，可交互式地查看膜电位、脉冲光栅图 (Raster Plot) 以及记录到的 LFP 轨迹。
## 结果展示 (Simulation Results)
`neuron.ipynb` 的主要目的是运行单次仿真，直观地展示神经元的微观动态（动作电位）、中观动态（群体发放）以及宏观信号（LFP）。

运行neuro 最终会生成一张包含三个子图的面板。
<img width="716" height="498" alt="25378063ce5618784db6c3d28a4fe742" src="https://github.com/user-attachments/assets/cd1b578b-979d-43ea-bd11-fb00aab04b79" />

* **子图 1 (Top): 单个神经元膜电位 (Neuron 0 Membrane Potential)**
  - **内容:** 展示了网络中第 0 号神经元（兴奋性）的电压随时间的变化
  - **现象:** 可以清晰地看到典型的 **Hodgkin-Huxley 动作电位**（尖峰）以及尖峰之间的**阈下波动**。
  - **意义:** 证明了模型构建的微观基础是具有生物物理真实性的，而非简化的人工神经元。

* **子图 2 (Middle): 群体脉冲光栅图 (Raster Plot)**
  - **内容:** 横轴为时间，纵轴为神经元索引（0-100）。每一个黑点代表该时刻该神经元发放了一个脉冲。
  - **现象:**
  - 点在图上的分布显示了神经元的**群体同步性 (Synchrony)**。
  - 如果点是随机散布的，说明网络处于**异步状态 (Asynchronous state)**，这是清醒大脑皮层的典型特征。
  - 如果出现竖直的条纹，说明网络发生了**同步振荡 (Oscillation)**（如 Gamma 振荡）。

  - **意义:** 反映了网络的中观动力学状态。

* **子图 3 (Bottom): 局部场电位 (Local Field Potential, LFP)**
  - **内容:** 展示了三个不同位置电极记录到的电压波动
  - **LFP 0 (蓝线):** 距离神经元群体最近
  - **LFP 1 (橙线):** 距离中等
  - **LFP 2 (绿线):** 距离最远

* **关键特征:**
1. **振幅衰减:** LFP 0 的振幅最大，LFP 2 最小。这验证了**容积导电理论中的  衰减规律**
2. **波形平滑:** 相比于子图1中尖锐的动作电位，LFP 信号看起来更平滑、更像“波”。这是因为 LFP 主要是突触电流的叠加，且高频成分随距离衰减更快（组织作为一个低通滤波器）
## 成员分工 (Team Contributions)
- 苗金成: 撰写报告，统筹代码。
- 杨逸健: 负责基于容积导电理论设计 LFP 监测逻辑，并处理空间坐标计算，处理主逻辑。
- 陈志盛: 负责开发数据生成流水线、npz 数据存储逻辑及文档撰写。
