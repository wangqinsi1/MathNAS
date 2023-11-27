# MathNAS: If Blocks Have a Role in Mathematical Architecture Design

This is Official **PyTorch implementation** for 2023-NeurIPS-MathNAS: If Blocks Have a Role in Mathematical Architecture Design. 

For **a quick overview** of this work you can look at the [poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202023/70987.png?t=1699703256.6154978) made by the two main authors, [Qinsi Wang*](https://wangqinsi1.github.io/) and [Jinghan Ke*](https://ah-miu.github.io/). 

For **more details and supplementary material content** please see the [paper](https://openreview.net/pdf?id=e1l4ZYprQH).

If you find it useful, please feel free to cite:
```bash
@inproceedings{qinsi2023mathnas,
  title={MathNAS: If Blocks Have a Role in Mathematical Architecture Design},
  author={Qinsi, Wang and Ke, JingHan and Liang, Zhi and Zhang, Sihai},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## Prerequisites

---

- Python 3.7 (Anaconda)
- PyTorch 1.9.0
- CUDA 11.1

## Installation

---

```bash
conda create --name mathnas python=3.7
conda activate mathnas
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install nas-bench-201
pip install -i https://pypi.gurobi.com gurobipy
```

## Contents

---

1. Experiments on MobileNet-V3 Search Space
2. Experiments on NAS-Bench-201 Search Space
3. Experiments on SuperViT Search Space
4. Experiments on SuperTransformer Search Space
5. Dynamic Running on Edge Device

### 1. Experiments on MobileNet-V3 Search Space

---

#### 1.1 Hardware Search

You can perform a latency-limited search on Raspberry Pi, jetson TX-2 CPU and jetson TX-2 GPU with the following code:
```bash
$ python main.py
		 --mode nas \
		 --search_space mobilenetv3 \
		 --device [raspberrypi/tx2cpu/tx2gpu] \
		 --latency_constraint [number of latency]
```

For example, 

`python main.py --mode nas --search_space mobilenetv3 --device tx2gpu --latency_constraint 500`

result:

```tex
search time is 0.51258 s
The search net is {'ks': [7, 5, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 7, 3, 5, 5], 'd': [4, 4, 4, 4, 4], 'e': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6], 'encodearch': [3, 4, 4, 4, 3]}
The latency of search net is 68.9 s
network saved!
```

The output file path of search result is  `results/mobilenetv3/[device]/[latency_constraint].yml`. 

#### 1.2 Network Performance Prediction

You can predict the latency and accuracy of any network through the following code:

```bash
$ python main.py
		 --mode predict \
		 --search_space mobilenetv3 \
		 --device [raspberrypi/tx2cpu/tx2gpu] \
		 --load_path [network.yml]
```

For example, 

`python main.py --mode predict --search_space mobilenetv3 --device tx2gpu --load_path ./results/mobilenetv3/tx2gpu/500.yml`

result:

```tex
The net arch is {'d': [4, 4, 4, 4, 4], 'e': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6], 'encodearch': [3, 4, 4, 4, 3], 'ks': [7, 5, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 7, 3, 5, 5]}
The predicted accuracy of the net is 79.3 %
The predicted latency of the net is 68.9 s
```

#### 1.3 Architecture Results (Table 1)

| Model       | Flops(M) | Top1 | Search Time | Config                                        |
| ----------- | -------- | ---- | ----------- | --------------------------------------------- |
| MathNAS-MB1 | 257      | 75.9 | 0.9s        | [link](./results/mobilenetv3/MathNAS-MB1.yml) |
| MathNAS-MB2 | 289      | 76.4 | 1.2s        | [link](./results/mobilenetv3/MathNAS-MB2.yml) |
| MathNAS-MB3 | 336      | 78.2 | 1.5s        | [link](./results/mobilenetv3/MathNAS-MB3.yml) |
| MathNAS-MB4 | 669      | 79.2 | 0.8s        | [link](./results/mobilenetv3/MathNAS-MB4.yml) |

#### 1.4 Validation

You can verify the accuracy of the searched network on ImageNet:

```bash
$ python valide/validate_mobilenetv3.py \
  --config_path [Path of neural architecture config file] \
  --imagenet_save_path [Path of ImageNet 1k]
```

### 2. Experiments on NAS-Bench-201 Search Space

---

#### 2.1 Hardware Search

You can perform a latency-limited and energy-limited search on FPGA,edgeGPU with the following code:

```bash
$ python main.py
		 --mode nas \
		 --search_space nasbench201 \
		 --device [fpga/edgegpu] \
		 --latency_constraint [number of latency] \
         --energy_constraint [number of energy]
```

The output file path of search result is  `results/nasbench201/[device]/[latency_constraint]_[energy_constraint].yml`. 

#### 2.2 Network Performance Prediction

You can predict the latency and accuracy of any network through the following code:

```bash
$ python main.py
		 --mode predict \
		 --search_space nasbench201 \
		 --device [fpga/edgegpu] \
		 --load_path [network.yml]
```

### 3. Experiments on SuperViT  Search Space

---

#### 3.1 Search

You can perform a FLOPs-limited search with the following code:

```bash
$ python main.py
		 --mode nas \
		 --search_space supervit \
		 --flops_constraint [number of FLOPs]
```

The output file path of search result is  `results/supervit/[flops_constraint].yml`. 

#### 3.2 Network Performance Prediction

You can predict the latency and accuracy of any network through the following code:

```bash
$ python main.py
		 --mode predict \
		 --search_space supervit \
		 --load_path [network.yml]
```

#### 3.3 Architecture Results (Table 2)

| Model      | Topk1(%) | Topk5(%) | FLOPs(M) | Param(M) | Config                                    |
| ---------- | -------- | -------- | -------- | -------- | ----------------------------------------- |
| MathNAS-T1 | 78.4     | 93.5     | 200      | 8.9      | [link](./results/supervit/MathNAS-T1.yml) |
| MathNAS-T2 | 79.6     | 94.3     | 325      | 9.3      | [link](./results/supervit/MathNAS-T2.yml) |
| MathNAS-T3 | 81.3     | 95.1     | 671      | 13.6     | [link](./results/supervit/MathNAS-T3.yml) |
| MathNAS-T4 | 82.0     | 95.7     | 1101     | 14.4     | [link](./results/supervit/MathNAS-T4.yml) |
| MathNAS-T5 | 82.5     | 95.8     | 1476     | 14.8     | [link](./results/supervit/MathNAS-T5.yml) |

#### 3.4 Validation

You can verify the accuracy of the searched network on ImageNet:

```bash
$ python valide/validate_nasvit.py \
		--config_path [Path of neural architecture config file] \
		--imagenet_save_path [Path of ImageNet 1k]
```

### 4. Experiments on SuperTransformer Search Space

---

#### 4.1 Hardware Search

You can perform a latency-limited search on Raspberry Pi, Intel Xeon CPU and Nvidia TITAN Xp GPU with the following code:

```bash
$ python main.py
		 --mode nas \
		 --search_space supertransformer  \
		 --device [raspberrypi/xeon/titanxp] \
		 --latency_constraint [number of latency]
```

The output file path of search result is  `results/supertransformer/[device]/[latency_constraint].yml`. 

#### 4.2 Network Performance Prediction

You can predict the latency and accuracy of any network through the following code:

```bash
$ python main.py
		 --mode predict \
		 --search_space supertransformer \
		 --device [raspberrypi/xeon/titanxp] \
		 --load_path [network.yml]
```

**NOTE**: You can test models by [BLEU score](https://github.com/mit-han-lab/hardware-aware-transformers#test-bleu-sacrebleu-score) and [Computing Latency](https://github.com/mit-han-lab/hardware-aware-transformers#test-latency-model-size-and-flops).

### 5. Dynamic Running on Edge Device

------

#### Test on CPU/GPU

Enter the dynamic folder, run the following code to achieve dynamic running on the CPU.

```python
from Runcpu import Run
Run(latencymax,latencymin,ILP)  
```

latencymax and latencymin represent the maximum and minimum latency accepted on the device, respectively.  ILP represents the hardware solver selection, ILP=1 (Gurobipy) or 0 (Linprog).

Similarly, run the following code to achieve dynamic running on the GPU.

```python
from Rungpu import Run
Run(latencymax,latencymin,ILP) 
```

The profiles of the blocks used by the CPU and GPU are in the CPUblock and GPUblock folders.
