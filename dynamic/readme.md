# Neural Architecture Solver based on Block-Architecture Accuracy Loss Model

This is  PyTorch implementation for  Neural Architecture Solver based on Block-Architecture Accuracy Loss Model.

### Prerequisites

------

- Python 3.7.6
- PyTorch 1.9.0
- CUDA 11.1
- Jetson TX2

### Installation

------

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install nas-bench-201
```

## 1.Dynamic Running on Edge Device

------

We provide the code that makes the network run dynamically on edge devices.

#### Test on CPU/GPU

Run the following code to achieve dynamic running on the CPU.

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
