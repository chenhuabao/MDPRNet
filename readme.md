## MDPRNet: Multi-stage Dual-domain Progressive Reconstruction Network

The official pytorch implementation of the paper **[Multi-stage Dual-domain Progressive Network with Synergistic Training for Sparse-view CT Reconstruction]**

#### Jingyuan Shao, Huabao Chen, Qiankun Li, Xiang Huang, Jiong Shu, Lingling Liu, Shaobin Dou, Hongzhi Wang

<p align="center">
<img src=/Visuals/MDPRNet.png width=70%>
</p>

### Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks, [NAFNet](https://github.com/megvii-research/NAFNet) and [MPRNet](https://github.com/swz30/MPRNet) 

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

```
git clone https://github.com/megvii-research/NAFNet
cd MDPRNet
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

## MDPRNet implementation
The implementation of our proposed MDPRNet, MDPRNet block, and the Learnable Global Attention Gate can be found in ```/MDPRNet/basicsr/models/archs/MDPRNet_arch.py```

##  Reconstruction on the AAPM
### 1. Data Preparation
##### Download the train set(from the AAPM dataset website) and place it in ```./datasets/AAPM/Data/train```,
##### Download the evaluation data (from the AAPM dataset website) and place it in ```./datasets/AAPM/Data/val```:
#### After downloading, it should be like this:


```bash
./datasets/
└── AAPM/
    ├── Data/
    │   ├── train/
    │   │   ├── L067_FD_1_1_0.PNG
    │   │   ├── L067_FD_1_1_1.PNG
    │   │   ....
    │   └── val/
    │       ├── L141_FD_1_1_396.PNG
    │       ├── L141_FD_1_1_397.PNG    
    └──     ....

```
* Use ```python scripts/data_preparation/aapm.py``` to make the data into lmdb format. the processed images will be saved in ```./datdsets/AAPM/train/```

### 2. Training

* To train the MDPRNet model:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/MDPRNet.yml --launcher pytorch
```


### 3. Evaluation


#### Note: Due to the file size limitation, we are not able to share the pre-trained models in this code submission. However, they will be provided with an open-source release of the code.


##### Testing the model

  * To evaluate the pre-trained model use this command:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt ./options/test/MDPRNet.yml --launcher pytorch
```