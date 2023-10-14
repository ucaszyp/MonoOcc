# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation



**a. Create a conda virtual environment and activate it.**
```shell
conda create -n MonooOcc python=3.8 -y
conda activate MonoOcc
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9

```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install -U openmim
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.18.1
pip install mmsegmentation==0.27.0
```

**e. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
pip install -v -e .
```

**f. Install timm.**
```shell
pip install timm==0.6.11
```

**g. Install InternImage.**
```shell
git clone https://github.com/OpenGVLab/InternImage
cd InternImage/segmentation/ops_dcnv3
# need PyTorch>=1.10.0
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

**h. Clone MonoOcc.**
```
git clone https://github.com/ucaszyp/MonoOcc.git
```

**i. Prepare pretrained resnet50 models.**
```shell
cd MonoOcc && mkdir ckpts && cd ckpts
```
Download the pretrained [resnet50](https://drive.google.com/file/d/1A4Efx7OQ2KVokM1XTbZ6Lf2Q5P-srsyE/view?usp=share_link).

Pretrained model of InternImage will be released soon
