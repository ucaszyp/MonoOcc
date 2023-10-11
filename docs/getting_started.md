# Prerequisites

**Please ensure you have prepared the environment and the SemanticKITTI dataset.**

# Train and Test

Train MonoOcc with temporal information with 4 GPUs 
```
./tools/dist_train.sh ./projects/configs/MonoOcc/MonoOcc-S.py 4
```
```
./tools/dist_train.sh ./projects/configs/MonoOcc/MonoOcc-L.py 4
```

Eval MonoOcc with temporal information with 4 GPUs
```
./tools/dist_test.sh ./projects/configs/MonoOcc/MonoOcc-S.py ./path/to/ckpts.pth 4
```

```
./tools/dist_test.sh ./projects/configs/MonoOcc/MonoOcc-L.py ./path/to/ckpts.pth 4
```