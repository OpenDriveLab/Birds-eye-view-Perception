# Usage

## Train & Eval
### Train

```
cd code
./tools/dist_train_video.sh ./path/to/config.py num_gpus
```

### Eval
```
cd code
./tools/dist_test_video.sh ./path/to/config.py ./path/to/ckpt.pth num_gpus
```