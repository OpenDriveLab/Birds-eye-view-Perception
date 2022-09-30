# Usage

## Train & Eval

We provide a baseline config to run bevformer on Waymo 1/5 training data.
```shell
cd experiment
sh ./tools/dist_train_video.sh projects/configs/bevformer/waymo_mini_r101_baseline.py 8
```
Logs and checkpoints will be saved at ```work_dirs/waymo_mini_r101_baseline```. 
After training, run the following command for evaluation.
```
sh ./tools/dist_test_video.sh projects/configs/bevformer/waymo_mini_r101_baseline.py work_dirs/waymo_mini_r101_baseline/latest.pth 8
```