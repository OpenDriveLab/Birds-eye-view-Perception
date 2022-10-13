#### 1. Installation
Please refer to [installation.md](./installation.md) for installation of environment.

#### 2. Data Preparation

Please refer to [data_preparation.md](./data_preparation.md) for preparation of dataset and pretrained model.

#### 3. Running Experiments

We provide a baseline config to run BEVFormer on Waymo 1/5 training data.
```shell
cd waymo_playground
sh ./tools/dist_train_video.sh projects/configs/bevformer/waymo_mini_r101_baseline.py 8
```
Logs and checkpoints will be saved at ```work_dirs/waymo_mini_r101_baseline```. 
After training, run the following command for evaluation.
```
sh ./tools/dist_test_video.sh projects/configs/bevformer/waymo_mini_r101_baseline.py work_dirs/waymo_mini_r101_baseline/latest.pth 8
```
Change the config file for running other experiments.
