PYTHONPATH=".":$PYTHONPATH python -m torch.distributed.launch --nproc_per_node=1 tools/testStu.py projects/configs/distiller/base_distill_small_with_pv-cwd_bev-l2-heatmap.py ckpts/base_distill_small_with_pv-cwd_bev-l2-heatmap_ep24.pth --launcher pytorch --eval bbox