# Train
python launch.py --config CONFIG_PATH --train --case CASE_NAME --gpu 0

example: python launch.py --config configs/base.yaml --train --case dtu_scan24 --gpu 0

# Test
python launch.py --config CONFIG_PATH --test --ckpt CASE_NAME

example: python launch.py --config exp/dtu_scan24/config.yaml --test --ckpt exp/dtu_scan24/epoch=0-step=10000.ckpt

# Acknowledgements
[instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl): Great NeuS implementation in PyTorch-Lightning!

[nerfacc](https://github.com/nerfstudio-project/nerfacc): Library for accelerating NeRF renders.

[CuMCubes](https://github.com/lzhnb/CuMCubes): Very fast marching cubes implementation using CUDA.