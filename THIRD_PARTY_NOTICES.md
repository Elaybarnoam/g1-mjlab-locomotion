# Third-party notices

This project is MIT-licensed. It integrates with, but does not relicense, the projects below.
Unless explicitly stated, dependencies are installed separately and are not vendored in the
source distribution.

| Component | Pinned basis | License | Use |
| --- | --- | --- | --- |
| [mjlab](https://github.com/mujocolab/mjlab) | commit `8ee51fbcf806a7419189f706d9e394cbeb7790fa`, version 1.6.0 | Apache-2.0 | Training environment and G1 task/model provider |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | 3.11.0 | Apache-2.0 | Native simulation and viewer |
| [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | 3.11.0 | Apache-2.0 | Batched GPU simulation through mjlab |
| [NVIDIA Warp](https://github.com/NVIDIA/warp) | 1.14.0 | Apache-2.0 | GPU simulation dependency |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | 5.5.0 | BSD-3-Clause | PPO/GAE implementation |
| [PyTorch](https://github.com/pytorch/pytorch) | 2.9.0+cu128 in the qualified run; 2.13 default training line | BSD-3-Clause | Neural-network runtime |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 1.24.3 for native playback | MIT | Exported-policy inference |
| [NumPy](https://github.com/numpy/numpy) | 2.5.1 for native playback | BSD-3-Clause | Numerical arrays |
| [ImageIO](https://github.com/imageio/imageio) | 2.37.0 | BSD-2-Clause | Video writing |
| [ImageIO-FFmpeg](https://github.com/imageio/imageio-ffmpeg) | 0.6.0 | BSD-2-Clause | MP4 encoding helper |
| [NVIDIA SOMA Retargeter sample motions](https://github.com/NVIDIA/soma-retargeter/tree/b3ef2708d84bfd1314ddb52d0db6c9c211df1f57/assets/motions) | commit `b3ef2708d84bfd1314ddb52d0db6c9c211df1f57`, sample `Neutral_walk_forward_002__A057` | Apache-2.0 for the ten bundled sample BVH files, [confirmed by the maintainer](https://github.com/NVIDIA/soma-retargeter/issues/13#issuecomment-4509181768) | Source of the processed walking-reference cycle and preview; the larger Bones-SEED dataset is excluded |

The Unitree G1 model and visual geometry shown in the curated experiment media are loaded through
the pinned [mjlab G1 asset directory](https://github.com/mujocolab/mjlab/tree/8ee51fbcf806a7419189f706d9e394cbeb7790fa/src/mjlab/asset_zoo/robots/unitree_g1),
introduced in mjlab's Apache-2.0-licensed initial public release. The source tree identifies no
separate asset license or notice for that directory. On that documented basis, the project retains
the unaltered simulation screenshots and video as attributed experiment output. Git does not store
the model mesh files, compiled MuJoCo model, or Unitree trademarks as standalone assets. The
`standing-v1-inference.zip` release includes the exact compiled model required for reproducible
playback and reproduces the
[published Unitree G1 binary-redistribution license](https://github.com/google-deepmind/mujoco_menagerie/blob/main/unitree_g1/LICENSE)
inside the archive.
The media identifies the depicted robot; no affiliation with or endorsement by
Unitree, the mjlab developers, Google DeepMind, NVIDIA, ETH Zurich, or OpenAI is implied.

Dependency wheels may include additional notices. Their installed license files remain
authoritative. Before redistributing a dependency or robot asset inside this repository or a
release archive, review and include all license and notice files shipped by that component.

The Apache-2.0 text distributed with NVIDIA SOMA Retargeter is reproduced at
[`release/NVIDIA_SOMA_APACHE_2_LICENSE.txt`](release/NVIDIA_SOMA_APACHE_2_LICENSE.txt) because this
repository vendors a processed derivative of one approved sample motion.
