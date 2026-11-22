# hypercomplex-network
This resipository contains building blocks and codes for hypercomplex network. 
You can design any architecture of choice with limitless number of hypercomplex component using the code in this resiprosity.

If using this code, please cite 
```
@article{bojesomo2024hypercomplex,
  author={Bojesomo, Alabi and Liatsis, Panos and Marzouqi, Hasan Al},
  journal={IEEE Signal Processing Magazine}, 
  title={Deep Hypercomplex Networks for Spatiotemporal Data Processing: Parameter efficiency and superior performance [Hypercomplex Signal and Image Processing]}, 
  year={2024},
  volume={41},
  number={3},
  pages={101-112},
  keywords={Training;Convolution;Algebra;Quaternions;Image processing;Data processing;Spatiotemporal phenomena},
  doi={10.1109/MSP.2024.3381808}}

@article{bojesomo2023octonion,
  author={Bojesomo, Alabi and Liatsis, Panos and Almarzouqi, Hasan},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Marine Debris Segmentation Using a Parameter Efficient Octonion-Based Architecture}, 
  year={2023},
  volume={20},
  number={},
  pages={1-5},
  keywords={Convolution;Plastics;Neural networks;Computer architecture;Deep learning;Marine vehicles;Feature extraction;Hypercomplex numbers;marine debris image segmentation;octonion neural networks (ONNs)},
  doi={10.1109/LGRS.2023.3321177}}

@article{bojesomo2020traffic,
      title={Traffic flow prediction using Deep Sedenion Networks}, 
      author={Alabi Bojesomo and Hasan Al-Marzouqi and Panos Liatsis },
      year={2020},
      eprint={2012.03874},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```

## AV-FFIA: hypercomplex audio-visual fish feeding intensity assessment

`avffia/` contains HyperAVNet, an efficient audio-visual classifier for the
[AV-FFIA dataset](https://github.com/FishMaster93/U-FFIA). It has hypercomplex backbones,
attention and heads, a hypercomplex component fusion block applied at configurable depths
(deep to early), pretrained PANNs, S3D, timm and other models as teachers or frozen extractors,
multi-level distillation, and U-FFIA's noise-robustness protocol. Target: beat the published
SOTA (0.917, cross-attention) at 2.1M params / 1.2 GMACs. See [docs/AVFFIA_DESIGN.md](docs/AVFFIA_DESIGN.md).

```bash
pip install -r requirements.txt
bash scripts/download_pretrained.sh --cache                 # pretrained checkpoints (docs/PRETRAINED_MODELS.md)
python -m pytest -q tests                                     # offline smoke tests
python train.py --build-manifest --video-root /data/AV-FFIA --audio-root /data/AV-FFIA_audio
python train.py -c configs/teacher_panns_s3d.yaml             # 1) teacher (PANNs CNN10 + S3D)
python train.py -c configs/student_hyper_kd.yaml              # 2) distilled 2.1M-param student
bash scripts/run_ablations.sh configs/student_hyper.yaml      # fusion-depth / algebra / operator ablations
```

Architecture and paper-writing guide (PDF): [docs/HyperAVNet_architecture.pdf](docs/HyperAVNet_architecture.pdf).
