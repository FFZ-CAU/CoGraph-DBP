# CoGraph-DBP

Rank fusion of complementary graph predictors for DNA-binding protein prediction.

Authors: Wentao Gong, Jiaying Zhang, Huihui Pang, Jingrui Li, and Feifan Zhang.

## Installation

Use Python 3.12 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The neural training configuration uses PyTorch 2.8.0, CUDA 12.8 and one 24 GB GPU
per process. Score fusion and evaluation run on CPU.

## Inputs

Training starts from per-protein feature arrays. Benchmark sequences are available
from [PLM-DBPs](https://github.com/suresh-pokharel/PLM-DBPs) and
[CNNCaps-DBP](https://github.com/YZYAlex/CNNCaps-DBP).
The prepared tables, feature arrays and graph-summary CSV files are available
from the corresponding authors upon request.

```text
data/
  pkl/Train-DBP.pkl
  pkl/Train-2104.pkl
  pkl/Test-DBP.pkl
  pkl/Test-296.pkl
  Train/Train-1698/{feature/feature-prot5-xl,contact,map,map-esm}/<id>.npy
  Train/Train-2104/{feature/feature-prot5-xl,contact,map,map-esm}/<id>.npy
  Test/Test-997/{feature/feature-prot5-xl,contact,map,map-esm}/<id>.npy
  Test/Test-296/{feature/feature-prot5-xl,contact,map,map-esm}/<id>.npy
esm2/
  {Train-1698,Train-2104,Test-997,Test-296}/<id>.npy
graph_features/
  {Train-1698,Train-2104,Test-997,Test-296}.csv
```

Each PKL file is a dictionary mapping a protein ID to `(sequence, label, length)`.
The label is 1 for DBPs and 0 otherwise.
ProtT5 and ESM2 residue arrays have 1024 and 1280 channels, respectively. Relation
arrays are square matrices aligned to the residue arrays. Graph-summary CSV files
contain `name` and the 19 columns defined in `training/structure_support.py`.
Inputs are cropped or padded to 1000 residues with top-k = 25.
The Train-1698 reference path keeps self-loops and mean pooling over all 1000
positions. Other neural paths use valid-residue positions.
ProtT5 relation matrices average the attention heads in encoder layer 24 and retain
only amino-acid token rows and columns, in the same order as the residue embeddings.

To generate these matrices from local ProtT5-XL-UniRef50 weights, use an empty
output directory:

```bash
python training/precompute_plm_relations.py --records data/pkl/Train-DBP.pkl --features data/Train/Train-1698/feature/feature-prot5-xl --model weights/prot_t5_xl_uniref50 --out-dir data/Train/Train-1698/map --device cuda:0
```

ProtT5 originates from [ProtTrans](https://github.com/agemagician/ProtTrans);
the ESM2 model is `esm2_t33_650M_UR50D` from
[ESM](https://github.com/facebookresearch/esm). Third-party resources retain their
respective licenses.

## Training

```bash
python scripts/pipeline.py train --data-root data --esm2-root esm2 --graph-features graph_features --out-dir runs/train --device cuda:0
```

This trains the reference, topology and residue paths, plus the auxiliary neural
models used by the structure path. It fits the two-stage structure correction and
then selects rank-fusion weights and fits beta calibration. Model files and scores
are written under `runs/train/<source-set>/`; structure parameters are written under
`runs/train/structure/`.

The configuration uses seed 525, the third fold of five-fold shuffled KFold,
25 epochs, batch size 16 and Adam with learning rate 5e-5. Checkpoints are selected
by validation MCC. Source-specific graph and pooling settings are listed in
`configs/training.json`; fusion candidates are listed in `configs/fusion.json`.

## Prediction and evaluation

```bash
python scripts/pipeline.py predict --source-set Train-1698 --records data/pkl/Test-DBP.pkl --split-root data/Test/Test-997 --esm2-dir esm2/Test-997 --features graph_features/Test-997.csv --run-dir runs/train --out-dir runs/test997 --device cuda:0
python scripts/prepare_scores.py --neural runs/test997/neural_scores.npz --records data/pkl/Test-DBP.pkl --output runs/test997/labels.npz
python scripts/fuse.py evaluate --predictions runs/test997/predictions.npz --labels runs/test997/labels.npz --output runs/test997/metrics.json
```

For Test-296, use `Train-2104` and the corresponding Test-296 input paths.
Predictions contain `names`, `probabilities` and `threshold`. The four path order is
reference, structure, topology and residue. Empirical quantile maps and beta
calibration are fitted on source-validation scores. The decision threshold is 0.5
when fusion validation AUC is at least 0.90; otherwise it maximizes expected F1 on
the predicted probabilities. Evaluation reports specificity, accuracy, precision,
recall, F1, ROC-AUC, trapezoidal PR-AUC and MCC.

Individual stages are available through `scripts/train.py`,
`scripts/predict_neural_paths.py`, `training/structure_path.py`,
`training/replay_structure.py`, `scripts/prepare_scores.py` and `scripts/fuse.py`.
Use `--help` to inspect their arguments.

## Citation and license

Citation information is provided in `CITATION.cff`. Source code is licensed under
the MIT License in `LICENSE`.
