# Independent CNN baselines (COS30082 plant disease project)

Two separate CNNs per backbone: one predicts **crop** (55 classes), one predicts **disease** (175 classes).
`combine.py` multiplies their probabilities to rank crop+disease compositions (323 seen + 10 unseen).

CSV files have no header: `file, crop, disease, comp, crop_name, disease_name, comp_name`.

## Setup
```
pip install torch torchvision scikit-learn pandas matplotlib pillow
python check_labels.py --img_dir PATH/TO/IMAGES
```

## 1. Pick model sizes (linear probe, minutes)
```
python linear_probe.py --img_dir PATH/TO/IMAGES
```
Writes `probe_results.csv`. Choose the smallest size per family where accuracy stops improving.

## 2. Train (one run = one task)
```
python train.py --task crop    --backbone resnet50 --img_dir PATH/TO/IMAGES
python train.py --task disease --backbone resnet50 --img_dir PATH/TO/IMAGES
```
Tuning levers: `--lr`, `--epochs`, `--bs`, `--wd`, `--aug light|strong`, `--label_smooth`,
`--freeze_epochs N` (head only for the first N epochs), `--img_size`.
Out of GPU memory: lower `--bs` to 16. DataLoader crashes on Windows: add `--workers 0`.

`python summarize.py` lists every run, best validation score first. Choose runs by **val_top1**,
not by test scores.

## 3. Combine and score
```
python combine.py --crop_run runs/<crop run> --disease_run runs/<disease run> --out combined/resnet50
```
Gives Top-1/Top-5 on seen and unseen, harmonic mean, and the seen confusion matrix.

## 4. Visualise
Optional frozen baseline (ImageNet weights untouched, linear head only), used by the notebook's section 4:
```
python linear_probe.py --img_dir PATH/TO/IMAGES --models efficientnet_b0 resnet50 convnext_tiny efficientnet_b3 --save_runs runs/frozen
```
Open `evaluation.ipynb` and run all cells. It reads the fine-tuned runs in `runs/main` (and `runs/frozen` if present):
training setup check, scoreboard with bootstrap confidence intervals and a verdict per metric, Top-k curves,
frozen vs fine-tuned, cost (size, training time, inference speed), then per-model detail, error sources,
per-unseen-composition accuracy, crop bias of the disease model, accuracy vs class size and example mistakes.

## Validation split
6800 train / 1200 val, stratified by composition (seed 42), shared by every script.
`python check_labels.py` writes the val file list to `split_val.csv` to compare with other team members' splits.
