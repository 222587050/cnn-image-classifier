#!/usr/bin/env bash
set -e
python src/train.py --name baseline
python src/train.py --name aug --augment
python src/train.py --name aug_bn_drop --augment --bn --dropout 0.3
python src/evaluate.py
