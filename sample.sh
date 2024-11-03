#!/bin/bash

# 仮想環境のアクティベート
source venv/Scripts/activate

python3 cifar100.py json/const_bs_decay_lr/cosine.json

deactivate