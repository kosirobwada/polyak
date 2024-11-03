#!/bin/bash

# 仮想環境のアクティベート
source venv/Scripts/activate

# JSONファイルのディレクトリを指定
JSON_DIR="/mnt/c/Users/kosir/workspace/SGD_momentum/json/const_bs_decay_lr"

# ディレクトリ内のすべてのjsonファイルに対してループ
for json_file in "$JSON_DIR"/*.json; do
  # Pythonスクリプトを実行
  python3 cifar100.py "$json_file"
done

# 仮想環境のデアクティベート
deactivate