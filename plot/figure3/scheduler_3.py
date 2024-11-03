import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import FuncFormatter

# ファイルパスを設定（適宜変更）
file_paths = {
    'constant': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_decay_lr\\constant\\lr_batch.csv',
    'x': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_incr_lr\\exp_growth\\min0.1_max0.2\\lr_batch.csv',
    'y': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_incr_lr\\exp_growth\\min0.1_max0.5\\lr_batch.csv',
    'z': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_incr_lr\\exp_growth\\min0.1_max1.0\\lr_batch.csv',
}

dataframes = {}
for key, path in file_paths.items():
    try:
        dataframes[key] = pd.read_csv(path)
        print(f"{key}: Loaded successfully.")
    except Exception as e:
        print(f"Failed to load {key} from {path}: {e}")

fig, ax1 = plt.subplots(figsize=(12, 6))

# Learning Rate のプロットと凡例ラベルの設定
labels = {
    'constant': 'constant',
    'x': r'$\eta_{max} = 0.2 (\gamma \approx 1.080)$',
    'y': r'$\eta_{max} = 0.5 (\gamma \approx 1.196)$',
    'z': r'$\eta_{max} = 1.0 (\gamma \approx 1.292)$'
}

for key, df in dataframes.items():
    ax1.plot(df['Epoch'], df['Learning Rate'], label=labels[key])

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Learning Rate')
ax1.set_title('Learning Rate and Batch Size Scheduler')
ax1.grid(True)

# Batch Size のプロット
ax2 = ax1.twinx() 
ax2.set_ylabel('Batch Size')

# y軸をlogスケールに設定
ax2.set_yscale('log')

# y軸を2^4から2^12までに限定し、2^6, 2^7, 2^8を目盛りに設定
ax2.set_ylim([2**(5/2), 2**(25/2)])
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"$2^{int(np.log2(x))}$"))

# バッチサイズデータをプロット
batch_size_line, = ax2.plot(df['Epoch'], df['Batch Size'], 'm--', label=r'Batch Size $(\delta = 2.0)$')

# 凡例の表示：ax1とax2の凡例を結合して表示
lines, labels = ax1.get_legend_handles_labels()  # ax1のラインとラベルを取得
ax1.legend(lines + [batch_size_line], labels + [r'Batch Size $(\delta = 2.0)$'], loc='upper left')

# 保存ディレクトリの作成
save_dir = 'experiment/momentum/figure3'
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'scheduler_3.png')

plt.tight_layout()
plt.savefig(save_path)
plt.show()
