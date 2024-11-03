import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import FuncFormatter

# 2^n形式で表示するフォーマット関数を定義
def log_format(y, _):
    if y == 2**6:
        return r'$2^6$'
    elif y == 2**4:
        return r'$2^4$'
    elif y == 2**8:
        return r'$2^8$'
    elif y == 2**10:
        return r'$2^{10}$'
    elif y == 2**12:
        return r'$2^{12}$'
    else:
        return ''

# ファイルパスを設定（適宜変更）
file_paths = {
    'constant': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_decay_lr\\constant\\lr_batch.csv',
    'linear': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_decay_lr\\linear\\lr_batch.csv',
    'polynomial (p=2.0)': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_decay_lr\\poly_2.0\\lr_batch.csv',
    'diminishing': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_decay_lr\\diminishing\\lr_batch.csv',
    'cosine': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_decay_lr\\cosine\\lr_batch.csv'
}

dataframes = {}
for key, path in file_paths.items():
    try:
        dataframes[key] = pd.read_csv(path)
        print(f"{key}: Loaded successfully.")
    except Exception as e:
        print(f"Failed to load {key} from {path}: {e}")

fig, ax1 = plt.subplots(figsize=(12, 6))

# Learning Rate のプロット
for key, df in dataframes.items():
    ax1.plot(df['Epoch'], df['Learning Rate'], label=key)

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Learning Rate')
ax1.set_title('Learning Rate and Batch Size Scheduler')
ax1.grid(True)

# Batch Size のプロット
ax2 = ax1.twinx() 
ax2.set_ylabel('Batch Size')

# y軸をlogスケールに設定
ax2.set_yscale('log')

# y軸を2^6から2^12までに限定し、2^6, 2^7, 2^8を目盛りに設定
ax2.set_ylim([2**(5/2), 2**(25/2)])

# 2^6, 2^7, 2^8を目盛りに設定
ax2.set_yticks([2**4, 2**6 , 2**8, 2**10, 2**12])
ax2.yaxis.set_major_formatter(FuncFormatter(log_format))

# バッチサイズデータをプロット
batch_size_line, = ax2.plot(df['Epoch'], df['Batch Size'], 'm--', label='Batch Size')

# 凡例の表示：ax1とax2の凡例を結合して表示
lines, labels = ax1.get_legend_handles_labels()  # ax1のラインとラベルを取得
ax1.legend(lines + [batch_size_line], labels + ['Batch Size'], loc='upper right', bbox_to_anchor=(1, 0.8))

# 保存ディレクトリの作成
save_dir = 'experiment/momentum/figure2'
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'scheduler_2.png')

plt.tight_layout()
plt.savefig(save_path)
plt.show()
