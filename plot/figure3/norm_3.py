import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter
import numpy as np

# 10^n形式で表示するフォーマット関数を定義
def log_format(y, _):
    if y == 0:
        return "0"
    exponent = int(np.log10(y))
    return r"$10^{{{}}}$".format(exponent)

file_paths = {
    'constant': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_decay_lr\\constant\\norm.csv',
    'x': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_incr_lr\\exp_growth\\min0.1_max0.2\\norm.csv',
    'y': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_incr_lr\\exp_growth\\min0.1_max0.5\\norm.csv',
    'z': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_incr_lr\\exp_growth\\min0.1_max1.0\\norm.csv',
}

dataframes = {}
for key, path in file_paths.items():
    dataframes[key] = pd.read_csv(path)
    
labels = {
    'constant': 'constant',
    'x': r'$\eta_{max} = 0.2 (\gamma \approx 1.080)$',
    'y': r'$\eta_{max} = 0.5 (\gamma \approx 1.196)$',
    'z': r'$\eta_{max} = 1.0 (\gamma \approx 1.292)$'
}

plt.figure(figsize=(12, 6))
for key, df in dataframes.items():
    plt.plot(df['Epoch'], df['Full Gradient Norm'], label=labels[key])

# y軸を対数スケールに設定
plt.yscale('log')

# y軸ラベルのフォーマットを10^0, 10^1, 10^2形式に変更
ax = plt.gca()
ax.yaxis.set_major_formatter(FuncFormatter(log_format))

plt.xlabel('Epoch')
plt.ylabel('Full Gradient Norm of Empirical Loss of Training')
plt.title('ResNet-18 on CIFAR100')
plt.legend()
plt.grid(True)

save_dir = 'experiment/momentum/figure3'
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'norm_3.png')

plt.savefig(save_path)
plt.show()
