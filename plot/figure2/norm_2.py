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
    'linear': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_decay_lr\\linear\\norm.csv',
    'polynomial (p=2.0)': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_decay_lr\\poly_2.0\\norm.csv',
    'diminishing': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_decay_lr\\diminishing\\norm.csv',
    'cosine': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_decay_lr\\cosine\\norm.csv'
}

dataframes = {}
for key, path in file_paths.items():
    dataframes[key] = pd.read_csv(path)

plt.figure(figsize=(12, 6))
for key, df in dataframes.items():
    plt.plot(df['Epoch'], df['Full Gradient Norm'], label=f'{key}')

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

save_dir = 'experiment/momentum/figure2'
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'norm_2.png')

plt.savefig(save_path)
plt.show()
