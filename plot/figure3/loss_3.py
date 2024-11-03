import pandas as pd
import matplotlib.pyplot as plt
import os

file_paths = {
    'constant': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_decay_lr\\constant\\train.csv',
    'x': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_incr_lr\\exp_growth\\min0.1_max0.2\\train.csv',
    'y': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_incr_lr\\exp_growth\\min0.1_max0.5\\train.csv',
    'z': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_incr_lr\\exp_growth\\min0.1_max1.0\\train.csv',
}

dataframes = {}
for key, path in file_paths.items():
    try:
        dataframes[key] = pd.read_csv(path)
        print(f"{key}: Loaded successfully.")
    except Exception as e:
        print(f"Failed to load {key} from {path}: {e}")

# Learning Rate のプロットと凡例ラベルの設定
labels = {
    'constant': 'constant',
    'x': r'$\eta_{max} = 0.2 (\gamma \approx 1.080)$',
    'y': r'$\eta_{max} = 0.5 (\gamma \approx 1.196)$',
    'z': r'$\eta_{max} = 1.0 (\gamma \approx 1.292)$'
}

plt.figure(figsize=(12, 6))
for key, df in dataframes.items():
    plt.plot(df['Epoch'], df['Train Loss'], label=labels[key])  # 修正箇所: labels[key]を使用
    
plt.xlabel('Epochs')
plt.ylabel('Empirical Loss Value for Training')
plt.yscale('log')  # 対数スケールを使用
plt.title('ResNet-18 on CIFAR100')
plt.legend(loc='best')
plt.grid(True)

save_dir = 'experiment/momentum/figure3'
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'loss_3.png')

plt.tight_layout()
plt.savefig(save_path)

plt.tight_layout()
plt.show()
