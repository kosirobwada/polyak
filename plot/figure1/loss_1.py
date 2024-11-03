import pandas as pd
import matplotlib.pyplot as plt
import os

file_paths = {
    'constant': 'result\\csv\\CIFAR100\\ResNet18\\polyak\\const_bs_decay_lr\\constant\\train.csv',
    'linear': 'result\\csv\\CIFAR100\\ResNet18\\polyak\\const_bs_decay_lr\\linear\\train.csv',
    'polynomial (p=2.0)': 'result\\csv\\CIFAR100\\ResNet18\\polyak\\const_bs_decay_lr\\poly_2.0\\train.csv',
    'diminishing': 'result\\csv\\CIFAR100\\ResNet18\\polyak\\const_bs_decay_lr\\diminishing\\train.csv',
    'cosine': 'result\\csv\\CIFAR100\\ResNet18\\polyak\\const_bs_decay_lr\\cosine\\train.csv'
}

dataframes = {}
for key, path in file_paths.items():
    try:
        dataframes[key] = pd.read_csv(path)
        print(f"{key}: Loaded successfully.")
    except Exception as e:
        print(f"Failed to load {key} from {path}: {e}")

plt.figure(figsize=(12, 6))
for key, df in dataframes.items():
    plt.plot(df['Epoch'], df['Train Loss'], label=key)
    
plt.xlabel('Epochs')
plt.ylabel('Empirical Loss Value for Training')
plt.yscale('log')  # 対数スケールを使用
plt.title('ResNet-18 on CIFAR100')
plt.legend(loc='best')
plt.grid(True)

save_dir = 'experiment/polyak/figure1'
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'loss_1.png')

plt.tight_layout()
plt.savefig(save_path)

plt.tight_layout()
plt.show()
