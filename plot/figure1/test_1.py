import pandas as pd
import matplotlib.pyplot as plt
import os

file_paths = {
    'constsant': 'result\\csv\\CIFAR100\\ResNet18\\polyak\\const_bs_decay_lr\\constant\\test.csv',
    'linear': 'result\\csv\\CIFAR100\\ResNet18\\polyak\\const_bs_decay_lr\\linear\\test.csv',
    'polynomial (p=2.0)': 'result\\csv\\CIFAR100\\ResNet18\\polyak\\const_bs_decay_lr\\poly_2.0\\test.csv',
    'diminishing': 'result\\csv\\CIFAR100\\ResNet18\\polyak\\const_bs_decay_lr\\diminishing\\test.csv',
    'cosine': 'result\\csv\\CIFAR100\\ResNet18\\polyak\\const_bs_decay_lr\\cosine\\test.csv'
}

dataframes = {}
for key, path in file_paths.items():
    dataframes[key] = pd.read_csv(path)

plt.figure(figsize=(12, 6))
for key, df in dataframes.items():
    plt.plot(df['Epoch'], df['Test Accuracy'], label=f'{key}')

plt.xlabel('Epoch')
plt.ylabel('Accuracy Score for Test')
plt.title('ResNet-18 on CIFAR100')
plt.legend()
plt.grid(True)

save_dir = 'experiment/polyak/figure1'
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'test_1.png')

# plt.tight_layout()
plt.savefig(save_path)

plt.show()
