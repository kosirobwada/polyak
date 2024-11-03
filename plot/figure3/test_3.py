import pandas as pd
import matplotlib.pyplot as plt
import os

file_paths = {
    'constant': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_decay_lr\\constant\\test.csv',
    'x': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_incr_lr\\exp_growth\\min0.1_max0.2\\test.csv',
    'y': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_incr_lr\\exp_growth\\min0.1_max0.5\\test.csv',
    'z': 'result\\csv\\CIFAR100\\ResNet18\\incr_bs_incr_lr\\exp_growth\\min0.1_max1.0\\test.csv',
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
    plt.plot(df['Epoch'], df['Test Accuracy'], label=labels[key])

plt.xlabel('Epoch')
plt.ylabel('Accuracy Score for Test')
plt.title('ResNet-18 on CIFAR100')
plt.legend()
plt.grid(True)

save_dir = 'experiment/momentum/figure3'
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'test_3.png')

# plt.tight_layout()
plt.savefig(save_path)

plt.show()
