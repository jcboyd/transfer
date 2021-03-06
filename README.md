# transfer

Transfer learning from ImageNet weights on 6 image datasets. Significant performance gains shown across three axes:

1. Deep (resnet-18) vs. very deep (densenet-161) networks
2. Freezing vs fine-tuning with a randomly-initialised linear head
3. Resizing low resolution data to approximate the network's native resolution

| Dataset  | Accuracy (resnet-18 frozen) | Accuracy (resnet-18 fine-tuned) | Accuracy (densenet-161 frozen) | Accuracy (densenet-161 fine-tuned) |
| --- | --- |--- |--- |--- |
| MNIST (28 x 28) | 74.54 | 98.81 | - | - |
| MNIST (56 x 56) | 85.04 | 99.1 | 89.49 | 99.40 |
| MNIST (112 x 112) | 90.49 | 99.24 | 94.79 | 99.54 |
| MNIST (224 x 224) | 92.14 | 99.25 | 95.56 | 99.50 |
| Fashion-MNIST (28 x 28) | 70.43 | 90.16 | - | - |
| Fashion-MNIST (56 x 56) | 78.61 | 91.71 | 80.06 | 92.94 |
| Fashion-MNIST (112 x 112) | 82.09 | 92.81 | 85.88 | 94.10 |
| Fashion-MNIST (224 x 224) |  82.43 | 92.16 | 86.97 | 94.39 |
| CIFAR10 (32 x 32) | 44.87 | 76.07 | 54.56 | 84.59 |
| CIFAR10 (64 x 64) | 63.66 | 87.90 | 72.74 | 93.34 |
| CIFAR10 (128 x 128) | 77.18 | 92.60 | 85.9 | 96.39 |
| CIFAR10 (256 x 256) | 72.65 | 92.37 | 77.56 | 96.90 |
| CIFAR100 (32 x 32) | 25.87 | 50.94 | 35.11 | 61.57 |
| CIFAR100 (64 x 64) | 42.74 | 67.79 | 54.77 | 78.39 |
| CIFAR100 (128 x 128) | 55.18 | 76.32 | 67.92 | 84.78 |
| CIFAR100 (256 x 256) | 47.66 | 76.06 | 57.04 | 83.98 |
| SVHN (32 x 32) | 33.06 | 89.61 | 39.31 | 94.36 |
| SVHN (64 x 64) | 37.00 | 92.80 | 42.38 | 95.61 |
| SVHN (128 x 128) | 42.26 | 94.12 | 50.71 | 96.52 |
| SVHN (256 x 256) | 40.24 | 94.33 | 48.16 | 96.76 |
| PCam (96 x 96) | 77.14 | 83.31 | 75.55 | 83.14 |
| PCam (192 x 192) | 80.50 | 84.66 | 81.29 | 86.21 |
