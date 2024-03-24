Official implementation of FedPBC for submission to *IEEE Transactions on Signal Processing*

The code is adapted from https://github.com/IBM/fedau, which is under MIT license.

The numerical section compares in total of 6 baseline algorithms with our proposed FedPBC, including FedAvg, FedAvg-known statistics, FedAvg-all, F3AST, FedAU, MIFA.

The corresponding method names in implementation are 'fedpbc', 'fedavg', 'fedknown', 'fedall', 'f3ast', 'mifa'.

#### examples
The detailed configurations can be found in config.py

python -m main --method fedavg --dataset cifar10 --


The results are stored under 'csvs' folder as csvs.