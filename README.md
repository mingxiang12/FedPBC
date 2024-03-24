### Official implementation of FedPBC for submission to *IEEE Transactions on Signal Processing*

The code is adapted from https://github.com/IBM/fedau, which is under MIT license.

The numerical section compares in total of 6 baseline algorithms with our proposed FedPBC, including FedAvg, FedAvg-known statistics, FedAvg-all, F3AST, FedAU, MIFA.

#### examples

python -m main --method fedavg --dataset cifar10 --lr 0.05 --fluctuate 1 --sigma 10. --prob_lower 0.02 --dirich_alpha 0.1 --num_clients 100 --time_varying 1

#### configurations

We list only the key parameters here:

* method: FedAvg, 'fedavg'; FedPBC, 'fedpbc'; FedAvg-known, 'fedknown'; FedAvg-all, 'fedall'; F3AST, 'f3ast'; MIFA: 'mifa'.
* fluctuate: 1: $$\gamma = 0.4$$; 2: $$\gamma = 0.2$$; 3: $$\gamma = 0.1$$.
* sigma: $$\sigma_0$$.
* prob_lower: $$\delta$$.
* dirich_alpha: Dirichlet $$\alpha$$

Please find the detailed configurations in 'config.py'.

The results are stored under 'csvs' folder as csvs.