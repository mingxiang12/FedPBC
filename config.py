import os
import torch
import argparse
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='svhn, cifar10, cinic10')  

parser.add_argument('--method', type=str, default='f3ast')  # algorithm
parser.add_argument('--time_varying', type=int, default=1)
parser.add_argument('--unreliable', type=str, default='bernoulli', help='bernoulli, markov-0.05, cyclic-100')  
parser.add_argument('--fluctuate', type=int, default=1, help='1: 0.4, 2: 0.2, 3: 0.1')
        
parser.add_argument('--lr', type=float, default=0.1)  
parser.add_argument('--lr-global', type=float, default=1.0) 
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--total_rounds', type=int, default=10000)
parser.add_argument('--seeds', type=str, default='1')  

parser.add_argument('--eval_freq', type=int, default=50)

parser.add_argument('--num_clients', type=int, default=100)

parser.add_argument('--gpu', type=int, default=1)  

parser.add_argument('--dirich_alpha', type=float, default=0.1)
parser.add_argument('--prob_lower', type = float, default=0.02)
parser.add_argument('--sigma', type=float, default=20.)


args = parser.parse_args()

print(', '.join(f'{k}={v}' for k, v in vars(args).items()))

algorithm  = args.method
unreliable = args.unreliable
time_varying = args.time_varying

gpu = bool(args.gpu) and torch.cuda.is_available()
device = 'cuda' if gpu else 'cpu'

dataset_dict ={
    'cifar10': 'CIFAR10',
    'svhn'   : 'SVHN',
    'cinic10': 'CINIC10'
}


if args.dataset == 'svhn':
    model_name = 'cnnsvhn'

elif args.dataset == 'cifar10':
    model_name = 'cnncifar10'

elif args.dataset == 'cinic10':
    model_name = 'cnncinic10'

else:
    raise NotImplementedError



dataset      = dataset_dict[args.dataset]


total_rounds = args.total_rounds * 5


seed_str     = args.seeds.split(',')
seeds        = [int(i) for i in seed_str]

dataset_file_path = os.path.join(os.path.dirname(__file__), 'raw_data')

dirichlet_alpha   = args.dirich_alpha

fluctuate         = args.fluctuate
sigma             = args.sigma
prob_lower        = args.prob_lower

num_clients       = args.num_clients
lr_local_init     = args.lr
lr_global         = args.lr_global


batch_size_train  = args.batch_size
eval_freq         = args.eval_freq


prefix = 'results' + '_' + dataset + '_' + model_name + '_' + algorithm + '_' + unreliable + '_lr' + str(lr_local_init) + \
                        '_lr_global' + str(lr_global) + '_dataAlpha' + str(dirichlet_alpha) + \
                        '_prob_lower'+ str(prob_lower) + '_fluctuate' + str(fluctuate) +  \
                        '_sigma'     + str(sigma)


if dataset == 'CIFAR10' or dataset == 'CINIC10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    transform_train_eval = None
    
elif dataset == 'SVHN':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
    ])
    transform_train_eval = None

else:
    pass