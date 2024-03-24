import torch
from torch.utils.data import DataLoader
from config import *
from dataset.dataset import *
from statistic.collect_stat import stats_collector
from util.util import data_participation_each_node, data_participation_each_node_per_round, DatasetSplit, WorkerSampler
import numpy as np
import random
from model.model import Model

def varying_dynamics_pbc():
    stat = stats_collector(prefix=prefix)
    
    for seed in seeds:

        # random_initializations
        random.seed(seed)
        np.random.seed(seed)  
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed)  
        torch.backends.cudnn.deterministic = True  

        data_train, data_test = load_data(dataset, dataset_file_path, 'cpu')
        data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=0)
        data_test_loader = DataLoader(data_test, batch_size=256, num_workers=0)
        dict_users, participation_prob_each_node_init, actual_label_distributions_each_node, num_labels = data_participation_each_node(data_train, num_clients)        

        step_size_local = lr_local_init

        model = Model(seed, step_size_local, model_name=model_name, device=device, flatten_weight=True)

        train_loader_list = []
        dataiter_list = []
        for n in range(num_clients):
            train_loader_list.append(
                DataLoader(DatasetSplit(data_train, dict_users[n]), batch_size=batch_size_train, shuffle=True))
            dataiter_list.append(iter(train_loader_list[n]))


        def sample_minibatch(n):
            try:
                images, labels = next(dataiter_list[n])
                if len(images) < batch_size_train:
                    dataiter_list[n] = iter(train_loader_list[n])
                    images, labels = next(dataiter_list[n])
            except StopIteration:
                dataiter_list[n] = iter(train_loader_list[n])
                images, labels = next(dataiter_list[n])

            return images, labels

        
        w_global = model.get_weight()   
        w_local = torch.stack([w_global.detach().to('cpu') for i in range(num_clients)])

        rounds = 0
             
        s = unreliable.split('-')
        method = s[0]
   
        while rounds < total_rounds:

            participation_prob_each_node = data_participation_each_node_per_round(participation_prob_each_node_init, rounds)
      
            if method == 'cyclic' and not (rounds / 5) % 100:
                worker_samplers = []
                for n in range(num_clients):
                    worker_samplers.append(WorkerSampler(unreliable, participation_prob_each_node[n]))
            
            elif method == 'cyclic':
                pass
                
            else:
                worker_samplers = []
                for n in range(num_clients):
                    worker_samplers.append(WorkerSampler(unreliable, participation_prob_each_node[n]))

            step_size_local_round = step_size_local / np.sqrt(rounds / 50+1) 
            model.update_learning_rate(step_size_local_round)

            participation = np.array([False for i in range(num_clients)])

            accumulated = 0
            w_accumulate = torch.zeros_like(w_global)

            for n in range(num_clients):
                worker_sampler = worker_samplers[n]

                if worker_sampler.sample():
                    participation[n] = True
                
                model.assign_weight(w_local[n].to('cuda'))
                model.model.train()

                for i in range(0, 5):
                    images, labels = sample_minibatch(n)

                    images, labels = images.to(device), labels.to(device)

                    if transform_train is not None:
                        images = transform_train(images).contiguous()  # contiguous() needed due to the use of ColorJitter in CIFAR transforms

                    model.optimizer.zero_grad()
                    output = model.model(images)
                    loss = model.loss_fn(output, labels)
                    loss.backward()
                    model.optimizer.step()

                w_tmp = model.get_weight()  
                w_tmp += (w_tmp - w_local[n].to('cuda')) * torch.tensor(lr_global).to(device)

                w_local[n] = w_tmp.to('cpu')

               
                if participation[n]:
                    w_accumulate += w_tmp 
                    accumulated += 1

            if accumulated > 0:
                w_global = torch.div(w_accumulate, torch.tensor(accumulated).to(device)).view(-1)
                accumulated = 0
            else:
                pass 

            for n in range(num_clients):
                if participation[n]:
                    w_local[n] = w_global.detach().to('cpu')

            rounds = rounds + 5

            w_eval = sum(w_local).to(device) / len(w_local)            

            if rounds % eval_freq == 0:
                stat.collect_stat_eval(seed, rounds, model, data_train_loader, data_test_loader, w_eval)

        torch.cuda.empty_cache()
