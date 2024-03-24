import torch
from torch.utils.data import DataLoader
from config import *
from dataset.dataset import *
from statistic.collect_stat import stats_collector
from util.util import data_participation_each_node, data_participation_each_node_per_round, DatasetSplit, WorkerSampler
import numpy as np
import random
from model.model import Model

def varying_dynamics():
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

        rounds = 0
        
        not_participate_count_at_node = []
        participate_intervals_at_node = []

        for n in range(num_clients):
            not_participate_count_at_node.append(0)
            participate_intervals_at_node.append([])
        
        if algorithm == 'mifa':
            update_per_node = []
            for n in range(num_clients):
                update_per_node.append(torch.zeros(w_global.shape[0]).to('cpu'))
            update_per_node = torch.stack(update_per_node)

        
        s = unreliable.split('-')
        method = s[0]

        if algorithm == 'f3ast':
            # long-term ratio
            r = np.array([1 for i in range(num_clients)])
        
        
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

            for n in range(num_clients):
                    worker_sampler = worker_samplers[n]

                    if worker_sampler.sample():
                        participation[n] = True

            if algorithm == 'f3ast':
                # determine the second sample set
                if sum(participation) < 10:
                    pass
                else:
                    r2 = 1 / r**2
                    r2_reducted = r2[participation]
                    top_10 = np.argsort(r2_reducted)[::-1][:10]
                    part = np.array([False for i in range(num_clients)])
                    part[top_10] = True
                    participation = part
                beta = 1e-2
                r = (1 - beta) * r + beta * participation
            
            for n in range(num_clients):
                
                if participation[n]:
                    model.assign_weight(w_global)
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
                    w_tmp -= w_global  

                    agg_weight = None
                    if algorithm == 'fedknown':
                        w_tmp /= worker_sampler.participation_prob
                        agg_weight = 1/worker_sampler.participation_prob
                    elif algorithm == 'fedau':
                        if len(participate_intervals_at_node[n]) > 0:
                            agg_weight = np.mean(participate_intervals_at_node[n])
                            w_tmp *= agg_weight
                    elif algorithm == 'f3ast':
                        w_tmp /= (r[n])
                    elif algorithm == 'mifa':
                        update_per_node[n] = w_tmp.to('cpu')

                    participate_intervals_at_node[n].append(not_participate_count_at_node[n] + 1) 
                    not_participate_count_at_node[n] = 0

                else:
                    participation[n] = False
                    not_participate_count_at_node[n] += 1

                    if not_participate_count_at_node[n] >= 50:
                        participate_intervals_at_node[n].append(not_participate_count_at_node[n])
                        not_participate_count_at_node[n] = 0
                    w_tmp = 0.0

                if algorithm != 'mifa':
                    if accumulated == 0:
                        w_accumulate = w_tmp
                    else:
                        w_accumulate += w_tmp

                    if algorithm != 'fedavg':
                        accumulated += 1
                    elif participation[n]:
                        accumulated += 1

            if algorithm != 'mifa':
                if accumulated > 0:
                    w_tmp_a = torch.div(w_accumulate, torch.tensor(accumulated).to(device)).view(-1)
                
                else:
                    w_tmp_a = torch.zeros(w_global.shape[0]).to(device)
            else:
                w_tmp_a = torch.mean(update_per_node, 0).to(device)

            w_global += torch.tensor(lr_global).to(device) * w_tmp_a

            rounds += 5

            w_eval = w_global

            if rounds % eval_freq == 0:
                stat.collect_stat_eval(seed, rounds, model, data_train_loader, data_test_loader, w_eval)



        torch.cuda.empty_cache()
