### This is a modified version of main.py from https://github.com/facebookresearch/GradientEpisodicMemory 
### The most significant changes are to the arguments: 1) allowing for new models and 2) changing the default settings in some cases

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import datetime
import argparse
import random
import uuid
import time
import os

import numpy as np

import torch
from torch.autograd import Variable
from metrics.metrics import confusion_matrix
import matplotlib.pyplot as plt

# continuum iterator #########################################################


def load_datasets(args):
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max())
        n_outputs = max(n_outputs, d_te[i][2].max())

    # Woody: Dataset summary: 60,000 examples of each task for train. 10,000 of each task for test. MNIST images are flattened with shape 784.
    # Woody: d_te[insert task number][0] gives you the rotation used to create the dataset, d_te[insert task number][1] gives the inputs, and d_te[insert task number][2] gives the labels
    # Woody: Remember labels in MNIST are in [0, 9], so a random baseline gets 10% accuracy. 
    #print(d_te[0][1].shape) # Woody: Rotations: torch.Size([10000, 784])
    #print(d_tr[0][1].shape) # Woody: Rotations: torch.Size([60000, 784])
    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)

"""
    Defines the data stream
"""
class Continuum:

    def __init__(self, data, args):
        self.data = data
        self.batch_size = args.batch_size
        n_tasks = len(data)
        task_permutation = range(n_tasks)

        if args.shuffle_tasks == 'yes':
            task_permutation = torch.randperm(n_tasks).tolist()

        sample_permutations = []

        for t in range(n_tasks):
            N = data[t][1].size(0)
            # Either train on all the task data or sub-sample it
            if args.samples_per_task <= 0:
                n = N
            else:
                n = min(args.samples_per_task, N)

            p = torch.randperm(N)[0:n]
            sample_permutations.append(p)

        self.permutation = []

        for t in range(n_tasks):
            task_t = task_permutation[t]
            for _ in range(args.n_epochs):
                # Create a random permutation for each epoch of the data
                # we are using for task_t
                task_p = [[task_t, i] for i in sample_permutations[task_t]]
                random.shuffle(task_p)
                self.permutation += task_p

        self.length = len(self.permutation)
        print(self.length)
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            ti = self.permutation[self.current][0] # Task index - note we are not using this for now!
            j = []
            i = 0
            while (((self.current + i) < self.length) and
                   (self.permutation[self.current + i][0] == ti) and
                   (i < self.batch_size)):
                j.append(self.permutation[self.current + i][1])
                i += 1
            self.current += i
            j = torch.LongTensor(j)
            # Get the randomly shuffled data-samples for task ti
            return self.data[ti][1][j], ti, self.data[ti][2][j]

# train handle ###############################################################


def eval_tasks(model, tasks, args, current_task, mistake_type="current"):
    model.eval()
    result = []
    mistakes = np.zeros(0)
    for i, task in enumerate(tasks):
        t = i
        x = task[1]
        y = task[2]
        rt = 0
        
        eval_bs = x.size(0)

        with torch.no_grad():  # torch 0.4+
            for b_from in range(0, x.size(0), eval_bs):
                b_to = min(b_from + eval_bs, x.size(0) - 1)
                if b_from == b_to:
                    xb = x[b_from].view(1, -1)
                    yb = torch.LongTensor([y[b_to]]).view(1, -1)
                else:
                    xb = x[b_from:b_to]
                    yb = y[b_from:b_to]
                if args.cuda:
                    xb = xb.cuda()
                # xb = Variable(xb, volatile=True)  # torch 0.4+
                # Get the model predictions!
                _, pb = torch.max(model(xb, t).data.cpu(), 1, keepdim=False)
                rt += (pb == yb).float().sum()

                # Track the numbers that are incorrectly predicted
                # Only track for tasks we has learned on!
                #if (t <= current_task):
                if (mistake_type == "current" and current_task == t):
                    gt_wrong = yb[pb != yb].detach().numpy()
                    mistakes = np.concatenate((mistakes, gt_wrong), axis=-1)

        result.append(rt / x.size(0))

    return result, mistakes


def life_experience(model, continuum, x_te, args):
    result_a = [] # Woody: result_a is a list of lists of average accuracy for each task at the speciic t!
    result_t = [] # Woody: result_t is a list of the current task indices
    mistakes_t = [] # Jon: mistakes_t is a list of lists of the mistakes made by the model over tasks 1...t

    current_task = 0
    time_start = time.time()
    

    for (i, (x, t, y)) in enumerate(continuum):
        # Log if we are switching tasks!
        if(((i % args.log_every) == 0) or (t != current_task)):
            print(i) # Woody debugging
            result, mistakes = eval_tasks(model, x_te, args, current_task)
            mistakes_t.append(mistakes)
            result_a.append(result)
            result_t.append(current_task)
            current_task = t

            # When we switch tasks is probably the best place to keep track of
            # the priorities for later visualization
            if (len(model.priorities) == model.memories):
                model.priority_tracker.append(model.priorities.copy())

        v_x = x.view(x.size(0), -1)
        v_y = y.long()

        if args.cuda:
            v_x = v_x.cuda()
            v_y = v_y.cuda()

        model.train()
        model.observe(Variable(v_x), t, Variable(v_y))

    # Get the mistakes at the end of training 
    result, mistakes = eval_tasks(model, x_te, args, current_task)
    mistakes_t.append(mistakes)
    result_a.append(result)
    result_t.append(current_task)

    time_end = time.time()
    time_spent = time_end - time_start

    return torch.Tensor(result_t), torch.Tensor(result_a), time_spent, mistakes_t

def visualize_priorities(priorities):
    # Plots a curve for each memory replay t 
    # as the priorities for that replay. Makes
    # sure to sort the priorities.
    for priority_t in priorities:
        sorted_p_t = sorted(priority_t)
        plt.plot(np.arange(len(sorted_p_t)), sorted_p_t)

    plt.show()

"""
    Jon
    ------------
    At the end of training, show a grid of histogram plots
    demonstrating the mistakes made on prior tasks 1,...,t at
    the time after learning task t
"""
def mistakes_histogram(mistakes_t):
    fig, ax = plt.subplots(4, 5, sharex='all', sharey='all')

    for row in range(4):
        for col in range(5):
            counts, _ = np.histogram(mistakes_t[5 * row + col], bins=np.arange(11))
            x = np.arange(10)
            ax[row, col].bar(x, counts)
            #plt.hist(mistakes, bins=np.arange(11))
    plt.xticks(x, x)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continuum learning')

    # Woody: extra args for caml
    parser.add_argument('--caml_priority', type=str, default='loss',
                        help='how to prioritize sampling in caml')
    parser.add_argument('--softmax_temperature', type=float, default=1.0,
                        help='temperature for softmax in replay buffer sampling')

    # model details
    parser.add_argument('--model', type=str, default='single',
                        help='model to train')
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--finetune', default='yes', type=str,help='whether to initialize nets in indep. nets')
    
    # optimizer parameters influencing all models
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='the amount of items received by the algorithm at one time (set to 1 across all experiments). Variable name is from GEM project.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')

    # memory parameters for GEM baselines
    parser.add_argument('--n_memories', type=int, default=0,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')

    # parameters specific to models in https://openreview.net/pdf?id=B1gTShAct7 
    
    parser.add_argument('--memories', type=int, default=5120, help='number of total memories stored in a reservoir sampling based buffer')

    parser.add_argument('--gamma', type=float, default=1.0,
                        help='gamma learning rate parameter') #gating net lr in roe 

    parser.add_argument('--batches_per_example', type=float, default=1,
                        help='the number of batch per incoming example')

    parser.add_argument('--s', type=float, default=1,
                        help='current example learning rate multiplier (s)')

    parser.add_argument('--replay_batch_size', type=float, default=20,
                        help='The batch size for experience replay. Denoted as k-1 in the paper.')

    parser.add_argument('--beta', type=float, default=1.0,
                        help='beta learning rate parameter') # exploration factor in roe
    
    # experiment parameters
    parser.add_argument('--cuda', type=str, default='no',
                        help='Use GPU?')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed of model')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models at the end of training')

    # data parameters
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--data_file', default='mnist_permutations.pt',
                        help='data file')
    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')
    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')
    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    args.finetune = True if args.finetune == 'yes' else False

    # taskinput model has one extra layer
    if args.model == 'taskinput':
        args.n_layers -= 1

    # unique identifier
    uid = uuid.uuid4().hex

    # initialize seeds    
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        print("Found GPU:", torch.cuda.get_device_name(0))
        torch.cuda.manual_seed_all(args.seed)

    # load data
    x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)
    n_outputs = n_outputs.item()  # outputs should not be a tensor, otherwise "TypeError: expected Float (got Long)"

    # set up continuum
    continuum = Continuum(x_tr, args)

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    if args.cuda:
        try:
            model.cuda()
        except:
            pass 

    # run model on continuum
    result_t, result_a, spent_time, mistakes_t = life_experience(
        model, continuum, x_te, args)

    # Jon:
    # Visualize the evolution of the priorities
    visualize_priorities(model.priority_tracker)

    # Visualize the class mistakes as a histogram
    # after training is completed.
    mistakes_histogram(mistakes_t)

    # prepare saving path and file name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    fname = args.model + '_' + args.data_file + '_'
    fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += '_' + uid
    fname = os.path.join(args.save_path, fname)

    # save confusion matrix and print one line of stats
    stats = confusion_matrix(result_t, result_a, fname + '.txt')
    one_liner = str(vars(args)) + ' # '
    one_liner += ' '.join(["%.3f" % stat for stat in stats])
    print(fname + ': ' + one_liner + ' # ' + str(spent_time))

    # save all results in binary file
    torch.save((result_t, result_a, model.state_dict(),
                stats, one_liner, args), fname + '.pt')
