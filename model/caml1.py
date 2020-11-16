# An implementation of CAML Algorithm 1 for CS330

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from .common import MLP, ResNet18
import random
from torch.nn.modules.loss import CrossEntropyLoss
from random import shuffle
import sys
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

from scipy.special import softmax

class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        
        nl, nh = args.n_layers, args.n_hiddens
        
        # We should see about trying to do a conv net!
        if (args.model_type == "CNN"):
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                # Need to calculate this!
                nn.Linear(7 * 7 * 64, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, n_outputs),
            )

        self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.model_type = args.model_type

        self.bce = CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)
        self.batchSize = int(args.replay_batch_size)

        self.memories = args.memories
        self.steps = int(args.batches_per_example)
        self.beta = args.beta
        self.gamma = args.gamma

        # allocate buffer
        self.M = []
        self.age = 0

        # Woody
        self.caml_priority = args.caml_priority
        if self.caml_priority == 'dynamic':
            self.dynamic_alpha = 0.05
        self.priorities = [] # This list should mirror the buffer self.M with the priority at each index
        self.temperature = args.softmax_temperature

        # Jon
        # Let us track how the priorities are changing and looking. We basically want
        # to generate some curves to just see how the priorities in the replay buffer
        # are evolving!
        self.priority_tracker = []
        
        # handle gpus if specified
        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda()
            if args.model_type == "CNN":
                self.features = self.features.cuda()
                self.classifier = self.classifier.cuda()


    def forward(self, x, t):
        if self.model_type == "CNN":
            x = x.view(-1, 28, 28) # Check this!
            x = x.unsqueeze(1) # Add the channel dim
            x = self.features(x)
            x = torch.flatten(x, 1)
            output = self.classifier(x)
        else:
            output = self.net(x)
        return output

    # Woody: added this softmax function with temperature
    def softmax(self, logits, temperature=1.0):
        return softmax(np.asarray(logits) / temperature) # as temperature increases, the distribution of softmax becomes more uniform

    """
        Compute the probability distribution P over the prioritized
        replay buffer. Also compute the importance weights.
        Note: For now the importance weighting does not work really at all!!
    """
    # alpha = 0.7 was some hyperparamter set in DQN
    def dqn_stochastic_sampling(self, priorities, alpha=0.6, beta=1):
        # Probability of sampling transition i:
        #   - P(i) = pi^(alpha) / sum(pk^(alpha))
        priority_powers = np.asarray(priorities) ** alpha
        probabilities = priority_powers / np.sum(priority_powers)
        # Compute the importance weights as:
        #   - wi (1 / len(priorities) * 1 / P(i))^(beta)
        # DOES NOT DO WELL MAY LOOK INTO LATER!
        importance_weights = (1./len(priorities) * 1. / probabilities) ** beta
        return probabilities#, importance_weights


    def getBatch(self,x,y,t):
        xi = Variable(torch.from_numpy(np.array(x))).float().view(1,-1)
        yi = Variable(torch.from_numpy(np.array(y))).long().view(1)
        if self.cuda:
            xi = xi.cuda()
            yi = yi.cuda()
        bxs = [xi]
        bys = [yi]
        # If using importance weights!
        # DOES NOT WORK NOW!
        #importance_weights_selected = [0] # Save this for the later setting idx[0] = max weight
        
        # Woody: add indices_sampled as a variable that keeps track of all the indices of examples in the buffer that were sampled
        indices_sampled = []
        if len(self.M) > 0:
            order = [i for i in range(0,len(self.M))]
            osize = min(self.batchSize,len(self.M))

            # Woody: prioritized sampling
            curr_probabilities = self.softmax(self.priorities, temperature=self.temperature)
            #curr_probabilities = self.dqn_stochastic_sampling(self.priorities)
            #print(curr_probabilities)
            for j in range(0,osize):
                # Old uniform sampling code
                #shuffle(order)
                #k = order[j]

                # Woody: prioritized sampling
                # Jon: Should we set replace = true?
                k = np.random.choice(np.arange(len(self.M)), replace=False, p=curr_probabilities) # a single index drawn from the curr_probabilities distribution
                indices_sampled.append(k)
                # Only if using importance weighting
                #importance_weights_selected.append(importance_weights[k])

                x,y,t = self.M[k]
                xi = Variable(torch.from_numpy(np.array(x))).float().view(1,-1)
                yi = Variable(torch.from_numpy(np.array(y))).long().view(1)
                # handle gpus if specified
                if self.cuda:
                    xi = xi.cuda()
                    yi = yi.cuda()
                bxs.append(xi)
                bys.append(yi)
        # Note exclude importance weights in future.
        # Include the max importance weight for the current train example index 0
        # importance_weights_selected[0] = max(importance_weights_selected)
        return bxs, bys, indices_sampled #, importance_weights_selected
                

    def observe(self, x, t, y):
        ### step through elements of x
        for i in range(0,x.size()[0]):
            self.age += 1
            xi = x[i].data.cpu().numpy()
            yi = y[i].data.cpu().numpy()

            # Odd that we are not doing zero grad for the optimizer but okay
            if (self.model_type == 'CNN'):
                self.features.zero_grad()
                self.classifier.zero_grad()
            else:
                self.net.zero_grad()

            before = deepcopy(self.net.state_dict())
            for step in range(0,self.steps):                
                weights_before = deepcopy(self.net.state_dict())
                # Draw batch from buffer:
                bxs, bys, indices_sampled = self.getBatch(xi,yi,t)

                loss = 0.0
                for idx in range(len(bxs)):
                    # Odd that we are not doing zero grad for the optimizer but okay
                    if (self.model_type == 'CNN'):
                        self.features.zero_grad()
                        self.classifier.zero_grad()
                    else:
                        self.net.zero_grad()
                    
                    bx = bxs[idx]
                    by = bys[idx] 
                    prediction = self.forward(bx,0)
                    loss = self.bce(prediction, by)

                    if idx != 0: # skip the idx of the current xi and yi since they are not from the replay buffer
                        # Update replay buffer priority
                        curr_replay_buffer_idx = indices_sampled[idx - 1]
                        if self.caml_priority == 'loss':
                            self.priorities[curr_replay_buffer_idx] = float(loss)
                        elif self.caml_priority == 'newest':
                            self.priorities[curr_replay_buffer_idx] = self.age
                        elif self.caml_priority == 'oldest':
                            self.priorities[curr_replay_buffer_idx] = -self.age
                        elif self.caml_priority == 'dynamic': 
                            self.priorities[curr_replay_buffer_idx] = float(loss)
                            #self.priorities[curr_replay_buffer_idx] = float(loss) + (self.dynamic_alpha * -self.age)
                        #elif self.priorities == 'loss+old':
                            #sum_old
                            #self.priorities[curr_replay_buffer_idx] = (1-alpha) * float(loss) / 

                    loss.backward()
                    self.opt.step()

                # If using dynamic prioritization, add one to all priority of all examples in the replay buffer
                if self.caml_priority == 'dynamic':
                    self.priorities = list(np.asarray(self.priorities) + self.dynamic_alpha)
                
                weights_after = self.net.state_dict()
                
                # Within batch Reptile meta-update:
                self.net.load_state_dict({name : weights_before[name] + ((weights_after[name] - weights_before[name]) * self.beta) for name in weights_before})
            
            after = self.net.state_dict()
            
            # Across batch Reptile meta-update:
            self.net.load_state_dict({name : before[name] + ((after[name] - before[name]) * self.gamma) for name in before})

            
            # Woody
            current_example_loss = 0.0
            if self.caml_priority == 'loss':
                with torch.no_grad():
                    curr_x = Variable(torch.from_numpy(np.array(xi))).float().view(1,-1)
                    curr_y = Variable(torch.from_numpy(np.array(yi))).long().view(1)
                    prediction = self.forward(curr_x, 0)
                    current_example_loss = self.bce(prediction, curr_y)

            # Reservoir sampling memory update: gives a random sample
            # of what we have seen up till now 
            if len(self.M) < self.memories:
                self.M.append([xi,yi,t])

                # Woody: update priorities
                if self.caml_priority == 'loss':
                    self.priorities.append(current_example_loss.item())
                elif self.caml_priority == 'newest':
                    self.priorities.append(self.age)
                elif self.caml_priority == 'oldest':
                    self.priorities.append(-self.age)
                elif self.caml_priority == 'dynamic': 
                    self.priorities.append(float(loss))
                    #self.priorities.append(float(loss) + (self.dynamic_alpha * -self.age))

            else:
                # Save these set of priorities to do some visualization on
                #self.priority_tracker.append(self.priorities.copy())

                # Jon: should we not do some sort of priority based removal like through a heap?
                p = random.randint(0,self.age) 
                if p < self.memories:
                    self.M[p] = [xi,yi,t]

                    # Woody update priorities
                    if self.caml_priority == 'loss':
                        #print ("adding new memory:", current_example_loss.item())
                        copy = self.priorities.copy()
                        self.priorities[p] = current_example_loss.item()
                        #print (np.linalg.norm(np.array(copy) - np.array(self.priorities)))
                    elif self.caml_priority == 'newest':
                        self.priorities[p] = self.age
                    elif self.caml_priority == 'oldest':
                        self.priorities[p] = -self.age
                    elif self.caml_priority == 'dynamic': 
                        self.priorities[p] = float(loss)
                        #self.priorities[p] = float(loss) + (self.dynamic_alpha * -self.age)

        #self.priority_tracker.append(self.priorities.copy())
                



