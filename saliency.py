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

from main import load_datasets

"""
    Stolen from CS231N
"""
def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, H*W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H*W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()
    
    # Make input tensor require gradient
    X.requires_grad_()
    
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = model(X, 0) # Not sure about the 0
    # loss = self.bce(prediction, by)
    # Gather just the correct scores
    # Not sure why we did this instead of the loss!
    scores = scores.gather(1, y.view(-1, 1),).squeeze()
    loss = torch.sum(scores)
    
    loss.backward()
    # Now actually get step
    X_grad = X.grad
    saliency = torch.abs(X_grad)
    return saliency

def main():
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

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)

    result_t, result_a, model_state_dict, stats, one_liner, args = torch.load('woody_results/online_mnist_rotations.pt_2020_11_01_11_19_37_f37e2305e6e04d61ab498c9bf252fe97.pt')
    #result_t, result_a, model_state_dict, stats, one_liner, args = torch.load('woody_results/caml1_mnist_rotations.pt_2020_11_01_13_58_46_0c7287daad494c818e6d5ce206b16b0b.pt')
    model.load_state_dict(model_state_dict)
    model.eval()
    
    if args.cuda:
        try:
            model.cuda()
        except:
            pass 

    # Test this saliency shit on two data points
    # From the final task train set
    task_num = 0
    saliency_idxes = [7, 1, 105]
    x = x_tr[task_num][1][saliency_idxes]
    y = x_tr[task_num][2][saliency_idxes]

    saliency = compute_saliency_maps(x, y, model)
    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.detach().numpy()
    saliency = saliency.reshape(-1, 28, 28)
    x = x.reshape(-1, 28, 28).detach().numpy()
    N = x.shape[0]
    for i in range(N):
        plt.subplot(2, N, i+1)
        plt.imshow(x[i])
        plt.axis('off')
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.Greens)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()

if __name__ == '__main__':
    main()

