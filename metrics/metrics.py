### We directly copied the metrics.py model file from the GEM project https://github.com/facebookresearch/GradientEpisodicMemory

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import torch


def task_changes(result_t):
    n_tasks = int(result_t.max() + 1)
    changes = []
    current = result_t[0]
    for i, t in enumerate(result_t):
        if t != current:
            changes.append(i)
            current = t

    return n_tasks, changes # Woody: n_tasks is number of tasks. changes is a list of indices where the task changes. 


def confusion_matrix(result_t, result_a, fname=None):
    nt, changes = task_changes(result_t)

    baseline = result_a[0] # Woody: this is called baseline because it is the performance on each task before any training has happened
    changes = torch.LongTensor(changes + [result_a.size(0)]) - 1 # Woody: I believe the subtract 1 is to grab the result right before a change and adding the final result_a.size(0) gives the final result
    # Woody: This selects the results from right before every change in task, so the evaluated test accuracy for each task after training on each task sequentially
    result = result_a.index_select(0, torch.LongTensor(changes))  # .index (torch<0.3.1) | .index_select (torch>0.4)

    # acc[t] equals result[t,t]
    # Woody: This is the Learned Accuracy (LA) metric, or the accuracy on each task right after it was learned
    acc = result.diag() # Woody: since result is a 2d vector, this acc variable is a 1d vector of the accuracy of task t right after training on task t (see https://pytorch.org/docs/stable/generated/torch.diag.html)
    fin = result[nt - 1] # Woody: list of final accuracies for each task after training on all data
    # bwt[t] equals result[T,t] - acc[t]
    bwt = result[nt - 1] - acc # Woody: this is the BTI metric (Backward Transfer Interference). This gives a list of length num_tasks with values final task accuracy after training on all data - final task accuracy after training on current task


    # Woody: We can ignore this forward transfer metric, as done in the MER paper. We use bwt as a metric for measuring catastrophic forgetting. Highly negative bwt means catastrophic forgetting. 
    # fwt[t] equals result[t-1,t] - baseline[t]
    fwt = torch.zeros(nt)
    for t in range(1, nt):
        fwt[t] = result[t - 1, t] - baseline[t]

    # Woody: The saved txt file prints a row of the baseline accuracies for each task. The next chunk is the accuracies after training on each task sequentially. 
    if fname is not None:
        f = open(fname, 'w')

        print(' '.join(['%.4f' % r for r in baseline]), file=f)
        print('|', file=f)
        for row in range(result.size(0)):
            print(' '.join(['%.4f' % r for r in result[row]]), file=f)
        print('', file=f)
        print('Diagonal Accuracy: %.4f' % acc.mean(), file=f) # Woody: This is the learned accuracy (LA metric
        print('Final Accuracy: %.4f' % fin.mean(), file=f) # Woody: This is the Retained Accuracy (RA) metric
        print('Backward: %.4f' % bwt.mean(), file=f) # Woody: This is the Backward Transfer Interference (BTI) metric
        print('Forward:  %.4f' % fwt.mean(), file=f) # Woody: We should ignore this metric. Forward transfer is not our concern
        f.close()

    stats = []
    # stats.append(acc.mean())
    stats.append(fin.mean())
    stats.append(bwt.mean())
    stats.append(fwt.mean())

    return stats
