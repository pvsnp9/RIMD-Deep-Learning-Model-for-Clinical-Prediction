import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBLoss():
    def __init__(self, number_of_class, beta=0.99):
        self.number_of_class = number_of_class
        self.beta = beta
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'beta=' + str(self.beta) \
            + ', num classes=' + str(self.number_of_class) + ')'


    def __call__(self, labels, logits, sample_per_class):
        effective_num = 1.0 - np.power(self.beta, sample_per_class)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.number_of_class

        labels_one_hot = F.one_hot(torch.tensor(labels.tolist()), self.number_of_class).float().to(self.device) #F.one_hot(labels, self.number_of_class).float()

        weights = torch.tensor(weights).float().to(self.device)
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,self.number_of_class)

        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights)
        return cb_loss

    
# if __name__ == '__main__':
#     cb = CBLoss(2)
#     print(cb)