import numpy as np
from math import pi
import random
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.lin1 = nn.Linear(1, 64)
        self.norm1 = nn.LayerNorm(64)
        self.lin2 = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.lin3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.norm1(self.lin1(x)))
       #  x = F.dropout(x, 0.5)
        x = F.relu(self.norm2(self.lin2(x)))
       #  x = F.dropout(x, 0.5)
        return self.lin3(x)


def query_sin_task(phase, amplitude, k=50):
    x = np.random.uniform(-5.0, 5.0, size=(k,))
    x = np.array(sorted(x))
    y = np.sin(x + phase) * amplitude
    return torch.tensor(x, dtype=torch.float).view(k, 1), torch.tensor(y, dtype=torch.float).view(k, 1)

if __name__ == '__main__':
    inner_lr = 0.01
    outer_lr = 0.001

    writer = SummaryWriter('runs/LayerNorm')

    model = RegressionModel()
    meta_model = RegressionModel()

    for ep in range(500000):
        phase = random.uniform(0, 2 * pi)
        amplitude = random.uniform(0.1, 5)

        x, y = query_sin_task(phase, amplitude, k=10)
        val_x, val_y = query_sin_task(phase, amplitude, k=50)

        meta_model.load_state_dict(model.state_dict())
        optimizer = optim.SGD(params=meta_model.parameters(), lr=inner_lr)
        for i in range(10):
            meta_model.train()
            pred = meta_model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss at ' + str(i), loss.item(), ep)

            meta_model.eval()
            test_out = meta_model(val_x)
            val_loss = F.mse_loss(test_out, val_y)

            writer.add_scalar('Val loss at ' + str(i), val_loss.item(), ep)

            if ep % 5000 == 0:
                res, = plt.plot(test_out.detach().numpy())
                plt.legend([res], [str(i) + '_result'])
        if ep % 5000 == 0:
            base, = plt.plot(val_y.numpy())
            plt.legend([base], 'Base truth')
            plt.savefig('runs/' + str(ep) + '_val')
            plt.clf()

        old_state_dict = model.state_dict()
        for p in old_state_dict:
            old_state_dict[p] = old_state_dict[p] * (1 - outer_lr) + meta_model.state_dict()[p] * outer_lr
        model.load_state_dict(old_state_dict)
