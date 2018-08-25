from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


from util import GutenbergConstructor, RedditDatasetConstructor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 100


class CNN_Classification(nn.Module):
    def __init__(self, embedding_weights, hidden_size, embedding_dim, n_words, output_size, input_size):
        super(CNN_Classification, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(n_words, embedding_dim)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
            # self.embedding.requires_grad = False

        self.dropout = nn.Dropout(0.5)
        self.max_pool = nn.MaxPool1d(2)

        self.cnn1 = nn.Conv1d(embedding_dim, hidden_size, 7, padding=3)
        self.norm1 = nn.LayerNorm([hidden_size, input_size])
        self.activ1 = nn.PReLU(hidden_size)

        self.cnn2 = nn.Conv1d(hidden_size, hidden_size*2, 7, padding=3)
        self.norm2 = nn.LayerNorm([hidden_size*2, input_size//2])
        self.activ2 = nn.PReLU(hidden_size*2)

        self.cnn3 = nn.Conv1d(hidden_size*2, hidden_size*4, 7, padding=3)
        self.norm3 = nn.LayerNorm([hidden_size*4, input_size//4])
        self.activ3 = nn.PReLU(hidden_size*4)

        self.penultimate_layer = nn.Linear(hidden_size*4, hidden_size*4)
        self.norm4 = nn.LayerNorm(hidden_size*4)
        self.activ4 = nn.PReLU(hidden_size*4)

        self.final_layer = nn.Linear(hidden_size*4, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.embedding(input)

        # Why do I have to do this
        x = self.cnn1(x.transpose(-1, -2))
        x = self.norm1(x)
        x = self.activ1(x)
        x = self.max_pool(x)

        x = self.cnn2(x)
        x = self.norm2(x)
        x = self.activ2(x)
        x = self.max_pool(x)

        x = self.cnn3(x)
        x = self.norm3(x)
        x = self.activ3(x)
        # Global max pooling
        x = F.max_pool1d(x, kernel_size=x.shape[-1]).view(-1, self.hidden_size*4)

        x = self.penultimate_layer(x)
        x = self.norm4(x)
        x = self.activ4(x)

        x = self.final_layer(x.view(x.shape[0], -1))
        return x


def reptile_author_recognition(examples_size=512, examples=10, different_authors=5, hidden_size=64):
    meta_env = RedditDatasetConstructor()
    inner_lr = 0.01
    outer_lr = 0.001

    writer = SummaryWriter('RedditAuthorRecognition/ex_size_{}_examples_{}_diff_authors_{}'.format(examples_size, examples, different_authors))

    model = CNN_Classification(meta_env.glove_embedding, hidden_size, 100, meta_env.n_words, different_authors, examples_size).to(device)
    meta_model = CNN_Classification(meta_env.glove_embedding, hidden_size, 100, meta_env.n_words, different_authors, examples_size).to(device)

    criterion = nn.CrossEntropyLoss()

    for ep in range(500000):
        x, y, val_x, val_y = meta_env.get_n_task(different_authors, examples, examples_size)

        meta_model.load_state_dict(model.state_dict())
        optimizer = optim.SGD(params=meta_model.parameters(), lr=inner_lr)
        for i in range(10):
            meta_model.train()
            pred = meta_model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = accuracy_score(np.argmax(pred.detach().cpu().numpy(), axis=-1), y.cpu().numpy())

            writer.add_scalar('Train loss at ' + str(i), loss.item(), ep)
            writer.add_scalar('Train accuracy at ' + str(i), accuracy, ep)

            meta_model.eval()
            test_out = meta_model(val_x)
            val_loss = criterion(test_out, val_y)

            val_accuracy = accuracy_score(np.argmax(test_out.detach().cpu().numpy(), axis=-1), val_y.cpu().numpy())

            writer.add_scalar('Train test loss at ' + str(i), val_loss.item(), ep)
            writer.add_scalar('Train test accuracy at ' + str(i), val_accuracy, ep)

        old_state_dict = model.state_dict()
        for p in old_state_dict:
            old_state_dict[p] = old_state_dict[p] * (1 - outer_lr) + meta_model.state_dict()[p] * outer_lr
        model.load_state_dict(old_state_dict)

        if ep % 100 == 0:
            x, y, val_x, val_y = meta_env.get_validation_task(different_authors, examples, examples_size)

            meta_model.load_state_dict(model.state_dict())
            optimizer = optim.SGD(params=meta_model.parameters(), lr=inner_lr)
            for i in range(10):
                meta_model.train()
                pred = meta_model(x)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                accuracy = accuracy_score(np.argmax(pred.detach().cpu().numpy(), axis=-1), y.cpu().numpy())

                writer.add_scalar('Val loss at ' + str(i), loss.item(), ep)
                writer.add_scalar('Val accuracy at ' + str(i), accuracy, ep)

                meta_model.eval()
                test_out = meta_model(val_x)
                val_loss = criterion(test_out, val_y)

                val_accuracy = accuracy_score(np.argmax(test_out.detach().cpu().numpy(), axis=-1), val_y.cpu().numpy())

                writer.add_scalar('Val test loss at ' + str(i), val_loss.item(), ep)
                writer.add_scalar('Val test accuracy at ' + str(i), val_accuracy, ep)

        if ep % 50000 == 0:
            torch.save(model.state_dict(), 'model_reddit_' + str(ep))


if __name__ == '__main__':
    reptile_author_recognition(examples_size=128, different_authors=20, examples=10)
