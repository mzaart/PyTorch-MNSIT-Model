import os
import json
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from .config import DATA_DIR, RESULTS_DIR
from .data import DataSetSamplers, ShuffledDataSet
from .models import DenseNetwork
from .training import RandomSearch, AccuracyScorer, FMeasureScorer


def network_factory(activation, out_func, h_dim=(), drop_out=None, **kwargs):
    return DenseNetwork(28 * 28, 10, activation, out_func, h_dim, drop_out)


# load data sets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# split train data to train and validation
train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
train_dataset = ShuffledDataSet(train_dataset)

samplers = DataSetSamplers(train_dataset, training_set_size=0.8, validation_set_size=0.2)

train_loader = torch.utils.data.DataLoader(train_dataset, sampler=samplers.training_sampler, batch_size=200,)
val_loader = torch.utils.data.DataLoader(train_dataset, sampler=samplers.validation_sampler, batch_size=200)

# construct test loader
test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=200, shuffle=True)

hyper_params = {
    'h_dim': [
        [100, 100],
        [200, 200],
        [100, 100, 100, 100],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [200, 200, 200]
    ],
    'dropout': [
        None,
        0.2,
        0.5
    ],
    'activation': [
        lambda: nn.ReLU()
    ],
    'out_func': [
        nn.Softmax(dim=1), nn.LogSoftmax(dim=1)
    ],
    'epochs': [
        10
    ],
    'optimizer_cls': [
        optim.SGD, optim.Adam
    ],
    'lr': [
        0.1, 0.01, 0.001
    ],
    'loss_func': [
        nn.NLLLoss(),
        nn.CrossEntropyLoss()
    ]
}

random_search = RandomSearch(10, network_factory, hyper_params, AccuracyScorer(), train_loader, val_loader)
model, score, params = random_search.get_tuned_model()


# evaluate model

with torch.no_grad():
    y_actual = torch.zeros(len(test_loader.dataset))
    y_pred = torch.zeros(len(test_loader.dataset))

    for i, (x, y) in enumerate(test_loader):
        x, y = Variable(x), Variable(y)
        x = x.view(-1, 28 * 28)
        pred = model(x).data.max(1)[1]  # get the index of the max log-probability
        batch_size = y.shape[0]
        y_actual[i * batch_size:(i + 1) * batch_size] = y
        y_pred[i * batch_size:(i + 1) * batch_size] = pred

        accuracy = AccuracyScorer().score(y_pred, y_actual)
        fmeasure = FMeasureScorer().score(y_pred, y_actual)
        print('Selected Model: ', model)
        print('Selected hyper params: ', params)
        print('Accuracy: ', accuracy)
        print('F-Measure: ', fmeasure)

        # save results

        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'model.pt'))
        with open(os.path.join(RESULTS_DIR, 'scores.json'), 'w') as fp:
            json.dump({
                'accuracy': accuracy,
                'fmeasure': fmeasure,
            }, fp)
