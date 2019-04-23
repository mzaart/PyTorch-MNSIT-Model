import torch
from .scorer import Scorer
from torch.autograd import Variable


class DenseNetworkScorer(Scorer):
    """
    Measures the performance of dense networks.
    """

    def __init__(self, model, hyper_params, scoring_func, train_loader, validation_loader):
        super().__init__(model, hyper_params, scoring_func, train_loader, validation_loader)

    def score(self):
        self._train()

        with torch.no_grad():
            y_actual = torch.zeros(len(self.validation_loader.dataset))
            y_pred = torch.zeros(len(self.validation_loader.dataset))

            for i, (x, y) in enumerate(self.validation_loader):
                x, y = Variable(x), Variable(y)
                x = x.view(-1, 28 * 28)
                pred = self.model(x).data.max(1)[1]  # get the index of the max log-probability
                batch_size = y.shape[0]
                y_actual[i*batch_size:(i+1)*batch_size] = y
                y_pred[i*batch_size:(i+1)*batch_size] = pred

            return self.scoring_func.score(y_pred, y_actual)

    def _train(self):
        optimizer_cls = self.hyper_params['optimizer_cls']
        lr = self.hyper_params['lr']
        weight_decay = self.hyper_params.get('weight_decay', 0)
        optimizer = optimizer_cls(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = self.hyper_params['loss_func']
        num_epochs = self.hyper_params['epochs']

        for epoch in range(num_epochs):
            for i, (x, y) in enumerate(self.train_loader):
                x, y = Variable(x), Variable(y)

                # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
                x = x.view(-1, 28 * 28)

                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
