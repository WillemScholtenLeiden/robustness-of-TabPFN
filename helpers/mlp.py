import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import accuracy_score


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super(SimpleNN, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32, device=next(self.parameters()).device)
            logits = self.forward(X)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def train_standard_nn(X_train, y_train, X_val=None, y_val=None, epochs=100, lr=0.001, device="cpu"):
    model = SimpleNN(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                train_preds = model.predict(X_train.cpu().numpy())
                train_acc = accuracy_score(y_train.cpu().numpy(), train_preds)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}")

    return model


class _MLPWrapper(nn.Module):

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self._model = None

    def _to_tensors(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long, device=self.device)
        return X.to(self.device), y.to(self.device)

    def forward(self, x):
        return self._model(x)

    def parameters(self, recurse=True):
        return self._model.parameters(recurse=recurse) if self._model is not None else super().parameters(recurse=recurse)

    def eval(self):
        if self._model is not None:
            self._model.eval()
        return super().eval()

    def train(self, mode=True):
        if self._model is not None:
            self._model.train(mode)
        return super().train(mode)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class StandardMLPClassifier(_MLPWrapper):
    def __init__(self, epochs=100, lr=0.001, device="cpu"):
        super().__init__(device=device)
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        X, y = self._to_tensors(X, y)
        self._model = train_standard_nn(X, y, epochs=self.epochs, lr=self.lr, device=self.device)
        return self


class AdversarialMLPClassifier(_MLPWrapper):
    def __init__(self, epochs=100, lr=0.001, eps=0.1, alpha=0.01, steps=10, device="cpu"):
        super().__init__(device=device)
        self.epochs = epochs
        self.lr = lr
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def fit(self, X, y):
        X, y = self._to_tensors(X, y)
        self._model = train_adversarial_nn(
            X, y, epochs=self.epochs, lr=self.lr,
            eps=self.eps, alpha=self.alpha, steps=self.steps, device=self.device,
        )
        return self


def train_adversarial_nn(X_train, y_train, X_val=None, y_val=None,
                         epochs=100, lr=0.001, eps=0.1, alpha=0.01, steps=10, device="cpu"):
    model = SimpleNN(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        # Generate adversarial examples using PGD
        X_adv = X_train.clone().detach()
        X_adv.requires_grad = True

        for _ in range(steps):
            if X_adv.grad is not None:
                X_adv.grad.zero_()

            outputs = model(X_adv)
            loss = criterion(outputs, y_train)
            loss.backward()

            with torch.no_grad():
                perturbation = alpha * X_adv.grad.sign()
                X_adv = X_adv + perturbation
                perturbation_total = X_adv - X_train
                perturbation_total = torch.clamp(perturbation_total, -eps, eps)
                X_adv = X_train + perturbation_total

            X_adv = X_adv.detach()
            X_adv.requires_grad = True

        # Train on adversarial examples
        optimizer.zero_grad()
        outputs = model(X_adv)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                train_preds = model.predict(X_train.cpu().numpy())
                train_acc = accuracy_score(y_train.cpu().numpy(), train_preds)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}")

    return model