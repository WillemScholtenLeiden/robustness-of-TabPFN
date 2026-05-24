import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer, make_classification
from ucimlrepo import fetch_ucirepo

def _prepare_dataset(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.25, 
    random_state: int = 42, 
    device: str = "cpu"
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)
    return X_train, y_train, X_test, y_test

def create_synthetic_dataset(
    n_samples: int = 100,
    n_features: int = 5,
    test_size: float = 0.25,
    random_state: int = 42,
    device: str = "cpu",
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=random_state,
    )
    return _prepare_dataset(X, y, test_size, random_state, device)


def load_breast_cancer_dataset(
    test_size: float = 0.25, 
    random_state: int = 42, 
    device: str = "cpu"
):
    data = load_breast_cancer()
    return _prepare_dataset(data.data, data.target, test_size, random_state, device)


def load_sonar_dataset(
    test_size: float = 0.25, 
    random_state: int = 42, 
    device: str = "cpu"
):
    dataset = fetch_ucirepo(id=151)
    X = dataset.data.features.values.astype(np.float64)
    y_raw = dataset.data.targets.values.ravel()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return _prepare_dataset(X, y, test_size, random_state, device)


def load_ionosphere_dataset(
    test_size: float = 0.25, 
    random_state: int = 42, 
    device: str = "cpu"
):
    dataset = fetch_ucirepo(id=52)
    X = dataset.data.features.values.astype(np.float64)
    y_raw = dataset.data.targets.values.ravel()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return _prepare_dataset(X, y, test_size, random_state, device)


def load_banknote_dataset(
    test_size: float = 0.25, 
    random_state: int = 42, 
    device: str = "cpu"
):
    dataset = fetch_ucirepo(id=267)
    X = dataset.data.features.values.astype(np.float64)
    y = dataset.data.targets.values.ravel().astype(int)
    return _prepare_dataset(X, y, test_size, random_state, device)

def load_haberman_dataset(
    test_size: float = 0.25,
    random_state: int = 42,
    device: str = "cpu",
):
    dataset = fetch_ucirepo(id=43)
    X = dataset.data.features.values.astype(np.float64)
    y_raw = dataset.data.targets.values.ravel()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return _prepare_dataset(X, y, test_size, random_state, device)


def load_spambase_dataset(
    test_size: float = 0.25,
    random_state: int = 42,
    device: str = "cpu",
):
    dataset = fetch_ucirepo(id=94)
    X = dataset.data.features.values.astype(np.float64)
    y = dataset.data.targets.values.ravel().astype(int)
    return _prepare_dataset(X, y, test_size, random_state, device)