import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import os

    import pandas as pd
    import numpy as np

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import mean_absolute_error, r2_score, classification_report

    from PIL import Image
    import torchvision.models as models
    import torchvision.transforms as T

    return (
        DataLoader,
        Image,
        Path,
        T,
        TensorDataset,
        models,
        nn,
        np,
        os,
        pd,
        torch,
        train_test_split,
    )


@app.cell
def _(Path):
    WDIR = 'data/cv-lego/'
    collections = ['harry-potter', 'jurassic-world', 'marvel', 'star-wars']
    TDIR = Path('data/cv-lego/test')
    INDEXES = Path('data/cv-lego/index.csv')
    return INDEXES, TDIR, WDIR, collections


@app.cell
def _(TDIR, WDIR, collections, os):
    for theme in collections:
        count_photo = 0
        dirs = os.walk(WDIR + theme)
        for dir in dirs:
            count_photo += len(dir[2])
        print(f'Коллекция {theme}, кол-во фото {count_photo}')
    print()
    print(f'Коллекция test, кол-во фото {len(os.listdir(TDIR))}')
    return


@app.cell
def _(INDEXES, pd):
    indexes = pd.read_csv(INDEXES)
    return (indexes,)


@app.cell
def _(indexes):
    indexes.class_id = indexes.class_id.map(lambda cl: cl - 1)
    return


@app.cell
def _(indexes):
    classes = indexes.class_id.nunique()
    classes
    return (classes,)


@app.cell
def _(pd):
    test_dataset = pd.read_csv('data/cv-lego/test.csv')
    test_dataset.class_id = test_dataset.class_id.map(lambda id: id - 1)
    return (test_dataset,)


@app.cell
def _(indexes):
    dataset = indexes.copy()
    dataset['vectorize'] = None
    return (dataset,)


@app.cell
def _(models):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    return (model,)


@app.cell
def _(model, nn):
    tv_model = nn.Sequential(*list(model.children())[:-1])
    return (tv_model,)


@app.cell
def _():
    mean_n_tv = [0.485, 0.456, 0.406]
    std_n_tv  = [0.229, 0.224, 0.225]
    return mean_n_tv, std_n_tv


@app.cell
def _(T, mean_n_tv, std_n_tv):
    preprocessing = T.Compose([
        T.Resize(256),
        T.CenterCrop(244),
        T.ToTensor(),
        T.Normalize(
            mean=mean_n_tv,
            std=std_n_tv
        )
    ])
    return (preprocessing,)


@app.cell
def _(Image, preprocessing, torch, tv_model):
    def img_to_vec(path):
        image = Image.open(path).convert("RGB")
        vec = preprocessing(image).unsqueeze(0)

        with torch.no_grad():
            vector = tv_model(vec)

        vector = vector.squeeze()
        return vector.numpy()

    return (img_to_vec,)


@app.cell
def _(WDIR, dataset, img_to_vec):
    dataset.vectorize = dataset.path.map(lambda path: img_to_vec(WDIR + path))
    return


@app.cell
def _(WDIR, img_to_vec, test_dataset):
    test_dataset['vectorize'] = test_dataset.path.map(lambda path: img_to_vec(WDIR + path))
    return


@app.cell
def _(dataset, torch, train_test_split):
    x_t = torch.tensor(dataset['vectorize'])
    y_t = torch.tensor(dataset.class_id)

    x_t_t, x_t_v, y_t_t, y_t_v = train_test_split(x_t, y_t, test_size=0.2, stratify=y_t)
    return x_t, x_t_t, x_t_v, y_t, y_t_t, y_t_v


@app.cell
def _(test_dataset, torch):
    test_x_t = torch.tensor(test_dataset['vectorize'])
    test_y_t = torch.tensor(test_dataset.class_id)
    return test_x_t, test_y_t


@app.cell
def _(DataLoader, TensorDataset, x_t, x_t_t, x_t_v, y_t, y_t_t, y_t_v):
    tensor_dataset_train = TensorDataset(x_t_t, y_t_t)
    tensor_dataset_valid = TensorDataset(x_t_v, y_t_v)
    tensor_dataset_full = TensorDataset(x_t, y_t)

    dataloader_train = DataLoader(tensor_dataset_train, batch_size=64, shuffle=True)
    dataloader_valid = DataLoader(tensor_dataset_valid, batch_size=64, shuffle=True)
    dataloader_full = DataLoader(tensor_dataset_full, batch_size=64)
    return dataloader_full, dataloader_train, dataloader_valid


@app.cell
def _(DataLoader, TensorDataset, test_x_t, test_y_t):
    tensor_dataset_test = TensorDataset(test_x_t, test_y_t)
    dataloader_test = DataLoader(tensor_dataset_test)
    return (dataloader_test,)


@app.cell
def _(classes, nn):
    class MLP(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.net = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, classes),
            )

        def forward(self, x):
            return self.net(x)

    return (MLP,)


@app.cell
def _(MLP, nn, np, torch, y_t):
    mlp = MLP()
    optim = torch.optim.AdamW(mlp.parameters(), lr=0.001)

    class_counts = np.bincount(y_t.numpy())
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    loss_func = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    return loss_func, mlp, optim


@app.cell
def _(nn, np):
    def train(model: nn.Module, optimizer, loss_func, data, epoch: int = 10):
        model.train()
        loss_mean = 0
        for epo in range(epoch):
            for xb, yb in data:
                pred = model(xb)
                loss = loss_func(pred, yb)
                loss_mean += loss.item()
            
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if epo % 10 == 0 or epo == epoch - 1:
                print(f'Эпоха: {epo}, ошибка: {np.mean(loss_mean)}')
                loss_mean = 0

    return (train,)


@app.cell
def _(dataloader_train, loss_func, mlp, optim, train):
    train(mlp, optim, loss_func, dataloader_train, 30)
    return


@app.cell
def _(nn, torch):
    def model_validate(model: nn.Module, loader: torch.utils.data.DataLoader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                outputs = model(xb)
                _, predicted = torch.max(outputs.data, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
            accuracy = 100 * correct / total
        print(f'\nТочность на валидации: {accuracy:.2f}')
        return accuracy

    return (model_validate,)


@app.cell
def _(dataloader_valid, mlp, model_validate):
    model_validate(mlp, dataloader_valid)
    return


@app.cell
def _(dataloader_full, loss_func, mlp, optim, train):
    train(mlp, optim, loss_func, dataloader_full, 30)
    return


@app.cell
def _(dataloader_test, mlp, model_validate):
    model_validate(mlp, dataloader_test)
    return


if __name__ == "__main__":
    app.run()
