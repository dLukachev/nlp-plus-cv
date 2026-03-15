import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


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
    import torchvision.transforms as transforms

    return (
        DataLoader,
        Image,
        LinearRegression,
        LogisticRegression,
        Path,
        TensorDataset,
        classification_report,
        mean_absolute_error,
        mo,
        models,
        nn,
        np,
        os,
        pd,
        r2_score,
        torch,
        train_test_split,
        transforms,
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
    indexes.head()
    return


@app.cell
def _(indexes):
    print(indexes.isna().sum())
    return


@app.cell
def _(indexes):
    indexes.class_id = indexes.class_id.map(lambda cl: cl - 1)
    return


@app.cell
def _(indexes):
    classes = indexes.class_id.nunique()
    classes
    return


@app.cell
def _(indexes):
    # смотрим чтобы были такие же названия как и в папках
    indexes.path.map(lambda name: name.split('/')[0]).unique()
    return


@app.cell
def _(indexes):
    dataset = indexes.copy()
    return (dataset,)


@app.cell
def _(dataset):
    dataset['vectorize'] = None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Преобразовываем все картинки в вектора для дальнейшего построения CNN
    """)
    return


@app.cell
def _(models):
    # model_ = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model_ = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    return (model_,)


@app.cell
def _(model_, torch):
    model = torch.nn.Sequential(*(
        list(model_.children())[:-1]
    ))
    return (model,)


@app.cell
def _(transforms):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return (preprocess,)


@app.cell
def _(Image, model, preprocess, torch):
    def img_to_vec(image_path):
        model.eval()
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            vector = model(img_tensor)

        return vector.squeeze().numpy()

    return (img_to_vec,)


@app.cell
def _(WDIR, dataset, img_to_vec):
    dataset['vectorize'] = dataset.path.map(lambda path: img_to_vec(WDIR + path))
    return


@app.cell
def _(WDIR, dataset):
    dataset.to_csv(WDIR + 'dataset.csv')
    return


@app.cell
def _(dataset):
    dataset.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Грузим датасет
    """)
    return


@app.cell
def _(dataset, np):
    vectors_for_linreg = np.stack(dataset["vectorize"].values)   # shape: (N, 2048)
    y = dataset["class_id"].values                   # shape: (N,)
    return vectors_for_linreg, y


@app.cell
def _(train_test_split, vectors_for_linreg, y):
    X_train, X_val, y_train, y_val = train_test_split(vectors_for_linreg, y)
    return X_train, X_val, y_train, y_val


@app.cell
def _(dataset):
    vectors = dataset.vectorize
    y_true = dataset.class_id
    return vectors, y_true


@app.cell
def _(torch, vectors, y_true):
    t_x = torch.tensor(vectors)
    t_y = torch.tensor(y_true)
    return t_x, t_y


@app.cell
def _(TensorDataset, t_x, t_y):
    x_tensor_dataset = TensorDataset(t_x, t_y)
    return (x_tensor_dataset,)


@app.cell
def _(x_tensor_dataset):
    x_tensor_dataset.tensors
    return


@app.cell
def _(DataLoader, x_tensor_dataset):
    x_train_dl = DataLoader(x_tensor_dataset, batch_size=32, shuffle=True)
    return (x_train_dl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Реализация модели
    """)
    return


@app.cell
def _(nn):
    # ps просто захотелось, так сначала бы начал с логистической регрессии, картинки бы так же преобразовал сначала

    class CV(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.net = nn.Sequential(
                nn.Linear(2048, 38),
            )

        def forward(self, x):
            return self.net(x)


    return (CV,)


@app.cell
def _(CV, nn, torch):
    cnn = CV()
    optim = torch.optim.AdamW(cnn.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    return cnn, loss_func, optim


@app.cell
def _(nn, np, x_train_dl):
    def train(model: nn.Module, optimizer, loss_func, epoch: int = 5):
        model.train()
        loss_mean = 0
        for epo in range(epoch):
            for xb, yb in x_train_dl:
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
def _(cnn, loss_func, optim, train):
    train(cnn, optim, loss_func, 300)
    return


@app.cell
def _(pd):
    test_df = pd.read_csv('data/cv-lego/test.csv')
    return (test_df,)


@app.cell
def _(WDIR, img_to_vec, test_df):
    test_df['x'] = test_df.path.map(lambda img: img_to_vec(WDIR + img))
    return


@app.cell
def _(test_df):
    test_df.class_id = test_df.class_id.map(lambda id: id - 1)
    return


@app.cell
def _(test_df):
    test_df.head()
    return


@app.cell
def _(test_df, torch):
    test_x_t = torch.tensor(test_df.x)
    test_y_t = torch.tensor(test_df.class_id)
    return test_x_t, test_y_t


@app.cell
def _(DataLoader, TensorDataset, test_x_t, test_y_t):
    test_ds = TensorDataset(test_x_t, test_y_t)
    test_dl = DataLoader(test_ds, batch_size=32)
    return (test_dl,)


@app.cell
def _(nn, torch):
    def model_testing(model: nn.Module, loader: torch.utils.data.DataLoader):
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
        print(f'\nТочность на тестовой выборке: {accuracy:.2f}')
        return accuracy

    return (model_testing,)


@app.cell
def _(cnn, model_testing, test_dl):
    model_testing(cnn, test_dl)
    return


@app.cell
def _():
    # Пробуем линейку
    return


@app.cell
def _(LinearRegression):
    linReg = LinearRegression()
    return (linReg,)


@app.cell
def _(X_train, linReg, y_train):
    linReg.fit(X_train, y_train)
    return


@app.cell
def _(X_val, linReg):
    y_pred_linReg = linReg.predict(X_val)
    return (y_pred_linReg,)


@app.cell
def _(mean_absolute_error, y_pred_linReg, y_val):
    mean_absolute_error(y_val, y_pred_linReg)
    return


@app.cell
def _(r2_score, y_pred_linReg, y_val):
    r2_score(y_val, y_pred_linReg)
    return


@app.cell
def _(LogisticRegression, X_train, y_train):
    logReg = LogisticRegression(max_iter=500).fit(X_train, y_train)
    return (logReg,)


@app.cell
def _(X_val, logReg):
    y_pred_logReg = logReg.predict(X_val)
    return (y_pred_logReg,)


@app.cell
def _(classification_report, y_pred_logReg, y_val):
    print(classification_report(y_val, y_pred_logReg))
    return


if __name__ == "__main__":
    app.run()
