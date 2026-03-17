import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    import pandas as pd
    import numpy as np

    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    from catboost import CatBoostClassifier

    import nltk
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer

    return (
        CatBoostClassifier,
        LogisticRegression,
        TfidfVectorizer,
        WordNetLemmatizer,
        classification_report,
        nltk,
        pd,
        train_test_split,
        wordnet,
    )


@app.cell
def _(nltk):
    nltk.download('wordnet')
    return


@app.cell
def _(pd):
    df_train_ = pd.read_csv('data/nlp-twitter/twitter_training.csv', header=None)
    return (df_train_,)


@app.cell
def _(df_train_):
    df_train_.columns = ["id", "topic", "sentiment", "text"]
    return


@app.cell
def _(pd):
    df_test_ = pd.read_csv('data/nlp-twitter/twitter_validation.csv', header=None)
    return (df_test_,)


@app.cell
def _(df_test_):
    df_test_.columns = ["id", "topic", "sentiment", "text"]
    return


@app.cell
def _(df_train_):
    df_train_
    return


@app.cell
def _(df_test_):
    df_test_
    return


@app.cell
def _(df_train_):
    df_train_.sentiment.unique()
    return


@app.cell
def _(df_train_):
    df_train_.isna().sum()
    return


@app.cell
def _(df_test_):
    df_test_.isna().sum()
    return


@app.cell
def _(df_train_):
    df_train = df_train_.dropna()
    return (df_train,)


@app.cell
def _(df_train):
    classes = df_train.sentiment.values.unique()
    return


@app.cell
def _(WordNetLemmatizer, wordnet):
    def text_lemmatizer(text):
        lemmatizer = WordNetLemmatizer()
        return lemmatizer.lemmatize(text, pos=wordnet.VERB)

    return (text_lemmatizer,)


@app.cell
def _(df_train, text_lemmatizer):
    df_train.sentiment = df_train.sentiment.map(lambda text: text_lemmatizer(text))
    return


@app.cell
def _(df_test_, text_lemmatizer):
    df_test_.sentiment = df_test_.sentiment.map(lambda text: text_lemmatizer(text))
    return


@app.cell
def _(TfidfVectorizer):
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    return (vectorizer,)


@app.cell
def _(df_train, vectorizer):
    X = vectorizer.fit_transform(df_train.text)
    return (X,)


@app.cell
def _(df_train):
    y = df_train.sentiment
    return (y,)


@app.cell
def _(df_test_, vectorizer):
    X_val, y_val = vectorizer.transform(df_test_.text), df_test_.sentiment
    return X_val, y_val


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    return X_test, X_train, y_test, y_train


@app.cell
def _(LogisticRegression):
    lg = LogisticRegression()
    return (lg,)


@app.cell
def _(X_train, lg, y_train):
    lg.fit(X_train, y_train)
    return


@app.cell
def _(X_test, lg):
    y_pred = lg.predict(X_test)
    return (y_pred,)


@app.cell
def _(classification_report, y_pred, y_test):
    print(classification_report(y_pred, y_test))
    return


@app.cell
def _(X, lg, y):
    lg.fit(X, y)
    return


@app.cell
def _(X_val, lg):
    y_pred_val = lg.predict(X_val)
    return (y_pred_val,)


@app.cell
def _(classification_report, y_pred_val, y_val):
    print(classification_report(y_pred_val, y_val))
    return


@app.cell
def _(CatBoostClassifier):
    cbc = CatBoostClassifier(max_depth=5, thread_count=-1, learning_rate=1, iterations=2000)
    return (cbc,)


@app.cell
def _(X_train, cbc, y_train):
    cbc.fit(X_train, y_train)
    return


@app.cell
def _(X_test, cbc):
    y_pred_cbc = cbc.predict(X_test)
    return (y_pred_cbc,)


@app.cell
def _(classification_report, y_pred_cbc, y_test):
    print(classification_report(y_pred_cbc, y_test))
    return


@app.cell
def _(X, cbc, y):
    cbc.fit(X, y)
    return


@app.cell
def _(X_val, cbc):
    y_pred_cbc_val = cbc.predict(X_val)
    return (y_pred_cbc_val,)


@app.cell
def _(classification_report, y_pred_cbc_val, y_val):
    print(classification_report(y_pred_cbc_val, y_val))
    return


@app.cell
def _():
    # Кетбуст проиграл тупо по скорости - 20 минут обучения, против 1< минуты у LogisticRegression

    print(f'Catboost, total: 20m 50s, accuracy: 95%')
    print(f'LogisticRegression, total: 1< min, accuracy: 97%')
    return


if __name__ == "__main__":
    app.run()
