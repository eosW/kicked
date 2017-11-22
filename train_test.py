import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib


def train(trainfile, modelfile, C):
    data = pd.read_csv(trainfile, index_col="RefId")

    X = data[data.columns[:-1]]
    Y = data["IsBadBuy"]

    imputer = Imputer(strategy="median", copy=False)
    X = imputer.fit_transform(X)

    model = LogisticRegression(C=C, class_weight="balanced", random_state=0)
    model.fit(X, Y)

    persistence = imputer, model
    joblib.dump(persistence, modelfile)


def test(filein, fileout, modelfile):
    imputer, model = joblib.load(modelfile)

    data = pd.read_csv(filein, index_col="RefId")

    X = imputer.transform(data)

    prob = model.predict_proba(X)
    prob_pos = prob[:, 1]

    res = pd.DataFrame(prob_pos, index=data.index, columns=["IsBadBuy"])
    res.to_csv(fileout)
