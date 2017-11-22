import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, make_scorer, precision_score, \
    recall_score, roc_curve
from matplotlib import pyplot as plt

if __name__=="__main__":
    data = pd.read_csv("data/processed_train.csv", index_col="RefId")

    X = data[data.columns[:-1]]
    Y = data["IsBadBuy"]

    X = Imputer(strategy="median", copy=False).fit_transform(X)

    # for C in [0.00001,0.0001,0.01,0.1,1]:
    for C in [0.01]:
        model = LogisticRegression(C=C,class_weight="balanced",random_state=0)

        prob = cross_val_predict(model, X, Y, method="predict_proba", cv=5, verbose=2)

        prob_pos = prob[:, 1]

        P = np.greater(prob_pos, 0.5)

        print(C)
        conf_mat = confusion_matrix(Y, P)
        precision = precision_score(Y, P)
        recall = recall_score(Y, P)
        f1 = f1_score(Y, P)
        auc = roc_auc_score(Y, prob_pos)
        # auc2 = roc_auc_score(Y, P)
        print("conf_mat")
        print(conf_mat)
        print("precision:{},recall:{},f1:{},auc:{},gini:{}".format(precision,recall,f1,auc,auc*2-1))
        print()

        # fpr,tpr,threshold = roc_curve(Y,prob_pos)
        # plt.plot(fpr, tpr)
        # plt.plot([0, 1], [0, 1])
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.show()
        # print()
