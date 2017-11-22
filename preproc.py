import pandas as pd
import numpy as np
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from dateutil import parser
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder


def process(filein, fileout, train=True):
    df = pd.read_csv(filein, index_col="RefId")

    transformed = []

    # columns for direct usage
    transformed.append(df[["VehicleAge", "VehOdo", "VehBCost", "WarrantyCost"]])

    # columns with extra replacement
    col_list = ["MMRAcquisitionAuctionAveragePrice", "MMRAcquisitionAuctionCleanPrice",
                "MMRAcquisitionRetailAveragePrice", "MMRAcquisitonRetailCleanPrice", "MMRCurrentAuctionAveragePrice",
                "MMRCurrentAuctionCleanPrice", "MMRCurrentRetailAveragePrice", "MMRCurrentRetailAveragePrice",
                "MMRCurrentRetailCleanPrice"]
    transformed.append(df[col_list].replace([0, 1], [np.nan, np.nan]))

    # replace small categories with "other"
    def mask_helper(series_name, threshold):
        series = df[series_name]
        df[series_name] = series.mask(series.map(series.value_counts()) < threshold, 'other')

    # print(df["VNZIP1"].value_counts())
    mask_helper("BYRNO", 20)

    # discretilize columns
    col_list = ["Auction", "Make", "Color", "Transmission", "WheelTypeID", "Nationality", "Size",
                "TopThreeAmericanName", "PRIMEUNIT", "AUCGUART", "BYRNO", "VNST", "IsOnlineSale"]
    if train:
        target = df[col_list].apply(pd.Categorical)
        cats = {}
        for col in col_list:
            cats[col] = target[col].cat.categories
        joblib.dump(cats,"preproc.model")
    else:
        cats = joblib.load("preproc.model")
        target = df[col_list].apply(lambda x : pd.Categorical(x,cats[x.name]))
    transformed.append(pd.get_dummies(target, prefix=col_list, columns=col_list))

    # apply BoW to columns model, trim, and submodel
    strings = df[["Model", "Trim", "SubModel"]].fillna("").apply(" ".join, axis=1)
    tokens = [x.split(" ") for x in strings]
    if train:
        dictionary = Dictionary(tokens)
        dictionary.filter_extremes(no_below=5)
        dictionary.save("model.dict")
    else:
        dictionary = Dictionary.load("model.dict")
    # print(list(sorted(dictionary.dfs.values())))
    corpora = [dictionary.doc2bow(x) for x in tokens]
    vectors = corpus2dense(corpora, len(dictionary))
    transformed.append(pd.DataFrame(vectors.astype(bool).T, index=df.index, columns=dictionary.values()))

    if "IsBadBuy" in df.columns:
        transformed.append(df["IsBadBuy"])
    processed = pd.concat(transformed, axis=1)
    processed.to_csv(fileout, na_rep="NULL")

