import gzip
import pickle
from collections import defaultdict
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
from numba import jit


class Model:

    def __init__(self):
        self.x = defaultdict()
        self.y = defaultdict()
        self.z = defaultdict()
        self.Vx = defaultdict()
        self.Vy = defaultdict()
        self.Vz = defaultdict()


@jit
def predict(coef, history, length):
    predictions = list()
    for t in range(length):
        # append to history prediction!
        yhat = coef[0]
        for i in range(1, len(coef)):
            yhat += coef[i] * history[-i]
        predictions.append(yhat)
        np.append(history, yhat)
    return predictions


def make_some_interesting(id_, test_data, column):
    to_predict = test_data[column]
    attr = column.split('_')[0]
    predictions = list()
    file = gzip.GzipFile(f"./Models/{id_}.pickle", "rb")
    models = pickle.load(file)
    file.close()
    history = models.__dict__[attr]["history"]
    coef = models.__dict__[attr]["coef"]
    length = len(history)
    for t in range(len(to_predict)):
        # append to history prediction!
        yhat = predict(coef, history)
        predictions.append(yhat)
        history.append(yhat)
        length += 1
    return pd.DataFrame([[x + y] for x, y in zip(predictions, to_predict.values)])


def main():
    test_data = pd.read_csv("test.csv")
    submission = pd.DataFrame()
    with Pool(1) as p:
        for id_ in test_data["sat_id"].unique().tolist():
            for_id = pd.concat(
                p.map(partial(make_some_interesting, id_, test_data.loc[test_data["sat_id"] == id_]),
                        ["x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]),
                ignore_index=True, axis=1)
            submission = pd.concat([submission, for_id], axis=0, ignore_index=True)
    submission.columns = ["x", "y", "z", "Vx", "Vy", "Vz"]
    submission["id"] = test_data["id"]
    submission = submission[["id", "x", "y", "z", "Vx", "Vy", "Vz"]]
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
