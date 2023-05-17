import numpy as np
import pickle
import os
import sklearn
import sklearn.utils

DATA_PATH = "" # TODO: set this to the path of the data folder

def load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def load_glove(normalize):
    with open(DATA_PATH + "/data.pickle", "rb") as f:
        data_dict = pickle.load(f)
        X, Y, words_train = data_dict["train"]
        X_dev, Y_dev, words_dev = data_dict["dev"]
        X_test, Y_test, words_test = data_dict["test"]

        X, Y = X[Y > -1], Y[Y > -1]
        X_dev, Y_dev = X_dev[Y_dev > -1], Y_dev[Y_dev > -1]
        X_test, Y_test = X_test[Y_test > -1], Y_test[Y_test > -1]

        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        X_dev = np.concatenate([X_dev, np.ones((X_dev.shape[0], 1))], axis=1)
        X_test = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1)

        if normalize:
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X_dev /= np.linalg.norm(X_dev, axis=1, keepdims=True)
            X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)
    return X, Y, X_dev, Y_dev, X_test, Y_test

def load_dictionary(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    k2v, v2k = {}, {}
    for line in lines:
        k, v = line.strip().split("\t")
        v = int(v)
        k2v[k] = v
        v2k[v] = k

    return k2v, v2k

def loadBiasInBios(encoder,is_finetuned):
    data = []
    path = DATA_PATH + 'biasbios/'
    path_pickle = path
    p2i, i2p = load_dictionary(path+"profession2index.txt")
    g2i, i2g = load_dictionary(path+"gender2index.txt")
    if encoder == 'roberta':
        path = path + 'roberta/'
    if is_finetuned:
        path = path + 'finetuned/'
    for mode in ["train", "dev", "test"]:
        X = np.load(path+"{}_cls.npy".format(mode))
        with open(path_pickle+"{}.pickle".format(mode), "rb") as f:
            bios_data = pickle.load(f)
            Y = np.array([1 if d["g"] == "f" else 0 for d in bios_data])
            Y_main = np.array([p2i[d["p"]] for d in bios_data])
            data.extend([X, Y, Y_main])
    return data

def load_moji():
    ratio = 0.8
    path = DATA_PATH + "emoji_sent_race_0.8/"
    size = 100000
    fnames = ["neg_neg.npy", "neg_pos.npy", "pos_neg.npy", "pos_pos.npy"]
    protected_labels = [0, 1, 0, 1]
    main_labels = [0, 0, 1, 1]
    output = []
    for mode in ["train","dev","test"]:
        path_in = path + "{}/".format(mode)
        X, Y_p, Y_m = [], [], []
        n1 = int(size * ratio / 2)
        n2 = int(size * (1 - ratio) / 2)
        #     print(n1, n2)
        for fname, p_label, m_label, n in zip(fnames, protected_labels, main_labels, [n1, n2, n2, n1]):
            #         print(path + '/' + fname)
            #         print(np.load(path + '/' + fname).shape)
            data = np.load(path_in + '/' + fname)[:n]
            for x in data:
                X.append(x)
            for _ in data:
                Y_p.append(p_label)
            for _ in data:
                Y_m.append(m_label)

        Y_p = np.array(Y_p)
        Y_m = np.array(Y_m)
        X = np.array(X)
        X, Y_p, Y_m = sklearn.utils.shuffle(X, Y_p, Y_m, random_state=0)
        output.extend([X,Y_p,Y_m])
    return output


def load_data(mode):
    if mode == "glove":
        print("Loading glove")
        x_train, y_train_gender, [], x_dev, Y_dev, [], X_test, Y_test, [] = load_glove(normalize=True)
        return x_train, y_train_gender, x_dev, Y_dev, X_test, Y_test
    elif mode == "moji":
        print("Loading moji")
        return load_moji()

    elif mode == "bert-finetuned":
        print("Loading bios-bert-finetuned")
        encoder = 'bert'
        is_finetuned = True

    elif mode == "bert-frozen":
        print("Loading bios-bert-frozen")
        encoder = 'bert'
        is_finetuned = False

    elif mode == "roberta-finetuned":
        print("Loading bios-roberta-finetuned")
        encoder = 'roberta'
        is_finetuned = True

    elif mode == "roberta-frozen":
        print("Loading bios-roberta-frozen")
        encoder = 'roberta'
        is_finetuned = False
    else:
        raise ("not implemented")
    return loadBiasInBios(encoder,is_finetuned)