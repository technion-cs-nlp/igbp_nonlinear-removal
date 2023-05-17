import os
import numpy as np
from typing import List
from collections import Counter, defaultdict
from sklearn.svm import LinearSVC, SVC
from torch import nn, optim, Tensor
from torch.utils.data import SequentialSampler, DataLoader, Dataset, TensorDataset
from sklearn.linear_model import SGDClassifier, SGDRegressor, Perceptron, LogisticRegression
import torch
from transformers import set_seed, is_wandb_available
from mdl import OnlineCodeMDLProbe
from os.path import join

def count_profs_and_gender(data: List[dict]):
    counter = defaultdict(Counter)
    for entry in data:
        gender, prof = entry["g"], entry["p"]
        counter[prof][gender] += 1

    return counter

def rms(arr):
    return np.sqrt(np.mean(np.square(arr)))

def get_TPR(y_main, y_hat_main, y_protected):
    all_y = list(Counter(y_main).keys())
    protected_vals = defaultdict(dict)
    for label in all_y:
        for i in range(2):
            used_vals = (y_main == label) & (y_protected == i)
            y_label = y_main[used_vals]
            y_hat_label = y_hat_main[used_vals]
            if not np.any(used_vals):
                protected_vals['proffesion:{}'.format(label)]['gender:{}'.format(i)] = 0.5
            else:
                protected_vals['proffesion:{}'.format(label)]['gender:{}'.format(i)] = (y_label == y_hat_label).mean()
    diffs = {}
    for k, v in protected_vals.items():
        vals = list(v.values())
        diffs[k] = vals[0] - vals[1]
    return protected_vals, diffs


def evaluate(mode,seed,method,n_iterations,x_train_cleaned,y_train,y_train_gender,x_dev_cleaned,y_dev,y_dev_gender,
                x_test_cleaned,y_test,y_test_gender,scores,calc_mdl):


    # Main Task Accuracy
    new_main_clf = LogisticRegression(warm_start=True, penalty='l2',
                                          solver="saga", multi_class='multinomial', fit_intercept=False,
                                          verbose=5, n_jobs=90, max_iter=7)
    if mode == 'moji':
        new_main_clf = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)
    new_main_clf.fit(x_train_cleaned, y_train)
    new_acc = new_main_clf.score(x_test_cleaned, y_test)
    scores[method]['acc'].append(new_acc*100)
    # Main Task TPR-GAP
    GAP_rms = rms(list(
        get_TPR(y_test, new_main_clf.predict(x_test_cleaned), y_test_gender)[1].values()))
    scores[method]['gap'].append(GAP_rms*100)

    #Leakage and Compression
    compression_nonlin, gender_nonlin_acc = 0,0
    if calc_mdl:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def build_probe(input_size, num_classes=2, probe_type='mlp'):
            probes = {
                'mlp': lambda: nn.Sequential(
                    nn.Linear(input_size, input_size),
                    nn.ReLU(),
                    nn.Linear(input_size, input_size),
                    nn.ReLU(),
                    nn.Linear(input_size, num_classes)
                ),
                'linear': lambda: nn.Linear(input_size, num_classes)
            }
            return probes[probe_type]()
        input_size = x_train_cleaned.shape[1]
        fractions = [2.0, 3.0, 4.4, 6.5, 9.5, 14.0, 21.0, 31.0, 45.7, 67.6, 100]
        def create_probe_lin():
            return build_probe(input_size, probe_type='linear')
        def create_probe_nonlin():
            return build_probe(input_size, probe_type='mlp')

        X_train_cleaned, Y_train = torch.FloatTensor(x_train_cleaned).to(device), torch.LongTensor(y_train).to(device)
        X_dev_cleaned, Y_dev = torch.FloatTensor(x_dev_cleaned).to(device), torch.LongTensor(y_dev).to(device)
        X_test_cleaned, Y_test = torch.FloatTensor(x_test_cleaned).to(device), torch.LongTensor(y_test).to(device)
        Y_train_gender, Y_test_gender, Y_dev_gender = torch.LongTensor(y_train_gender).to(device), torch.LongTensor(
            y_test_gender).to(device), torch.LongTensor(y_dev_gender).to(device)
        train_dataset = TensorDataset(X_train_cleaned, Y_train_gender)
        dev_dataset = TensorDataset(X_dev_cleaned, Y_dev_gender)
        test_dataset = TensorDataset(X_test_cleaned, Y_test_gender)

        online_code_probe_lin = OnlineCodeMDLProbe(create_probe_lin, fractions, device)
        reporting_rootlin = join(os.getcwd(), f'online_coding_tstlin{mode}{method}{seed}.pkl')
        uniform_cdl_lin, online_cdl_lin, gender_lin_acc = online_code_probe_lin.evaluate(train_dataset, test_dataset,
                                                                                         dev_dataset,
                                                                                         reporting_root=reporting_rootlin,
                                                                                         verbose=True,
                                                                                         device=device,
                                                                                         num_train_epochs=1,# should be 10 i think
                                                                                         train_batch_size=64)
        compression_lin = uniform_cdl_lin / online_cdl_lin


        online_code_probe_nonlin = OnlineCodeMDLProbe(create_probe_nonlin, fractions, device)
        reporting_rootnonlin = join(os.getcwd(), f'online_coding_tstnonlin{mode}{method}{seed}.pkl')
        uniform_cdl_nonlin, online_cdl_nonlin, gender_nonlin_acc = online_code_probe_nonlin.evaluate(train_dataset,
                                                                                                     test_dataset,
                                                                                                     dev_dataset,
                                                                                                     reporting_root=reporting_rootnonlin,
                                                                                                     verbose=True,
                                                                                                     device=device,
                                                                                                     num_train_epochs=1,# should be 10 i think
                                                                                                     train_batch_size=64)
        compression_nonlin = uniform_cdl_nonlin / online_cdl_nonlin

        scores[method]['acc'].append(round(new_acc * 100, 2))
        scores[method]['gap'].append(round(GAP_rms * 100, 2))

        scores[method]['lleakage'].append(round(gender_lin_acc * 100, 2))
        scores[method]['mdl'].append(round(compression_lin, 3))

        scores[method]['nlleakage'].append(round(gender_nonlin_acc * 100, 2))
        scores[method]['nl_mdl'].append(round(compression_nonlin, 3))

    print(method, scores[method])

    #print('----------uploading to wandb and saving--------------')
    #np.save(f"/home/shadi.isk/nullspace_projection-master/data/exp/igbp-final/{mode}/scores_{seed}.npy", np_score)

def calc_downstream_result(mode,x_train,y_train,x_dev,y_dev,y_dev_gender):
    new_main_clf = LogisticRegression(warm_start=True, penalty='l2',
                                      solver="saga", multi_class='multinomial', fit_intercept=False,
                                      verbose=5, n_jobs=90, max_iter=4)
    if mode == 'moji':
        new_main_clf = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)

    new_main_clf.fit(x_train, y_train)
    dev_acc = new_main_clf.score(x_dev, y_dev)
    dev_GAP = rms(list(
        get_TPR(y_dev, new_main_clf.predict(x_dev), y_dev_gender)[1].values()))
    return dev_GAP, dev_acc
