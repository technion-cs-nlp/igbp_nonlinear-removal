import os
import random
import copy
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from transformers import set_seed, is_wandb_available
import torch
from torch import nn, optim, Tensor
from torch.utils.data import SequentialSampler, DataLoader, Dataset
from tqdm import tqdm, trange, tqdm_notebook
from torch.utils.data import TensorDataset, DataLoader
from evaluation_metrics import calc_downstream_result
import sys
sys.path.append('..')

import wandb

def save_in_word2vec_format(vecs: np.ndarray, words: np.ndarray, fname: str):


    with open(fname, "w", encoding = "utf-8") as f:

        f.write(str(len(vecs)) + " " + "300" + "\n")
        for i, (v ,w) in enumerate(zip(vecs, words)):

            vec_as_str = " ".join([str(x) for x in v])
            f.write(w + " " + vec_as_str + "\n")

def evaluate_probe(
                   eval_dataset: Dataset,
                   model,
                   loss_fn=None,
                   verbose=False,
                   collate_fn=None,
                   device=None):
    """Evaluate a probe model on a given dataset.
    :param args: Arguments for probe training
    :param eval_dataset: The dataset to evaluate on
    :param model: The (trained) probe model
    :param loss_fn: The loss function, Default: CrossEntropyLoss
    :param verbose: If true, prints progress bars and logs. Default: False
    :param collate_fn: A collate function passed to PyTorch `DataLoader`
    :param device:
    :return: A tuple (loss, out_label_ids, preds)
    * loss: mean loss over all samples
    * out_label_ids: The dataset labels, of shape (N,)
    * preds: The model predictions, of shape (N, C), where C is the number of classes that the model predicts
    """
    if eval_dataset is None:
        raise ValueError('eval_dataset cannot be None')
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction='sum')

    tr_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    criterion = loss_fn
    test_sampler = SequentialSampler(eval_dataset)
    test_dataloader = DataLoader(eval_dataset, batch_size=64,
                                 sampler=test_sampler, collate_fn=collate_fn)

    for batch in tqdm(test_dataloader, desc='Evaluating', disable=not verbose):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            embedding = batch[0]
            labels = batch[1]

            outputs = model(embedding)

        loss = criterion(outputs, labels)
        tr_loss += loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = outputs.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    return tr_loss, out_label_ids, preds


def train_probe(probe_classifier, X, y, X_dev, y_dev, epochs = 1,early_stop = 0.85,num_early_stopping=3):
    loss_fn = nn.CrossEntropyLoss()
    train_dataset = TensorDataset(X, y)  # create your datset
    dl_train = DataLoader(train_dataset, batch_size=256, shuffle=True)
    # Training parameters
    # probe_classifier = probe
    criterion = loss_fn
    optimizer = optim.AdamW(probe_classifier.parameters(), lr=0.0002)
    verbose = True
    train_iterator = trange(0, epochs, disable=not verbose)
    epochs_without_improvement = 0
    max_dev_acc = 0
    tr_loss = 0.0
    for _ in train_iterator:
        acc = 0
        total_size = 0
        mean_loss = 0

        num_batches = 0
        epoch_iterator = tqdm(dl_train, desc='Iteration', disable=True)
        probe_classifier.train()
        for step, (embedding, labels) in enumerate(epoch_iterator):
            # Forward propagation
            optimizer.zero_grad()
            outputs = probe_classifier(embedding)

            # Backward propagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate loss and accuracy
            l_item = loss.item()
            _, prediction = torch.max(outputs, 1)
            correct = (prediction == labels).sum().item()
            acc += correct
            tr_loss += l_item
            mean_loss += l_item
            total_size += embedding.size()[0]
            num_batches += 1
        if X_dev is not None:
            # Validation
            probe_classifier.eval()
            with torch.no_grad():
                Y_probs = probe_classifier(X_dev)
                Y_preds = torch.argmax(Y_probs, axis=1)
                dev_acc = ((Y_preds == y_dev).sum() / len(Y_preds)).item()
                if max_dev_acc == 0:
                    max_dev_acc = dev_acc
            if dev_acc > early_stop:
                return dev_acc
            if dev_acc > (max_dev_acc+0.02) and dev_acc < early_stop:
                epochs_without_improvement = 0
                max_dev_acc = dev_acc
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= num_early_stopping:
                    return dev_acc
    return dev_acc

def build_probe(input_size, num_classes, width_list, device):
    if len(width_list)==0: #linear
        return nn.Sequential(nn.Linear(input_size, num_classes)).to(device)
    layers = [nn.Linear(input_size, width_list[0] * input_size), nn.ReLU()]
    for i in range(len(width_list)-1):
        layers.append(nn.Linear(width_list[i] * input_size, width_list[i+1] * input_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width_list[-1] * input_size, num_classes))
    return nn.Sequential(*layers).to(device)

class ManualLossProjection:
    """
    outputs is the vector with shape (batch_size,C)
    target shape is the same (batch_size), whose entries are integers from 0 to C-1
    """
    def __init__(self) -> None:

        self.sftmx = nn.Softmax(dim=1)
    def __call__(self, outputs, target):
        loss = 0.
        n_batch, n_class = outputs.shape
        # print(n_class)
        pt = torch.max(self.sftmx(outputs),dim=1)[0].unsqueeze(0).T
        #pt = torch.gather(self.sftmx(outputs), 1, target.unsqueeze(1))
        loss = (1/2)*torch.pow((torch.log(pt)-torch.log(1-pt)),2).sum()
        return loss/n_batch
def choose_loss_fn(loss_fn_name):
    if loss_fn_name == "projection":
        loss_fn = ManualLossProjection()
    elif loss_fn_name == "ce":
        loss_fn = nn.CrossEntropyLoss()
    else:
        print("loss function not implemented. choose between 'projection' and 'ce'")
        raise NotImplementedError
    return loss_fn



def clean_glove_representations(x_train,y_train_gender,x_dev,y_dev_gender,x_test,
                          max_epochs=1,early_stop=1,num_probes=100, lambdaa = 0.02,batch_size=32,path = f"probe_list.pth",
                          probe_arch = 'non_lin',device=None,loss_fn_name="projection",mode='bios'):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    X_train = torch.FloatTensor(x_train).to(device)
    X_dev = torch.FloatTensor(x_dev).to(device)
    Y_train_gender, Y_dev_gender = torch.LongTensor(y_train_gender).to(device), torch.LongTensor(y_dev_gender).to(device)
    X_test =  torch.FloatTensor(x_test).to(device)
    train_len, dev_len, test_len = len(x_train), len(x_dev), len(X_test)
    num_classes = 2
    input_size = len(X_train[0])
    # main
    ##-------------------------##

    probe_list = []
    probe = build_probe(input_size, num_classes, probe_arch, device)

    # train probe
    number_of_probes = 0
    dev_leakage = train_probe(probe, X_train, Y_train_gender, X_dev, Y_dev_gender, epochs=max_epochs,early_stop = early_stop)
    if is_wandb_available():
        wandb.log({
            'Iteration': 0,
            'Dev Leakage': round(dev_leakage * 100, 3),
        })
    X = torch.cat((X_train, X_dev, X_test), axis=0).to(device)
    maj = (y_train_gender==1).sum()/len(y_train_gender)

    train_dataset = TensorDataset(X)
    iterations_while_under = 0
    loss_fn = choose_loss_fn(loss_fn_name)

    sftmx = nn.Softmax(dim=1)
    while number_of_probes < num_probes:
        dl_train = DataLoader(train_dataset, batch_size=batch_size)
        epoch_iterator = dl_train
        optimizer = optim.AdamW(probe.parameters(), lr=0.001)
        X_cleaned = torch.FloatTensor([]).to(device)
        for step, (embedding,) in enumerate(epoch_iterator):
            probe.eval()
            # Forward propagation
            optimizer.zero_grad()
            embedding.requires_grad = True
            outputs = probe(embedding)
            labels = torch.argmax(outputs,dim=1)
            # compute loss
            loss = loss_fn(outputs, labels).mean()

            # Zero all existing gradients
            probe.zero_grad()

            # Calculate gradients
            loss.backward()
            # probabilies
            embedding_grad = embedding.grad.data * batch_size
            adv_grad = torch.nan_to_num(embedding_grad.detach())
            if loss_fn_name == "projection":
                pt = torch.max(sftmx(outputs.detach()), dim=1)[0].unsqueeze(0).T
                scalar = -torch.log(1 / pt - 1)
                theta = adv_grad / scalar
                theta_norm2 = torch.bmm(theta.unsqueeze(1), theta.unsqueeze(2)).squeeze(2)
                perturbation = torch.nan_to_num(adv_grad.detach() / theta_norm2.detach())
            elif loss_fn_name == "ce":
                perturbation = -1*lambdaa * adv_grad
            else:
                raise("not implemented")
            perturbed_embedding = embedding.detach() - perturbation
            X_cleaned = torch.cat([X_cleaned, perturbed_embedding])
            del loss
            del embedding
            del adv_grad
            torch.cuda.empty_cache()

        X_train_cleaned = X_cleaned[:train_len].to(device)
        X_dev_cleaned = X_cleaned[train_len:train_len + dev_len].to(device)
        X_test_cleaned = X_cleaned[train_len + dev_len:]
        del train_dataset

        # ----evaluate probe----
        with torch.no_grad():
            probe.eval()
            Y_probs = probe(X_dev_cleaned)
            Y_preds = torch.argmax(Y_probs, axis=1)
            old_probe_acc = (Y_preds == Y_dev_gender).sum() / len(Y_preds)
            print("After projection probe acc = {}".format(old_probe_acc)) # this should be close to 0.5

        # -----fineTune-----
        number_of_probes += 1
        probe_list.append(copy.deepcopy(probe))
        del probe

        probe = build_probe(input_size, num_classes, probe_arch, device)
        probe.train()
        dev_probe_acc = train_probe(probe, X_train_cleaned, Y_train_gender, X_dev_cleaned, Y_dev_gender, epochs=max_epochs)

        if is_wandb_available():
            wandb.log({
                'Iteration': number_of_probes,
                'Dev Leakage': round(dev_probe_acc * 100, 3),
            })

        if dev_probe_acc <= maj :
            print("****** : ", number_of_probes)
            break


        train_dataset = TensorDataset(X_cleaned)  # create your datset
        del X_cleaned
        torch.cuda.empty_cache()

    torch.save(probe_list,path)
    return X_train_cleaned.detach().cpu().numpy(), X_dev_cleaned.detach().cpu().numpy(), X_test_cleaned.detach().cpu().numpy(), probe_list

def clean_representations(x_train,y_train,y_train_gender,x_dev,y_dev,y_dev_gender,x_test,
                          max_epochs=1,early_stop=1,num_probes=100, lambdaa = 0.02,batch_size=32,path = f"probe_list.pth",
                          probe_arch = [1],device=None,loss_fn_name="projection",mode='bios', reach_max_probes = True, save=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    X_train, Y_train, Y_train_gender = torch.FloatTensor(x_train).to(device), torch.LongTensor(y_train).to(device), torch.LongTensor(y_train_gender).to(device)
    X_dev  , Y_dev  , Y_dev_gender   = torch.FloatTensor(x_dev).to(device)  , torch.LongTensor(y_dev).to(device)  , torch.LongTensor(y_dev_gender).to(device)
    X_test =  torch.FloatTensor(x_test).to(device)
    train_len, dev_len, test_len = len(x_train), len(x_dev), len(X_test)

    num_classes = 2 # TODO : if bigger than 2 make adjustment
    input_size = len(X_train[0])
    majority = (y_train_gender==1).sum()/len(y_train_gender)

    probe_list = []
    number_of_probes = 0
    probe = build_probe(input_size, num_classes, probe_arch, device)

    # train probe
    dev_leakage = train_probe(probe, X_train, Y_train_gender, X_dev, Y_dev_gender, epochs=max_epochs,early_stop = early_stop)

    # build dataset
    X = torch.cat((X_train, X_dev, X_test), axis=0).to(device)
    train_dataset = TensorDataset(X)


    should_stop_counter = 0
    loss_fn = choose_loss_fn(loss_fn_name)
    sftmx = nn.Softmax(dim=1)
    dev_org_GAP, dev_org_accuracy = calc_downstream_result(mode,x_train,y_train,x_dev,y_dev,y_dev_gender)
    if is_wandb_available():
        wandb.log({
            'Iteration': 0,
            'Dev_Accuracy': round(dev_org_accuracy * 100, 3),
            'Dev_Gap': round(dev_org_GAP * 100, 3),
            'Dev Leakage': round(dev_leakage * 100, 3),
            'Accuracy Drop': round(0, 4)
        })
    #IGBP loop
    while number_of_probes < num_probes:
        dl_train = DataLoader(train_dataset, batch_size=batch_size)
        optimizer = optim.AdamW(probe.parameters(), lr=0.001)
        X_cleaned = torch.FloatTensor([]).to(device)
        probabilites_before = torch.FloatTensor([]).to(device)
        for step, (embedding,) in enumerate(dl_train):
            probe.eval()
            # Forward propagation
            optimizer.zero_grad()
            embedding.requires_grad = True
            outputs = probe(embedding)
            labels = torch.argmax(outputs,dim=1)
            # compute loss
            loss = loss_fn(outputs, labels).mean()

            # Zero all existing gradients
            probe.zero_grad()

            # Calculate gradients
            loss.backward()
            # probabilies
            embedding_grad = embedding.grad.data * batch_size
            adv_grad = torch.nan_to_num(embedding_grad.detach())
            if loss_fn_name == "projection":
                pt = torch.max(sftmx(outputs.detach()), dim=1)[0].unsqueeze(0).T
                probabilites_before = torch.cat([probabilites_before,pt])
                scalar = -torch.log(1 / pt - 1)
                theta = adv_grad / scalar
                theta_norm2 = torch.bmm(theta.unsqueeze(1), theta.unsqueeze(2)).squeeze(2)
                perturbation = torch.nan_to_num(adv_grad.detach() / theta_norm2.detach())
            elif loss_fn_name == "ce":
                perturbation = -1*lambdaa * adv_grad
            else:
                raise("not implemented")
            perturbed_embedding = embedding.detach() - perturbation
            X_cleaned = torch.cat([X_cleaned, perturbed_embedding])
            del loss
            del embedding
            del adv_grad
            torch.cuda.empty_cache()
        X_train_cleaned = X_cleaned[:train_len].to(device)
        X_dev_cleaned = X_cleaned[train_len:train_len + dev_len].to(device)
        X_test_cleaned = X_cleaned[train_len + dev_len:]
        del train_dataset

        number_of_probes += 1
        probe_list.append(copy.deepcopy(probe))
        del probe
        # train new probe
        probe = build_probe(input_size, num_classes, probe_arch ,device)
        probe.train()
        dev_probe_acc = train_probe(probe, X_train_cleaned, Y_train_gender, X_dev_cleaned, Y_dev_gender, epochs=max_epochs,early_stop=early_stop)


        dev_acc, dev_GAP = calc_downstream_result(mode,X_train_cleaned.detach().cpu().numpy(),y_train,X_dev_cleaned.detach().cpu().numpy(),y_dev,y_dev_gender)
        dev_accuracy_drop = (dev_org_accuracy - dev_acc) * 100 / dev_org_accuracy

        if is_wandb_available():
            wandb.log({
                'Iteration': number_of_probes,
                'Dev_Accuracy': round(dev_acc * 100, 3),
                'Dev_Gap': round(dev_GAP * 100, 3),
                'Dev Leakage': round(dev_probe_acc * 100, 3),
                'Accuracy Drop': round(dev_accuracy_drop, 4)
            })

        if not reach_max_probes and (dev_accuracy_drop > 2 or dev_probe_acc <= (majority+0.02)):
            should_stop_counter += 1
            if should_stop_counter >= 2:
                print("IGBP Algorithm Finished, Number of Total Debiasing Probes: ", number_of_probes)
                break
        else:
            should_stop_counter = 0
        train_dataset = TensorDataset(X_cleaned)  # create your datset
        del X_cleaned
        torch.cuda.empty_cache()

    if save:
        torch.save(probe_list,path)

    return X_train_cleaned.detach().cpu().numpy(), X_dev_cleaned.detach().cpu().numpy(), X_test_cleaned.detach().cpu().numpy(), probe_list


def clean_test_representations(x_test,lambdaa = 0.02,batch_size=32,path = f"probe_list.pth",device=None,loss_fn_name="projection"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test = torch.FloatTensor(x_test).to(device)

    # to load back
    probe_list = torch.load(path)
    loss_fn = choose_loss_fn(loss_fn_name)
    sftmx = nn.Softmax(dim=1)
    test_dataset = TensorDataset(X_test)
    for i, probe in enumerate(probe_list):
        probe = probe.to(device)
        optimizer = optim.AdamW(probe.parameters(), lr=0.001)
        norm = 0
        dl_test = DataLoader(test_dataset, batch_size=batch_size)
        X_test_cleaned = torch.FloatTensor([]).to(device)
        for step, embedding in enumerate(dl_test):

            probe.eval()
            # Forward propagation
            optimizer.zero_grad()
            embedding.requires_grad = True
            outputs = probe(embedding)
            labels = torch.argmax(outputs,dim=1)
            # compute loss
            loss = loss_fn(outputs, labels).mean()

            # Zero all existing gradients
            probe.zero_grad()

            # Calculate gradients
            loss.backward()
            # probabilies
            pt = torch.max(sftmx(outputs.detach()), dim=1)[0].unsqueeze(0).T
            embedding_grad = embedding.grad.data * batch_size
            adv_grad = torch.nan_to_num(embedding_grad.detach())
            norm += (torch.norm(adv_grad.detach(), dim=1).sum()).item()
            if loss_fn_name == "projection":
                scalar = -torch.log(1 / pt - 1)
                theta = adv_grad / scalar
                theta_norm2 = torch.bmm(theta.unsqueeze(1), theta.unsqueeze(2)).squeeze(2)
                perturbation = torch.nan_to_num(adv_grad.detach() / theta_norm2.detach())
            elif loss_fn_name == "ce":
                perturbation = -1 * lambdaa * adv_grad
            else:
                raise("Not Implemented")
            perturbed_embedding = embedding.detach() - perturbation.detach()
            del loss
            del embedding
            del adv_grad
            torch.cuda.empty_cache()
            X_test_cleaned = torch.cat([X_test_cleaned, perturbed_embedding])
        test_dataset = TensorDataset(X_test_cleaned)

    x_test_cleaned = X_test_cleaned.detach().cpu().numpy()
    return x_test_cleaned



