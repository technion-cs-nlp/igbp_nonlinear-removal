import os
import sys
sys.path.append("..")
sys.path.append("../src")
sys.path.append("../data/embeddings")
sys.path.append("../data/biasbios")
sys.path.append("../data/embeddings/biasbios")
import numpy as np
from debias_representation import clean_representations, clean_test_representations
import random
import wandb
import torch
from evaluation_metrics import evaluate
import warnings
warnings.filterwarnings("ignore")
import argparse
from data_loading import load_dataset, load_glove, load_dictionary, load_data


DATA_PATH = "" # TODO: set this to the path of the data folder
RESULTS_PATH = "" # TODO: set this to the path of the results folder



def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_probes", type=int, default=100, help='The number of iterations to run igbp for')
    parser.add_argument('--probe_arch', help='The architecture of the probe used in igbp.expects list of hidden layers sizes. for example: one hidden layer of the same input size [1], two hidden layers with double the input size [2,2]',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument("--loss", default="projection", help='The loss function to use. expects "projection" to run igbp or "ce" to run regular adversarial shifting')
    parser.add_argument("--mode", default="bert-frozen", help='The dataset to run igbp on. expects "moji" for the twitter-sentiment task. for bias in bios : "bert-frozen" to run igbp on a frozen bert model or "bert-finetuned" to run igbp on a finetuned bert model . same for roberta')
    parser.add_argument("--save", type=int, default=0, help='Whether to save the model or not')
    parser.add_argument("--reach_max_probes", type=int, default=1, help='Whether to stop igbp when the number of probes reaches the max number of probes or when the accuracy drops below the early stop threshold')
    parser.add_argument("--n_seeds", type=int, default=1, required=False, help='The number of seeds to run igbp for')
    parser.add_argument("--batch_size",type=int, default=256, required=False, help='The batch size to use for igbp')
    parser.add_argument("--project_name", default="Attemp1", required=False, help='The name of the wandb project to use')



    return parser.parse_args()

if __name__=="__main__":
    args = prepare_args()

    n_seeds = args.n_seeds
    num_probes = args.n_probes
    batch_size = args.batch_size
    probe_arch = args.probe_arch
    loss = args.loss
    mode = args.mode
    save = bool(args.save)
    reach_max_probes = bool(args.reach_max_probes)
    project_name = args.project_name
    calc_mdl = False
    max_epochs = 10
    early_stop = 0.80

    config = dict(
        num_probes=num_probes,
        batch_size=batch_size,
        architecture=probe_arch,
        mode=mode,
        loss = loss
    )

    print('Loading data')
    """Load Data"""
    x_train, y_train_gender, y_train, x_dev, y_dev_gender, y_dev, x_test, y_test_gender, y_test = load_data(mode)


    print('Cuda Available: ',torch.cuda.is_available())

    method = 'igbp'
    scores = {method: {'acc': [], 'gap': [], 'lleakage': [], 'mdl': [], 'nlleakage': [],
                       'nl_mdl': []}}


    project_name = 'igbp'

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        folder_path = RESULTS_PATH + f"igbp/{mode}/{probe_arch}/{seed}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        model_path = folder_path + "/probe_list.pth"


        run_name = f"{mode}_{seed}"
        run = wandb.init(project=project_name, name=run_name)


        x_train_cleaned, x_dev_cleaned, x_test_cleaned, probe_list = clean_representations(x_train,y_train,y_train_gender,x_dev,y_dev,y_dev_gender,x_test,
                            max_epochs=max_epochs, early_stop=early_stop,num_probes=num_probes, batch_size=batch_size,
                              probe_arch=probe_arch, path = model_path, loss_fn_name=loss,mode=mode,reach_max_probes=reach_max_probes,save=save)
        #save all cleaned data
        if save:
            np.save(folder_path + "/x_train_cleaned".format(mode,seed), x_train_cleaned)
            np.save(folder_path + "/x_dev_cleaned".format(mode,seed), x_dev_cleaned)
            np.save(folder_path + "/x_test_cleaned".format(mode,seed), x_test_cleaned)
            torch.save(probe_list, model_path)

        #evaluate
        evaluate(mode,seed,method,num_probes,x_train_cleaned,y_train,y_train_gender,x_dev_cleaned,y_dev,y_dev_gender,
                    x_test_cleaned,y_test,y_test_gender,scores,calc_mdl)
