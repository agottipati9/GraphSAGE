from __future__ import print_function
import json
import numpy as np
import pandas as pd

from networkx.readwrite import json_graph
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

''' To evaluate the embeddings, we run a logistic regression.
Run this script after running unsupervised training.
Baseline of using features-only can be run by setting data_dir as 'feat'
Example:
  python eval_scripts/tda_eval.py no > results.txt
  python eval_scripts/tda_eval.py yes > results_tda.txt
'''

def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    np.random.seed(1)
    from sklearn.linear_model import SGDRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.multioutput import MultiOutputRegressor
    dummy = MultiOutputRegressor(DummyRegressor())
    dummy.fit(train_embeds, train_labels)
    sgd_opt = MultiOutputRegressor(SGDRegressor(loss="squared_loss"), n_jobs=10)
    sgd_opt.fit(train_embeds, train_labels)
    for i in range(test_labels.shape[1]):
        print("RMSE score", mean_squared_error(test_labels[:,i], sgd_opt.predict(test_embeds)[:,i])**0.5)
    for i in range(test_labels.shape[1]):
        print("Random Baseline RMSE score", mean_squared_error(test_labels[:,i], dummy.predict(test_embeds)[:,i])**0.5)

if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on PPI data.")
    parser.add_argument("tda", help="Either yes or no")
    args = parser.parse_args()
    use_tda = args.tda
    data_file = 'graph_embeds_tda' if use_tda == 'yes' else 'graph_embeds'
    print("Loading data...")
    print("Using TDA?: ", use_tda)        
    embeds = np.load('/notebooks/subgraphs/{}.npy'.format(data_file))
    labels = np.load('/notebooks/subgraphs/graph_labels.npy')
    train_embeds, test_embeds, train_labels, test_labels = train_test_split(embeds, labels, test_size=0.20, random_state=42)
    print("Running regression..")
    run_regression(train_embeds, train_labels, test_embeds, test_labels)