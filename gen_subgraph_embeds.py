#!/usr/bin/env python

import numpy as np
import json
import pickle
import networkx as nx
from networkx.readwrite import json_graph
import glob
import os
import pandas as pd
from ripser import Rips
from sklearn.preprocessing import MinMaxScaler

def node_to_graph(file_name='val', label_op='avg'):
    # Load in node embeddings, node labels, subgraph
    nodes = np.load("/notebooks/subgraphs/{}.npy".format(file_name))
    labels = json.load(open("/notebooks/example_data/toy-ppi-class_map.json"))
    with open("/notebooks/subgraphs/{}_graph.pkl".format(file_name), 'rb') as f:
        G = pickle.load(f)

    # Get subgraph node labels
    graph_labels = []
    for n in G.nodes():
        graph_labels.append(labels[str(n)])

    # Generate graph embeddings and label
    graph_embed_tda = np.mean(tda_features(nodes), axis=0)
    graph_embed = np.mean(nodes, axis=0)
    graph_labels = np.array(graph_labels)
    if label_op == 'avg':
        graph_labels = np.mean(graph_labels, axis=0)
    elif label_op == 'sum':
        graph_labels = np.sum(graph_labels, axis=0)
    else:
        graph_labels = np.sum(graph_labels, axis=0)
        for i in range(graph_labels.shape[0]):
            graph_labels[i] = min(graph_labels[i], 1)
    return graph_embed, graph_embed_tda, graph_labels

def tda_features(nodes):
    # rips = Rips(maxdim=2)
    feat_cols = ['feat-{}'.format(i) for i in range(nodes.shape[1])]
    embeds = pd.DataFrame(nodes, columns=feat_cols)
    rips = Rips()
    scaler = MinMaxScaler()

    # Transform
    print("Generating rips barcodes... This may take a while.")
    diagrams = rips.fit_transform(embeds)
    birth_dim0 = diagrams[0][:, 0]
    birth_dim1 = diagrams[1][:, 0]
    lifetime_dim0_pts = diagrams[0][:, 1] - diagrams[0][:, 0]
    lifetime_dim1_pts = diagrams[1][:, 1] - diagrams[1][:, 0]

    # Replace NaN in dim0
    i = np.argwhere(~np.isfinite(lifetime_dim0_pts))
    if (len(i) > 0):
        print('Cleaning dim0...')
        lifetime_dim0_pts[i] = lifetime_dim0_pts.min() # Set NaNs to lowest real value
        lifetime_dim0_pts[i] = lifetime_dim0_pts.max() + 1.0 # Replace NaNs with largest value

    # Replace NaN in dim0
    i = np.argwhere(~np.isfinite(lifetime_dim1_pts))
    if (len(i) > 0):
        print('Cleaning dim1...')
        lifetime_dim1_pts[i] = lifetime_dim1_pts.min() # Set NaNs to lowest real value
        lifetime_dim1_pts[i] = lifetime_dim1_pts.max() + 1.0 # Replace NaNs with largest value

    # Remove 0s
    birth_dim0[birth_dim0 <= 0] = 1e-7
    birth_dim1[birth_dim1 <= 0] = 1e-7

    # Weight birth times
    birth_dim0 = np.reciprocal(birth_dim0)
    birth_dim1 = np.reciprocal(birth_dim1)

    # MinMax scaling
    birth_dim0 = scaler.fit_transform(birth_dim0.reshape(-1, 1))
    birth_dim1 = scaler.fit_transform(birth_dim1.reshape(-1, 1))
    lifetime_dim0_pts = scaler.fit_transform(lifetime_dim0_pts.reshape(-1, 1))
    lifetime_dim1_pts = scaler.fit_transform(lifetime_dim1_pts.reshape(-1, 1))

    # Concatenate tda features to embeds
    embeds['birth_dim0'] = pd.Series(data=birth_dim0)
    embeds['lifetime_dim0'] = pd.Series(data=lifetime_dim0_pts)
    embeds['birth_dim1'] = pd.Series(data=birth_dim1)
    embeds['lifetime_dim1'] = pd.Series(data=lifetime_dim1_pts)
    # embeds['birth_dim2'] = pd.Series(data=diagrams[2][:, 0])
    # embeds['lifetime_dim2'] = pd.Series(data=lifetime_dim2_pts)
    embeds.fillna(0, inplace=True)
    return embeds.values

def get_graph_data(label_op='avg'):
    data = []
    data_tda = []
    labels = []
    files = glob.glob('/notebooks/subgraphs/' + '*.npy')
    for f in files:
        filepath, file_ = os.path.split(f)
        file_name, _ = os.path.splitext(file_)
        if 'graph_embeds' not in file_name and 'graph_labels' not in file_name:
            graph_embed, graph_embed_tda, graph_labels = node_to_graph(file_name=file_name, label_op=label_op)
            data.append(graph_embed)
            data_tda.append(graph_embed_tda)
            labels.append(graph_labels)
    return np.array(data), np.array(data_tda), np.array(labels)

data, data_tda, labels = get_graph_data()

# Save subgraph embeddings
with open('/notebooks/subgraphs/graph_embeds.npy', 'wb') as f:
    np.save(f, data)
with open('/notebooks/subgraphs/graph_embeds_tda.npy', 'wb') as f:
    np.save(f, data_tda)
with open('/notebooks/subgraphs/graph_labels.npy', 'wb') as f:
    np.save(f, labels)
