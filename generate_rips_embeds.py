#!/usr/bin/env python

import pandas as pd
import numpy as np
from ripser import Rips

feat_cols = ['feat-{}'.format(i) for i in range(256)]
train_embed = pd.read_csv('point_cloud_embeds.csv', names=feat_cols)
print("SHAPE: ", train_embed.shape)
print("COLS: ", train_embed.columns)

# rips = Rips(maxdim=2)
rips = Rips()

# Transform
diagrams = rips.fit_transform(train_embed)
lifetime_dim0_pts = diagrams[0][:, 1] - diagrams[0][:, 0] 
lifetime_dim1_pts = diagrams[1][:, 1] - diagrams[1][:, 0]

# Concatenate embeds
train_embed['birth_dim0'] = pd.Series(data=diagrams[0][:, 0])
train_embed['lifetime_dim0'] = pd.Series(data=lifetime_dim0_pts)
train_embed['birth_dim1'] = pd.Series(data=diagrams[1][:, 0])
train_embed['lifetime_dim1'] = pd.Series(data=lifetime_dim1_pts)
# train_embed['birth_dim2'] = pd.Series(data=diagrams[2][:, 0])
# train_embed['lifetime_dim2'] = pd.Series(data=lifetime_dim2_pts)
print("COLS: ", train_embed.columns)

# Save embeds
df = pd.DataFrame(data=train_embed.astype(float))
df.to_csv('tda_embeds.csv', sep=',', header=False, index=False)