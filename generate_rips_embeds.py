#!/usr/bin/env python

import pandas as pd
import numpy as np
from ripser import Rips

feat_cols = ['feat-{}'.format(i) for i in range(256)]
train_embed = pd.read_csv('point_cloud_embeds.csv', names=feat_cols)
# print("SHAPE: ", train_embed.shape)
# print("COLS: ", train_embed.columns)

# rips = Rips(maxdim=2)
rips = Rips()

# Transform
print("Generating rips barcodes... This may take a while.")
diagrams = rips.fit_transform(train_embed)
lifetime_dim0_pts = diagrams[0][:, 1] - diagrams[0][:, 0] 
lifetime_dim1_pts = diagrams[1][:, 1] - diagrams[1][:, 0]

# Replace NaN in dim0
i = np.argwhere(~np.isfinite(lifetime_dim0_pts))
if (len(i) > 0):
    print('Cleaning dim0...')
    lifetime_dim0_pts[i] = lifetime_dim0_pts.min() # Set NaNs to lowest real value
    lifetime_dim0_pts[i] = lifetime_dim0_pts.max() + .01 # Replace NaNs with largest value

# Replace NaN in dim0
i = np.argwhere(~np.isfinite(lifetime_dim1_pts))
if (len(i) > 0):
    print('Cleaning dim1...')
    lifetime_dim1_pts[i] = lifetime_dim1_pts.min() # Set NaNs to lowest real value
    lifetime_dim1_pts[i] = lifetime_dim1_pts.max() + .01 # Replace NaNs with largest value

# Concatenate embeds
train_embed['birth_dim0'] = pd.Series(data=diagrams[0][:, 0])
train_embed['lifetime_dim0'] = pd.Series(data=lifetime_dim0_pts)
train_embed['birth_dim1'] = pd.Series(data=diagrams[1][:, 0])
train_embed['lifetime_dim1'] = pd.Series(data=lifetime_dim1_pts)
# train_embed['birth_dim2'] = pd.Series(data=diagrams[2][:, 0])
# train_embed['lifetime_dim2'] = pd.Series(data=lifetime_dim2_pts)
# print("COLS: ", train_embed.columns)

# Save embeds
print('Saving tda_embeds.csv...')
df = pd.DataFrame(data=train_embed.astype(float))
df.to_csv('tda_embeds.csv', sep=',', header=False, index=False)
