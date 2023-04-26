#!/bin/env python

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import PIL

parser = argparse.ArgumentParser(description="Plot runtimes")
parser.add_argument("logfiles", type=str, nargs="+", help="Directory with log files")

args = parser.parse_args()
log = args.logfiles[0]
df = pd.read_csv(log)

# Number of runs
num_runs = df.groupby(['Approach', 'Rank', 'Chkpt ID', 'Chunk Size']).count()['Uncompressed Size'].iloc[0]

# Add Run columns
num_tile = df.shape[0] / num_runs
df['Run'] = np.tile(list(range(1,num_runs+1)), int(num_tile)).tolist()

# Rename TreeLowOffset to Tree
df['Approach'] = df['Approach'].replace({'TreeLowOffset': 'Tree'})

# Sort by approach
sort_dict = {'Full': int(3), 'Basic': int(2), 'List': int(1), 'Tree': int(0)}
df['Approach IDX'] = df['Approach'].apply(lambda x: sort_dict[x])
df.sort_values(by='Approach IDX')

# Exclude inital checkpoint and chunk sizes greater than 512
test = df[df['Chkpt ID'] > 0]
test = test[test['Chunk Size'] < 1024]

# Group data and take sum over checkpoint IDs
grouped_df = test.groupby(['Approach', 'Approach IDX', 'Chunk Size', 'Run']).agg('sum')[['Uncompressed Size', 'Compressed Size', 'Compression Runtime']]
agg_ratio = grouped_df['Uncompressed Size'] / grouped_df['Compressed Size']
agg_thrpt = grouped_df['Uncompressed Size'] / grouped_df['Compression Runtime']

# Build separate frames for deduplication ratio
ratio_frame = agg_ratio.to_frame().reset_index()
ratio_frame.columns = ['Approach', 'Approach IDX', 'Chunk Size', 'Run', 'Deduplication Ratio']
ratio_frame['Compression Throughput'] = ratio_frame['Deduplication Ratio'] / (10**9)
ratio_frame['X ticks'] = np.log2(ratio_frame['Chunk Size'] / 32)
ratio_frame = ratio_frame.sort_values(by='Approach IDX')
#print(ratio_frame[['Deduplication Ratio', 'Approach', 'Chunk Size']])

# Build separate frames for deduplication throughput
thrpt_frame = agg_thrpt.to_frame().reset_index()
thrpt_frame.columns = ['Approach', 'Approach IDX', 'Chunk Size', 'Run', 'Compression Throughput']
thrpt_frame['Compression Throughput'] = thrpt_frame['Compression Throughput'] / (10**9)
thrpt_frame['X ticks'] = np.log2(thrpt_frame['Chunk Size'] / 32)
thrpt_frame = thrpt_frame.sort_values(by='Approach IDX')
#print(thrpt_frame[['Compression Throughput', 'Approach', 'Chunk Size']])

# Hatch marks and colorblind friendly palette
hatches = ['', '//', '..', '\\\\', 'o', '///', '...', '\\\\\\', 'oo', '////']
petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]

# Create figure and subplot axes
fig,ax = plt.subplots(1,2, sharex=True, figsize=(6.4, 3.6))
ax1 = ax[0]
ax2 = ax[1]

# Plot ratio and throughput
bplot = sns.barplot(data=ratio_frame, x='X ticks', y='Deduplication Ratio', hue='Approach', ax=ax1, palette=petroff10)
lplot = sns.barplot(data=thrpt_frame, x='X ticks', y='Compression Throughput', hue='Approach', ax=ax2, errorbar='sd', palette=petroff10)

# Add hatch marks
for i,thisbar in enumerate(bplot.patches):
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[int(i/5)])
for i,thisbar in enumerate(lplot.patches):
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[int(i/5)])

# Set labels and remove legend
xticklabels = ['32', '64', '128', '256', '512']
ax1.set_xticklabels(xticklabels, rotation=45, fontsize=12)
ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=12)
ax1.set_ylabel('Deduplication Ratio', fontsize=12)
ax1.legend().remove()
ax2.set_xticklabels(xticklabels, rotation=45, fontsize=12)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=12)
ax2.set_ylabel('Deduplication Throughput (GB/s)', fontsize=12)
ax2.legend().remove()

# Remove subplots xlabels, set single xlabel, and update legend
bplot.set(xlabel=None)
lplot.set(xlabel=None)
fig.supxlabel("Chunk Size", fontsize=12, y=0.05)
ax1.legend(frameon=False, loc='upper right')

plt.tight_layout()
plt.show()
