#!/bin/env python

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

parser = argparse.ArgumentParser(description="Plot runtimes")
parser.add_argument("logfiles", type=str, nargs="+", help="Directory with log files")

args = parser.parse_args()
log = args.logfiles[0]
df = pd.read_csv(log)

# Isolate only Full and Tree approach
df = df[df['Approach'] != 'TreeLowRoot']
df = df[df['Approach'] != 'List']
df = df[df['Approach'] != 'Basic']

# Update Approach column to use Tree instaed of TreeLowOffset
df['Approach'] = df['Approach'].replace({'TreeLowOffset': 'Tree'})

# Sort entries by approach
sort_dict = {'Full': int(3), 'Basic': int(2), 'List': int(1), 'Tree': int(0), 'bitcomp': 4, 'cascaded': 5, 'deflate': 6, 'gdeflate': 7, 'lz4': 8, 'snappy': 9, 'zstd': 10}
df['Approach IDX'] = df['Approach'].apply(lambda x: sort_dict[x])
df.sort_values(by='Approach IDX')

# Group data so we can sum data across Ranks
grouped_df = df.groupby(['Scale', 'Approach', 'Approach IDX', 'Run']).agg('sum')

# Hatches and colorblind friendly color palette
hatches = ['', '//', '..', '\\\\', 'o', '///', '...', '\\\\\\', 'oo', '////']
petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]

# Create Figure
axes = plt.subplot(1,1,1)

# Reset index and resort frame by Approach
frame = grouped_df.reset_index()
frame = frame.sort_values(by='Approach IDX')

# Adjust Deduplicated size to use TB instead of bytes
frame['Deduplicated Size'] = frame['Deduplicated Size'] / (10**12)

# Plot Deduplicated size
bplot = sns.barplot(data=frame, x='Scale', y='Deduplicated Size', hue='Approach', palette=petroff10)

# Adjust legend to remove frame, title, use the correct font size, and position the legend
axes.legend(frameon=False, title='', loc='upper right', bbox_to_anchor=(0.9,1), prop={'size': 12})

# Get max value for the tree approach at 64 processes
ymax = frame[(frame['Scale'] == 64) & (frame['Approach'] == 'Tree')]['Deduplicated Size'].max()

# Create inset axis for Tree approach. Needed to actually see the values due to the large difference in size
axins = axes.inset_axes([0.15, 0.45, 0.47, 0.47])

# Plot inset axis
zplot = sns.barplot(data=frame, x='Scale', y='Deduplicated Size', hue='Approach', ax=axins, palette=petroff10)
axins.set_ylim(top=ymax*1.1)
axins.set_xlabel('# of Processes (GPUs)', fontsize=12)
axins.set_ylabel('Total Checkpoint Size (TB)', fontsize=12)
axins.set_xticklabels(axins.get_xticklabels(), fontsize=12)
axins.set_yticklabels(axins.get_yticklabels(), fontsize=12)
axins.set_title("Zoom")
axins.legend().remove()
inset_patch, (line1, line2, line3, line4) = axes.indicate_inset_zoom(axins, edgecolor="black")
line1.set(visible=True)
line2.set(visible=False)
line3.set(visible=True)
line4.set(visible=False)

# Add hatch marks to both plots
for i,thisbar in enumerate(bplot.patches):
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[int(i/7)])
for i,thisbar in enumerate(zplot.patches):
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[int(i/7)])

# Update figure labels
axes.set_xticklabels(axes.get_xticklabels(), fontsize=12)
axes.set_yticklabels(axes.get_yticklabels(), fontsize=12)
axes.set_xlabel('# of Processes (GPUs)', fontsize=12)
axes.set_ylabel('Total Checkpoint Size (TB)', fontsize=12)
plt.tight_layout()

# Create second figure for throughput
fig2,axes2 = plt.subplots(1,1)

# Change units for throughput to GB
frame['Deduplication Throughput'] = frame['Deduplication Throughput'] / (10**9)

# Plot throughput
thrpt = sns.barplot(data=frame, x='Scale', y='Deduplication Throughput', hue='Approach', ax=axes2, palette=petroff10)
axes2.set_ylabel('Deduplication Throughput (TB/s)', fontsize=12)
axes2.set_xlabel('# of Processes (GPUs)', fontsize=12)
axes2.set_xticklabels(axes2.get_xticklabels(), fontsize=12)
axes2.set_yticklabels(axes2.get_yticklabels(), fontsize=12)
axes2.legend(frameon=False, title='', prop={'size': 12})

# Add hatches
for i,thisbar in enumerate(thrpt.patches):
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[int(i/7)])

plt.tight_layout()

# Show figures
plt.show()

