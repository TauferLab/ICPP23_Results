#!/bin/env python

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

parser = argparse.ArgumentParser(description="Plot runtimes")
parser.add_argument("logfiles", type=str, nargs="+", help="Directory with log files")

args = parser.parse_args()
log = args.logfiles[0]
df = pd.read_csv(log)

# Number of runs
num_runs = df.groupby(['Scenario','Approach','Rank','Chkpt ID','Number of Chkpts','Chunk Size']).count()['Uncompressed Size'].iloc[0]

# Add Run columns
num_tile = df.shape[0] / num_runs
df['Run'] = np.tile(list(range(1,num_runs+1)), int(num_tile)).tolist()

# Replace TreeLowOffset with Tree
df['Approach'] = df['Approach'].replace({'TreeLowOffset': 'Tree'})

# Add indices for Approach and Graph (used for sorting)
sort_dict = {'Full': int(3), 'Basic': int(2), 'List': int(1), 'Tree': int(0), 'bitcomp': 4, 'cascaded': 5, 'deflate': 6, 'gdeflate': 7, 'lz4': 8, 'snappy': 9, 'zstd': 10}
graph_dict = {'Message Race': int(0), 'Unstructured Mesh': int(1), 'AsiaOSM': int(2), 'Hugebubbles': int(3)}
df['Approach IDX'] = df['Approach'].apply(lambda x: sort_dict[x])
df['Graph IDX'] = df['Scenario'].apply(lambda x: graph_dict[x])
df.sort_values(by='Approach IDX')

# Exclude initial checkpoint
init = df[df['Chkpt ID'] == 0]
frame = df[df['Chkpt ID'] > 0]

# Iterate through each checkpoint frequency
for freq in [5, 10, 20]:
    # Create frame only containing entries with N checkpoints
    print('Number of Checkpoints: ', freq)
    test = frame[frame['Number of Chkpts'] == freq]
    
#    init_df = init.groupby(['Scenario', 'Approach', 'Approach IDX', 'Run']).agg('sum')[['Uncompressed Size', 'Compressed Size', 'Compression Runtime']]
#    init_ratio = init_df['Uncompressed Size'] / init_df['Compressed Size']
#    init_ratio_frame = init_ratio.to_frame().reset_index()
#    init_ratio_frame.columns = ['Scenario', 'Approach', 'Approach IDX', 'Run', 'Deduplication Ratio']
#    init_ratio_frame['Compression Throughput'] = init_ratio_frame['Deduplication Ratio'] / (10**9)
#    init_ratio_frame = init_ratio_frame[init_ratio_frame['Approach'] != 'bitcomp']
#    init_ratio_frame = init_ratio_frame.sort_values(by='Approach IDX')
    
    # Group frame and take sum across Checkpoint IDs
    grouped_df = test.groupby(['Scenario', 'Graph IDX', 'Approach', 'Approach IDX', 'Run']).agg('sum')[['Uncompressed Size', 'Compressed Size', 'Compression Runtime']]
    agg_ratio = grouped_df['Uncompressed Size'] / grouped_df['Compressed Size']
    agg_thrpt = grouped_df['Uncompressed Size'] / grouped_df['Compression Runtime']
    
    # Create ratio frame
    ratio_frame = agg_ratio.to_frame().reset_index()
    ratio_frame.columns = ['Scenario', 'Graph IDX', 'Approach', 'Approach IDX', 'Run', 'Deduplication Ratio']
    ratio_frame['Compression Throughput'] = ratio_frame['Deduplication Ratio'] / (10**9)
    ratio_frame = ratio_frame[ratio_frame['Approach'] != 'bitcomp']
    ratio_frame = ratio_frame.sort_values(by=['Approach IDX', 'Graph IDX'])
    #print(ratio_frame.groupby(['Scenario', 'Approach']).agg('mean'))
    
    # Create throughput frame
    thrpt_frame = agg_thrpt.to_frame().reset_index()
    thrpt_frame.columns = ['Scenario', 'Graph IDX', 'Approach', 'Approach IDX', 'Run', 'Compression Throughput']
    thrpt_frame['Compression Throughput'] = thrpt_frame['Compression Throughput'] / (10**9)
    thrpt_frame = thrpt_frame[thrpt_frame['Approach'] != 'bitcomp']
    thrpt_frame = thrpt_frame.sort_values(by=['Approach IDX', 'Graph IDX'])
    #print(thrpt_frame.groupby(['Scenario', 'Approach']).agg('mean'))
    
    # Hatches and colorblind friendly palette
    hatches = ['', '//', '..', '\\\\', 'o', '///', '...', '\\\\\\', 'oo', '////']
    petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
    
    # Create figure with split in y-axis
    fig,(ax1top, ax1bot) = plt.subplots(2,1, sharex=True, gridspec_kw={'hspace':0.05, 'height_ratios': [1, 2]})
    fig.subplots_adjust(top=0.956, bottom=0.257, left=0.121, right=0.977, hspace=0.2, wspace=0.2)
    
    # Plot ratio twice for y-axis split
    bplot_top = sns.barplot(data=ratio_frame, x='Approach', y='Deduplication Ratio', hue='Scenario', ax=ax1top, palette=petroff10)
    bplot_bot = sns.barplot(data=ratio_frame, x='Approach', y='Deduplication Ratio', hue='Scenario', ax=ax1bot, palette=petroff10)
    labels = ax1bot.get_xticklabels()
    
    # Remove spines so that the subplots look like one plot
    sns.despine(ax=ax1bot, left=False, right=False, top=True)
    sns.despine(ax=ax1top, left=False, right=False, bottom=True, top=False)
    
    # Adjust limits for split
    ax1top.set_ylim(bottom=190)
    ax1bot.set_ylim(0, 190)
    
    # Add break marks
    ax = ax1top
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1-d, 1+d), (-d, +d), **kwargs)        # top-left diagonal
    ax2 = ax1bot
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1-d, 1+d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    
    # Remove one of the legends
    ax1bot.legend().remove()
    ax1top.legend(frameon=False, title='', loc='upper center', prop={'size': 12})
    
    # Adjust plot labels and parameters
    ax1bot.tick_params(top=False, labeltop=False)
    ax1top.tick_params(top=False, labeltop=False)
    ax1top.set_ylabel('')
    ax1top.set_xlabel('')
    ax1top.set_yticklabels(ax1top.get_yticklabels(), fontsize=14)
    ax1bot.set_ylabel('')
    ax1bot.set_xlabel('')
    labels = ['Tree', 'List', 'Basic', 'Full', 'Cascaded', 'Deflate', 'GDeflate', 'LZ4', 'Snappy', 'Zstd']
    ax1top.set_xticklabels([], rotation=60)
    ax1bot.set_xticklabels(labels, rotation=60, fontsize=14)
    ax1bot.set_yticklabels(ax1bot.get_yticklabels(), fontsize=14)
    fig.supylabel('Deduplication Ratio', fontsize=14)
    fig.supxlabel('Approach', fontsize=14)
    
    # Create figure for throughput
    fig2,ax2 = plt.subplots(1,1)

    # Plot throughput
    lplot = sns.barplot(data=thrpt_frame, x='Approach', y='Compression Throughput', hue='Scenario', ax=ax2, errorbar='sd', errwidth=2.0, palette=petroff10)
    ax2.set_ylabel('Deduplication Throughput (GB/s)', fontsize=14)
    ax2.set_xlabel('Approach', fontsize=14)
    ax2.set_ylim(0, 500)
    ax2.set_xticklabels(labels, rotation=60, fontsize=14)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=14)
    ax2.legend().remove()
    
    # Add hatch marks for both figures
    for i,thisbar in enumerate(bplot_top.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[int(i/10)])
    for i,thisbar in enumerate(bplot_bot.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[int(i/10)])
    for i,thisbar in enumerate(lplot.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[int(i/10)])
    
    plt.tight_layout()
    plt.show()
