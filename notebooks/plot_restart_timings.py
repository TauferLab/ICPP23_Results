import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D
from matplotlib import gridspec
import numpy as np
import argparse
import pickle
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import copy

def plot_clustered_stacked_frames(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

#    color_cycle = axe._get_lines.prop_cycler
    colorlist = plt.rcParams['axes.prop_cycle'].by_key()['color']

    offset = 1
    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      color=[colorlist[0]] + colorlist[offset:],
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots
        offset += len(df.columns)
#        for i in range(len(df.columns)):
#            next(color_cycle)

    hatches = ['xx', '//', '\\\\', '++', '--', '**', 'oo']
    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + (1 / float(n_df + 1)) * i / float(n_col) + 0.1)
#                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col) - 0.2)
#                rect.set_hatch(H * int(i / n_col)) #edited part     
                if j > 0:
                    rect.set_hatch(hatches[int((i+j)/2)+1]) #edited part     
                rect.set_width(1 / float(n_df + 1.5))
    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    labellist = []
#    labellist.append(dfall[0].columns[0])
    for i in range(len(dfall)):
        cols = dfall[i].columns
        for j in range(0, len(cols)):
            if j != 0:
                labellist.append(labels[i] + ' ' + cols[j])
            else:
                labellist.append(cols[j])
    print(labellist)

    handles, labels = axe.get_legend_handles_labels()
    print(labels)
    labels = labellist
    oc_set = set()
    res = []
    for idx,val in enumerate(labels):
        if val not in oc_set:
            oc_set.add(val)
        else:
            res.append(idx)
    filtered_labels = []
    filtered_handles = []
    for i in range(len(labels)):
        if i not in res:
            filtered_labels.append(labels[i])
            filtered_handles.append(handles[i])
        
    plt.legend(filtered_handles, filtered_labels)

#    # Add invisible data to add another legend
#    n=[]        
#    for i in range(n_df):
#        n.append(axe.bar(0, 0, color="gray", hatch=H * i))
#
##    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.8])
#    l1 = axe.legend(h[:n_col], l[:n_col], loc=[0.01, -0.21])
#    if labels is not None:
##        l2 = plt.legend(n, labels, loc=[1.01, 0.0]) 
#        l2 = plt.legend(n, labels, loc=[0.81, -0.21]) 
#    axe.add_artist(l1)
    return axe

def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
#                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col) - 0.2)
                rect.set_x(rect.get_x() + (1 / float(n_df + 1)) * i / float(n_col) + 0.1)
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))
    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

#    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.8])
    l1 = axe.legend(h[:n_col], l[:n_col], loc=[-0.10, -0.21])
    if labels is not None:
#        l2 = plt.legend(n, labels, loc=[1.01, 0.0]) 
        l2 = plt.legend(n, labels, loc=[0.81, -0.21]) 
    axe.add_artist(l1)
    return axe

def get_frame(master_frame, impl, chunk_size, alg):
    fl = []
    for i in range(int(master_frame['Checkpoint'].max())):
        if i%2 == 0:
            frame = master_frame.loc[(master_frame['Implementation'] == impl) & (master_frame['Chunk size'] == chunk_size) & (master_frame['Checkpoint'] == str(i))]
            if alg == 'full':
                diff = frame.filter(like='Full checkpoint diff')
                wrte = frame.filter(like='Full checkpoint write')
                diff.columns = [i]
                wrte.columns = [i]
                frame = diff + wrte
            elif alg == 'list':
                diff = frame.filter(like='list diff')
                wrte = frame.filter(like='list write')
                diff.columns = [i]
                wrte.columns = [i]
                frame = diff + wrte
            elif alg == 'tree':
                diff = frame.filter(like='tree diff')
                wrte = frame.filter(like='tree write')
                diff.columns = [i]
                wrte.columns = [i]
                frame = diff + wrte
            frame.columns = [i]
            fl.append(frame)
    df = pd.concat(fl, axis=1)
    return df


parser = argparse.ArgumentParser(description="Plot runtimes")
parser.add_argument("logfiles", type=str, nargs="+", help="Log files")
parser.add_argument("--hashfunc", type=str, nargs=1, default="MD5", help="Which hash function to plot the data (MD5|Murmur3C")
parser.add_argument("--vary_chunk_size", action='store_true', help="Plot for different chunk sizes")
parser.add_argument("--num_ranks", type=int, default=1, help='Number of ranks for input files')
parser.add_argument("--title", type=str, default='')
parser.add_argument("--ipdps", action='store_true', help='IPDPS')

args = parser.parse_args()
print(args.hashfunc)

logs = args.logfiles

if args.vary_chunk_size:
    logs = args.logfiles

    frames = []
    for log in logs:
        implementation = 'GPU'
        if 'Serial' in log:
            implementation = 'Serial'
        elif 'CPU' in log:
            implementation = 'CPU'
        elif 'GPU' in log:
            implementation = 'GPU'
    
        file_name = log.split('.')
    
        chunksize_idx = file_name.index('chunk_size')
        chunk_size = file_name[chunksize_idx+1]
    
        checkpoint_idx = file_name.index('dat')
        checkpoint = file_name[checkpoint_idx-1]
    
        df = pd.read_csv(log, header=None)
        df.columns = ['Full checkpoint diff time', 'Full checkpoint write time', 'Full checkpoint data', 'Full checkpoint metadata', 'Hash list diff time', 'Hash list write time', 'Hash list data', 'Hash list metadata', 'Merkle tree diff time', 'Merkle tree write time', 'Merkle tree data', 'Merkle tree metadata']
        df['Chunk size'] = pd.Series([chunk_size for i in range(len(df.index))])
        df['Checkpoint'] = pd.Series([checkpoint[-1] for i in range(len(df.index))])
        df['Implementation'] = pd.Series([implementation for i in range(len(df.index))])
        frames.append(df)
    master_frame = pd.concat(frames, axis=0)
    print(master_frame[['Hash list data', 'Hash list metadata', 'Merkle tree data', 'Merkle tree metadata', 'Chunk size', 'Checkpoint']])
#    fig = plt.figure()
    list128_frames = []
    tree128_frames = []
    list4096_frames = []
    tree4096_frames = []
    list_frames = {'128': [], '256': [], '512': [], '1024': [], '2048': [], '4096': []}
    tree_frames = {'128': [], '256': [], '512': [], '1024': [], '2048': [], '4096': []}
    for i in range(int(master_frame['Checkpoint'].max())):
#        if i % 1 == 0:
        if i > 0:
            list_128  = pd.DataFrame(master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '128') & (master_frame['Checkpoint'] == str(i))].filter(like='time').filter(like='list').mean()).T
            list_128.columns = ['Calculate Checkpoint', 'Write Checkpoint']
            list_128.index = [i]
            list_frames['128'].append(list_128)
            tree_128  = pd.DataFrame(master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '128') & (master_frame['Checkpoint'] == str(i))].filter(like='time').filter(like='tree').mean()).T
            tree_128.columns = ['Calculate Checkpoint', 'Write Checkpoint']
            tree_128.index = [i]
            tree_frames['128'].append(tree_128)

            list_256  = pd.DataFrame(master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '256') & (master_frame['Checkpoint'] == str(i))].filter(like='time').filter(like='list').mean()).T
            list_256.columns = ['Calculate Checkpoint', 'Write Checkpoint']
            list_256.index = [i]
            list_frames['256'].append(list_256)
            tree_256  = pd.DataFrame(master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '256') & (master_frame['Checkpoint'] == str(i))].filter(like='time').filter(like='tree').mean()).T
            tree_256.columns = ['Calculate Checkpoint', 'Write Checkpoint']
            tree_256.index = [i]
            tree_frames['256'].append(tree_256)

            list_512  = pd.DataFrame(master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '512') & (master_frame['Checkpoint'] == str(i))].filter(like='time').filter(like='list').mean()).T
            list_512.columns = ['Calculate Checkpoint', 'Write Checkpoint']
            list_512.index = [i]
            list_frames['512'].append(list_512)
            tree_512  = pd.DataFrame(master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '512') & (master_frame['Checkpoint'] == str(i))].filter(like='time').filter(like='tree').mean()).T
            tree_512.columns = ['Calculate Checkpoint', 'Write Checkpoint']
            tree_512.index = [i]
            tree_frames['512'].append(tree_512)

            list_1024  = pd.DataFrame(master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '1024') & (master_frame['Checkpoint'] == str(i))].filter(like='time').filter(like='list').mean()).T
            list_1024.columns = ['Calculate Checkpoint', 'Write Checkpoint']
            list_1024.index = [i]
            list_frames['1024'].append(list_1024)
            tree_1024  = pd.DataFrame(master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '1024') & (master_frame['Checkpoint'] == str(i))].filter(like='time').filter(like='tree').mean()).T
            tree_1024.columns = ['Calculate Checkpoint', 'Write Checkpoint']
            tree_1024.index = [i]
            tree_frames['1024'].append(tree_1024)

            list_2048  = pd.DataFrame(master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '2048') & (master_frame['Checkpoint'] == str(i))].filter(like='time').filter(like='list').mean()).T
            list_2048.columns = ['Calculate Checkpoint', 'Write Checkpoint']
            list_2048.index = [i]
            list_frames['2048'].append(list_2048)
            tree_2048  = pd.DataFrame(master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '2048') & (master_frame['Checkpoint'] == str(i))].filter(like='time').filter(like='tree').mean()).T
            tree_2048.columns = ['Calculate Checkpoint', 'Write Checkpoint']
            tree_2048.index = [i]
            tree_frames['2048'].append(tree_2048)

            list_4096  = pd.DataFrame(master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '4096') & (master_frame['Checkpoint'] == str(i))].filter(like='time').filter(like='list').mean()).T
            list_4096.columns = ['Calculate Checkpoint', 'Write Checkpoint']
            list_4096.index = [i]
            list_frames['4096'].append(list_4096)
            tree_4096  = pd.DataFrame(master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '4096') & (master_frame['Checkpoint'] == str(i))].filter(like='time').filter(like='tree').mean()).T
            tree_4096.columns = ['Calculate Checkpoint', 'Write Checkpoint']
            tree_4096.index = [i]
            tree_frames['4096'].append(tree_4096)

    listframes128  = pd.concat(list_frames['128'])
    listframes256  = pd.concat(list_frames['256'])
    listframes512  = pd.concat(list_frames['512'])
    listframes1024 = pd.concat(list_frames['1024'])
    listframes2048 = pd.concat(list_frames['2048'])
    listframes4096 = pd.concat(list_frames['4096'])

    treeframes128  = pd.concat(tree_frames['128'])
    treeframes256  = pd.concat(tree_frames['256'])
    treeframes512  = pd.concat(tree_frames['512'])
    treeframes1024 = pd.concat(tree_frames['1024'])
    treeframes2048 = pd.concat(tree_frames['2048'])
    treeframes4096 = pd.concat(tree_frames['4096'])
    print(listframes128)
    print(listframes128.sum())
    framelist = []
    framelist.append(pd.concat(list_frames['128']))
    framelist.append(pd.concat(tree_frames['128']))
    framelist.append(pd.concat(list_frames['256']))
    framelist.append(pd.concat(tree_frames['256']))
    framelist.append(pd.concat(list_frames['512']))
    framelist.append(pd.concat(tree_frames['512']))
    framelist.append(pd.concat(list_frames['1024']))
    framelist.append(pd.concat(tree_frames['1024']))
    framelist.append(pd.concat(list_frames['2048']))
    framelist.append(pd.concat(tree_frames['2048']))
    framelist.append(pd.concat(list_frames['4096']))
    framelist.append(pd.concat(tree_frames['4096']))
    fig = plt.figure()
    listframe = pd.concat([listframes128.sum().T, listframes256.sum().T, listframes512.sum().T, listframes1024.sum().T, listframes2048.sum().T, listframes4096.sum().T], axis=1)
    listframe.columns = ['128', '256', '512', '1024', '2048', '4096']
    treeframe = pd.concat([treeframes128.sum().T, treeframes256.sum().T, treeframes512.sum().T, treeframes1024.sum().T, treeframes2048.sum().T, treeframes4096.sum().T], axis=1)
    treeframe.columns = ['128', '256', '512', '1024', '2048', '4096']
    print(pd.concat([listframes128.sum().T, listframes256.sum().T, listframes512.sum().T, listframes1024.sum().T, listframes2048.sum().T, listframes4096.sum().T], axis=1))
    print(pd.concat([treeframes128.sum().T, treeframes256.sum().T, treeframes512.sum().T, treeframes1024.sum().T, treeframes2048.sum().T, treeframes4096.sum().T], axis=1))
    print(listframe.T)
    print(treeframe.T)
    plot_clustered_stacked_frames([listframe.T, treeframe.T], ['List', 'Tree'])
    plt.title(args.title + ": Checkpoint Time (10 Checkpoints)", fontsize=14);
    plt.xlabel("Chunk Size (Bytes)", fontsize=14)
    plt.ylabel("Total Checkpoint Time (s)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
elif args.num_ranks != 1:
    logs = args.logfiles
    frames = []
    for log in logs:
        implementation = 'GPU'
        if 'Serial' in log:
            implementation = 'Serial'
        elif 'CPU' in log:
            implementation = 'CPU'
        elif 'GPU' in log:
            implementation = 'GPU'
    
        file_name = log.split('.')
    
        chunksize_idx = file_name.index('chunk_size')
        chunk_size = file_name[chunksize_idx+1]
    
        checkpoint_idx = file_name.index('dat')
        checkpoint = file_name[checkpoint_idx-1]
    
        df = pd.read_csv(log, header=None)
        df.columns = ['Full checkpoint diff time', 'Full checkpoint write time', 'Full checkpoint data', 'Full checkpoint metadata', 'Hash list diff time', 'Hash list write time', 'Hash list data', 'Hash list metadata', 'Merkle tree diff time', 'Merkle tree write time', 'Merkle tree data', 'Merkle tree metadata']
        if 'Rank' in log:
            rank_pos = log.find("Rank")
            nrank_pos = log.find("proc")
            rank = log[rank_pos+4]
            nranks = log[nrank_pos-1]
        else:
            rank = '0'
            nranks = '1'

        
        df['NumRanks'] = pd.Series([nranks for x in range(len(df.index))])
        df['Rank'] = pd.Series([rank for x in range(len(df.index))])
        df['Chunk size'] = pd.Series([chunk_size for i in range(len(df.index))])
        df['Checkpoint'] = pd.Series([checkpoint[-1] for i in range(len(df.index))])
        df['Implementation'] = pd.Series([implementation for i in range(len(df.index))])
        frames.append(df)
    master_frame = pd.concat(frames, axis=0)

    ranks1  = master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '128') & (master_frame['NumRanks'] == str(1))]
    ranks2  = master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '128') & (master_frame['NumRanks'] == str(2))]
    ranks4  = master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '128') & (master_frame['NumRanks'] == str(4))]
    ranks8  = master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '128') & (master_frame['NumRanks'] == str(8))]
    ranks1['Full checkpoint total time']   = ranks1['Full checkpoint diff time']   + ranks1['Full checkpoint write time']
    ranks1['Hash list total time']   = ranks1['Hash list diff time']   + ranks1['Hash list write time']
    ranks1['Merkle tree total time'] = ranks1['Merkle tree diff time'] + ranks1['Merkle tree write time']
    ranks2['Full checkpoint total time']   = ranks2['Full checkpoint diff time']   + ranks2['Full checkpoint write time']
    ranks2['Hash list total time']   = ranks2['Hash list diff time']   + ranks2['Hash list write time']
    ranks2['Merkle tree total time'] = ranks2['Merkle tree diff time'] + ranks2['Merkle tree write time']
    ranks4['Full checkpoint total time']   = ranks4['Full checkpoint diff time']   + ranks4['Full checkpoint write time']
    ranks4['Hash list total time']   = ranks4['Hash list diff time']   + ranks4['Hash list write time']
    ranks4['Merkle tree total time'] = ranks4['Merkle tree diff time'] + ranks4['Merkle tree write time']
    ranks8['Full checkpoint total time']   = ranks8['Full checkpoint diff time']   + ranks8['Full checkpoint write time']
    ranks8['Hash list total time']   = ranks8['Hash list diff time']   + ranks8['Hash list write time']
    ranks8['Merkle tree total time'] = ranks8['Merkle tree diff time'] + ranks8['Merkle tree write time']

    testframe = ranks1[['Checkpoint', 'Rank', 'Full checkpoint total time']]
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby(['index', 'Checkpoint']).max()
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby('index').sum()['Full checkpoint total time']
    rank1fullframe = testframe

    testframe = ranks2[['Checkpoint', 'Rank', 'Full checkpoint total time']]
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby(['index', 'Checkpoint']).max()
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby('index').sum()['Full checkpoint total time']
    rank2fullframe = testframe

    testframe = ranks4[['Checkpoint', 'Rank', 'Full checkpoint total time']]
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby(['index', 'Checkpoint']).max()
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby('index').sum()['Full checkpoint total time']
    rank4fullframe = testframe

    testframe = ranks8[['Checkpoint', 'Rank', 'Full checkpoint total time']]
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby(['index', 'Checkpoint']).max()
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby('index').sum()['Full checkpoint total time']
    rank8fullframe = testframe

    full_list = [rank1fullframe.rename("1"), rank2fullframe.rename("2"), rank4fullframe.rename("4"), rank8fullframe.rename("8")]
    fullframe = pd.concat(full_list, axis=1)

#    rank1listframe = ranks1[['Checkpoint', 'Rank', 'Hash list total time']]
#    rank1listframe.reset_index(inplace=True)
#    rank1listframe = rank1listframe.groupby(['Rank', 'index']).sum()
#    rank1listframe.reset_index(inplace=True)
#    rank1listframe = rank1listframe.groupby('index').max()['Hash list total time']

#    rank2listframe = ranks2[['Checkpoint', 'Rank', 'Hash list total time']]
#    rank2listframe.reset_index(inplace=True)
#    rank2listframe = rank2listframe.groupby(['Rank', 'index']).sum()
#    rank2listframe.reset_index(inplace=True)
#    rank2listframe = rank2listframe.groupby('index').max()['Hash list total time']

#    rank4listframe = ranks4[['Checkpoint', 'Rank', 'Hash list total time']]
#    rank4listframe.reset_index(inplace=True)
#    rank4listframe = rank4listframe.groupby(['Rank', 'index']).sum()
#    rank4listframe.reset_index(inplace=True)
#    rank4listframe = rank4listframe.groupby('index').max()['Hash list total time']

#    rank8listframe = ranks8[['Checkpoint', 'Rank', 'Hash list total time']]
#    rank8listframe.reset_index(inplace=True)
#    rank8listframe = rank8listframe.groupby(['Rank', 'index']).sum()
#    rank8listframe.reset_index(inplace=True)
#    rank8listframe = rank8listframe.groupby('index').max()['Hash list total time']

    testframe = ranks1[['Checkpoint', 'Rank', 'Hash list total time']]
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby(['index', 'Checkpoint']).max()
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby('index').sum()['Hash list total time']
    rank1listframe = testframe

    testframe = ranks2[['Checkpoint', 'Rank', 'Hash list total time']]
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby(['index', 'Checkpoint']).max()
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby('index').sum()['Hash list total time']
    rank2listframe = testframe

    testframe = ranks4[['Checkpoint', 'Rank', 'Hash list total time']]
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby(['index', 'Checkpoint']).max()
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby('index').sum()['Hash list total time']
    rank4listframe = testframe

    testframe = ranks8[['Checkpoint', 'Rank', 'Hash list total time']]
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby(['index', 'Checkpoint']).max()
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby('index').sum()['Hash list total time']
    rank8listframe = testframe

    list_list = [rank1listframe.rename("1"), rank2listframe.rename("2"), rank4listframe.rename("4"), rank8listframe.rename("8")]
    listframe = pd.concat(list_list, axis=1)

#    rank1treeframe = ranks1[['Checkpoint', 'Rank', 'Merkle tree total time']]
#    rank1treeframe.reset_index(inplace=True)
#    rank1treeframe = rank1treeframe.groupby(['Rank', 'index']).sum()
#    rank1treeframe.reset_index(inplace=True)
#    rank1treeframe = rank1treeframe.groupby('index').max()['Merkle tree total time']

#    rank2treeframe = ranks2[['Checkpoint', 'Rank', 'Merkle tree total time']]
#    rank2treeframe.reset_index(inplace=True)
#    rank2treeframe = rank2treeframe.groupby(['Rank', 'index']).sum()
#    rank2treeframe.reset_index(inplace=True)
#    rank2treeframe = rank2treeframe.groupby('index').max()['Merkle tree total time']
#
#    rank4treeframe = ranks4[['Checkpoint', 'Rank', 'Merkle tree total time']]
#    rank4treeframe.reset_index(inplace=True)
#    rank4treeframe = rank4treeframe.groupby(['Rank', 'index']).sum()
#    rank4treeframe.reset_index(inplace=True)
#    rank4treeframe = rank4treeframe.groupby('index').max()['Merkle tree total time']
#
#    rank8treeframe = ranks8[['Checkpoint', 'Rank', 'Merkle tree total time']]
#    rank8treeframe.reset_index(inplace=True)
#    rank8treeframe = rank8treeframe.groupby(['Rank', 'index']).sum()
#    rank8treeframe.reset_index(inplace=True)
#    rank8treeframe = rank8treeframe.groupby('index').max()['Merkle tree total time']

    testframe = ranks1[['Checkpoint', 'Rank', 'Merkle tree total time']]
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby(['index', 'Checkpoint']).max()
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby('index').sum()['Merkle tree total time']
    rank1treeframe = testframe

    testframe = ranks2[['Checkpoint', 'Rank', 'Merkle tree total time']]
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby(['index', 'Checkpoint']).max()
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby('index').sum()['Merkle tree total time']
    rank2treeframe = testframe

    testframe = ranks4[['Checkpoint', 'Rank', 'Merkle tree total time']]
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby(['index', 'Checkpoint']).max()
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby('index').sum()['Merkle tree total time']
    rank4treeframe = testframe

    testframe = ranks8[['Checkpoint', 'Rank', 'Merkle tree total time']]
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby(['index', 'Checkpoint']).max()
    testframe.reset_index(inplace=True)
    testframe = testframe.groupby('index').sum()['Merkle tree total time']
    rank8treeframe = testframe

    print(rank8treeframe)
    print(type(rank8treeframe))
    tree_list = [rank1treeframe.rename("1"), rank2treeframe.rename("2"), rank4treeframe.rename("4"), rank8treeframe.rename("8")]
    treeframe = pd.concat(tree_list, axis=1)

    print(fullframe)
    print(listframe)
    print(treeframe)

    avgfull = fullframe.mean()
    avglist = listframe.mean()
    avgtree = treeframe.mean()
    avgfull.rename("Full")
    avglist.rename("List")
    avgtree.rename("Tree")
    print(avgfull)
    print(avglist)
    print(avgtree)
    df = pd.concat([avgfull, avglist, avgtree], axis=1)
    df.columns = ['Full', 'List', 'Tree']
#    print(df)
#    df.plot.bar()
#    plt.xlabel("Number of Processes", fontsize=14)
#    plt.ylabel("Total Time Checkpointing (s)", fontsize=14)
#    plt.xticks(fontsize=14, rotation=0)
#    plt.yticks(fontsize=14)
#    plt.title(args.title + ": MPI Scaling", fontsize=14)

    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1,4])
    axes = ['', '']
    axes[0] = fig.add_subplot(spec[0])
    axes[1] = fig.add_subplot(spec[1])
    hatches = ['xx', '//', '\\\\', '++', '--', '**', 'oo']
    df.plot.bar(ax=axes[0])
    df.plot.bar(ax=axes[1])

    for axe in axes:
        h,l = axe.get_legend_handles_labels() # get the handles we want to modify
        for i in range(0, 2, 3): # len(h) = n_col * n_df
            for j, pa in enumerate(h[i:i+3]):
                for rect in pa.patches: # for each index
                    rect.set_width(1 / float(3 + 4.5))

    for bars, hatch in zip(axes[0].containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    for bars, hatch in zip(axes[1].containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    list_max = df['List'].max()
    tree_max = df['Tree'].max()
    cutoff = max(list_max, tree_max) * 1.1
    low_cutoff   = df['Full'].min() * 0.9
    upper_cutoff = df['Full'].max() * 1.1
#    low_cutoff = 21
    axes[0].set_ylim(low_cutoff, upper_cutoff)
    axes[1].set_ylim(0, cutoff)
    axes[0].spines['bottom'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[0].xaxis.tick_top()
    axes[0].tick_params(labeltop=False)
    axes[1].xaxis.tick_bottom()

    d = .01
    kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False)
    axes[0].plot((-d, +d), (-d, +d), **kwargs)
    axes[0].plot((1-d, 1+d), (-d, +d), **kwargs)
    kwargs.update(transform=axes[1].transAxes)
    axes[1].plot((-d, +d), (1-d, 1+d), **kwargs)
    axes[1].plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')

    fig.suptitle(args.title + ": MPI Scaling", fontsize=14)
    fig.supxlabel("Number of Processes", fontsize=14)
    fig.supylabel("Total Time Checkpointing (s)", fontsize=14)
    for a in axes:
        for label in (a.get_xticklabels() + a.get_yticklabels()):
            label.set_fontsize(14)
    plt.legend(fontsize=14)
    axes[0].get_legend().remove()
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14)

#    fig,axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
#    listframe.boxplot(ax=axes[0])
#    treeframe.boxplot(ax=axes[1])
#    axes[0].set_title("List", fontsize=14)
#    axes[1].set_title("Tree", fontsize=14)
#    for a in axes:
#        for label in (a.get_xticklabels() + a.get_yticklabels()):
#            label.set_fontsize(14)
#    fig.supxlabel("Number of Processes", fontsize=14)
#    fig.supylabel("Total Time Checkpointing (s)", fontsize=14)
#    fig.suptitle(args.title + ": MPI Scaling", fontsize=14)

#    testframe.groupby('index').sum().boxplot(column='Hash list total time', by='index')
#    print(ranks8[['Checkpoint', 'Hash list total time']].set_index('Checkpoint', inplace=True))
#    ranks8[['Checkpoint', 'Hash list total time']].boxplot(column='Hash list total time', by='Checkpoint')
    ranksumlist1 = (ranks1['Hash list diff time']   + ranks1['Hash list write time']  )
    ranksumtree1 = (ranks1['Merkle tree diff time'] + ranks1['Merkle tree write time'])
    ranksumlist2 = (ranks2['Hash list diff time']   + ranks2['Hash list write time']  )
    ranksumtree2 = (ranks2['Merkle tree diff time'] + ranks2['Merkle tree write time'])
    ranksumlist4 = (ranks4['Hash list diff time']   + ranks4['Hash list write time']  )
    ranksumtree4 = (ranks4['Merkle tree diff time'] + ranks4['Merkle tree write time'])
    ranksumlist8 = (ranks8['Hash list diff time']   + ranks8['Hash list write time']  )
    ranksumtree8 = (ranks8['Merkle tree diff time'] + ranks8['Merkle tree write time'])
#    print(ranksumlist1.groupby()
#    print(ranksumtree1.groupby()
#    print(ranksumlist2.groupby()
#    print(ranksumtree2.groupby()
#    print(ranksumlist4.groupby()
#    print(ranksumtree4.groupby()
#    print(ranksumlist1.groupby()
#    print(ranksumtree1.groupby()
#    frame_dict = {'1': [ranksumlist1, ranksumtree1], '2': [ranksumlist2, ranksumtree2], '4': [ranksumlist4, ranksumtree4], '8': [ranksumlist8, ranksumtree8]}
#    df = pd.DataFrame.from_dict(frame_dict, columns=['List', 'Tree'], orient='index')
#    print(df)
#    df.plot.bar()
#    plt.legend()
#    plt.title(args.title + ": Total Checkpoint Size", fontsize=14)
#    plt.ylabel("Total Checkpoint Size (GB)", fontsize=14)
#    plt.xlabel("Number of Processes", fontsize=14)
#    plt.xticks(rotation=0, fontsize=14)
#    plt.yticks(fontsize=14)
#    plt.legend(fontsize=14)
elif args.ipdps:
    print("IPDPS")
    frames = []
    for log in logs:
        implementation = ''
        mode = 'baseline'
        if 'Serial' in log:
            implementation = 'Serial'
        elif 'CPU' in log:
            implementation = 'CPU'
        elif 'GPU' in log:
            implementation = 'GPU'
        if 'local' in log:
            mode = 'baseline'
        elif 'global' in log:
            mode = 'history'
    
        file_name = log.split('.')
    
        chunksize_idx = file_name.index('chunk_size')
        chunk_size = file_name[chunksize_idx+1]
    
        checkpoint_idx = file_name.index('dat')
        checkpoint = file_name[checkpoint_idx-1]
    
#        if int(checkpoint[-1]) % 2 == 1:
        if True:
            df = pd.read_csv(log)
            df['Full total time'] = df['Copy full chkpt to GPU'] + df['Restart full chkpt']
            df['List total time'] = df['Copy list chkpt to GPU'] + df['Restart list chkpt']
            df['Tree total time'] = df['Copy tree chkpt to GPU'] + df['Restart tree chkpt']
            df['Chunk size'] = pd.Series([chunk_size for i in range(len(df.index))])
            df['Checkpoint'] = pd.Series([checkpoint[-1] for i in range(len(df.index))])
            df['Implementation'] = pd.Series([implementation for i in range(len(df.index))])
            df['Mode'] = pd.Series([mode for i in range(len(df.index))])
            frames.append(df)
    master_frame = pd.concat(frames, axis=0)
    
    full_diff_df = copy.deepcopy(master_frame[['Full total time', 'Copy full chkpt to GPU', 'Restart full chkpt', 'Chunk size', 'Checkpoint', 'Implementation']])
    full_diff_df['Strategy'] = pd.Series(['Full' for x in range(len(full_diff_df.index))])
    full_diff_df.columns = ['Total time', 'Copy time', 'Restart time', 'Chunk size', 'Checkpoint', 'Implementation', 'Strategy']
#    print(full_diff_df)
    
    naive_list_diff_df = copy.deepcopy(master_frame[['List total time', 'Copy list chkpt to GPU', 'Restart list chkpt', 'Chunk size', 'Checkpoint', 'Implementation']])
    naive_list_diff_df['Strategy'] = pd.Series(['Naive List' for x in range(len(naive_list_diff_df.index))])
    naive_list_diff_df.columns = ['Total time', 'Copy time', 'Restart time', 'Chunk size', 'Checkpoint', 'Implementation', 'Strategy']
#    print(naive_list_diff_df)
    
    list_diff_df = copy.deepcopy(master_frame[['List total time', 'Copy list chkpt to GPU', 'Restart list chkpt', 'Chunk size', 'Checkpoint', 'Implementation']])
    list_diff_df['Strategy'] = pd.Series(['List' for x in range(len(list_diff_df.index))])
    list_diff_df.columns = ['Total time', 'Copy time', 'Restart time', 'Chunk size', 'Checkpoint', 'Implementation', 'Strategy']
#    print(list_diff_df)
    
    tree_diff_df = copy.deepcopy(master_frame[['Tree total time', 'Copy tree chkpt to GPU', 'Restart tree chkpt', 'Chunk size', 'Checkpoint', 'Implementation']])
    tree_diff_df['Strategy'] = pd.Series(['Tree' for x in range(len(tree_diff_df.index))])
    tree_diff_df.columns = ['Total time', 'Copy time', 'Restart time', 'Chunk size', 'Checkpoint', 'Implementation', 'Strategy']
#    print(tree_diff_df)
    
    
#    print(tree_diff_df.columns)
    fullframe = full_diff_df.groupby(['Chunk size', 'Strategy', 'Checkpoint']).mean()
    fullframe.reset_index(inplace=True)
    naive_listframe = naive_list_diff_df.groupby(['Chunk size', 'Strategy', 'Checkpoint']).mean()
    naive_listframe.reset_index(inplace=True)
    listframe = list_diff_df.groupby(['Chunk size', 'Strategy', 'Checkpoint']).mean()
    listframe.reset_index(inplace=True)
    treeframe = tree_diff_df.groupby(['Chunk size', 'Strategy', 'Checkpoint']).mean()
    treeframe.reset_index(inplace=True)
    list_max = listframe['Total time'].max()
    tree_max = treeframe['Total time'].max()
    cutoff = max(list_max, tree_max) * 1.1
    low_cutoff = fullframe['Total time'].min() * 0.9
#    upper_cutoff = fullframe['Total time'].max() * 1.1
    upper_cutoff = 0.4
    if cutoff < low_cutoff:
        fig = plt.figure()
        spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1,4])
        axes = ['', '']
        axes[0] = fig.add_subplot(spec[0])
        axes[1] = fig.add_subplot(spec[1])
        print(pd.concat([fullframe, naive_listframe, listframe, treeframe], axis=0))
        hatches = ['xx', '//', '\\\\', '++', '--', '**', 'oo']
#        sns.barplot(x='Checkpoint', y='Total time', hue='Strategy', data=pd.concat([fullframe, listframe, treeframe], axis=0), ci=None, ax=axes[0])
#        sns.barplot(x='Checkpoint', y='Total time', hue='Strategy', data=pd.concat([fullframe, listframe, treeframe], axis=0), ci=None, ax=axes[1])
        sns.barplot(x='Checkpoint', y='Total time', hue='Strategy', data=pd.concat([fullframe, naive_listframe, listframe, treeframe], axis=0), ci=None, ax=axes[0])
        sns.barplot(x='Checkpoint', y='Total time', hue='Strategy', data=pd.concat([fullframe, naive_listframe, listframe, treeframe], axis=0), ci=None, ax=axes[1])
        print("Plot bars")

        for axe in axes:
            h,l = axe.get_legend_handles_labels() # get the handles we want to modify
            for i in range(0, 2, 3): # len(h) = n_col * n_df
                for j, pa in enumerate(h[i:i+3]):
                    for rect in pa.patches: # for each index
                        rect.set_width(1 / float(4 + 1.5))

#        for bars, hatch in zip(axes[0].containers, hatches):
#            for bar in bars:
#                bar.set_hatch(hatch)
#        for bars, hatch in zip(axes[1].containers, hatches):
#            for bar in bars:
#                bar.set_hatch(hatch)
        axes[0].set_ylim(low_cutoff, upper_cutoff)
        axes[1].set_ylim(0, cutoff)
        axes[0].spines['bottom'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        axes[0].xaxis.tick_top()
        axes[0].tick_params(labeltop=False)
        axes[1].xaxis.tick_bottom()

        d = .01
        kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False)
        axes[0].plot((-d, +d), (-d, +d), **kwargs)
        axes[0].plot((1-d, 1+d), (-d, +d), **kwargs)
        kwargs.update(transform=axes[1].transAxes)
        axes[1].plot((-d, +d), (1-d, 1+d), **kwargs)
        axes[1].plot((1-d, 1+d), (1-d, 1+d), **kwargs)
        axes[0].set_ylabel('')
        axes[1].set_ylabel('')
        axes[0].set_xlabel('')
        axes[1].set_xlabel('')

        fig.suptitle(args.title + ": Checkpoint Time (128 Byte Chunks)", fontsize=14)
        fig.supxlabel("Checkpoint number", fontsize=14)
        fig.supylabel("Time (s)", fontsize=14)
        for a in axes:
            for label in (a.get_xticklabels() + a.get_yticklabels()):
                label.set_fontsize(14)
        plt.legend(fontsize=12)
        axes[0].get_legend().remove()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    else:
        fig = plt.figure()
        sns.barplot(x='Checkpoint', y='Total time', hue='Strategy', data=pd.concat([fullframe, naive_listframe, listframe, treeframe], axis=0), ci=None)
        ax = fig.get_axes()[0]
        h,l = ax.get_legend_handles_labels() # get the handles we want to modify
        for i in range(0, 2, 3): # len(h) = n_col * n_df
            for j, pa in enumerate(h[i:i+3]):
                for rect in pa.patches: # for each index
                    rect.set_width(1 / float(4 + 1.5))
        plt.title(args.title + ": Checkpoint Time (128 Byte Chunks)", fontsize=14)
        plt.xlabel("Checkpoint number", fontsize=14)
        plt.ylabel("Time (s)", fontsize=14)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
#        hatches = ['xx', '//', '\\\\', '++', '--', '**', 'oo']
#        for bars, hatch in zip(ax.containers, hatches):
#            for bar in bars:
#                bar.set_hatch(hatch)
else:
    frames = []
    for log in logs:
        implementation = ''
        if 'Serial' in log:
            implementation = 'Serial'
        elif 'CPU' in log:
            implementation = 'CPU'
        elif 'GPU' in log:
            implementation = 'GPU'
    
        file_name = log.split('.')
    
        chunksize_idx = file_name.index('chunk_size')
        chunk_size = file_name[chunksize_idx+1]
    
        checkpoint_idx = file_name.index('dat')
        checkpoint = file_name[checkpoint_idx-1]
    
#        if int(checkpoint[-1]) % 2 == 1:
        if True:
            df = pd.read_csv(log)
            df['Full total time'] = df['Copy full chkpt to GPU'] + df['Restart full chkpt']
            df['List total time'] = df['Copy list chkpt to GPU'] + df['Restart list chkpt']
            df['Tree total time'] = df['Copy tree chkpt to GPU'] + df['Restart tree chkpt']
            df['Chunk size'] = pd.Series([chunk_size for i in range(len(df.index))])
            df['Checkpoint'] = pd.Series([checkpoint[-1] for i in range(len(df.index))])
            df['Implementation'] = pd.Series([implementation for i in range(len(df.index))])
            frames.append(df)
    master_frame = pd.concat(frames, axis=0)
    
    full_diff_df = copy.deepcopy(master_frame[['Full total time', 'Copy full chkpt to GPU', 'Restart full chkpt', 'Chunk size', 'Checkpoint', 'Implementation']])
    full_diff_df['Strategy'] = pd.Series(['Full' for x in range(len(full_diff_df.index))])
    full_diff_df.columns = ['Total time', 'Copy time', 'Restart time', 'Chunk size', 'Checkpoint', 'Implementation', 'Strategy']
#    print(full_diff_df)
    
    list_diff_df = copy.deepcopy(master_frame[['List total time', 'Copy list chkpt to GPU', 'Restart list chkpt', 'Chunk size', 'Checkpoint', 'Implementation']])
    list_diff_df['Strategy'] = pd.Series(['List' for x in range(len(list_diff_df.index))])
    list_diff_df.columns = ['Total time', 'Copy time', 'Restart time', 'Chunk size', 'Checkpoint', 'Implementation', 'Strategy']
#    print(list_diff_df)
    
    tree_diff_df = copy.deepcopy(master_frame[['Tree total time', 'Copy tree chkpt to GPU', 'Restart tree chkpt', 'Chunk size', 'Checkpoint', 'Implementation']])
    tree_diff_df['Strategy'] = pd.Series(['Tree' for x in range(len(tree_diff_df.index))])
    tree_diff_df.columns = ['Total time', 'Copy time', 'Restart time', 'Chunk size', 'Checkpoint', 'Implementation', 'Strategy']
#    print(tree_diff_df)
    
    
#    print(tree_diff_df.columns)
    fullframe = full_diff_df.groupby(['Chunk size', 'Strategy', 'Checkpoint']).mean()
    fullframe.reset_index(inplace=True)
    listframe = list_diff_df.groupby(['Chunk size', 'Strategy', 'Checkpoint']).mean()
    listframe.reset_index(inplace=True)
    treeframe = tree_diff_df.groupby(['Chunk size', 'Strategy', 'Checkpoint']).mean()
    treeframe.reset_index(inplace=True)
    list_max = listframe['Total time'].max()
    tree_max = treeframe['Total time'].max()
    print(pd.concat([fullframe, listframe, treeframe], axis=0))
    cutoff = max(list_max, tree_max) * 1.1
    low_cutoff = fullframe['Total time'].min() * 0.9
#    upper_cutoff = fullframe['Total time'].max() * 1.1
    upper_cutoff = 0.4
    if cutoff < low_cutoff:
        fig = plt.figure()
        spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1,4])
        axes = ['', '']
        axes[0] = fig.add_subplot(spec[0])
        axes[1] = fig.add_subplot(spec[1])
        print(pd.concat([fullframe, listframe, treeframe], axis=0))
        hatches = ['xx', '//', '\\\\', '++', '--', '**', 'oo']
        sns.barplot(x='Checkpoint', y='Total time', hue='Strategy', data=pd.concat([fullframe, listframe, treeframe], axis=0), ci=None, ax=axes[0])
        sns.barplot(x='Checkpoint', y='Total time', hue='Strategy', data=pd.concat([fullframe, listframe, treeframe], axis=0), ci=None, ax=axes[1])

        for axe in axes:
            h,l = axe.get_legend_handles_labels() # get the handles we want to modify
            for i in range(0, 2, 3): # len(h) = n_col * n_df
                for j, pa in enumerate(h[i:i+3]):
                    for rect in pa.patches: # for each index
                        rect.set_width(1 / float(3 + 1.5))

        for bars, hatch in zip(axes[0].containers, hatches):
            for bar in bars:
                bar.set_hatch(hatch)
        for bars, hatch in zip(axes[1].containers, hatches):
            for bar in bars:
                bar.set_hatch(hatch)
        axes[0].set_ylim(low_cutoff, upper_cutoff)
        axes[1].set_ylim(0, cutoff)
        axes[0].spines['bottom'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        axes[0].xaxis.tick_top()
        axes[0].tick_params(labeltop=False)
        axes[1].xaxis.tick_bottom()

        d = .01
        kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False)
        axes[0].plot((-d, +d), (-d, +d), **kwargs)
        axes[0].plot((1-d, 1+d), (-d, +d), **kwargs)
        kwargs.update(transform=axes[1].transAxes)
        axes[1].plot((-d, +d), (1-d, 1+d), **kwargs)
        axes[1].plot((1-d, 1+d), (1-d, 1+d), **kwargs)
        axes[0].set_ylabel('')
        axes[1].set_ylabel('')
        axes[0].set_xlabel('')
        axes[1].set_xlabel('')

        fig.suptitle(args.title + ": Checkpoint Time (128 Byte Chunks)", fontsize=14)
        fig.supxlabel("Checkpoint number", fontsize=14)
        fig.supylabel("Time (s)", fontsize=14)
        for a in axes:
            for label in (a.get_xticklabels() + a.get_yticklabels()):
                label.set_fontsize(14)
        plt.legend(fontsize=14)
        axes[0].get_legend().remove()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    else:
        fig = plt.figure()
        sns.barplot(x='Checkpoint', y='Total time', hue='Strategy', data=pd.concat([fullframe, listframe, treeframe], axis=0), ci=None)
        ax = fig.get_axes()[0]
        plt.title(args.title + ": Checkpoint Time (128 Byte Chunks)", fontsize=14)
        plt.xlabel("Checkpoint number", fontsize=14)
        plt.ylabel("Time (s)", fontsize=14)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        hatches = ['xx', '//', '\\\\', '++', '--', '**', 'oo']
        for bars, hatch in zip(ax.containers, hatches):
            for bar in bars:
                bar.set_hatch(hatch)

plt.tight_layout()
plt.show()

