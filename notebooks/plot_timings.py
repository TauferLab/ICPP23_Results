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
import os

def plot_clustered_stacked_frames(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)
    alg_labels = labels

#    color_cycle = axe._get_lines.prop_cycler
    colorlist = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cmap = matplotlib.cm.get_cmap('tab20')
#    colors = plt.cm.Spectral(np.linspace(0,1,30))
    colors = cmap(np.linspace(0,1, 20))
    axe.set_prop_cycle('color', colors)

    offset = 0
    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
#                      color=colorlist[offset:],
#                      color=[colorlist[3]] + colorlist[offset:],
                      color=colors,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots
        offset += len(df.columns)-1
#        for i in range(len(df.columns)):
#            next(color_cycle)

    hatches = ['xx', '//', '\\\\', '++', '--', '**', 'oo']
    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
#                rect.set_x(rect.get_x() + (1 / float(n_df + 1)) * i / float(n_col) + 0.1)
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col) + 0.025)
                rect.set_hatch(H * 2*int(i / n_col)) #edited part     
#                if j > 0:
#                    rect.set_hatch(hatches[int((i+j)/2)+1]) #edited part     
#                rect.set_hatch(hatches[int((i+j)/2)+1]) #edited part     
                rect.set_width(1 / float(n_df + 1.5))
    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    print(df.index)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)
    axe.set_prop_cycle('color', colors)

    labellist = []
#    labellist.append(dfall[0].columns[0])
    metadata_labels = []
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
        
#    plt.legend(filtered_handles, filtered_labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
#    plt.legend(filtered_handles, filtered_labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.legend(handles, labels)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * 2*i))

    print(labels)
    labels = alg_labels
    print(alg_labels)
#    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.8])
#    l1 = axe.legend(h[:n_col], l[:n_col], loc=[0.01, -0.21])
#    l1 = axe.legend(h[:n_col], l[:n_col], loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=3)
#    l1 = axe.legend(h[:n_col], l[:n_col], loc='center left', bbox_to_anchor=(1.0, 0.5))
    l1 = axe.legend(h[:n_col], l[:n_col], loc='upper center')
    if labels is not None:
#        l2 = plt.legend(n, labels, loc=[1.01, 0.0]) 
#        l2 = plt.legend(n, labels, loc=[0.81, -0.21]) 
        l2 = plt.legend(n, labels) 
    axe.add_artist(l1)
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
parser.add_argument("--restart", action='store_true', help='Use restart timing logs')
parser.add_argument("--plot-restart", action='store_true', help='Plot restart timers')
parser.add_argument("--mpi", action='store_true', help='Plot scaling tests')

args = parser.parse_args()
print(args.hashfunc)

logs = args.logfiles

if args.mpi:
    frames = {1:[], 2:[], 4:[], 8:[], 16:[], 32:[], 64:[], 128:[]}
    data_frames = {}
    directories = args.logfiles
    if args.restart:
        checkpoint = 2
        for directory in directories:
            name = directory.split('/')
            scale = ''.join(filter(str.isdigit, name[1]))
            for root,dirs,files in os.walk(directory):
                frame_list = []
                for file in files:
                    if file.endswith(".restart_timing.csv"):
                        df = pd.read_csv(directory+'/'+file, header=None)
                        df.columns = ['Config', 'Chkpt ID', 'Chunk Size', 'Copy Diff', 'Restart Time']
                        df = df.sort_values(['Config']).reset_index(drop=True)
                        df['Total'] = df.loc[:, 'Copy Diff':'Restart Time'].sum(1)
                        df['Scale'] = scale
                        df['Run'] = df.index
                        filename = file.split('.')
                        rank = 0;
                        for i in range(len(filename)):
                            if 'Rank' in filename[i]:
                                rank = ''.join(filter(str.isdigit, filename[i]))
                                df['Rank'] = rank
                                break
                        frames[int(scale)].append(df)
            data_frames[int(scale)] = pd.concat(frames[int(scale)])                
        alg_config = []
        scale_config = []
        total_sizes = []
        ranks = []
        runs = []
        # Grouped
        clean_frames = []
        for scale in [1, 2, 4, 8, 16, 32 ,64]:
            df = data_frames[int(scale)]
            df = df[df['Chkpt ID'] == checkpoint]
            df['Scale'] = scale
            df = df.groupby(['Config', 'Run']).max(numeric_only=True)[['Scale', 'Total']]
            clean_frames.append(df)
        full_frame = pd.concat(clean_frames)
        print(full_frame.reset_index())
        g = sns.barplot(data=full_frame.reset_index(), hue='Scale', y='Total', x='Config')
        plt.xlabel("Checkpoint Algorithm")
        plt.ylabel("Total Time (s)")
        plt.title(args.title + ": Restart checkpoint " + str(checkpoint) + " (Show rank with most overhead)")
    else:
        for directory in directories:
            name = directory.split('/')
            scale = ''.join(filter(str.isdigit, name[1]))
            for root,dirs,files in os.walk(directory):
                frame_list = []
                for file in files:
                    if file.endswith(".timing.csv"):
                        df = pd.read_csv(directory+'/'+file, header=None)
                        df.columns = ['Config', 'Chkpt ID', 'Chunk Size', 'Calc Diff', 'Gather Diff', 'Copy Diff']
                        df = df.sort_values(['Config']).reset_index(drop=True)
                        df['Total'] = df.loc[:, 'Calc Diff':'Copy Diff'].sum(1)
                        df['Scale'] = scale
                        df['Run'] = df.index
                        filename = file.split('.')
                        rank = 0;
                        for i in range(len(filename)):
                            if 'Rank' in filename[i]:
                                rank = ''.join(filter(str.isdigit, filename[i]))
                                df['Rank'] = rank
                                break
                        frames[int(scale)].append(df)
            data_frames[int(scale)] = pd.concat(frames[int(scale)])                
        alg_config = []
        scale_config = []
        total_sizes = []
        ranks = []
        runs = []
        # Grouped
        print(data_frames[int(2)].groupby(['Config', 'Run', 'Rank']).sum(numeric_only=True))
        print(data_frames[int(2)].groupby(['Config', 'Run', 'Rank']).sum(numeric_only=True).groupby(['Config', 'Run']).max(numeric_only=True)['Total'])
        clean_frames = []
        for scale in [1, 2, 4, 8, 16, 32 ,64]:
            df = data_frames[int(scale)].groupby(['Config', 'Run', 'Rank']).sum(numeric_only=True)
            df['Scale'] = scale
            df = df.groupby(['Config', 'Run']).max(numeric_only=True)[['Scale', 'Total']]
            clean_frames.append(df)
        full_frame = pd.concat(clean_frames)
        print(full_frame)
        print(full_frame.reset_index())
        g = sns.barplot(data=full_frame.reset_index(), hue='Scale', y='Total', x='Config')
        plt.xlabel("Checkpoint Algorithm")
        plt.ylabel("Total Time (s)")
        plt.title(args.title + ": Time spent on checkpointing (Rank with most Overhead)")

#    # Plot (no groups)
#    for config in ['Full', 'List', 'TreeLowOffset', 'TreeLowRoot']:
##    for config in ['Full', 'List']:
##        for scale in [1, 2, 4, 8, 16, 32, 64, 128]:
#        for scale in [1, 2, 4, 8]:
#            df = data_frames[int(scale)]
#            config_frame = df[df['Config'] == config]
##            print(config_frame)
##            print(config_frame.groupby(['Run', 'Rank']).max(numeric_only=True))
##            print(config_frame.groupby(['Run', 'Rank']).mean(numeric_only=True))
##            config_frame = config_frame.groupby(level=0).mean(numeric_only=True)
#            for i in range(len(list(config_frame['Total']))):
#                if list(config_frame['Chkpt ID'])[i] > 0:
#                    alg_config.append(config)
#                    scale_config.append(scale)
#                    ranks.append((list(config_frame['Rank'])[i]))
#                    runs.append((list(config_frame['Run'])[i]))
#                    total_sizes.append((list(config_frame['Total'])[i]))
##            total_sizes.append(config_frame['Total'].sum()/(1024**2))
#
#    df = pd.DataFrame.from_dict({'Config': alg_config, 'Scale': scale_config, 'Run': runs, 'Rank': ranks, 'Total': total_sizes})
##    test_frame = df[['Config', 'Run', 'Total']]
##    test_frame.index = pd.MultiIndex.from_frame(df[['Scale', 'Config']])
##    print(test_frame.groupby(['Scale', 'Config']).mean())
##    test_frame.plot.bar(y="Total")
##    print(test_frame)
##    print(pd.MultiIndex.from_frame(df[['Scale', 'Rank']]))
##    print(df)
##    print(df.to_string())
##    figure = plt.figure()
##    df.plot.bar(y='Total')
##    g = sns.barplot(data=df, hue="Scale", y="Total", x="Config")
##    g = sns.catplot(data=df, hue="Scale", y="Total", x="Config")
##    g = sns.violinplot(data=df, hue="Scale", y="Total", x="Config")
##    g = sns.boxplot(data=df, hue="Scale", y="Total", x="Config")
##    g = sns.scatterplot(data=df, hue="Scale", y="Total", x="Config")
##    approaches = ['Full', 'Basic', 'List', 'Tree Low Offset', 'Tree Low Root']
##    scales = [1, 2, 4, 8, 32]
#
##    plt.xlabel("Checkpoint Algorithm")
##    plt.ylabel("Total Time (s)")
##    plt.yscale('log')
##    plt.ylim(bottom=1, top=60000)
##    h,l = g.get_legend_handles_labels()
##    g.legend(h, ['1', '2', '4', '8', '16', '32', '64', '128'], loc='lower left', title="Scale")
##    g.legend(h, ['1', '2', '4', '8', '16', '32', '64', '128'], title="Scale")
##    g.legend(h, ['1', '2', '4', '8', '32'], title="Scale")
##    plt.title(args.title + ": Sum of Checkpoints (Excluding Chkpt 0)")
    plt.tight_layout()
    plt.show()

elif args.plot_restart:
    logs = args.logfiles

    frames = []
    for log in logs:
        df = pd.read_csv(log, header=None)
        df.columns = ['Config', 'Chkpt ID', 'Chunk Size', 'Copy Diff', 'Restart Time']
        frames.append(df)
    master_frame = pd.concat(frames)
    configs = ['Full', 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot']
    selected_configs = ['Full' , 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot']
    chunk_sizes = [128, 256, 512, 1024, 2048, 4096]
    master_frame['Total'] = master_frame.loc[:, 'Copy Diff':'Restart Time'].sum(1)
    fig_data = plt.figure()
    data_g = sns.barplot(data=master_frame, hue="Config", y="Total", x="Chkpt ID")
#    plt.yscale('log')
    plt.ylim(bottom=0, top=3)
    plt.xlabel("Checkpoint ID")
    plt.ylabel("Time (s)")
    plt.title(args.title + " Restart Time")
#    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

elif args.vary_chunk_size:
    logs = args.logfiles

    frames = []
    for log in logs:
        df = pd.read_csv(log, header=None)
        if args.restart:
            df.columns = ['Config', 'Chkpt ID', 'Chunk Size', 'Copy Diff', 'Restart Time']
        else:
            df.columns = ['Config', 'Chkpt ID', 'Chunk Size', 'Calc Diff', 'Gather Diff', 'Copy Diff']
        df.sort_values(by=['Config', 'Chkpt ID'], inplace=True)
        df['Run'] = pd.DataFrame([0, 1, 2, 3, 4]*5)
        frames.append(df)
    master_frame = pd.concat(frames)
    test_frame = master_frame.groupby(['Config', 'Chunk Size', 'Run']).sum(1)
    if args.restart:
        test_frame['Total'] = test_frame.loc[:, 'Copy Diff':'Restart Time'].sum(1)
    else:
        test_frame['Total'] = test_frame.loc[:, 'Calc Diff':'Copy Diff'].sum(1)
    print(test_frame)
    test_frame = test_frame.reset_index()
    print(test_frame)
    sort_criteria = {"Full": 0, "Basic": 1, "List": 2, 'TreeLowOffset': 3, 'TreeLowRoot': 4}
    test_frame['sort index'] = test_frame['Config'].map(sort_criteria)
    print(test_frame)
    test_frame.sort_values(by=['sort index'], inplace=True)
    print(test_frame)
    fig = plt.figure()
    g = sns.barplot(data=test_frame, hue="Chunk Size", y="Total", x="Config")
#    fig2 = plt.figure()
#    configs = ['Full', 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot']
#    selected_configs = ['Full' , 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot']
#    chunk_sizes = [128, 256, 512, 1024, 2048, 4096]
#    if args.restart:
#        master_frame['Total'] = master_frame.loc[:, 'Copy Diff':'Restart Time'].sum(1)
#    else:
#        master_frame['Total'] = master_frame.loc[:, 'Calc Diff':'Copy Diff'].sum(1)
#    avg_frames = []
#    alg_config = []
#    chunk_sizes = []
#    total_sizes = []
#    print(master_frame)
#    g = sns.barplot(data=master_frame, hue="Chunk Size", y="Total", x="Config")
    h,l = g.get_legend_handles_labels()
    g.legend(h, ['  128', '  256', '  512', '1024', '2048', '4096'], loc='lower left', title="Chunk size")

    plt.xlabel("Checkpoint Algorithm")
    plt.ylabel("Total Runtime (s)")
    plt.ylim(bottom=0, top=5)
#    plt.yscale('log')
    if args.restart:
        plt.title("Restart " + args.title + ": Sum of 10 Checkpoints")
    else:
        plt.title(args.title + ": Sum of 10 Checkpoints")
    plt.tight_layout()
    plt.show()

else:
    logs = args.logfiles
    frames = []
    for log in logs:
        df = pd.read_csv(log, header=None)
        if args.restart:
            df.columns = ['Config', 'Chkpt ID', 'Chunk Size', 'Copy Diff', 'Restart Time']
            df.loc[:,'Total Time'] = df[['Copy Diff', 'Restart Time']].sum(axis=1)
        else:
            df.columns = ['Config', 'Chkpt ID', 'Chunk Size', 'Calc Diff', 'Gather Diff', 'Copy Diff']
            df.loc[:,'Total Time'] = df[['Calc Diff', 'Gather Diff', 'Copy Diff']].sum(axis=1)
        print(df)
        frames.append(df)
    master_frame = pd.concat(frames)
    print(master_frame)
    test_frame = master_frame
    test_frame = test_frame[test_frame['Chunk Size'] == 128]
    if args.restart:
        new_frame = test_frame[['Copy Diff', 'Restart Time']].groupby(test_frame.index).sum(numeric_only=True)
    else:
        new_frame = test_frame[['Calc Diff', 'Gather Diff', 'Copy Diff']].groupby(test_frame.index).sum(numeric_only=True)
    new_frame['Config'] = test_frame[test_frame['Chkpt ID'] == 0]['Config']
    grouped_frame = new_frame.groupby('Config').mean()
    grouped_frame = grouped_frame.reindex(['Full', 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot'])
    grouped_frame.reset_index(level=0, inplace=True)
    grouped_frame['Config'] = ['Full', 'Basic', 'List', 'Tree\nLow Offset', 'Tree\nLow Root']
#    print(grouped_frame)
    chart = grouped_frame.plot(kind='bar', stacked=True, x='Config')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=0)
    plt.xlabel("Configuration")
    plt.ylabel("Time (s)")
    plt.ylim(bottom=0, top=4)
#    plt.ylim(bottom=0, top=20)
    if args.restart:
        plt.title("Restart " + args.title + " Runtime: Sum of 10 Checkpoints")
    else:
        plt.title(args.title + " Runtime: Sum of 10 Checkpoints")
    plt.tight_layout()

#    plotting_frames = []
#    plot_frame = new_frame[['Config', 'Calc Diff']]
#    plot_frame.insert(1, 'Timer', 'Calc Diff')
#    plot_frame.columns = ['Config', 'Timer', 'Time (s)']
#    plotting_frames.append(plot_frame)
#    plot_frame = new_frame[['Config', 'Gather Diff']]
#    plot_frame.insert(1, 'Timer', 'Gather Diff')
#    plot_frame.columns = ['Config', 'Timer', 'Time (s)']
#    plotting_frames.append(plot_frame)
#    plot_frame = new_frame[['Config', 'Copy Diff']]
#    plot_frame.insert(1, 'Timer', 'Copy Diff')
#    plot_frame.columns = ['Config', 'Timer', 'Time (s)']
#    plotting_frames.append(plot_frame)
#    plot_frame = pd.concat(plotting_frames)
#    print(plot_frame)
#    test_fig = plt.figure()
#    test_g = sns.barplot(data=plot_frame, x='Config', y='Time (s)', hue='Timer') 
#    test_fig2 = plt.figure()
#    test_g2 = sns.barplot(data=plot_frame, hue='Config', y='Time (s)', x='Timer') 
    plt.show()

##    new_frame['Total Time'] = new_frame[['Calc Diff', 'Gather Diff', 'Copy Diff']].sum(axis=1)
##    print(new_frame)
##    test_fig = plt.figure()
##    test_g = sns.barplot(data=new_frame, x='Config', y='Total Time') 
#    configs = ['Full', 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot']
#    selected_configs = ['Full' , 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot']
#    overview_frames = []
#    trimmed_frames = []
#    for config in configs:
#        config_frame0 = master_frame[master_frame['Config'] == config]
##        config_frame = config_frame0[config_frame0['Chkpt ID'] % 2 == 1]
#        config_frame = config_frame0
#        config_frame = config_frame.set_index('Chkpt ID')
#        config_frame = config_frame.groupby('Chkpt ID').mean()
#        print(config_frame)
##        trimmed_config_frame  = config_frame[['Calculate Diff', 'Collect Diff', 'Copy Diff', 'Total Time']]
#        if args.restart:
#            trimmed_config_frame  = config_frame[['Copy Diff', 'Restart Time']]
#        else:
#            trimmed_config_frame  = config_frame[['Calc Diff', 'Gather Diff', 'Copy Diff']]
#        print(trimmed_config_frame)
#        if config in selected_configs:
#            trimmed_frames.append(trimmed_config_frame)
#
#    fig = plt.figure()
#    g = sns.barplot(data=master_frame, hue="Config", y="Total Time", x="Chkpt ID")
#
##    plot_clustered_stacked_frames(trimmed_frames, selected_configs)
#    ax = fig.get_axes()
#    plt.xlabel("Checkpoint #")
#    plt.ylabel("Time (s)")
#    if args.restart:
#        plt.title("Restart " + args.title + " Runtime")
#    else:
#        plt.title(args.title + " Checkpoint Runtime")
#    cm = plt.get_cmap('gist_rainbow')
#    NUM_COLORS=18
#    colors = plt.cm.Spectral(np.linspace(0,1,30))
#    ax[0].set_prop_cycle('color', colors)
#    plt.ylim(bottom=0, top=0.55)
#    plt.tight_layout()
#    plt.show()

#    full_max = frames[0]['Total Time'].max()
#    basic_list_max = frames[1]['Total Time'].max()
#    list_max = frames[2]['Total Time'].max()
#    tree_max = frames[4]['Total Time'].max()
#    cutoff = max(list_max, tree_max) * 1.1
#    low_cutoff = frames[0]['Total Time'].min() * 0.9
#    upper_cutoff = max(basic_list_max, full_max) * 1.1
#    cutoff = 0.15
#    low_cutoff = 0.15
#    print(cutoff)
#    print(low_cutoff)
#    print(upper_cutoff)
#    fig = plt.figure()
#    spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1,4])
#    axes = ['', '']
#    axes[0] = fig.add_subplot(spec[0])
#    axes[1] = fig.add_subplot(spec[1])
##    print(pd.concat([fullframe, basiclistframe, listframe, treeframe], axis=0))
#    hatches = ['xx', '//', '\\\\', '++', '--', '**', 'oo']
#    sns.barplot(x='Chkpt ID', y='Total Time', hue='Config', data=pd.concat(frames, axis=0), ci=None, ax=axes[0])
#    sns.barplot(x='Chkpt ID', y='Total Time', hue='Config', data=pd.concat(frames, axis=0), ci=None, ax=axes[1])
#
#    for axe in axes:
#        h,l = axe.get_legend_handles_labels() # get the handles we want to modify
#        for i in range(0, 2, 3): # len(h) = n_col * n_df
#            for j, pa in enumerate(h[i:i+3]):
#                for rect in pa.patches: # for each index
#                    rect.set_width(1 / float(5 + 1.5))
#
##    for bars, hatch in zip(axes[0].containers, hatches):
##        for bar in bars:
##            bar.set_hatch(hatch)
##    for bars, hatch in zip(axes[1].containers, hatches):
##        for bar in bars:
##            bar.set_hatch(hatch)
#    axes[0].set_ylim(low_cutoff, upper_cutoff)
#    axes[1].set_ylim(0, cutoff)
#    axes[0].spines['bottom'].set_visible(False)
#    axes[1].spines['top'].set_visible(False)
#    axes[0].xaxis.tick_top()
#    axes[0].tick_params(labeltop=False)
#    axes[1].xaxis.tick_bottom()
#
#    d = .01
#    kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False)
#    axes[0].plot((-d, +d), (-d, +d), **kwargs)
#    axes[0].plot((1-d, 1+d), (-d, +d), **kwargs)
#    kwargs.update(transform=axes[1].transAxes)
#    axes[1].plot((-d, +d), (1-d, 1+d), **kwargs)
#    axes[1].plot((1-d, 1+d), (1-d, 1+d), **kwargs)
#    axes[0].set_ylabel('')
#    axes[1].set_ylabel('')
#    axes[0].set_xlabel('')
#    axes[1].set_xlabel('')
#
#    fig.suptitle(args.title + ": Checkpoint Time (128 Byte Chunks)", fontsize=14)
##    fig.supxlabel("Checkpoint number", fontsize=14)
#    fig.supylabel("Time (s)", fontsize=14)
#    for a in axes:
#        for label in (a.get_xticklabels() + a.get_yticklabels()):
#            label.set_fontsize(14)
#    plt.legend(ncol=2, fontsize=12, loc='center', bbox_to_anchor=(0.5, -0.3))
#    axes[0].get_legend().remove()
#    plt.xticks(fontsize=14)
#    plt.yticks(fontsize=14)

