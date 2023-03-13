import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")
#print(matplotlib.style.available)
#matplotlib.style.use('seaborn-colorblind')
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
import numpy as np
import argparse
import pickle
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns


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
            offset = 0
            for rect in pa.patches: # for each index
#                rect.set_x(rect.get_x() + (1 / float(n_df + 0.75)) * i / float(n_col) + 0.1)
                rect.set_x(rect.get_x() + (1 / float(n_df + 0.5)) * i / float(n_col) - (1/float(2*n_df)))
#                rect.set_x(rect.get_x() + i*0.0125)
#                rect.set_x(rect.get_x()+.25)
#                offset += 1
                rect.set_hatch(H * 2*int(i / n_col)) #edited part     
#                if j > 0:
#                    rect.set_hatch(hatches[int((i+j)/2)+1]) #edited part     
#                rect.set_hatch(hatches[int((i+j)/2)+1]) #edited part     
                rect.set_width(1 / float(n_df + 1.25))
#                rect.set_width(0.25)
    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    print(df.index)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)
    axe.set_prop_cycle('color', colors)
    axe.set_yscale('log')

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

parser = argparse.ArgumentParser(description="Plot runtimes")
parser.add_argument("logfiles", type=str, nargs="+", help="Log files")
parser.add_argument("--attr", type=str, nargs=1, help="Which attribute to show on x-axis")
parser.add_argument("--collected", action='store_true', help="Whether the data has been collected into a single file")
parser.add_argument("--vary_chunk_size", action='store_true', help="Plot for different chunk sizes")
parser.add_argument("--title", type=str, default='')
parser.add_argument("--num_ranks", type=int, default=1, help='Number of ranks for input files')
parser.add_argument("--ipdps", action='store_true', help="IPDPS plots")
parser.add_argument("--metadata-breakdown", action='store_true', help="Plot breakdown of metadata")
parser.add_argument("--split-graphs", action='store_true', help="Plot Data and Metadata separately")
parser.add_argument("--mpi", action='store_true', help="Visualize scaling tests")

args = parser.parse_args()

if args.mpi:
    frames = {1:[], 2:[], 4:[], 8:[], 16:[], 32:[], 64:[], 128:[]}
    data_frames = {}
    directories = args.logfiles
    for directory in directories:
        name = directory.split('/')
        scale = ''.join(filter(str.isdigit, name[1]))
        for root,dirs,files in os.walk(directory):
            frame_list = []
            for file in files:
                if file.endswith("size.csv"):
                    df = pd.read_csv(directory+'/'+file, header=None)
                    df.columns = ['Config', 'Chkpt ID', 'Chunk Size', 'Data', 'Metadata', 'Header', 'First\nOccurrences', 'Duplicate\nChkpt Map', 'Chkpt 0', 'Chkpt 1', 'Chkpt 2', 'Chkpt 3', 'Chkpt 4', 'Chkpt 5', 'Chkpt 6', 'Chkpt 7', 'Chkpt 8', 'Chkpt 9']
                    df['Scale'] = scale
                    filename = file.split('.')
                    rank = 0;
                    if len(list(df.index)) > 20:
                        print(file)
                    for i in range(len(filename)):
                        if 'Rank' in filename[i]:
                            rank = ''.join(filter(str.isdigit, filename[i]))
                            df['Rank'] = rank
                            break
                    df = df.sort_values(['Config']).reset_index(drop=True)
                    frames[int(scale)].append(df)
        data_frames[int(scale)] = pd.concat(frames[int(scale)])                
        data_frames[int(scale)]['Total'] = data_frames[int(scale)].loc[:, 'Data':'Metadata'].sum(1)
    alg_config = []
    scale_config = []
    total_sizes = []
    for config in ['Full', 'List', 'TreeLowOffset', 'TreeLowRoot']:
#        for scale in [1, 2, 4, 8, 16, 32, 64, 128]:
        for scale in [1, 2, 4, 8, 32]:
            df = data_frames[int(scale)]
            config_frame = df[df['Config'] == config]
#            config_frame = config_frame[config_frame['Chkpt ID'] == 1]
#            print(config_frame.groupby('Chkpt ID').sum(numeric_only=True))
            config_frame = config_frame.groupby(level=0).sum(numeric_only=True)
#            config_frame = config_frame.set_index('Chkpt ID')
#            config_frame = config_frame.groupby('Chkpt ID').mean(numeric_only=True)
            for i in range(len(list(config_frame['Total']))):
                alg_config.append(config)
                scale_config.append(scale)
                total_sizes.append((list(config_frame['Total'])[i])/(1024**2))
#            total_sizes.append(config_frame['Total'].sum()/(1024**2))
    df = pd.DataFrame.from_dict({'Config': alg_config, 'Scale': scale_config, 'Total': total_sizes})
    print(df)
    figure = plt.figure()
    g = sns.barplot(data=df, hue="Scale", y="Total", x="Config")
    plt.xlabel("Checkpoint Algorithm")
    plt.ylabel("Total Size (MB)")
    plt.yscale('log')
#    plt.ylim(bottom=1, top=60000)
    h,l = g.get_legend_handles_labels()
#    g.legend(h, ['1', '2', '4', '8', '16', '32', '64', '128'], loc='lower left', title="Scale")
#    g.legend(h, ['1', '2', '4', '8', '16', '32', '64', '128'], title="Scale")
    g.legend(h, ['1', '2', '4', '8', '32'], title="Scale")
    plt.title(args.title + ": Sum of Checkpoints (Excluding Chkpt 0)")
    plt.tight_layout()

elif args.metadata_breakdown:
    logs = args.logfiles
    frames = []
    for log in logs:
        df = pd.read_csv(log, header=None)
        df.columns = ['Config', 'Chkpt ID', 'Chunk Size', 'Data', 'Metadata', 'Header', 'First\nOccurrences', 'Duplicate\nChkpt Map', 'Chkpt 0', 'Chkpt 1', 'Chkpt 2', 'Chkpt 3', 'Chkpt 4', 'Chkpt 5', 'Chkpt 6', 'Chkpt 7', 'Chkpt 8', 'Chkpt 9']
        frames.append(df)
    master_frame = pd.concat(frames)
#    print(master_frame)
    configs = ['Full', 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot']
    selected_configs = ['Full' , 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot']
    overview_frames = []
    trimmed_frames = []
    for config in configs:
        config_frame0 = master_frame[master_frame['Config'] == config]
#        config_frame = config_frame0[config_frame0['Chkpt ID'] % 2 == 1]
        config_frame = config_frame0
        config_frame = config_frame.set_index('Chkpt ID')
        config_frame = config_frame.groupby('Chkpt ID').mean()
#        print(config_frame)
        trimmed_config_frame  = config_frame[[ 'Header', 'First\nOccurrences', 'Duplicate\nChkpt Map', 'Chkpt 0', 'Chkpt 1', 'Chkpt 2', 'Chkpt 3', 'Chkpt 4', 'Chkpt 5', 'Chkpt 6', 'Chkpt 7', 'Chkpt 8', 'Chkpt 9']]
        config_overview = config_frame[['Data', 'Metadata']]
        if config in selected_configs:
            overview_frames.append(config_overview)
            trimmed_frames.append(trimmed_config_frame)

    fig_overview = plt.figure()
    plot_clustered_stacked_frames(overview_frames, selected_configs)
    plt.xlabel("Checkpoint #")
    plt.ylabel("Size (Bytes)")
    plt.ylim(bottom=1000, top=10**10)
    plt.title(args.title + " Overview")

    fig = plt.figure()
    plot_clustered_stacked_frames(trimmed_frames, selected_configs)
    ax = fig.get_axes()
    plt.xlabel("Checkpoint #")
    plt.ylabel("Size (Bytes)")
    plt.ylim(bottom=10)
    plt.title(args.title + " Metadata")
    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS=18
    colors = plt.cm.Spectral(np.linspace(0,1,30))
    ax[0].set_prop_cycle('color', colors)
    plt.show()

elif args.split_graphs:
    logs = args.logfiles

    frames = []
    for log in logs:
        df = pd.read_csv(log, header=None)
        df.columns = ['Config', 'Chkpt ID', 'Chunk Size', 'Data', 'Metadata', 'Header', 'First\nOccurrences', 'Duplicate\nChkpt Map', 'Chkpt 0', 'Chkpt 1', 'Chkpt 2', 'Chkpt 3', 'Chkpt 4', 'Chkpt 5', 'Chkpt 6', 'Chkpt 7', 'Chkpt 8', 'Chkpt 9']
        frames.append(df)
    master_frame = pd.concat(frames)
    configs = ['Full', 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot']
    selected_configs = ['Full' , 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot']
    chunk_sizes = [128, 256, 512, 1024, 2048, 4096]
    master_frame['Total'] = master_frame.loc[:, 'Data':'Metadata'].sum(1)
    master_frame['Data'] = master_frame.loc[:,'Data']/(1024**2)
    master_frame['Metadata'] = master_frame.loc[:,'Metadata']/(1024**2)
#    avg_frames = []
#    alg_config = []
#    chunk_sizes = []
#    total_sizes = []
#    chkpt_idx = []
#    data_list = []
#    metadata_list = []

#    fig,axes = plt.subplots(1, 2, sharey=True)
#    data_g = sns.barplot(data=master_frame, hue="Config", y="Data", x="Chkpt ID", ax=axes[0])
#    data_g.legend_.remove()
#    metadata_g = sns.barplot(data=master_frame, hue="Config", y="Metadata", x="Chkpt ID", ax=axes[1])
#    data_g.set_ylabel("Data (Bytes)")
#    metadata_g.set_ylabel("Metadata (Bytes)")
#    plt.suptitle(args.title)
#    for ax in axes:
#        ax.set_ylim([1, 1*(10**10)])
#        ax.set_yscale('log')

    fig_data = plt.figure()
    data_g = sns.barplot(data=master_frame, hue="Config", y="Data", x="Chkpt ID")
    plt.yscale('log')
    plt.ylim(bottom=.001, top=5*(10**3))
    plt.xlabel("Checkpoint ID")
    plt.ylabel("Total Size (MB)")
    plt.title(args.title + " Data: Sum of 10 Checkpoints")
    plt.legend(loc='lower left')
    fig_metadata = plt.figure()
    metadata_g = sns.barplot(data=master_frame, hue="Config", y="Metadata", x="Chkpt ID")
    plt.yscale('log')
    plt.ylim(bottom=.001, top=5*(10**3))
    plt.xlabel("Checkpoint ID")
    plt.ylabel("Total Size (MB)")
    plt.title(args.title + " Metadata: Sum of 10 Checkpoints")
    plt.legend(loc='lower left')

    plt.tight_layout()


elif args.vary_chunk_size:
    logs = args.logfiles

    frames = []
    for log in logs:
        df = pd.read_csv(log, header=None)
        df.columns = ['Config', 'Chkpt ID', 'Chunk Size', 'Data', 'Metadata', 'Header', 'First\nOccurrences', 'Duplicate\nChkpt Map', 'Chkpt 0', 'Chkpt 1', 'Chkpt 2', 'Chkpt 3', 'Chkpt 4', 'Chkpt 5', 'Chkpt 6', 'Chkpt 7', 'Chkpt 8', 'Chkpt 9']
        frames.append(df)
    master_frame = pd.concat(frames)
#    print(master_frame)
    configs = ['Full', 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot']
    selected_configs = ['Full' , 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot']
    chunk_sizes = [128, 256, 512, 1024, 2048, 4096]
    master_frame['Total'] = master_frame.loc[:, 'Data':'Metadata'].sum(1)
    print(master_frame[master_frame['Config'] == 'TreeLowOffset'])
    avg_frames = []
    alg_config = []
    chunk_sizes = []
    total_sizes = []

    for config in ['Full', 'Basic', 'List', 'TreeLowOffset', 'TreeLowRoot']:
        for chunk_size in [128, 256, 512, 1024, 2048, 4096]:
#            print(str(config) + ": " + str(chunk_size))
            config_frame = master_frame[master_frame['Config'] == config]
            config_frame = config_frame[config_frame['Chunk Size'] == chunk_size]
            config_frame = config_frame.set_index('Chkpt ID')
            config_frame = config_frame.groupby('Chkpt ID').mean(numeric_only=True)
#            print(config_frame)
            alg_config.append(config)
            chunk_sizes.append(config_frame.loc[0,'Chunk Size'])
            total_sizes.append(config_frame['Total'].sum()/(1024**2))
    print(alg_config)
    print(chunk_sizes)
    print(total_sizes)
    df = pd.DataFrame.from_dict({'Config': alg_config, 'Chunk Size': chunk_sizes, 'Total': total_sizes})
    print(df)
    figure = plt.figure()
    g = sns.barplot(data=df, hue="Chunk Size", y="Total", x="Config")
    plt.xlabel("Checkpoint Algorithm")
    plt.ylabel("Total Size (MB)")
#    plt.yscale('log')
    plt.ylim(bottom=1, top=60000)
    h,l = g.get_legend_handles_labels()
    g.legend(h, ['  128', '  256', '  512', '1024', '2048', '4096'], loc='lower left', title="Chunk size")
#    g.legend(['  128', '  256', '  512', '1024', '2048', '4096'], loc='lower left')
#    shift = max([t.get_window_extent().width for t in legend.get_texts()])
#    for t in legend.get_texts():
#        t.set_ha('right')
#        t.set_position((shift,0))
    plt.title(args.title + ": Sum of 10 Checkpoints")

#    full_frames = []
#    basic_frames = []
#    list_frames = []
#    treelowoffset_frames = []
#    treelowroot_frames = []
#    overview_frames = []
#    trimmed_frames = []
#    for config in configs:
#        config_frame0 = master_frame[master_frame['Config'] == config]
##        config_frame = config_frame0[config_frame0['Chkpt ID'] % 2 == 1]
#        config_frame = config_frame0
#        config_frame = config_frame.set_index('Chkpt ID')
#        config_frame = config_frame.groupby('Chkpt ID').mean()
##        print(config_frame)
#        trimmed_config_frame  = config_frame[[ 'Header', 'First\nOccurrences', 'Duplicate\nChkpt Map', 'Chkpt 0', 'Chkpt 1', 'Chkpt 2', 'Chkpt 3', 'Chkpt 4', 'Chkpt 5', 'Chkpt 6', 'Chkpt 7', 'Chkpt 8', 'Chkpt 9']]
#        config_overview = config_frame[['Data', 'Metadata']]
#        if config in selected_configs:
#            overview_frames.append(config_overview)
#            trimmed_frames.append(trimmed_config_frame)
#
#    fig_overview = plt.figure()
#    plot_clustered_stacked_frames(overview_frames, selected_configs)
#    plt.xlabel("Checkpoint #")
#    plt.ylabel("Size (Bytes)")
#    plt.ylim(bottom=1000, top=10**10)
#    plt.title(args.title + " Overview")
#
#    fig = plt.figure()
#    plot_clustered_stacked_frames(trimmed_frames, selected_configs)
#    ax = fig.get_axes()
#    plt.xlabel("Checkpoint #")
#    plt.ylabel("Size (Bytes)")
#    plt.ylim(bottom=10)
#    plt.title(args.title + " Metadata")
#    cm = plt.get_cmap('gist_rainbow')
#    NUM_COLORS=18
#    colors = plt.cm.Spectral(np.linspace(0,1,30))
#    ax[0].set_prop_cycle('color', colors)
#    plt.show()

elif args.collected:
    logs = args.logfiles
    frames = []
    for log in logs:
        df = pd.read_csv(log)
#        if 'list' in log:
#            df.columns = ['List Data', 'List Metadata']
#        elif 'tree' in log:
#            df.columns = ['Tree Data', 'Tree Metadata']
        frames.append(df)
#        print(df)
    print(frames)
    plot_clustered_stacked(frames, ['List', 'Tree'])
#    frame = frames[0]
#    for i in range(1, len(frames)):
#        frame = frame.join(frames[i])
#    print(frame)
#    frame.plot.bar()
    plt.ylabel("Size (Bytes)", fontsize=14)
    plt.xlabel("Checkpoint #", fontsize=14)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14)
    plt.yscale('log')
    plt.tight_layout()
    plt.ylim(bottom=0)
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
    print(ranks1)
    print(ranks2)
    print(ranks4)
    print(ranks8)
    ranksumfull1 = (ranks1['Full checkpoint data']   + ranks1['Full checkpoint metadata']).sum()/(1024*1024*1024)
    ranksumlist1 = (ranks1['Hash list data']   + ranks1['Hash list metadata']).sum()/(1024*1024*1024)
    ranksumtree1 = (ranks1['Merkle tree data'] + ranks1['Merkle tree metadata']).sum()/(1024*1024*1024)
    ranksumfull2 = (ranks2['Full checkpoint data']   + ranks2['Full checkpoint metadata']).sum()/(1024*1024*1024)
    ranksumlist2 = (ranks2['Hash list data']   + ranks2['Hash list metadata']).sum()/(1024*1024*1024)
    ranksumtree2 = (ranks2['Merkle tree data'] + ranks2['Merkle tree metadata']).sum()/(1024*1024*1024)
    ranksumfull4 = (ranks4['Full checkpoint data']   + ranks4['Full checkpoint metadata']).sum()/(1024*1024*1024)
    ranksumlist4 = (ranks4['Hash list data']   + ranks4['Hash list metadata']).sum()/(1024*1024*1024)
    ranksumtree4 = (ranks4['Merkle tree data'] + ranks4['Merkle tree metadata']).sum()/(1024*1024*1024)
    ranksumfull8 = (ranks8['Full checkpoint data']   + ranks8['Full checkpoint metadata']).sum()/(1024*1024*1024)
    ranksumlist8 = (ranks8['Hash list data']   + ranks8['Hash list metadata']).sum()/(1024*1024*1024)
    ranksumtree8 = (ranks8['Merkle tree data'] + ranks8['Merkle tree metadata']).sum()/(1024*1024*1024)
    print((ranks1['Full checkpoint data']   + ranks1['Full checkpoint metadata']).sum()/(1024*1024*1024))
    print((ranks1['Hash list data']   + ranks1['Hash list metadata']).sum()/(1024*1024*1024))
    print((ranks1['Merkle tree data'] + ranks1['Merkle tree metadata']).sum()/(1024*1024*1024))
    print((ranks2['Full checkpoint data']   + ranks2['Full checkpoint metadata']).sum()/(1024*1024*1024))
    print((ranks2['Hash list data']   + ranks2['Hash list metadata']).sum()/(1024*1024*1024))
    print((ranks2['Merkle tree data'] + ranks2['Merkle tree metadata']).sum()/(1024*1024*1024))
    print((ranks4['Full checkpoint data']   + ranks4['Full checkpoint metadata']).sum()/(1024*1024*1024))
    print((ranks4['Hash list data']   + ranks4['Hash list metadata']).sum()/(1024*1024*1024))
    print((ranks4['Merkle tree data'] + ranks4['Merkle tree metadata']).sum()/(1024*1024*1024))
    print((ranks8['Full checkpoint data']   + ranks8['Full checkpoint metadata']).sum()/(1024*1024*1024))
    print((ranks8['Hash list data']   + ranks8['Hash list metadata']).sum()/(1024*1024*1024))
    print((ranks8['Merkle tree data'] + ranks8['Merkle tree metadata']).sum()/(1024*1024*1024))
    frame_dict = {'1': [ranksumfull1, ranksumlist1, ranksumtree1], '2': [ranksumfull2, ranksumlist2, ranksumtree2], '4': [ranksumfull4, ranksumlist4, ranksumtree4], '8': [ranksumfull8, ranksumlist8, ranksumtree8]}
    df = pd.DataFrame.from_dict(frame_dict, columns=['Full', 'List', 'Tree'], orient='index')
    print(df)
#    df.plot.bar()

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
#    low_cutoff   = df['Full'].min() * 0.9
    low_cutoff   = df['List'].max() * 1.1
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
    for a in axes:
        for label in (a.get_xticklabels() + a.get_yticklabels()):
            label.set_fontsize(14)
    plt.legend(fontsize=14)
    axes[0].get_legend().remove()
    plt.suptitle(args.title + ": Total Checkpoint Size", fontsize=14)
    plt.ylabel("Total Checkpoint Size (GB)", fontsize=14)
    plt.xlabel("Number of Processes", fontsize=14)
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)

#    metadata1rank = ranks1[
    

#    frames = {'128': {'list': [], 'tree': []}}
#    for log in logs:
#        file_name = log.split('.')
#        chunksize_idx = file_name.index('chunk_size')
#        chunk_size = file_name[chunksize_idx+1]
#    
#        checkpoint_idx = file_name.index('dat')
#        checkpoint = file_name[checkpoint_idx-1]
#    
##        print(log)
#        df = pd.read_csv(log, header=None)
#        df = df.div(1024*1024)
#        df.columns = ['Full checkpoint diff time', 'Full checkpoint write time', 'Full checkpoint data', 'Full checkpoint metadata', 'Hash list diff time', 'Hash list write time', 'Hash list data', 'Hash list metadata', 'Merkle tree diff time', 'Merkle tree write time', 'Merkle tree data', 'Merkle tree metadata']
#        rank_pos = log.find("Rank")
#        df['Rank'] = pd.Series([log[rank_pos+4] for x in range(len(df.index))])
#        df['Checkpoint'] = pd.Series([log[log.find('.dat')-1] for x in range(len(df.index))])
##        print(df)
#        
##        frames[chunk_size]['full'].append(df[['Full checkpoint data', 'Full checkpoint metadata', 'Rank', 'Checkpoint']])
#        frames[chunk_size]['list'].append(df[['Hash list data', 'Hash list metadata', 'Rank', 'Checkpoint']])
#        frames[chunk_size]['tree'].append(df[['Merkle tree data', 'Merkle tree metadata', 'Rank', 'Checkpoint']])
#    
##    print(frames)
#    collected_frames = {'128': {'0': [], '1': [],'2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}}
#    mpi_frames = {'128': []}
#    for chunk_size,chunk_frames in frames.items():
#        for k,v in chunk_frames.items():
#            frame_list = []
#            frame_dict = {'0': [], '1': [],'2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []} 
#            for i in range(0, len(v), 2):
#                v[i].columns = ['Data', 'Metadata', 'Rank', 'Checkpoint']
#                f = pd.DataFrame(v[i]['Data'] + v[i]['Metadata'])
#                f.columns = ['Rank' + v[i]['Rank'].iloc[[0]] + ' Checkpoint ' + v[i]['Checkpoint'].iloc[[0]]]
#                frame_list.append(f)
#                frame_dict[v[i]['Checkpoint'].iloc[[0]][0]].append(f.iloc[[0]])
#            df = pd.concat(frame_list, axis=1)
##            print(df)
#            fr_list = []
#            for chkpt_id, f_list in frame_dict.items():
#                if len(f_list) > 0:
#                    fr_list.append( pd.concat(f_list, axis=1) )
#            df.index = range(0, 2*(len(df.index)), 2)
#            fd = df.filter(like='Checkpoint 0')
#            f_list = []
#            for i in range(2, 10, 2):
##                print(df.filter(like='Checkpoint ' + str(i)))
#                collected_frames[chunk_size][str(i)].append(df.filter(like='Checkpoint ' + str(i)))
#                ndf = df.filter(like='Checkpoint ' + str(i)).loc[[i]]
#                ndf.columns = [ i for i in range(len(ndf.columns))]
#                f_list.append(ndf)
#            new_df = pd.concat(f_list, axis=0)
#            new_df.columns = ['Rank ' + str(j) for j in range(args.num_ranks)]
#            mpi_frames[chunk_size].append(new_df)
##            print(new_df)
##    print(collected_frames['128'])
#    print(mpi_frames['128'])
#
#    plot_clustered_stacked(mpi_frames['128'], labels=['List', 'Tree'])
#    plt.ylabel("Size (MB)", fontsize=14)
#    plt.xlabel("Checkpoint Number", fontsize=14)
#    plt.xticks(fontsize=14, rotation=0)
#    plt.suptitle(args.title, fontsize=14)
#    plt.yticks(fontsize=14)
###    plt.yscale('log')
    plt.tight_layout()
#    plt.ylim(bottom=1)

else:
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
        if len(df.columns) == 12:
            df.columns = ['Full checkpoint diff time', 'Full checkpoint write time', 'Full checkpoint data', 'Full checkpoint metadata', 'Hash list diff time', 'Hash list write time', 'Hash list data', 'Hash list metadata', 'Merkle tree diff time', 'Merkle tree write time', 'Merkle tree data', 'Merkle tree metadata']
        else:
            df.columns = ['Full checkpoint diff time', 'Full checkpoint collect time', 'Full checkpoint write time', 'Full checkpoint data', 'Full checkpoint metadata', 'Hash list diff time', 'Hash list collect time', 'Hash list write time', 'Hash list data', 'Hash list metadata', 'Merkle tree diff time', 'Merkle tree collect time', 'Merkle tree write time', 'Merkle tree data', 'Merkle tree metadata']
        df['Chunk size'] = pd.Series([chunk_size for i in range(len(df.index))])
        df['Checkpoint'] = pd.Series([checkpoint[-1] for i in range(len(df.index))])
        df['Implementation'] = pd.Series([implementation for i in range(len(df.index))])
        frames.append(df)
    master_frame = pd.concat(frames, axis=0)
    print(master_frame[['Full checkpoint data', 'Full checkpoint metadata', 'Hash list data', 'Hash list metadata', 'Merkle tree data', 'Merkle tree metadata', 'Chunk size', 'Checkpoint']])
    fig = plt.figure()
    list128_frames = []
    tree128_frames = []
    list4096_frames = []
    tree4096_frames = []
    for i in range(int(master_frame['Checkpoint'].max())+1):
#        if i % 1 == 0:
        if True:
            list_128  = master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '128') & (master_frame['Checkpoint'] == str(i))].filter(like='data').filter(like='list').iloc[[0]].div(1024*1024)
            list_128.columns = ['Data', 'Metadata']
            list_128.index = [i]
            tree_128  = master_frame.loc[(master_frame['Implementation'] == 'GPU') & (master_frame['Chunk size'] == '128') & (master_frame['Checkpoint'] == str(i))].filter(like='data').filter(like='tree').iloc[[0]].div(1024*1024)
            tree_128.columns = ['Data', 'Metadata']
            tree_128.index = [i]
            list128_frames.append(list_128)
            tree128_frames.append(tree_128)
#    print(pd.concat(list128_frames))
#    print([pd.concat(list128_frames), pd.concat(tree128_frames)])
#    framelist = [pd.concat(list128_frames), pd.concat(tree128_frames), pd.concat(list4096_frames), pd.concat(tree4096_frames)]
#    plot_clustered_stacked(framelist, ['List: 128B Chunks', 'Tree: 128B Chunks', 'List: 4KB Chunks', 'Tree: 4KB Chunks'])
    framelist = [pd.concat(list128_frames), pd.concat(tree128_frames)]
    print(framelist[0])
    print(framelist[1])
#    plot_clustered_stacked(framelist, ['List', 'Tree'])
    plot_clustered_stacked_frames(framelist, ['List', 'Tree'])
    plt.xlabel("Checkpoint Number", fontsize=14)
    plt.ylabel("Checkpoint Size (MB)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
#    plt.title(args.title + ": Checkpoint size", fontsize=14, pad=40)
    plt.title(args.title + ": Checkpoint size (128 Byte Chunks)", fontsize=14, pad=30)
    plt.tight_layout()
    plt.ylim(bottom=0,top=300)

plt.show()

