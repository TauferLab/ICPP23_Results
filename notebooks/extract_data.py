#!/bin/env python

import os
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Plot runtimes")
parser.add_argument("logfiles", type=str, nargs="+", help="Directory with log files")
parser.add_argument("--type", type=str, nargs=1, help="Which type of file (nvcomp|dedup)")
parser.add_argument("--output", type=str, nargs=1, default='', help='Output filename')
parser.add_argument("--chunksize", type=int, nargs=1, default='0', help='Chunks size')
parser.add_argument("--scenario", type=str, nargs=1, default='', help='Scenario name')
parser.add_argument("--num-chkpts", type=int, nargs=1, default=5, help='Number of checkpoints')
parser.add_argument("--size-time-file", action='store_true', help='Input csv is a old checkpoint log')
parser.add_argument("--scale", type=int, nargs=1, default=1, help='# of GPUs used for runs')
parser.add_argument("--num-runs", type=int, nargs=1, default=1, help='# of runs')


args = parser.parse_args()
directories = args.logfiles
print(args.type)

if args.size_time_file:
    num_runs = args.num_runs[0]
    size_frames = []
    time_frames = []
    scale = args.scale[0]
    for directory in directories:
#        print(directory)
        name = directory.split('/')
        for root,dirs,files in os.walk(directory):
            frame_list = []
            for file in files:
                if file.endswith("size.csv") and os.path.getsize(directory+'/'+file) > 0:
                    split_str = file.split('.')
                    index = 0
                    for i in range(len(split_str)):
                        substr = split_str[i]
                        if 'Rank' in substr:
                            index = i
                            break
#                    print(directory+'/'+file)
                    rank = int(split_str[index][split_str[index].index('Rank')+4:])
                    temp_df = pd.read_csv(directory+'/'+file, names=['Approach', 'Chkpt ID', 'Chunk Size', 'Data', 'Metadata', 'Header', 'Bytes for First Occurrence Metadata', 'Size of repeat map', 'Bytes for Chkpt 0', 'Bytes for Chkpt 1', 'Bytes for Chkpt 2', 'Bytes for Chkpt 3', 'Bytes for Chkpt 4', 'Bytes for Chkpt 5', 'Bytes for Chkpt 6', 'Bytes for Chkpt 7', 'Bytes for Chkpt 8', 'Bytes for Chkpt 9'])
                    temp_df['Rank'] = rank
                    temp_df = temp_df[temp_df['Chkpt ID'] > 0]
                    size_frames.append(temp_df)
                elif file.endswith(".timing.csv") and os.path.getsize(directory+'/'+file) > 0:
                    split_str = file.split('.')
                    index = 0
                    for i in range(len(split_str)):
                        substr = split_str[i]
                        if 'Rank' in substr:
                            index = i
                            break
#                    print(directory+'/'+file)
                    rank = int(split_str[index][split_str[index].index('Rank')+4:])
                    temp_df = pd.read_csv(directory+'/'+file, names=['Approach', 'Chkpt ID', 'Chunk Size', 'Comparison Time', 'Gather Time', 'Copy Time'])
                    temp_df['Rank'] = rank
                    temp_df = temp_df[temp_df['Chkpt ID'] > 0]
                    time_frames.append(temp_df)
    mapping = {'Full': 0, 'Basic': 1, 'List': 2, 'TreeLowOffset': 3, 'TreeLowRoot': 4, 0:5, 1:6, 2:7, 3:8, 4:9, 5:10, 6:11, 7:12, 8:13, 9:14, 10:15, 11:16, 12:17, 13:18, 14:19, 15:20, 16:21, 17:22, 18:23, 19:24, 20:25}
    dedup_df = pd.concat(size_frames)
#    print(dedup_df.to_string())
        
    if args.scenario != '':
        dedup_df['Scenario'] = args.scenario[0]
    dedup_df['Number of Chkpts'] = args.num_chkpts[0]
    dedup_df = dedup_df.sort_values(by=['Chkpt ID', 'Approach'], key=lambda o: o.apply(lambda x: mapping[x]))
    dedup_df = dedup_df.reset_index(drop=True)
    dedup_df = dedup_df.reset_index()
    dedup_df['Run'] = dedup_df['index'] % num_runs
    grouped_df = dedup_df.groupby(['Approach', 'Run']).sum()
    grouped_df['Deduplicated Size'] = grouped_df['Data'] + grouped_df['Metadata']
    grouped_df = grouped_df.reset_index()
    uncompr_size = grouped_df[grouped_df['Approach'] == 'Full'].iloc[0]['Deduplicated Size']
    grouped_df['Deduplication Ratio'] = uncompr_size / grouped_df['Deduplicated Size']
#    print(grouped_df)

    time_df = pd.concat(time_frames)
    if args.scenario != '':
        time_df['Scenario'] = args.scenario[0]
    time_df['Number of Chkpts'] = args.num_chkpts[0]
    time_df = time_df.sort_values(by=['Chkpt ID', 'Approach'], key=lambda o: o.apply(lambda x: mapping[x]))
    time_df = time_df.reset_index(drop=True)
    time_df = time_df.reset_index()
    time_df['Run'] = time_df['index'] % num_runs
    grouped_time_df = time_df.groupby(['Approach', 'Run', 'Rank']).sum()
    grouped_time_df['Deduplicated Time'] = grouped_time_df['Comparison Time'] + grouped_time_df['Gather Time'] + grouped_time_df['Copy Time']
    grouped_time_df = grouped_time_df.reset_index()
    grouped_time_df = grouped_time_df.groupby(['Approach', 'Run']).max().reset_index()
    grouped_time_df['Deduplication Throughput'] = (uncompr_size/scale) / grouped_time_df['Deduplicated Time']

    clean_df = grouped_df[['Approach', 'Run', 'Deduplicated Size', 'Deduplication Ratio']]
    clean_df['Deduplication Throughput'] = grouped_time_df['Deduplication Throughput']
    clean_df['Scale'] = scale
    print(clean_df)

    if args.output != '':
        if os.path.isfile(args.output[0]):
            clean_df.to_csv(args.output[0], mode='a', index=False, header=False)
        else:
            clean_df.to_csv(args.output[0], index=False)
else:
    if args.type[0] == 'nvcomp':
        data_dict = {'Scenario': [], 'Approach': [], 'Rank': [], 'Chkpt ID': [], 'Number of Chkpts': [], 'Chunk Size': [], 'Uncompressed Size': [], 'Compressed Size': [], 'Max Memory Usage': [], 'Compression Throughput': [], 'Compression Runtime': []}
        nvcomp_df = []
        for directory in directories:
            print(directory)
            name = directory.split('/')
            scale = ''.join(filter(str.isdigit, name[1]))
            for root,dirs,files in os.walk(directory):
                frame_list = []
                for file in files:
                    if file.endswith(".csv") and os.path.getsize(directory+'/'+file) > 0:
                        split_str = file.split('.')
                        index = 0
                        for i in range(len(split_str)):
                            substr = split_str[i]
                            if 'Rank' in substr:
                                index = i
                                break
                        print(directory+'/'+file)
                        temp_df = pd.read_csv(directory+'/'+file)
                        for i in range(temp_df.shape[0]):
                            if args.scenario != '':
                                data_dict['Scenario'].append(args.scenario[0])
                            data_dict['Rank'].append(int(split_str[index][split_str[index].index('Rank')+4:]))
                            data_dict['Chkpt ID'].append(int(split_str[index+1]))
                            data_dict['Number of Chkpts'].append(args.num_chkpts[0]);
                            data_dict['Chunk Size'].append(args.chunksize[0])
                            data_dict['Approach'].append(split_str[-2])
                            data_dict['Uncompressed Size'].append(temp_df.loc[i,'Ucompressed size in bytes'])
                            data_dict['Compressed Size'].append(temp_df.loc[i,'Compressed size in bytes'])
                            data_dict['Max Memory Usage'].append(temp_df.loc[i,'Max GPU Memory (B)'])
                            data_dict['Compression Throughput'].append(temp_df.loc[i,'Compression throughput (uncompressed) in GB/s']*10**9)
                            data_dict['Compression Runtime'].append(temp_df.loc[i, 'Computation time'] + temp_df.loc[0, 'Copy time'])
        for k,v in data_dict.items():
            print(k + " " + str(len(v)))
        nvcomp_df = pd.DataFrame(data_dict)
        print(nvcomp_df)
        nvcomp_df['Compression Ratio'] = nvcomp_df['Uncompressed Size'] / nvcomp_df['Compressed Size']
    #    nvcomp_df['Compression Runtime'] = nvcomp_df['Uncompressed Size'] / nvcomp_df['Compression Throughput']
    #    nvcomp_df['Compression Runtime'] = nvcomp_df['Computation Time'] + nvcomp_df['Copy Time']
        nvcomp_df = nvcomp_df.sort_values(by=['Chkpt ID', 'Approach'])
        nvcomp_df = nvcomp_df[['Scenario', 'Approach','Rank','Chkpt ID','Number of Chkpts','Chunk Size','Uncompressed Size','Compressed Size','Max Memory Usage','Compression Throughput','Compression Ratio','Compression Runtime']]
        print(nvcomp_df.columns)
        print(nvcomp_df)
    
        if args.output != '':
            if os.path.isfile(args.output[0]):
                nvcomp_df.to_csv(args.output[0], mode='a', index=False, header=False)
            else:
                nvcomp_df.to_csv(args.output[0], index=False)
    
    elif args.type[0] == 'dedup':
        print("Deduplication files")
        frames = []
        for directory in directories:
            print(directory)
            name = directory.split('/')
            scale = ''.join(filter(str.isdigit, name[1]))
            for root,dirs,files in os.walk(directory):
                frame_list = []
                for file in files:
                    if not file.endswith("size.csv"):
                        if not file.endswith("timing.csv"):
                            split_str = file.split('.')
                            index = 0
                            for i in range(len(split_str)):
                                substr = split_str[i]
                                if 'Rank' in substr:
                                    index = i
                                    break
                            temp_df = pd.read_csv(directory+'/'+file)
                            temp_df['Rank'] = int(split_str[index][split_str[index].index('Rank')+4:])
                            frames.append(temp_df)
        mapping = {'Full': 0, 'Basic': 1, 'List': 2, 'TreeLowOffset': 3, 'TreeLowRoot': 4, 0:5, 1:6, 2:7, 3:8, 4:9, 5:10, 6:11, 7:12, 8:13, 9:14, 10:15, 11:16, 12:17, 13:18, 14:19, 15:20, 16:21, 17:22, 18:23, 19:24, 20:25}
        dedup_df = pd.concat(frames)
        
        if args.scenario != '':
            dedup_df['Scenario'] = args.scenario[0]
        dedup_df['Number of Chkpts'] = args.num_chkpts[0]
        print(dedup_df)
        dedup_df = dedup_df.sort_values(by=['Chkpt ID', 'Approach'], key=lambda o: o.apply(lambda x: mapping[x]))
        dedup_df['Compression Ratio'] = dedup_df['Uncompressed Size'] / (dedup_df['Data Size'] + dedup_df['Metadata Size'])
    #    dedup_df['Compression Runtime'] = dedup_df['Setup Time'] + dedup_df['Comparison Time'] + dedup_df['Gather Time']
        dedup_df['Compression Runtime'] = dedup_df['Setup Time'] + dedup_df['Comparison Time'] + dedup_df['Gather Time'] + dedup_df['Write Time']
        dedup_df['Compression Throughput'] = dedup_df['Uncompressed Size'] / dedup_df['Compression Runtime']
        dedup_df = dedup_df.rename(columns={'Max GPU Memory (B)': 'Max Memory Usage'})
        print(dedup_df[['Approach', 'Comparison Time', 'Gather Time', 'Write Time']].to_string())
        clean_df = dedup_df[['Scenario', 'Approach','Rank','Chkpt ID','Number of Chkpts','Chunk Size','Uncompressed Size','Compressed Size','Max Memory Usage','Compression Throughput','Compression Ratio','Compression Runtime']]
        print(clean_df)
        if args.output != '':
            if os.path.isfile(args.output[0]):
                clean_df.to_csv(args.output[0], mode='a', index=False, header=False)
            else:
                clean_df.to_csv(args.output[0], index=False)


