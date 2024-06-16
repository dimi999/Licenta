# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:36:14 2024

@author: Andrei
"""

import pandas as pd 
import numpy as np
import sys
from dahuffman import HuffmanCodec
from utilities import read_timestamps, get_bytes

from transform_primitives import delta_of_delta, delta

def compress_huffman():
    timeseries = ['grok_asg_anomaly.csv', 'occupancy_t4013.csv','Sunspots.csv', 'monthly-beer-production.csv', 'monthly-housing.csv', 'cpu_utilization.csv', 'art-price.csv', 'Gold.csv', 'Electric_Production.csv', 'daily-temperatures.csv', 'oil.csv', 'rogue_agent_key_updown.csv']
    for x in timeseries:
        df = pd.read_csv(f'timeseries/{x}')
        timestamps = read_timestamps(df)
        metrics = (df.iloc[:, 1]).astype(np.float64).values

        metrics[np.isnan(metrics) == True] = 0
        timestamps[np.isnan(timestamps) == True] = 0

        dict_timestamps = dict()
        dict_timestamps[timestamps[0]] = 1
        if timestamps[1] not in dict_timestamps:
            dict_timestamps[timestamps[1]] = 1
        else:
            dict_timestamps[timestamps[1]] += 1

        for i in range(2, len(timestamps)):
            val = delta_of_delta(timestamps[i-2], timestamps[i-1], timestamps[i])
            if val in dict_timestamps:
                dict_timestamps[val] += 1
            else:
                dict_timestamps[val] = 1
        
        codec_timestamps = HuffmanCodec.from_frequencies(dict_timestamps)

        dict_metrics = dict()
        dict_metrics[metrics[0]] = 1
        
        for i in range(1, len(metrics)):
            val = delta(0, metrics[i-1], metrics[i])
            if val in dict_metrics:
                dict_metrics[val] += 1
            else:
                dict_metrics[val] = 1
        
        codec_metrics = HuffmanCodec.from_frequencies(dict_metrics)

        delta_timestamps = [timestamps[0], timestamps[1]]

        for i in range(2, len(timestamps)):
            val = delta_of_delta(timestamps[i-2], timestamps[i-1], timestamps[i])
            delta_timestamps.append(val)
        
        delta_metrics = [metrics[0]]

        for i in range(1, len(metrics)):
            val = delta(0, metrics[i-1], metrics[i])
            delta_metrics.append(val)

        compressed_timestamps = codec_timestamps.encode(delta_timestamps)
        compressed_metrics = codec_metrics.encode(delta_metrics)

        size_of_timestamps = sys.getsizeof(dict_timestamps) * 8 + len(compressed_timestamps) * 8
        size_of_metrics = sys.getsizeof(dict_metrics) * 8 + len(compressed_metrics) * 8

        with open(f"Huffman_res.txt", "a") as f:
            compresion_ratio_timestamps = (len(timestamps) * 64) / size_of_timestamps
            compresion_ratio_metrics = (len(metrics) * 64) / size_of_metrics
            ratio = (len(timestamps) * 64 + len(metrics) * 64) / (size_of_timestamps + size_of_metrics)
            f.write(f"{x},  timestamps: {compresion_ratio_timestamps}, metrics: {compresion_ratio_metrics}, full ratio {ratio}\n")


def compress_huffman_block():
    timeseries = ['grok_asg_anomaly.csv', 'occupancy_t4013.csv','Sunspots.csv', 'monthly-beer-production.csv', 'monthly-housing.csv', 'cpu_utilization.csv', 'art-price.csv', 'Gold.csv', 'Electric_Production.csv', 'daily-temperatures.csv', 'oil.csv', 'rogue_agent_key_updown.csv']
    dict_timestamps = dict()
    dict_metrics = dict()
    for x in timeseries:
        df = pd.read_csv(f'timeseries/{x}')
        timestamps = read_timestamps(df)
        metrics = (df.iloc[:, 1]).astype(np.float64).values

        metrics[np.isnan(metrics) == True] = 0
        timestamps[np.isnan(timestamps) == True] = 0

        delta_timestamps = [timestamps[0], timestamps[1]]
        for i in range(2, len(timestamps)):
            val = delta_of_delta(timestamps[i-2], timestamps[i-1], timestamps[i])
            delta_timestamps.append(val)
        
        delta_metrics = [metrics[0]]
        for i in range(1, len(metrics)):
            val = delta(0, metrics[i-1], metrics[i])
            delta_metrics.append(val)

        #get bytes of timestamps
        for x in delta_timestamps:
            bytes = get_bytes(x)

            for byte in bytes:
                if byte in dict_timestamps:
                    dict_timestamps[byte] += 1
                else:
                    dict_timestamps[byte] = 1
        
        #get bytes of metrics
        for x in delta_metrics:
            bytes = get_bytes(x)

            for byte in bytes:
                if byte in dict_metrics:
                    dict_metrics[byte] += 1
                else:
                    dict_metrics[byte] = 1

    codec_timestamps = HuffmanCodec.from_frequencies(dict_timestamps)
    codec_metrics = HuffmanCodec.from_frequencies(dict_metrics)

    print(codec_metrics.print_code_table())
    print(codec_timestamps.print_code_table())

    for x in timeseries:
        df = pd.read_csv(f'timeseries/{x}')
        timestamps = read_timestamps(df)
        metrics = (df.iloc[:, 1]).astype(np.float64).values

        metrics[np.isnan(metrics) == True] = 0
        timestamps[np.isnan(timestamps) == True] = 0

        delta_timestamps = [timestamps[0], timestamps[1]]
        for i in range(2, len(timestamps)):
            val = delta_of_delta(timestamps[i-2], timestamps[i-1], timestamps[i])
            delta_timestamps.append(val)
        
        delta_metrics = [metrics[0]]
        for i in range(1, len(metrics)):
            val = delta(0, metrics[i-1], metrics[i])
            delta_metrics.append(val)

        timestamps_bytes = []

        for i in range(len(delta_timestamps)):
            bytes = get_bytes(delta_timestamps[i])
            for byte in bytes:
                timestamps_bytes.append(byte)
        
        metrics_bytes = []

        for i in range(len(delta_metrics)):
            bytes = get_bytes(delta_metrics[i])
            for byte in bytes:
                metrics_bytes.append(byte)

        compressed_timestamps = codec_timestamps.encode(timestamps_bytes)
        compressed_metrics = codec_metrics.encode(metrics_bytes)

        size_of_timestamps = len(compressed_timestamps) * 8
        size_of_metrics = len(compressed_metrics) * 8

        # with open(f"Huffman_res_block.txt", "a") as f:
        #     compresion_ratio_timestamps = (len(timestamps) * 64) / size_of_timestamps
        #     compresion_ratio_metrics = (len(metrics) * 64) / size_of_metrics
        #     ratio = (len(timestamps) * 64 + len(metrics) * 64) / (size_of_timestamps + size_of_metrics)
        #     f.write(f"{x},  timestamps: {compresion_ratio_timestamps}, metrics: {compresion_ratio_metrics} ratio {ratio}\n")


def compress_huffman_block_individual():
    timeseries = ['grok_asg_anomaly.csv', 'occupancy_t4013.csv','Sunspots.csv', 'monthly-beer-production.csv', 'monthly-housing.csv', 'cpu_utilization.csv', 'art-price.csv', 'Gold.csv', 'Electric_Production.csv', 'daily-temperatures.csv', 'oil.csv', 'rogue_agent_key_updown.csv']
    for x in timeseries:
        df = pd.read_csv(f'timeseries/{x}')
        timestamps = read_timestamps(df)
        dict_timestamps = dict()
        dict_metrics = dict()
        metrics = (df.iloc[:, 1]).astype(np.float64).values

        metrics[np.isnan(metrics) == True] = 0
        timestamps[np.isnan(timestamps) == True] = 0

        delta_timestamps = [timestamps[0], timestamps[1]]
        for i in range(2, len(timestamps)):
            val = delta_of_delta(timestamps[i-2], timestamps[i-1], timestamps[i])
            delta_timestamps.append(val)
        
        delta_metrics = [metrics[0]]
        for i in range(1, len(metrics)):
            val = delta(0, metrics[i-1], metrics[i])
            delta_metrics.append(val)


        metrics_bytes = []
        timestamps_bytes = []

        #get bytes of timestamps
        for y in delta_timestamps:
            bytes = get_bytes(y)

            for byte in bytes:
                timestamps_bytes.append(byte)
                if byte in dict_timestamps:
                    dict_timestamps[byte] += 1
                else:
                    dict_timestamps[byte] = 1
        
        #get bytes of metrics
        for y in delta_metrics:
            bytes = get_bytes(y)

            for byte in bytes:
                metrics_bytes.append(byte)
                if byte in dict_metrics:
                    dict_metrics[byte] += 1
                else:
                    dict_metrics[byte] = 1

        codec_timestamps = HuffmanCodec.from_frequencies(dict_timestamps)
        codec_metrics = HuffmanCodec.from_frequencies(dict_metrics)

        compressed_timestamps = codec_timestamps.encode(timestamps_bytes)
        compressed_metrics = codec_metrics.encode(metrics_bytes)

        size_of_timestamps = len(compressed_timestamps) * 8 + sys.getsizeof(dict_timestamps) * 8
        size_of_metrics = len(compressed_metrics) * 8 + sys.getsizeof(dict_metrics) * 8

        with open(f"Huffman_res_block_individual.txt", "a") as f:
            compresion_ratio_timestamps = (len(timestamps) * 64) / size_of_timestamps
            compresion_ratio_metrics = (len(metrics) * 64) / size_of_metrics
            ratio = (len(timestamps) * 64 + len(metrics) * 64) / (size_of_timestamps + size_of_metrics)
            f.write(f"{x},  timestamps: {compresion_ratio_timestamps}, metrics: {compresion_ratio_metrics} ratio {ratio}\n")
        
compress_huffman()
compress_huffman_block()
compress_huffman_block_individual()



