import random
import pandas as pd
import numpy as np
from utilities import read_timestamps, float_to_bits, int_to_bits, bits_to_int, bits_to_float, bits_to_int_unsigned
from transform_primitives import xor, delta_of_delta

timestamps = ['monthly-beer-production.csv', 'monthly-housing.csv', 'Twitter_volume_AMZN.csv', 'nyc_taxi.csv', 'network.csv', 'cpu_utilization.csv', 'art-price.csv', 'Electric_Production.csv', 'Gold.csv', 'Electric_Production.csv', 'daily-temperatures.csv', 'oil.csv', 'transactions.csv']

def compress_timestamps(timestamps):
    compressed_timestamps = ''
    compressed_timestamps += int_to_bits(timestamps[0])
    compressed_timestamps += int_to_bits(timestamps[1])

    for i in range(2, len(timestamps)):
        crt_val = delta_of_delta(timestamps[i - 2], timestamps[i - 1], timestamps[i])
        if crt_val == 0:
            compressed_timestamps += '0'
        elif -63 <= crt_val <= 64:
            compressed_timestamps += '10' + int_to_bits(crt_val)[-7:]  
        elif -255 <= crt_val <= 256:
            compressed_timestamps += '110' + int_to_bits(crt_val)[-9:]
        elif -2047 <= crt_val <= 2048:
            compressed_timestamps += '1110' + int_to_bits(crt_val)[-12:]
        else:
            compressed_timestamps += '1111' + int_to_bits(crt_val)
    return compressed_timestamps

def compress_metrics(metrics):
    compressed_metrics = ''
    compressed_metrics += float_to_bits(metrics[0])
    prev_lead_zeros = 0
    prev_trail_zeros = 0

    for i in range(1, len(metrics)):
        crt_val = xor(0, metrics[i - 1], metrics[i])
        leading_zeros = 0
        trailing_zeros = 0
        bitstring = float_to_bits(crt_val)
        sz = len(compressed_metrics)
        
        for bit in bitstring:
            if bit == '0':
                leading_zeros += 1
            else:
                break
        for bit in reversed(bitstring):
            if bit == '0':
                trailing_zeros += 1
            else:
                break
        
        if crt_val == 0:
            compressed_metrics += '0'
        else:
            compressed_metrics += '1'
            if leading_zeros >= prev_lead_zeros and trailing_zeros >= prev_trail_zeros:
                compressed_metrics += '0'
                compressed_metrics += bitstring[prev_lead_zeros:64 - prev_trail_zeros]
            else:
                compressed_metrics += '1'
                leading_zeros = min(leading_zeros, 31)
                compressed_metrics += int_to_bits(leading_zeros)[-5:]
                compressed_metrics += int_to_bits(64 - trailing_zeros - leading_zeros)[-6:]
                compressed_metrics += bitstring[leading_zeros:64 - trailing_zeros]
            prev_lead_zeros = leading_zeros
            prev_trail_zeros = trailing_zeros

    return compressed_metrics

def decompress_timestamps(compressed_timestamps):
    timestamps = []
    timestamps.append(bits_to_int(compressed_timestamps[:64]))
    timestamps.append(bits_to_int(compressed_timestamps[64:128]))
    i = 128

    while i < len(compressed_timestamps):
        if compressed_timestamps[i] == '0':
            timestamps.append(2 * timestamps[-1] - timestamps[-2])
            i += 1
        elif compressed_timestamps[i + 1] == '0':
            val = bits_to_int(compressed_timestamps[i + 2:i + 9])
            timestamps.append(val + 2 * timestamps[-1] - timestamps[-2])
            i += 9
        elif compressed_timestamps[i + 2] == '0':
            val = bits_to_int(compressed_timestamps[i + 3:i + 12])
            timestamps.append(val + 2 * timestamps[-1] - timestamps[-2])
            i += 12
        elif compressed_timestamps[i + 3] == '0':
            val = bits_to_int(compressed_timestamps[i + 4:i + 16])
            timestamps.append(val + 2 * timestamps[-1] - timestamps[-2])
            i += 16
        else:
            val = bits_to_int(compressed_timestamps[i + 4:i + 68])
            timestamps.append(val + 2 * timestamps[-1] - timestamps[-2])
            i += 68

    return timestamps


def decompress_metrics(compressed_metrics):
    metrics = []
    metrics.append(bits_to_float(compressed_metrics[:64]))
    i = 64
    prev_lead_zeros = 0
    prev_trail_zeros = 0

    while i < len(compressed_metrics):
        if compressed_metrics[i] == '0':
            metrics.append(metrics[-1])
            i += 1
        else:
            i += 1
            if compressed_metrics[i] == '0':
                x = bits_to_float('0' * prev_lead_zeros + compressed_metrics[i + 1: i + 65 - prev_trail_zeros - prev_lead_zeros] + '0' * prev_trail_zeros)
                metrics.append(xor(0, metrics[-1], x))
                pasi = 65 - prev_trail_zeros - prev_lead_zeros
                partial = compressed_metrics[i + 1: i + 65 - prev_trail_zeros - prev_lead_zeros]
                for j in partial:
                    if j == '1':
                        break
                    prev_lead_zeros += 1

                for j in reversed(partial):
                    if j == '1':
                        break
                    prev_trail_zeros += 1
                i += pasi
            else:
                leading_zeros = bits_to_int_unsigned(compressed_metrics[i + 1:i + 6])
                important_bits = bits_to_int_unsigned(compressed_metrics[i + 6:i + 12])
                x = bits_to_float('0' * leading_zeros + compressed_metrics[i + 12: i + 12 + important_bits] + '0' * (64 - leading_zeros - important_bits))
                metrics.append(xor(0, metrics[-1], x))
                prev_lead_zeros = leading_zeros
                prev_trail_zeros = 64 - leading_zeros - important_bits
                i += 12 + important_bits
    return metrics

df = pd.read_csv('Gold.csv')
timestamps = read_timestamps(df)
compressed_timestamps = compress_timestamps(timestamps)
print(len(timestamps) * 64 / len(compressed_timestamps))

# for x in timestamps:
#     sample = x
#     print(sample)

#     df = pd.read_csv(sample)
#     timestamps = read_timestamps(df)
#     metrics = (df.iloc[:, 1]).astype(np.float64).values

#     metrics[np.isnan(metrics)==True] = 0
#     timestamps[np.isnan(timestamps)==True] = 0

#     compressed_timestamps = compress_timestamps(timestamps)
#     compressed_metrics = compress_metrics(metrics)
#     total_comrpessoin = compressed_timestamps + compressed_metrics

#     with open('Gorilla_res.txt', 'a') as f:
#         f.write(sample)
#         f.write(' ')
#         f.write(str(len(timestamps) * 128 / len(total_comrpessoin)))
#         f.write('\n')





# original_timestamps = decompress_timestamps(compressed_timestamps)
# original_metrics = decompress_metrics(compressed_metrics)

# for i in range(len(original_timestamps)):
#     if original_timestamps[i] != timestamps[i]:
#         print(i, original_timestamps[i], timestamps[i])

# for i in range(len(original_metrics)):
#     if original_metrics[i] != metrics[i]:
#         print(i, original_metrics[i], metrics[i])