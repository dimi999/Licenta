from utilities import read_timestamps, float_to_bits, bits_to_float, int_to_bits, bits_to_int, unsigned_to_signed
import numpy as np
from transform_primitives import delta, delta_of_delta, rev_delta_of_delta, xor, delta_xor, rev_delta, xor_inverse, delta_inverse, delta_xor_inverse, rev_delta_inverse, delta_of_delta_inverse, rev_delta_of_delta_inverse  
from compression_primitives import offset, bitmask, trailing_zero
import dahuffman
import pandas as pd
from time import time

transformers = [delta, delta_of_delta, rev_delta_of_delta, xor, delta_xor, rev_delta]
inverse_transformers = [delta_inverse, delta_of_delta_inverse, rev_delta_of_delta_inverse, xor_inverse, delta_xor_inverse, rev_delta_inverse]
schemas = [(offset, i) for i in range(7)] + [(bitmask, i) for i in range(7)] + [(trailing_zero, 0)]

timeseries = ['grok_asg_anomaly.csv', 'occupancy_t4013.csv','Sunspots.csv', 'monthly-beer-production.csv', 'monthly-housing.csv', 'cpu_utilization.csv', 'art-price.csv', 'Gold.csv', 'Electric_Production.csv', 'daily-temperatures.csv', 'oil.csv', 'rogue_agent_key_updown.csv']
timeseries = timeseries + [ 
    "occupancy_6005.csv",
    "cpu_utilization_77.csv",
    "ambient_temperature.csv",
    "2014_apple_stock.csv",
    "load_balancer_spikes.csv",
]

#sample randomly at each run 5 series from the timeseries
test = np.random.choice(timeseries, 5, replace=False)


freq = dict()
for i in range(6):
    for j in range(15):
        freq[(i,j)] = 0

def compress_timestamps(timestamps):
    header = int_to_bits(len(timestamps) - 2)
    data = int_to_bits(timestamps[0]) + int_to_bits(timestamps[1])
    last_2 = [timestamps[0], timestamps[1]]

    for i in range(2, len(timestamps)):
        val = delta_of_delta(last_2[0], last_2[1], timestamps[i])
        if val == 0:
            header += '0'
        else:
            header += '1'
            if val > 0:
                val -= 1
            if -3 <= val <= 4:
                data += '1'
                data += int_to_bits(val)[-3:]
            elif -31 <= val <= 32:
                data += '01'
                data += int_to_bits(val)[-6:]
            else:
                ct_bitmask, compressed_bitmask = bitmask(int_to_bits(val), 0)
                ct_trailing_zeros, compressed_trailing = trailing_zero(int_to_bits(val))

                if ct_bitmask < ct_trailing_zeros:
                    data += '001'
                    data += compressed_bitmask
                else:
                    data += '000'
                    data += compressed_trailing
        last_2 = [last_2[1], timestamps[i]]
    
    return header + data

def decompress_timestamps(compressed):
    size = int(compressed[:64], 2)
    header = compressed[64:64+size]
    data = compressed[64+size:]

    orig_timestamps = []

    orig_timestamps.append(int(data[:64], 2))
    orig_timestamps.append(int(data[64:128], 2))

    last_2 = [orig_timestamps[0], orig_timestamps[1]]

    data_idx = 128

    for x in header:
        if x == '0':
            orig_timestamps.append(2 * last_2[1] - last_2[0])
        else:
            if data[data_idx] == '1':
                val = bits_to_int(data[data_idx+1:data_idx+4])
                if val >= 0:
                    val += 1
                orig_timestamps.append(val + 2 * last_2[1] - last_2[0])
                data_idx += 4
            else:
                if data[data_idx+1] == '1':
                    val = bits_to_int(data[data_idx+2:data_idx+8])
                    if val >= 0:
                        val += 1
                    orig_timestamps.append(val + 2 * last_2[1] - last_2[0])
                    data_idx += 8
                else:
                    val = 0
                    if data[data_idx+2] == '1':
                        last_bit = data_idx + 11
                        mask = data[data_idx+3:data_idx+11]
                        for bit in mask:
                            if bit == '0':
                                val <<= 8
                            else:
                                crt_byte = data[last_bit:last_bit+8]
                                val = (val << 8) + int(crt_byte, 2)
                                last_bit += 8
                        data_idx = last_bit + 8
                    else:
                        ct_zero_bytes = int(data[data_idx+3:data_idx+6], 2)
                        ct_non_zero_bytes = int(data[data_idx+6:data_idx+9], 2) + 1

                        for i in range(ct_non_zero_bytes):
                            crt_byte = data[data_idx+9+i*8:data_idx+9+(i+1)*8]
                            val = (val << 8) + int(crt_byte, 2)
                        val = (val << 8 * ct_zero_bytes)
                        data_idx += 9 + ct_non_zero_bytes * 8
                    val = unsigned_to_signed(val)
                    if val >= 0:
                        val += 1
                    orig_timestamps.append(val + 2 * last_2[1] - last_2[0])
                        
        last_2 = [last_2[1], orig_timestamps[-1]]
    return orig_timestamps

for name in timeseries:
    df = pd.read_csv(name)
    timestamps = read_timestamps(df)
    metrics = (df.iloc[:, 1]).astype(np.float64).values
    metrics[np.isnan(metrics)==True] = 0
    
    last_2 = [metrics[0], metrics[1]]

    for x in metrics[2:]:
        optimal = 100
        idx1, idx2 = 0, 0
        for (i, trans) in enumerate(transformers):
            for (j, schema) in enumerate(schemas):
                val = trans(last_2[0], last_2[1], x)
                val = float_to_bits(val)
                if schema[0].__name__ == 'offset':
                    ct, compressed = schema[0](val, schema[1], 3)
                elif schema[0].__name__ == 'bitmask':
                    ct, compressed = schema[0](val, schema[1])
                else:
                    ct, compressed = schema[0](val)
                if ct < optimal:
                    optimal = ct
                    idx1 = i
                    idx2 = j
        freq[(idx1, idx2)] += 1
        last_2 = [last_2[1], x]

codec = dahuffman.HuffmanCodec.from_frequencies(freq)
codec.print_code_table()
codes = dict()
rcodes = dict()

for x in codec.get_code_table():
    sz, val = codec.get_code_table()[x]
    val = bin(val)[2:]
    if len(val) < sz:
        val = '0' * (sz - len(val)) + val
    codes[x] = val
    rcodes[val] = x

compress_time = 0
decompress_time = 0
count = 0

for name in test:
    df = pd.read_csv(f'timeseries/{name}')
    timestamps = read_timestamps(df)
    metrics = (df.iloc[:, 1]).astype(np.float64).values
    timestamps[np.isnan(timestamps)==True] = 0
    metrics[np.isnan(metrics)==True] = 0

    count += len(metrics)

    last_2 = [metrics[0], metrics[1]]
    sum = 128
    compressed_block = float_to_bits(metrics[0]) + float_to_bits(metrics[1])

    compress_time -= time()

    compressed_timestamps = compress_timestamps(timestamps)

    for x in metrics[2:]:
        optimal = 100
        idx1, idx2 = 0, 0
        crt = ''

        for (i, trans) in enumerate(transformers):
            for (j, schema) in enumerate(schemas):
                val = trans(last_2[0], last_2[1], x)
                val = float_to_bits(val)
                if schema[0].__name__ == 'offset':
                    ct, compressed = schema[0](val, schema[1], 3)
                elif schema[0].__name__ == 'bitmask':
                    ct, compressed = schema[0](val, schema[1])
                else:
                    ct, compressed = schema[0](val)
                if ct < optimal:
                    optimal = ct
                    crt = compressed
                    idx1 = i
                    idx2 = j
        
        compressed_block += codes[(idx1, idx2)] + crt
        sum += len(codes[(idx1, idx2)])
        sum += optimal
        last_2 = [last_2[1], x]
    
    compress_time += time()

    decompress_time -= time()
    decompressed_timestamps = decompress_timestamps(compressed_timestamps)
    step = 128
    last_2 = [bits_to_float(compressed_block[:64]), bits_to_float(compressed_block[64:128])]
    orig_block = [last_2[0], last_2[1]]
    while step < len(compressed_block):
        crt_encoding = ''
        while crt_encoding not in rcodes:
            crt_encoding += compressed_block[step]
            step += 1
        idx1, idx2 = rcodes[crt_encoding]
        compresser, offsetParam = schemas[idx2]
        trans = inverse_transformers[idx1]

        if compresser.__name__ == 'bitmask':
            mask = compressed_block[step: step + 8 - offsetParam]
            val = ''
            step = step + 8 - offsetParam
            for j in mask:
                if j == '0':
                    val += '00000000'
                else:
                    val += compressed_block[step:step + 8]
                    step += 8
            val += (offsetParam * 8) * '0'
            val = bits_to_float(val)
            orig_block.append(trans(last_2[0], last_2[1], val))
        elif compresser.__name__ == 'offset':
            leading_zeros = 64 - offsetParam * 8 - 3 * 7
            val = '0' * leading_zeros + compressed_block[step:step+3 * 7] + '0' * (offsetParam * 8)
            value = bits_to_float(val)
            value = trans(last_2[0], last_2[1], value)
            orig_block.append(value)
            step = step + 3 * 7
        else:
            if compressed_block[step:step+4] == '1111':
                val = trans(last_2[0], last_2[1], 0.0)
                step = step + 4
                orig_block.append(val)
            else:
                ct_zero_bytes = int(compressed_block[step:step+3], 2)
                ct_non_zero_bytes = int(compressed_block[step+3:step+6], 2) + 1

                val = 8 * (8 - ct_zero_bytes - ct_non_zero_bytes) * '0'

                for j in range(ct_non_zero_bytes):
                    crt_byte = compressed_block[step+6+j*8:step+6+(j+1)*8]
                    val += crt_byte
                val += ct_zero_bytes * 8 * '0'
                step += 6 + ct_non_zero_bytes * 8
                val = bits_to_float(val)
                val = trans(last_2[0], last_2[1], val)
                
                orig_block.append(val)
        last_2 = [last_2[1], orig_block[-1]]
    decompress_time += time()

    # for i in range(len(metrics)):
    #     if metrics[i] != orig_block[i]:
    #         print("Metrics error", metrics[i], orig_block[i])
    #         break
    
    # for i in range(len(timestamps)):
    #     if timestamps[i] != decompressed_timestamps[i]:
    #         print("Timestamps error")
    #         break

    
    print(name, (len(metrics) * 64 + len(timestamps) * 64) / (sum + len(compressed_timestamps)))

print("Compression time: ", compress_time / count * 10000)
print("Decompression time: ", decompress_time / count * 10000)
print(count)