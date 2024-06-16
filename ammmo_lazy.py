from constants import parameters_dict
from utilities import read_timestamps, float_to_bits, int_to_bits, bits_to_int, bits_to_float, bits_to_int_unsigned, unsigned_to_signed
from transform_primitives import delta, rev_delta, rev_delta_of_delta, xor, delta_of_delta, delta_xor, delta_inverse, rev_delta_inverse, delta_of_delta_inverse, rev_delta_of_delta_inverse, xor_inverse, delta_xor_inverse
from compression_primitives import offset, trailing_zero, bitmask
import pandas as pd
import numpy as np
from time import time

#timeseries = ['grok_asg_anomaly.csv', 'occupancy_t4013.csv','Sunspots.csv', 'monthly-beer-production.csv', 'monthly-housing.csv', 'cpu_utilization.csv', 'art-price.csv', 'Gold.csv', 'Electric_Production.csv', 'daily-temperatures.csv', 'oil.csv', 'rogue_agent_key_updown.csv']
#timeseries = ['exchange-2_cpc_results.csv', 'ec2_request_latency.csv', 'cpu_utilization_asg.csv', 'ambient_temperature.csv'] + timeseries
#timeseries = ['2014_apple_stock.csv', 'finance-charts-apple.csv', 'Nominal and Real Fed Funds Rate.csv']
timeseries = ['grok_asg_anomaly.csv', 'occupancy_t4013.csv','Sunspots.csv', 'monthly-beer-production.csv', 'monthly-housing.csv', 'cpu_utilization.csv', 'art-price.csv', 'Gold.csv', 'Electric_Production.csv', 'daily-temperatures.csv', 'oil.csv', 'rogue_agent_key_updown.csv']
timeseries = timeseries + ['AIZ_stocks.csv', 'cpu_utilization_77.csv', 'nz_weather.csv', 'occupancy_6005.csv', 'load_balancer_spikes.csv', '2014_apple_stock.csv']

# for i in range(10):
    #sample randomly at each run 5 series from the timeseries
test = np.random.choice(timeseries, 5, replace=False)
possibilities = parameters_dict[3][0]
transTypes = [parameters_dict[1][2], parameters_dict[1][5], parameters_dict[1][0]]
encodings = ['00', '01', '10', '11']

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
    return orig_timestamps, size

def compress_metrics(metrics):
    data = float_to_bits(metrics[0]) + float_to_bits(metrics[1])
    last_2 = [metrics[0], metrics[1]]
    header = ''
    for i in range(2, len(metrics)):
        nr = metrics[i]
        opt = 100
        opt_compressed = ''
        idx_op = -1

        val = 0
        if nr == last_2[1]:
            header += '0'
            last_2 = [last_2[1], nr]
            continue
            
        header += '1'

        for (j, (transformer, compresser, param)) in  enumerate(possibilities):
            val = transTypes[transformer](last_2[0], last_2[1], nr)
            val = float_to_bits(val)
            if compresser.__name__ == 'bitmask':
                ct_bits, compressed = compresser(val, 0)
            elif compresser.__name__ == 'trailing_zero':
                ct_bits, compressed = compresser(val)
                
            if ct_bits < opt:
                opt = ct_bits
                opt_compressed = compressed
                idx_op = j
                    
        last_2 = [last_2[1], metrics[i]]
        data += encodings[idx_op] + opt_compressed
    data = header + data
    return data

def decompress_metrics(compressed_block, size):
    header = compressed_block[:size]
    transTypes = [parameters_dict[4][2], parameters_dict[4][5], parameters_dict[4][0]]
    encodings = {
        '00': 0,
        '01': 1,
        '10': 2,
        '11': 3
    }
    possibilities = parameters_dict[3][0]
    offBitmask = 0
    proper_block = compressed_block[size:]
    first, second = proper_block[:64], proper_block[64:128]
    first, second = bits_to_float(first), bits_to_float(second)

    orig_block = [first, second]
    prev = [first, second]
    step = 128

    for x in header:
        if x == '0':
            orig_block.append(prev[1])
            prev = [prev[1], orig_block[-1]]
        else:
            encoding = proper_block[step:step+2]
            idx = encodings[encoding]
            step += 2
            transIdx, compresser, mode = possibilities[idx]

            if compresser.__name__ == 'bitmask':
                mask = proper_block[step: step + 8 - offBitmask]
                val = ''
                step = step + 8 - offBitmask
                for j in mask:
                    if j == '0':
                        val += '00000000'
                    else:
                        val += proper_block[step:step + 8]
                        step += 8
                val += (offBitmask * 8) * '0'
                val = bits_to_float(val)
                orig_block.append(transTypes[transIdx](prev[0], prev[1], val))
            else:
                if proper_block[step:step+4] == '1111':
                    val = transTypes[transIdx](prev[0], prev[1], 0.0)
                    step = step + 4
                    orig_block.append(val)
                else:
                    ct_zero_bytes = int(proper_block[step:step+3], 2)
                    ct_non_zero_bytes = int(proper_block[step+3:step+6], 2) + 1

                    val = 8 * (8 - ct_zero_bytes - ct_non_zero_bytes) * '0'

                    for j in range(ct_non_zero_bytes):
                        crt_byte = proper_block[step+6+j*8:step+6+(j+1)*8]
                        val += crt_byte
                    val += ct_zero_bytes * 8 * '0'
                    step += 6 + ct_non_zero_bytes * 8
                    val = bits_to_float(val)
                    val = transTypes[transIdx](prev[0], prev[1], val)
                    
                    orig_block.append(val)
            
            prev = [prev[1], orig_block[-1]]

    return orig_block



compress_time = 0
decompress_time = 0
count = 0


for x in test:
    sample = x
    print(sample)

    df = pd.read_csv(f'timeseries/{sample}')
    timestamps = read_timestamps(df)
    metrics = (df.iloc[:, 1]).astype(np.float64).values

    metrics[np.isnan(metrics)==True] = 0
    timestamps[np.isnan(timestamps)==True] = 0

    count += len(metrics)

    compress_time -= time()
    compressed_timestamps = compress_timestamps(timestamps)
    compressed_metrics = compress_metrics(metrics)
    compress_time += time()

    decompress_time -= time()
    decompressed_timestamps, size = decompress_timestamps(compressed_timestamps)
    decompressed_metrics = decompress_metrics(compressed_metrics, size)
    decompress_time += time()

    # for i in range(len(metrics)):
    #     if metrics[i] != decompressed_metrics[i]:
    #         print("Metrics error")
    #         break
    
    # for i in range(len(timestamps)):
    #     if timestamps[i] != decompressed_timestamps[i]:
    #         print("Timestamps error")
    #         break

#     total_comrpessoin = compressed_timestamps + compressed_metrics

#     # with open('AMMMO_LAZY_res.txt', 'a') as f:
#     #     f.write(sample)
#     #     f.write(' ')
#     #     f.write(f'Compresia timestamps: {str(len(timestamps) * 64 / len(compressed_timestamps))}, Compresia metrics: {str(len(metrics) * 64 / len(compressed_metrics))}, Compresia totala: {str((len(timestamps) * 64 + len(metrics) * 64) / len(total_comrpessoin))}')
#     #     f.write('\n')

# print("Compression time: ", compress_time / count * 10000)
# print("Decompression time: ", decompress_time / count * 10000)
# print(count)