from constants import parameters_dict
from utilities import read_timestamps, float_to_bits, int_to_bits, bits_to_int, bits_to_float, bits_to_int_unsigned
from transform_primitives import delta, rev_delta, rev_delta_of_delta, xor, delta_of_delta, delta_xor, delta_inverse, rev_delta_inverse, delta_of_delta_inverse, rev_delta_of_delta_inverse, xor_inverse, delta_xor_inverse
from compression_primitives import offset, trailing_zero, bitmask
import pandas as pd
import numpy as np


timeseries = ['grok_asg_anomaly.csv', 'occupancy_t4013.csv','Sunspots.csv', 'monthly-beer-production.csv', 'monthly-housing.csv', 'cpu_utilization.csv', 'art-price.csv', 'Gold.csv', 'Electric_Production.csv', 'daily-temperatures.csv', 'oil.csv', 'rogue_agent_key_updown.csv']
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


# for x in timeseries:
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

#     with open('AMMMO_LAZY_res.txt', 'a') as f:
#         f.write(sample)
#         f.write(' ')
#         f.write(f'Compresia timestamps: {str(len(timestamps) * 64 / len(compressed_timestamps))}, Compresia metrics: {str(len(metrics) * 64 / len(compressed_metrics))}, Compresia totala: {str((len(timestamps) * 64 + len(metrics) * 64) / len(total_comrpessoin))}')
#         f.write('\n')
