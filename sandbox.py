import gymnasium as gym
import numpy as np
from random import sample
import pandas as pd
from utilities import split_in_blocks, float_to_bits, int_to_bits, read_timestamps, bits_to_int, unsigned_to_signed, bits_to_int_unsigned, bits_to_float
from transform_primitives import delta_of_delta
from compression_primitives import bitmask, trailing_zero
from constants import parameters_dict
import math
import struct

class CompressionEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), high=np.array([5, 5, 4, 4, 4, 7, 1, 1, 1]), shape=(9,),  dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=np.array([0 for i in range(34 * 8)] ), high=np.array([255 for i in range(34 * 8)]), shape=(34 * 8,),  dtype=np.float32)
        self.idx = 0
        self.timeseries = ['monthly-beer-production.csv', 'monthly-housing.csv', 'Twitter_volume_AMZN.csv', 'nyc_taxi.csv', 'network.csv', 'cpu_utilization.csv', 'art-price.csv', 'Gold.csv', 'Electric_Production.csv', 'daily-temperatures.csv', 'oil.csv', 'transactions.csv']
        self.timestamps = []
        self.timestamps_ratio = 0
        self.metrics = []
        self.paramters = parameters_dict
        self.last_2_metric = []
        self.compressed_block = ''
        self.get_ts()
        

    def get_ts(self):
        my_ts = sample(self.timeseries, 1)
        #my_ts = ['monthly-housing.csv']
        df = pd.read_csv(my_ts[0])
        print(my_ts[0])
        self.timestamps = read_timestamps(df)
        self.timestamps_ratio, compression = self.compress_timestamps(self.timestamps)
        
        self.metrics = (df.iloc[:, 1]).astype(np.float64).values
        self.metrics[np.isnan(self.metrics)==True] = 0
        self.timestamps[np.isnan(self.timestamps)==True] = 0
        self.last_2_metric = self.metrics[:2]
        self.metrics = split_in_blocks(self.metrics[2:])

        self.state = self.metrics[0]

    def compress_timestamps(self, timestamps):
        sum = 192
        header = int_to_bits(len(timestamps) - 2)
        data = int_to_bits(timestamps[0]) + int_to_bits(timestamps[1])
        last_2 = [timestamps[0], timestamps[1]]

        for i in range(2, len(timestamps)):
            val = delta_of_delta(last_2[0], last_2[1], timestamps[i])
            if val == 0:
                header += '0'
                sum += 1
            else:
                header += '1'
                sum += 1
                if val > 0:
                    val -= 1
                if -3 <= val <= 4:
                    data += '1'
                    sum += 4
                    data += int_to_bits(val)[-3:]
                elif -31 <= val <= 32:
                    data += '01'
                    sum += 8
                    data += int_to_bits(val)[-6:]
                else:
                    ct_bitmask, compressed_bitmask = bitmask(int_to_bits(val), 0)
                    ct_trailing_zeros, compressed_trailing = trailing_zero(int_to_bits(val))

                    if ct_bitmask < ct_trailing_zeros:
                        data += '001'
                        sum += ct_bitmask + 3
                        data += compressed_bitmask
                    else:
                        data += '000'
                        sum += ct_trailing_zeros + 3
                        data += compressed_trailing
            last_2 = [last_2[1], timestamps[i]]
        
        return (len(timestamps) * 64 / sum), header + data

    def decompress_timestamps(self, compressed):
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


    def bytes(self, vec):
        res = []
        for x in vec:
            y = float_to_bits(x)
            for start in range(0, 64, 8):
                byte = y[start:start+8]
                byte = int(byte, 2)
                res.append(byte)
        return np.asarray(res).astype(np.float32)

    def decompress_metrics(self, compressed_block):
        params = compressed_block[:21]
        trans1 = bits_to_int_unsigned(params[:3])
        trans2 = bits_to_int_unsigned(params[3:6])
        compress1 = bits_to_int_unsigned(params[6:9])
        compress2 = bits_to_int_unsigned(params[9:12])
        compress3 = bits_to_int_unsigned(params[12:15])
        offByteShift1 = bits_to_int_unsigned(params[15:18])
        offByteShift2 = bits_to_int_unsigned(params[18:19])
        offByteShift3 = bits_to_int_unsigned(params[19:20])
        offBitmask = bits_to_int_unsigned(params[20:21])
        encoding = {
            '010': (0, 0),
            '011': (0, 1),
            '100': (0, 2),
            '101': (1, 0),
            '110': (1, 1),
            '111': (1, 2),
            '001': (-1, -1),
            '000': (3, 3)
        }

        transVec = [self.paramters[7][trans1], self.paramters[7][trans2]]
        compressVec = [self.paramters[4][compress1], self.paramters[5][compress2], self.paramters[6][compress3]]

        proper_block = compressed_block[21:]

        orig_block = []
        prev = self.last_2_metric
        step = 0

        while step < len(proper_block):
            encoding_type = proper_block[step:step+3]
            crt_params = encoding[encoding_type]

            if crt_params == (3, 3):
                idx = int(proper_block[step+3])
                orig_block.append(transVec[idx](prev[0], prev[1], 0))
                step += 4
            elif crt_params == (-1, -1):
                orig_block.append(bits_to_float(proper_block[step+3:step+67]))
                step += 67
            else:
                idx1, idx2 = crt_params
                if compressVec[idx2][0].__name__ == 'bitmask':
                    mask = proper_block[step + 3: step + 11 - offBitmask]
                    val = ''
                    step = step + 11 - offBitmask
                    for j in mask:
                        if j == '0':
                            val += '00000000'
                        else:
                            # if len(proper_block[step:step + 8]) != 8:
                            #     print('Error here', self.idx)
                            val += proper_block[step:step + 8]
                            step += 8
                    val += (offBitmask * 8) * '0'
                    # if len(val) != 64:
                    #     print(val, len(val), compressVec[idx2][0].__name__, mask, offBitmask)
                    val = bits_to_float(val)
                    orig_block.append(transVec[idx1](prev[0], prev[1], val))
                elif compressVec[idx2][0].__name__ == 'offset':
                    offset = offByteShift1
                    match compressVec[idx2][1]:
                        case 1:
                            offset = offByteShift1
                        case 2:
                            offset = offByteShift1 - offByteShift2
                        case 3:
                            offset = offByteShift1 - offByteShift2 - offByteShift3
                    leading_zeros = 64 - offset * 8 - compressVec[idx2][1] * 7
                    val = '0' * leading_zeros + proper_block[step+3:step+3+compressVec[idx2][1] * 7] + '0' * (offset * 8)
                    # if len(proper_block[step+3:step+3+compressVec[idx2][1] * 7]) != compressVec[idx2][1] * 7:
                    #     print('Error here', self.idx)
                    # if len(val) != 64:
                    #     print(val, len(val), compressVec[idx2][0].__name__, offset, leading_zeros)
                    value = bits_to_float(val)
                    value = transVec[idx1](prev[0], prev[1], value)
                    orig_block.append(value)
                    step = step + 3 + compressVec[idx2][1] * 7
                else:
                    ct_zero_bytes = int(proper_block[step+3:step+6], 2)
                    ct_non_zero_bytes = int(proper_block[step+6:step+9], 2) + 1

                    val = 8 * (8 - ct_zero_bytes - ct_non_zero_bytes) * '0'

                    for j in range(ct_non_zero_bytes):
                        crt_byte = proper_block[step+9+j*8:step+9+(j+1)*8]
                        # if len(crt_byte) != 8:
                        #         print('Error here', self.idx)
                        val += crt_byte
                    val += ct_zero_bytes * 8 * '0'
                    step += 9 + ct_non_zero_bytes * 8
                    # if len(val) != 64:
                    #     print(val, len(val), compressVec[idx2][0].__name__, ct_zero_bytes, ct_non_zero_bytes)
                    val = bits_to_float(val)
                    val = transVec[idx1](prev[0], prev[1], val)
                    #print(val, prev[0], prev[1], val, transVec[idx1])
                    
                    orig_block.append(val)
            #print(orig_block[-1])
            prev = [prev[1], orig_block[-1]]
        return orig_block


    def step(self, action):
        params = np.round(action).astype(int)
        last_2_metric = self.last_2_metric
        bits_count = self.compress(params)

        ###################
        self.last_2_metric = last_2_metric
        decompressed = self.decompress_metrics(self.write_params(action) + self.compressed_block)

        for i in range(len(decompressed)):
            if self.state[i] != decompressed[i]:
                print('Error', i, self.state[i], decompressed[i], self.idx)
                break

        ###################

        reward = self.evaluate(self.state, bits_count + 21)
        self.idx += 1
        if self.idx >= len(self.metrics):
            state = np.pad(self.state, (0, 32 - len(self.state)), 'constant', constant_values=(-1e14, ))
            return self.bytes(np.concatenate([self.last_2_metric, state])), reward, True, False, {'block': self.compressed_block, 'params': self.write_params(action)}

        self.state = self.metrics[self.idx]

        state = np.pad(self.state, (0, 32 - len(self.state)), 'constant', constant_values=(-1e14, ))
        return self.bytes(np.concatenate([self.last_2_metric, state])), reward, False, False, {'block': self.compressed_block, 'params': self.write_params(action)}

    def write_params(self, action):
        params = np.round(action).astype(int)
        params_bits = ''
        for i in range(6):
            params_bits += int_to_bits(params[i])[-3:]
        for i in range(6, 9):
            params_bits += int_to_bits(params[i])[-1:]
        return params_bits

    def reset(self, seed=None):
        self.get_ts()
        return self.bytes(np.concatenate([self.last_2_metric, self.state])), {}

    def evaluate(self, original, compressed):
        val = (compressed / (len(original) * 64))
        return -math.tan(val) / 1.2 + 0.6

    def compress(self, params):
        self.compressed_block = ''
        transTypes = [self.paramters[1][params[0]], self.paramters[2][params[1]]]
        compressTypes = [self.paramters[4][params[2]], self.paramters[5][params[3]], self.paramters[6][params[4]]]
        #print(transTypes, compressTypes)
        encoding = {
            (0, 0): '010',
            (0, 1): '011',
            (0, 2): '100',
            (1, 0): '101',
            (1, 1): '110',
            (1, 2): '111',
            (-1, -1): '001',
            (3, 3): '000'
        }
        offByteShift1 = params[5]
        offByteShift2 = params[6]
        offByteShift3 = params[7]
        offBitmask = params[8] 
        #endian = params[10]

        sum = 0

        for i in range(len(self.state)):
            nr = self.state[i]
            opt = 64
            opt_compressed = float_to_bits(nr)
            idx_op1 = idx_op2 = -1

            last_2 = self.last_2_metric
            val = 0
            for (it, transformer) in enumerate(transTypes):
                val = transformer(last_2[0], last_2[1], nr)

                if val == 0 and struct.pack('d', val) == struct.pack('d', 0.0):
                    opt = 4
                    if it == 0:
                        opt_compressed = '0'
                    else:
                        opt_compressed = '1'
                    idx_op1 = idx_op2 = 3
                    continue

                val = float_to_bits(val)

                for (j, (compresser, param)) in  enumerate(compressTypes):
                    ct_bits = 64
                    if compresser.__name__ == 'offset' and param == 1:
                        ct_bits, compressed = compresser(val, offByteShift1, 1)
                    elif compresser.__name__ == 'offset' and param == 2:
                        ct_bits, compressed = compresser(val, offByteShift1 - offByteShift2, 2)
                    elif compresser.__name__ == 'offset' and param == 3:
                        ct_bits, compressed = compresser(val, offByteShift1 - offByteShift2 - offByteShift3, 3)
                    elif compresser.__name__ == 'bitmask':
                        ct_bits, compressed = compresser(val, offBitmask)
                    elif compresser.__name__ == 'trailing_zero':
                        ct_bits, compressed = compresser(val)
                        
                    if ct_bits < opt:
                        opt = ct_bits
                        opt_compressed = compressed
                        idx_op1 = it
                        idx_op2 = j
                        
            self.last_2_metric = [last_2[1], self.state[i]]
            sum += opt + 3
            self.compressed_block += encoding[(idx_op1, idx_op2)] + opt_compressed
        return sum


    def render(self):
        #useless
        pass

    def close(self):
        #useless
        pass

env = CompressionEnv()
action = env.action_space.sample()
state, reward, done, truncated, info = env.step(action)

sum = reward
ct = 1
maxi = 0
while not done:
    action = env.action_space.sample()
    state, reward, done, truncarted, info = env.step(action)
    maxi = max(maxi, reward)
    sum += reward
    ct += 1
print(sum / ct, maxi, ct)