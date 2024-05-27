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
        self.action_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0]), high=np.array([3, 5, 5, 5, 7, 1, 1, 5]), shape=(8,),  dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=np.array([0 for i in range(34 * 8)] ), high=np.array([1 for i in range(34 * 8)]), shape=(34 * 8,),  dtype=np.float32)
        self.idx = 0
        self.timeseries = ['grok_asg_anomaly.csv', 'occupancy_t4013.csv','Sunspots.csv', 'monthly-beer-production.csv', 'monthly-housing.csv', 'cpu_utilization.csv', 'art-price.csv', 'Gold.csv', 'Electric_Production.csv', 'daily-temperatures.csv', 'oil.csv', 'rogue_agent_key_updown.csv']
        self.timestamps = []
        self.timestamps_ratio = 0
        self.metrics = []
        self.paramters = parameters_dict
        self.last_2_metric = []
        self.compressed_block = ''
        self.get_ts()
        

    def get_ts(self):
        my_ts = sample(self.timeseries, 1)
        #print(my_ts[0])
        df = pd.read_csv(my_ts[0])
        self.timestamps = read_timestamps(df)
        self.timestamps_ratio, compression = self.compress_timestamps(self.timestamps)
        self.idx = 0
        
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
                res.append(byte / 255)
        return np.asarray(res).astype(np.float32)

    def decompress_metrics(self, compressed_block):
        params = compressed_block[:19]
        header = compressed_block[19:51]
        majorMode = bits_to_int_unsigned(params[:2])
        transTypes = [self.paramters[4][bits_to_int_unsigned(params[2:5])], self.paramters[4][bits_to_int_unsigned(params[5:8])], self.paramters[4][bits_to_int_unsigned(params[8:11])]]
        encodings = {
            '00': 0,
            '01': 1,
            '10': 2,
            '11': 3
        }
        possibilites = self.paramters[3][majorMode]
        offByteShift1 = bits_to_int_unsigned(params[11:14])
        offByteShift2 = bits_to_int_unsigned(params[14:15])
        offByteShift3 = bits_to_int_unsigned(params[15:16])
        offBitmask = bits_to_int_unsigned(params[16:19])
        proper_block = compressed_block[51:]

        orig_block = []
        prev = self.last_2_metric
        step = 0

        for x in header:
            if x == '0':
                orig_block.append(prev[1])
                prev = [prev[1], orig_block[-1]]
            else:
                encoding = proper_block[step:step+2]
                idx = encodings[encoding]
                step += 2
                transIdx, compresser, mode = possibilites[idx]

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
                elif compresser.__name__ == 'offset':
                    offset = offByteShift1
                    match mode:
                        case 1:
                            offset = offByteShift1
                        case 2:
                            offset = offByteShift1 - offByteShift2
                        case 3:
                            offset = offByteShift1 - offByteShift2 - offByteShift3
                    leading_zeros = 64 - offset * 8 - mode * 7
                    val = '0' * leading_zeros + proper_block[step:step+mode * 7] + '0' * (offset * 8)
                    value = bits_to_float(val)
                    value = transTypes[transIdx](prev[0], prev[1], value)
                    orig_block.append(value)
                    step = step + mode * 7
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


    def step(self, action):
        params = np.round(action).astype(int)
        prev_2 = self.last_2_metric
        bits_count = self.compress(params)

        # ###########
        # #decompress
        # self.last_2_metric = prev_2
        # orig_block = self.decompress_metrics(self.compressed_block)

        # for i in range(32):
        #     if orig_block[i] != self.state[i]:
        #         print('Error')
        #         print(orig_block[i], self.state[i])
        # self.last_2_metric = [self.state[-2], self.state[-1]]
        # ###########

        reward = self.evaluate(self.state, bits_count)
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
        params_bits += int_to_bits(params[0])[-2:]
        for i in range(4):
            params_bits += int_to_bits(params[i+1])[-3:]
        params_bits += int_to_bits(params[5])[-1:]
        params_bits += int_to_bits(params[6])[-1:]
        params_bits += int_to_bits(params[7])[-3:]
        return params_bits

    def reset(self, seed=None):
        self.get_ts()
        return self.bytes(np.concatenate([self.last_2_metric, self.state])), {}

    def evaluate(self, original, compressed):
        val = (compressed / (len(original) * 64))
        return -math.tan(val) / 1.3 + 0.8

    def compress(self, params):
        self.compressed_block = ''
        header = self.write_params(params)
        majorMode = params[0]
        transTypes = [self.paramters[1][params[1]], self.paramters[1][params[2]], self.paramters[1][params[3]]]
        offByteShift1 = params[4]
        offByteShift2 = params[5]
        offByteShift3 = params[6]
        offBitmask = params[7] 
        #endian = params[10]
        posibilities = self.paramters[3][majorMode]
        encodings = ['00', '01', '10', '11']

        for i in range(len(self.state)):
            nr = self.state[i]
            opt = 100
            opt_compressed = ''
            idx_op = -1

            last_2 = self.last_2_metric
            val = 0
            if nr == last_2[1]:
                header += '0'
                self.last_2_metric = [last_2[1], nr]
                continue
                
            header += '1'

            for (j, (transformer, compresser, param)) in  enumerate(posibilities):
                val = transTypes[transformer](last_2[0], last_2[1], nr)
                val = float_to_bits(val)
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
                    idx_op = j
                        
            self.last_2_metric = [last_2[1], self.state[i]]
            self.compressed_block += encodings[idx_op] + opt_compressed
        self.compressed_block = header + self.compressed_block
        return len(self.compressed_block)


    def render(self):
        #useless
        pass

    def close(self):
        #useless
        pass