import struct
import numpy as np
import pandas as pd

def split_in_blocks(ts, block_size = 32):
    blocks = []
    
    for i in range(0, len(ts), block_size):
        if len(ts[i:i + block_size]) < block_size:
            blocks.append(np.concatenate([ts[i:], [0] * (block_size - len(ts[i:i + block_size]))]))
        else:
            blocks.append(ts[i:i + block_size])
    
    blocks = np.array(blocks)
    return blocks

def get_last_n(block, size=2):
    return block[-size:]

def float_to_bits(number):
    if number == 0 and number == struct.pack('d', 0.0):
        return '0' * 64
    packed_bytes = struct.pack('>d', number)
    packed_bytes = packed_bytes.hex()
    packed_bytes = int(packed_bytes, 16)
    binary_string = "{:08b}".format(packed_bytes)

    if len(binary_string) != 64:
        for i in range(64 - len(binary_string)):
            binary_string = '0' + binary_string    
    return binary_string

#Convert a tring of bits to the number it represents in 2 complement
def bits_to_int(bits):
    if bits[0] == '1':
        bits = bits[1:]
        bits = ''.join(['1' if x == '0' else '0' for x in bits])
        return -int(bits, 2) - 1
    else:
        return int(bits, 2)

def bits_to_int_unsigned(bits):
    return int(bits, 2)

def bits_to_float(bits):
    if len(bits) != 64:
        print(bits)

    bits = int(bits, 2)
    bits = struct.pack('>Q', bits)
    bits = struct.unpack('>d', bits)[0]
    return bits


def int_to_bits(number):
    if number == 0:
        return '0' * 64
    packed_bytes = struct.pack('>q', number)
    packed_bytes = packed_bytes.hex()
    packed_bytes = int(packed_bytes, 16)
    binary_string = "{:08b}".format(packed_bytes)

    if len(binary_string) != 64:
        for i in range(64 - len(binary_string)):
            binary_string = '0' + binary_string
            
    return binary_string

def unsigned_to_signed(number):
    if number <= 2 ** 63 - 1:
        return number
    return -1 * (2 ** 64 - number)

def count_bits(num):
     binary = bin(num)[2:]
     return len(binary)

def get_bytes(num):
    #return the value of the bytes of num
    bits = ''
    if isinstance(num, int):
        bits = int_to_bits(num)
    else:
        bits = float_to_bits(num)

    bytes = []
    for i in range(0, 64, 8):
        bytes.append(bits_to_int_unsigned(bits[i:i+8])) 
    
    return bytes

def double_xor(a, b):
    a = float_to_bits(a)
    b = float_to_bits(b)
    result = ''
    for i in range(64):
        if a[i] == b[i]:
            result += '0'
        else:
            result += '1'

    result = int(result, 2)
    result = struct.pack('>Q', result)
    result = struct.unpack('>d', result)[0]
    return result

def read_timestamps(df):
    try:
        timestamps = pd.to_datetime(df['DATE'], format='%Y-%m-%d').astype(np.int64).values // 1000000
    except:
        try:
            timestamps = pd.to_datetime(df['DATE'], format='%d-%m-%Y').astype(np.int64).values // 1000000
        except:
            try:
                timestamps = pd.to_datetime(df['DATE'], format='%m-%d-%Y').astype(np.int64).values // 1000000
            except:
                try:
                    timestamps = pd.to_datetime(df['DATE'], format='%Y/%d/%m').astype(np.int64).values // 1000000
                except:
                    try:
                        timestamps = pd.to_datetime(df['DATE'], format='%d/%m/%Y').astype(np.int64).values // 1000000
                    except:
                        try:
                            timestamps = pd.to_datetime(df['DATE'], format='%m/%d/%Y').astype(np.int64).values // 1000000
                        except:
                            try:
                                timestamps = pd.to_datetime(df['DATE'], format='%Y-%m').astype(np.int64).values // 1000000
                            except:
                                timestamps = pd.to_datetime(df['DATE'], format='%Y-%m-%d %H:%M:%S').astype(np.int64).values // 1000000
    return timestamps
    
