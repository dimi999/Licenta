import struct
import numpy as np

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
    if number == 0:
        return '0' * 64
    packed_bytes = struct.pack('>d', number)
    packed_bytes = packed_bytes.hex()
    packed_bytes = int(packed_bytes, 16)
    binary_string = "{:08b}".format(packed_bytes)

    if len(binary_string) != 64:
        for i in range(64 - len(binary_string)):
            binary_string = '0' + binary_string
            
    return binary_string

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

def count_bits(num):
     binary = bin(num)[2:]
     return len(binary)

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