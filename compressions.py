from BitVector import BitVector
import numpy as np
from transform_primitives import delta_of_delta
from compression_primitives import bitmask, trailing_zero, count_bytes

def timestamps_compression(times):
    ct = 0
    bv = BitVector(size=0)
    processed_times = delta_of_delta(times)
    bv += BitVector(intVal=processed_times[0], size=64)
    bv += BitVector(intVal=processed_times[1], size=64)
    for i in range(2, len(processed_times)):
        ct += 1
        if ct % 100 == 0:
            print(ct)
        if processed_times[i] == 0:
            bv += BitVector(intVal=0)
        else:
            bv += BitVector(intVal=1)
            
            if processed_times[i] > 0:
                processed_times[i] -= 1
            #We lose one bit of information by keeping the 0 value unoccupied
            #So we will decrease the positive values by 1 to also use that value
            
            mask, new_value_mask = bitmask(int(abs(processed_times[i])))
            ct_zero_bytes, ct_non_zero_bytes, new_value_trail = trailing_zero(int(abs(processed_times[i])))
            
            #We suppose that both bitmask and the count of 0 bytes and !0 bytes fit in a single byte
            #So we only make the comparison between the values
            if processed_times[i] < 0:
                bv += BitVector(intVal=1)  
            else:
                bv += BitVector(intVal=0)
            
            if count_bytes(new_value_mask) < count_bytes(new_value_trail):
                bv += BitVector(intVal=1) #control bits
                bv += BitVector(intVal=mask, size=7)
                bv += BitVector(intVal=new_value_mask, size=8 * count_bytes(new_value_mask))
            else:
                bv += BitVector(bitstring='01') #control bits we have 6 more bits left fot bitmask
                bv += BitVector(intVal=ct_zero_bytes, size=3)
                bv += BitVector(intVal=ct_non_zero_bytes, size=3)
                bv += BitVector(intVal=new_value_trail, size=8 * count_bytes(new_value_trail))

    size_bv = BitVector(intVal=len(bv), size=64) 
    sz = len(bv)
    bv = size_bv + BitVector(intVal=0, size=8 - sz % 8) + bv
    print(len(bv))
    FILEOUT = open('timestamps.bits', 'wb')
    bv.write_to_file(FILEOUT)
    FILEOUT.close()

def decompress_header(filename):
    sz = 0
    poz = 0
    compressed_timestamps = BitVector(filename=filename)
    bv1 = compressed_timestamps.read_bits_from_file(64)
    for i in range(64):
        sz += (1 << (63 - i)) * bv1[i]
        poz += 1
    poz += 8 - sz % 8
    bv1 = compressed_timestamps.read_bits_from_file(8 - sz % 8 + sz)
    
    return sz, bv1[8 - sz % 8:]
    
def timestamps_decompression(filename):
    poz = 0
    last_val = 0
    last_delta = 0
    timestamps = []
    
    size, times = decompress_header(filename)
    
    val1 = times[:64]
    timestamps.append(int(val1))
    val2 = times[64:128]
    last_val = int(val2)
    last_delta = last_val - int(val1)
    timestamps.append(last_val)
    poz = 128
    
    while poz < len(times):
        if times[poz] == 0:
            last_val = last_val + last_delta
            timestamps.append(last_val)
            poz += 1
        else:
            poz += 1
            
            sign = 0
            if times[poz] == 0:
                sign = 1
            else:
                sign = -1
            poz += 1
            
            if times[poz] == 0: # trailing zero
                poz += 2
                ct_zero_bytes = 0
                ct_non_zero_bytes = 0
                for i in range(3):
                    ct_zero_bytes += times[poz + i] * (1 << (2 - i))
                poz += 3
                for i in range(3):
                    ct_non_zero_bytes += times[poz + i] * (1 << (2 - i))
                poz += 3
                original_value = 0
                for i in range(ct_non_zero_bytes * 8):
                    original_value += times[poz + i] * (1 << (ct_non_zero_bytes * 8 - 1 - i))
                
                original_value *= sign
                poz += ct_non_zero_bytes * 8
                original_value <<= (8 * ct_zero_bytes)
                
                if sign == 1:
                    original_value += 1
                
                last_delta = last_delta + original_value
                last_val = last_val + last_delta
                
                #print(original_value, sign, last_delta, last_val, poz)
                
                timestamps.append(last_val)
            else: #bitmask
                poz += 1
                ct_non_zero_bytes = 0
                original_value = 0
                for i in range(7):
                    if times[poz + i] == 1:
                        ct_non_zero_bytes += 1
                
                j = 0
                for i in range(7):
                    if times[poz + i] == 0:
                        original_value <<= 8
                    else:
                        start = poz + 7 + 8 * j
                        end = poz + 7 + 8 * (j + 1)
                        for it in range(start, end):
                            original_value += (1 << (7 - (it - start)))
                        j += 1
                
                poz += 7 + 8 * j
                original_value *= sign
                
                if sign == 1:
                    original_value += 1
                
                last_delta = last_delta + original_value
                last_val = last_val + last_delta
                
                #print(original_value, sign, last_delta, last_val, poz)
                
                timestamps.append(last_val)
        
    return np.array(timestamps)
                
    
timestamps = np.linspace(0, 1000, 201).astype(int)
timestamps[103] -= 1
timestamps[125] += 1
print(timestamps)
timestamps_compression(timestamps)


timestamps = timestamps_decompression('timestamps.bits')
print(timestamps)
