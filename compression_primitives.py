from utilities import count_bits

def offset(value, offset):
    bits = value
    ok = True
    for i in range(offset):
        if get_byte(bits, i) != 0:
            ok = False
    
    if ok == False:
        return 64 #Offset too big, compression becomes lossly
    
    #return (value >> (offset * 8))
    pad = 0
    for i in range(64 - offset * 8):
        if bits[i] == '0':
            pad += 1
        else:
            break
    if 64 - offset * 8 - pad == 0:
        return 1
    return 64 - offset * 8 - pad
    
    
def bitmask(value, offset):
    ok = True
    bits = value
    for i in range(offset):
        if get_byte(bits, i) != 0:
            ok = False
    
    if ok == False:
        return 64
        #return "Offset too big, compression becomes lossly"
    
    ct_bytes = 8 - offset
    mask = 0
    new_value = 0
    ct_non_zero_bytes = 0
    
    for i in range(ct_bytes):
        crt_byte = get_byte(bits, offset + i)
        if crt_byte != 0:
            mask += (1 << i)
            new_value += (crt_byte << (8 * ct_non_zero_bytes)) 
            ct_non_zero_bytes += 1
    #return mask, new_value
    return 8 - offset + ct_non_zero_bytes * 8
            
def trailing_zero(value):
    ct_bytes = 8
    ct_non_zero_bytes = 8
    ct_zero_bytes = 0
    bits = value
    i = 0
    
    crt_byte = get_byte(bits, i)
    while i < ct_bytes and crt_byte == 0:
        #new_value = new_value >> 8
        ct_zero_bytes += 1
        ct_non_zero_bytes -= 1
        crt_byte = get_byte(bits, i)
        i += 1
            
    return count_bits(ct_zero_bytes) + count_bits(ct_non_zero_bytes) + 8 * ct_non_zero_bytes   

def count_bytes(value):
    if value == 0:
        return 1
    return value.bit_length() // 8 + (1 if value.bit_length() % 8 else 0)

def get_byte(bitstring, byte_no): 
    #gets byte byte_no
    start = 64 - (byte_no + 1) * 8
    byte = bitstring[start : start + 8]
    value = int(byte, 2)
    return value