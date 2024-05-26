from utilities import count_bits, int_to_bits

def offset(value, offset, mode):
    if offset < 0 or offset * 8 + mode * 7 > 64:
        return 64, ''
    bits = value
    ok = True
    for i in range(offset):
        if get_byte(bits, i) != 0:
            ok = False
    
    if ok == False:
        return 64, '' #Offset too big, compression becomes lossly
    
    pad = 0
    for i in range(64 - offset * 8):
        if bits[i] == '0':
            pad += 1
        else:
            break

    if mode * 7 < 64 - offset * 8 - pad:
        return 64, ''
    
    new_value = ''

    for i in range(0, 7 * mode):
        new_value = bits[63 - (i + offset * 8)] + new_value

    return 7 * mode, new_value
    
    
def bitmask(value, offset):
    ok = True
    bits = value
    for i in range(offset):
        if get_byte(bits, i) != 0:
            ok = False
    
    if ok == False:
        return 64, ''
        #return "Offset too big, compression becomes lossly"
    
    ct_bytes = 8 - offset
    mask = ''
    new_value = ''
    ct_non_zero_bytes = 0
    
    for i in range(ct_bytes):
        crt_byte = get_byte(bits, offset + i)
        if crt_byte != 0:
            mask = '1' + mask
            start = 64 - (offset + i + 1) * 8
            new_value = bits[start:start + 8] + new_value
            ct_non_zero_bytes += 1
        else:
            mask = '0' + mask

    i = 0
    # while mask[i] == '0':
    #     mask = mask[1:]
    return len(mask) + ct_non_zero_bytes * 8, mask + new_value
            
def trailing_zero(value):
    ct_bytes = 8
    ct_non_zero_bytes = 0
    ct_zero_bytes = 0
    bits = value
    i = 0

    if len(bits) != 64:
        print("ERROR", bits)
    
    if bits == '0'  * 64:
        print("ERROR", "REICEIVED 0")

    crt_byte = get_byte(bits, i)
    while i < ct_bytes and crt_byte == 0:
        ct_zero_bytes += 1
        i += 1
        crt_byte = get_byte(bits, i)
    i = 7
    while i > ct_zero_bytes:
        crt_byte = get_byte(bits, i)
        if crt_byte != 0:
            break
        i -= 1
    ct_non_zero_bytes = 8 - (7 - i) - ct_zero_bytes
    new_value = value[64 - 8 * (ct_non_zero_bytes + ct_zero_bytes): 64 - 8 * ct_zero_bytes]
    return 6 + 8 * ct_non_zero_bytes, int_to_bits(ct_zero_bytes)[-3:] + int_to_bits(ct_non_zero_bytes - 1)[-3:] + new_value

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

#print(bitmask(int_to_bits(68722000896), 1))