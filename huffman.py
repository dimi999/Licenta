# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:36:14 2024

@author: Andrei
"""

import pandas as pd 
import numpy as np
from dahuffman import HuffmanCodec

from transform_primitives import delta_of_delta, delta

df = pd.read_csv("Gold.csv")
timestamps = pd.to_datetime(df['DATE'], format='%Y/%m/%d').astype(np.int64).values
values = df['VALUE'].astype(np.float64).to_numpy()

timestamps = delta_of_delta(timestamps)
print(timestamps)

values = delta(values)
print(values)

#%%

huff_dict = dict()


for i in range(64):
    huff_dict[i] = 64 - i

#for x in timestamps:
#    if x in huff_dict:
#        huff_dict[x] += 1
#    else:
#        huff_dict[x] = 1
        
#for x in values:
#    if x in huff_dict:
#        huff_dict[x] += 1
#    else:
#        huff_dict[x] = 1
        
#print(len(huff_dict))
codec2 = HuffmanCodec.from_frequencies(huff_dict)
codec2.print_code_table()

#%%
codec = HuffmanCodec.from_frequencies(huff_dict)
codec.print_code_table()

#%%
encoded = codec.encode(timestamps)
print(len(encoded))

#%%
import pickle
with open("huffman_gold.bits", "wb") as binary_file:
    binary_file.write(encoded)

with open('gold_huffman.pkl', 'wb') as fp:
    pickle.dump(huff_dict, fp)
    
#%%
with open('gold_huffman.pkl', 'rb') as fp:
    read_dict = pickle.load(fp)
codec2 = HuffmanCodec.from_frequencies(read_dict)
decoded = codec2.decode(encoded)



