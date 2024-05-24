import pandas as pd 
import numpy as np
from compressions import timestamps_compression, timestamps_decompression
df = pd.read_csv("Gold.csv")

timestamps = pd.to_datetime(df['DATE'], format='%Y/%m/%d').astype(np.int64).values
orig = timestamps.copy()
timestamps_compression(timestamps)
#%%
timestamps2 = timestamps_decompression('timestamps.bits')

for i in range(len(timestamps2)):
    if timestamps2[i] != orig[i]:
        print(i)
print(len(timestamps) * 8 * 4)

#%%

