import pandas as pd
import dahuffman
from utilities import int_to_bits

data = "a asd asfasfas asd asdef as fasfas aa asa  a a a aa"

codec = dahuffman.HuffmanCodec.from_data(data)
dictio = codec.get_code_table()

for x in dictio:
    sz, val = dictio[x]
    val = int_to_bits(val)[-sz:]
    print(x, val)