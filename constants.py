from transform_primitives import delta, rev_delta, rev_delta_of_delta, xor, delta_of_delta, delta_xor
from compression_primitives import offset, trailing_zero, bitmask

parameters_dict = {
    1: {
        0: delta,
        1: rev_delta,
        2: xor,
        3: delta_of_delta,
        4: rev_delta_of_delta,
        5: delta_xor
    },
    2: {
        0: delta,
        1: rev_delta,
        2: xor,
        3: delta_of_delta,
        4: rev_delta_of_delta,
        5: delta_xor
    },
    3: {
        0: delta,
        1: rev_delta,
        2: xor,
        3: delta_of_delta,
        4: rev_delta_of_delta,
        5: delta_xor
    },
    4: {
        0: (offset, 1),
        1: (offset, 2),
        2: (offset, 3),
        3: (bitmask, 1),
        4: (trailing_zero, 0)
    }, 
    5: {
        0: (offset, 1),
        1: (offset, 2),
        2: (offset, 3),
        3: (bitmask, 1),
        4: (trailing_zero, 0)
    },
    6: {
        0: (offset, 1),
        1: (offset, 2),
        2: (offset, 3),
        3: (bitmask, 1),
        4: (trailing_zero, 0)
    }
}
