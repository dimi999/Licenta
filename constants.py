from transform_primitives import delta, rev_delta, rev_delta_of_delta, xor, delta_of_delta, delta_xor, delta_inverse, rev_delta_inverse, delta_of_delta_inverse, rev_delta_of_delta_inverse, xor_inverse, delta_xor_inverse
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
        0: (offset, 1),
        1: (offset, 2),
        2: (offset, 3),
        3: (bitmask, 1),
        4: (trailing_zero, 0)
    }, 
    3: {
        0: [(0, bitmask, 1), (1, bitmask, 1), (2, bitmask, 1), (0, trailing_zero, 1)],
        1: [(0, bitmask, 1), (1, offset, 2), (2, bitmask, 1), (0, trailing_zero, 1)],
        2: [(1, offset, 1), (1, offset, 2), (1, offset, 3), (0, trailing_zero, 1)],
        3: [(0, bitmask, 1), (1, offset, 1), (2, offset, 2), (0, trailing_zero, 1)],
    },
    4: {
        0: delta_inverse,
        1: rev_delta_inverse,
        2: xor_inverse,
        3: delta_of_delta_inverse,
        4: rev_delta_of_delta_inverse,
        5: delta_xor_inverse
    }
}
