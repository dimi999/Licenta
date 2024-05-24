def delta(Xi_2, Xi_1, Xi): 
    return Xi - Xi_1

def rev_delta(Xi_2, Xi_1, Xi): 
    return Xi_1 - Xi

def xor(Xi_2, Xi_1, Xi):
    return Xi ^ Xi_1

def delta_of_delta(Xi_2, Xi_1, Xi):
    return (Xi - Xi_1) - (Xi_1 - Xi_2)

def rev_delta_of_delta(Xi_2, Xi_1, Xi): 
    return (Xi_1 - Xi_2) - (Xi - Xi_1)

def delta_xor(Xi_2, Xi_1, Xi): 
    return xor(0, Xi - Xi_1, Xi_1 - Xi_2)