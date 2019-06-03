import numpy as np
import math
import time

start = time.time()

MAX_N = 40


def str_to_row(s):
    return [int(x) for x in s.split()]

inp = open("input.txt", "r")
# out = open("output.txt", "w")
MAX_N = int(inp.readline())
A = []
for line in inp:
    A.append(str_to_row(line))

A = np.array(A, dtype=np.float64)
print(A)



# Matrix for jacobi calculation input and output
#A = np.zeros((MAX_N - 2, MAX_N - 2))
A = np.pad(A, pad_width=1, mode='constant', constant_values=1)
# Matrix for jacobi calculation output temp
(row_num, col_num) = A.shape
print("row num: ", row_num)
B = np.zeros((row_num, col_num))

# Do jacobi
converge = False
iteration_num = 0
#while (converge == False):
for i in range(1000):
    iteration_num = iteration_num + 1
    diffnorm = 0.0

    # for convinience, use padding border
    A_padding = A #np.pad(A, pad_width=1, mode='constant', constant_values=0)

    for i in range(row_num):
        for j in range(col_num):
            # because we do padding, index changed
            idx_i_A = i
            idx_j_A = j
            # Do jacobi
            B[i][j] = 0.25 * (A_padding[idx_i_A , idx_j_A]
                              + A_padding[idx_i_A, idx_j_A]
                              + A_padding[idx_i_A, idx_j_A]
                              + A_padding[idx_i_A, idx_j_A])
            # simple converge test
            diffnorm += math.sqrt((B[i, j] - A[i, j]) * (B[i, j] - A[i, j]))
    A = np.copy(B)

    # check converge
    if diffnorm <= 0.00001:
        print('Converge, iteration : %d' % iteration_num)
        print('Error : %f' % diffnorm)
        converge = True

end = time.time()
print('execution time : ')
print(end - start)
print('A:')
print(A)
