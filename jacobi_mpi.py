# Ax=b
from pprint import pprint
from numpy import zeros, diag, diagflat, dot, array, float64, ones, ndarray, arange, reshape, empty
import time
import sys
from mpi4py import MPI
MAX_ITER = 100

def call_dist(x, y, dim):
    sum = 0.0
    for i in range(dim):
        sum += (x[i] + y[i]) * (x[i] - y[i])
    return sum


def verify(no_rows, no_cols, no_procs, my_rank):
    if no_rows != no_cols:
        MPI.Finalize()
        if my_rank == 0:
            print("Matrix A should be square ...\n")
        sys.exit(-1)
    if no_rows % no_procs != 0:
        MPI.Finalize()
        if my_rank == 0:
            print("Matrix A can't be stripped even ...\n")
        sys.exit(-1)


def jacobi(A, b, N=100, x=None):
    if x is None:
        x = zeros(len(A[0]))
    D = diag(A)
    R = A - diagflat(D)
    for i in range(N):
        x = (b - dot(R, x)) / D
    return x


def str_to_row(s):
    return [float64(x) for x in s.split()][:-1]

def iteration(x_new, x_old, x_bloc, a_recv,b_recv, no_rows_bloc, no_rows, no_cols, my_rank, comm):
    iter = 0
    for irow in range(no_rows_bloc):
        # print(x_bloc[irow])
        # print(b_recv[irow])
        x_bloc[irow] = b_recv[irow]
    comm.Allgather(x_bloc, x_new)
    # print(x_new)
    # print(call_dist(x_old, x_new, no_rows))
    while iter < MAX_ITER:
        #przepisanie starego do nowego
        for irow in range(no_rows):
            x_old[irow] = x_new[irow]

        irow = 0
        for irow in range(no_rows_bloc):
            global_rows_idx = (my_rank * no_rows_bloc) + irow
            x_bloc[irow] = b_recv[irow]
            for icol in range(no_cols):
                if icol == global_rows_idx:
                    continue
                x_bloc[irow] -= x_old[icol] * a_recv[irow * no_cols + icol]
                # pprint(x_bloc)
            x_bloc[irow] /= a_recv[irow * no_cols + global_rows_idx]
        comm.Allgather(x_bloc, x_new)
        iter += 1
    return x_new, x_bloc

def main():
    inp = open("input.txt", "r")
    out = open("output.txt", "w")
    n = int(inp.readline())
    A = []
    b = []
    for line in inp:
        A.append(str_to_row(line))
        b.append([float64(x) for x in line.split()][-1])
    A = array(A, dtype=float64)
    a_input = A.flatten('C')
    b = array(b, dtype=float64)
    b_input = b.flatten('C')
    guess = ones(n)
    guess = array(guess, dtype=float64)
    start = time.time()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    #ilosc kolumn
    comm.bcast(n, root=0)
    #ilosc wierszy
    comm.bcast(n, root=0)
    verify(n, n, size, rank)

    # iteracja
    no_rows_bloc = int(n / size)
    nn = no_rows_bloc*n
    x_new = empty([n])
    x_old = empty([n])
    x_block = empty([no_rows_bloc])
    a_recv = empty([nn])
    b_recv = empty([nn])
    comm.Scatter(a_input, a_recv, root=0)
    comm.Scatter(b_input, b_recv, root=0)
    x_new, x_block = iteration(x_new, x_old, x_block, a_recv, b_recv, no_rows_bloc, n, n, rank, comm)

    end = time.time()
    if rank == 0:
        print("A:")
        pprint(A)
        print("b:")
        pprint(b)
        print("x:")
        pprint(x_new)
        print('execution time : ')
        print(end - start)

    sys.exit(0)
if __name__ == "__main__":
    main()
