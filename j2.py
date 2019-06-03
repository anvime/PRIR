## mpiexec -n 4 python ex-2.16.py

# Jacobi code
# version of parallel code using sendrecv and null proceses.

# --------------------------------------------------------------------

from mpi4py import MPI
try:
    import numpy as np
except ImportError:
    raise SystemExit

# --------------------------------------------------------------------
def str_to_row(s):
    return [int(x) for x in s.split()]

inp = open("input.txt", "r")
# out = open("output.txt", "w")
MAX_N = int(inp.readline())
A = []
for line in inp:
    A.append(str_to_row(line))
#    b.append(int(line[-2:-1]))

A = np.array(A, dtype=np.float64)
print(A)


n = MPI.COMM_WORLD.Get_size()

# compute number of processes and myrank
p = MPI.COMM_WORLD.Get_size()
myrank = MPI.COMM_WORLD.Get_rank()

# compute size of local block
m = n/p
if myrank < (n - p * m):
    m = m + 1

#compute neighbors
if myrank == 0:
    left = MPI.PROC_NULL
else:
    left = myrank - 1
if myrank == p - 1:
    right = MPI.PROC_NULL
else:
    right = myrank + 1

# allocate local arrays
n = int(n)
m = int(m)
A = np.empty((n+2, m+2), dtype='d', order='fortran')
B = np.empty((n, m),     dtype='d', order='fortran')


# main loop
print(A)
converged = False
while not converged:
    # compute,  B = 0.25 * ( N + S + E + W)
    N, S = A[:-2, 1:-1], A[2:, 1:-1]
    E, W = A[1:-1, :-2], A[1:-1, 2:]
    np.add(N, S, B)
    np.add(E, B, B)
    np.add(W, B, B)
    B *= 0.25
    A[1:-1, 1:-1] = B
    # communicate
    tag = 0
    MPI.COMM_WORLD.Sendrecv([B[:, -1], MPI.DOUBLE], right, tag,
                            [A[:,  0], MPI.DOUBLE], left,  tag)
    MPI.COMM_WORLD.Sendrecv((B[:,  0], MPI.DOUBLE), left,  tag,
                            (A[:, -1], MPI.DOUBLE), right, tag)
    # convergence
    myconv = np.allclose(B, 0)
    loc_conv = np.asarray(myconv, dtype='i')
    glb_conv = np.asarray(0, dtype='i')
    MPI.COMM_WORLD.Allreduce([loc_conv, MPI.INT],
                             [glb_conv, MPI.INT],
                             op=MPI.LAND)
    converged = bool(glb_conv)

# ---