import sys
import time
from numpy import linalg as la

import numpy as np
from mpi4py import MPI

inp = open('input_500.txt', 'r')
out = open('output.txt', 'w')


def get_step_and_master_count(n, size):
    #count number of rows per process
    step = n / size
    return step, int(step + n % size)
 
def rank_from_row(m_size, size, index):
    #size - nr procesu
    step, master_count = get_step_and_master_count(m_size, size)

    if index < master_count:
        return 0

    index -= master_count
    return 1 + index // step #process rank


def gauss(data, comm, n):

    def eliminate(l, r, current_index):

        for indx in indexes:
            if indx == current_index:
                continue
            rhs[indx] -= r * lhs[indx, current_index]
            lhs[indx] -= l * lhs[indx, current_index]

    def send(indx, rnk):
        comm.Bcast([lhs[indx], MPI.DOUBLE], root=rnk)
        comm.Bcast([rhs[indx], MPI.DOUBLE], root=rnk)

        eliminate(lhs[indx], rhs[indx], indx)

    def receive(indx, rnk):

        comm.Bcast([l_row, MPI.DOUBLE], root=rnk)
        comm.Bcast([r_row, MPI.DOUBLE], root=rnk)

        eliminate(l_row, r_row, indx)
    indexes = []

    lhs = np.zeros((n, n))
    rhs = np.identity(n)

    for line in data:
        ind = int(line[-1])
        ii = 1#[line.split()][-1]
        indexes.append(ind)
        lhs[ind] = line[:-1]


    l_row = np.zeros(n, dtype=np.float64)
    r_row = np.zeros(n, dtype=np.float64)

    for i in range(n): 
        rank = rank_from_row(n, comm.size, i)

        if i in indexes:
            rhs[i] /= lhs[i, i]
            lhs[i] /= lhs[i, i]
            send(i, rank)

        else:
            receive(i, rank)

        comm.Barrier()
    # back substitution
    for i in range(n - 1, -1, -1): 
        rank = rank_from_row(n, comm.size, i)

        if i in indexes:
            send(i, rank)
        else:
            receive(i, rank)

        comm.Barrier()

    return [rhs.tolist()[ind] + [ind] for ind in indexes]


def str_to_row(s):
    return [int(x) for x in s.split()][:-1]


def _do_inversion(com, rws, m_size):
    inv = gauss(np.array(rws, dtype=np.float64), com, m_size)
    return inv

def write_matrix(matrix, out):
    out.write('{}\n'.format(len(matrix)))
    for row in matrix:
        out.write(('{:.3f}\t' * len(matrix) + '\n').format(*row))

class Timer(object):
    def __init__(self, message):
        self.message = message
        self.start = time.time()

    def finish(self):
        print("-" * 20 + "| {0}: {1:.3f} s |".format(self.message, (time.time() - self.start)) + "-" * 20)


time_count = Timer("time_count")

comm = MPI.COMM_WORLD
master = 0
A, b,rows = [], [], []

if comm.rank == master:
    t = Timer('TOTAL')
    #wczytaj macierz
    m_size = int(inp.readline())

    for line in inp:
        A.append(str_to_row(line))
        b.append([int(x) for x in line.split()][-1])

    a = np.array(A, dtype=np.float64)
    #obl wyznacznik
    det = la.det(np.array(A))

    t_inv = Timer('inversion')

    if not det:
        print("Comm.rank: ", comm.rank, " Error! No inversion")
        comm.bcast(0, root=master)
        sys.exit()

        #collective communication techniques. 
        #During a broadcast, one process sends the same data to all processes in a communicator. 
        #One of the main uses of broadcasting 
        #is to send out user input to a parallel program, or send out configuration parameters to all processes.
    comm.bcast(m_size, root=master)

    #The number in a communicator does not change once it is created. 
    # That number is called the size of the communicator. 
    # At the same time, each process inside a communicator has a unique number to identify it.
    #  This number is called the rank of the process. 
    #  The rank of a process always ranges from 0 to size−1.

    step = m_size / comm.size
    master_count =  int(step + m_size % comm.size)

    #na koncu doczepiamy ~rank~
    rows = [A[i] + [i] for i in range(master_count)]
    current = master_count
    for proc in range(1, comm.size): #0 to master
        for i in range(int(step)):
            comm.send(A[int(current + i)] + [current + i], dest=proc)  # send row with index as last element
        current += step

    inversed = _do_inversion(comm, rows, m_size)
    A_inv = inversed

    for proc in range(1, comm.size):
        A_inv += comm.recv(source=proc)
    A_inv.sort(key=lambda row: row[-1])
    A_inv = np.delete(A_inv, -1, 1)#usun ostatni el kazdej z list
    t_inv.finish()

    write_matrix(A_inv, out)
    #wynik -> il skalarny 
    x = np.dot(A_inv, b)
    xl =x.tolist()
    print(x)
    #zapis do pliku
    #out.write(('{}\n'+'{:.3f}\t'*len(x)).format(1, *x))

    t.finish()
    print("count:", comm.size)
    time_count.finish()
else:
    #t = Timer('proc{}'.format(comm.rank))
    n = comm.bcast(None, root=master)
    step = n / comm.size
    if not step:
        sys.exit()
    for i in range(int(step)):
        rows.append(comm.recv(source=master))
    inversed = _do_inversion(comm, rows, n)
    comm.send(inversed, dest=master)
    #t.finish()
