# Ax=b
from pprint import pprint
from numpy import zeros, diag, diagflat, dot, array, float64, ones
import time
import sys


def jacobi(A, b, N=100, x=None):
    if x is None:
        x = zeros(len(A[0]))
    D = diag(A)
    R = A - diagflat(D)
    for i in range(N):
        x = (b - dot(R, x)) / D
    return x


def str_to_row(s):
    return [float64(x) for x in s.split(" ")][:-1]

if __name__== "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        inp = open(file_name, "r")
    else:
        inp = open("input.txt", "r")

    out = open("output.txt", "w")

    n = int(inp.readline())
    A = []
    b = []

    for line in inp:
        A.append(str_to_row(line))
        b.append([float64(x) for x in line.split(" ")][-1])
    A = array(A, dtype=float64)
    b = array(b, dtype=float64)
    guess = ones(n)
    guess = array(guess, dtype=float64)

    start = time.time()
    sol = jacobi(A, b, N=25, x=guess)
    end = time.time()

    print("A:")
    pprint(A)
    print("b:")
    pprint(b)
    print("x:")
    pprint(sol)
    print('execution time : ')
    print(end - start)
