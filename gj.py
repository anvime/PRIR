import numpy as np
inp_file = 'input.txt'

def str_to_row(s):
    return [int(x) for x in s.split()]

def gauss_jordan(m, eps = 1.0/(10**10)):
    (h, w) = (len(m), len(m[0]))

    for y in range(0,h):
        print("------my m", m)
        maxrow = y
        for y2 in range(y+1, h):    # Find max pivot
            print("y2: ", y2, "m[y2][y]:", m[y2][y])
            if abs(m[y2][y]) > abs(m[maxrow][y]):
                maxrow = y2
        print("m[maxrow]: ", m[maxrow], "m[y]:", m[y])
        hlp = list()
        for j in range(len(m[y])):
            hlp.append(m[y][j])
            m[y][j]= m[maxrow][j]
        for j in range(len(m[y])):
            m[maxrow][j]= hlp[j]    
        # m[maxrow] = m[y]
        # m[y] = m[maxrow]
        # m[maxrow] = k
        # m[y], m[maxrow] = m[maxrow], m[y]
        print("m[maxrow]2:", m[maxrow], "m[y]:", m[y])
        print("AFTER m: ", m, " y:", y )
        print("m[y][y]: ", m[y][y], "eps: ", eps)
        if abs(m[y][y]) <= eps:     # Singular?
            print("abs:", abs(m[y][y]))
            return False
        for y2 in range(y+1, h):    # Eliminate column y
            c = m[y2][y] / m[y][y]
            print("c:", c)
            print("m:", m)
            for x in range(y, w):
                m[y2][x] -= m[y][x] * c
    print("??;")   
  

    for y in range(h-1, 0-1, -1): # Backsubstitute
        c  = m[y][y]

        for y2 in range(0,y):
            for x in range(w-1, y-1, -1):
                m[y2][x] -=  m[y][x] * m[y2][y] / c
        m[y][y] /= c
        for x in range(h, w):       # Normalize row y
            m[y][x] /= c

    print("??", m)   
    return True


inp = open(inp_file, 'r')
A, b, rows = [], [], []
n = inp.readline()

for line in inp:
    A.append(str_to_row(line))
#    b.append(int(line[-2:-1]))

a = np.array(A, dtype=np.float64)

print(a)
gauss_jordan(a)
