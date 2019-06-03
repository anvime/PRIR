import numpy as np
import time

inp_file = 'input_500.txt'

def str_to_row(s):
    return [int(x) for x in s.split()]

class Timer(object):
    def __init__(self, message):
        self.message = message
        self.start = time.time()

    def finish(self):
        print("-" * 20 + "| {0}: {1:.3f} s |".format(self.message, (time.time() - self.start)) + "-" * 20)


def gauss_jordan(m, eps = 1.0/(10**10)):
    (height, w) = (len(m), len(m[0]))

    for y in range(0,height):
        maxrow = y
        #find pivot
        for y2 in range(y+1, height):    # Find max pivot
            if abs(m[y2][y]) > abs(m[maxrow][y]):
                maxrow = y2
        tmp = list()
        #zamien y-ty wiersz na miejsca z tym zawierajacym max value
        for j in range(len(m[y])):
            tmp.append(m[y][j])
            m[y][j]= m[maxrow][j]
        for j in range(len(m[y])):
            m[maxrow][j]= tmp[j]
        #jesli na diagolanli bedzie <e-10        
        if abs(m[y][y]) <= eps: 
            return False
        #wszystko w kolumnie pod diagonala dzielimy przez wartosc na diagonali    
        for y2 in range(y+1, height):    # Eliminate column y
            c = m[y2][y] / m[y][y]
            #wszystko w danym wierszu od doagolnali w prawo dzielimy..
            for x in range(y, w):
                m[y2][x] -= m[y][x] * c
        print

        #od, do, krok - tutaj iterowanie w dół :)
        #bierzemy warosc z diag.
    for y in range(height-1, 0-1, -1): # Backsubstitute
        c  = m[y][y]

        for y2 in range(0,y):
            for x in range(w-1, y-1, -1):
                m[y2][x] -=  m[y][x] * m[y2][y] / c
        m[y][y] /= c
        for x in range(height, w):       # Normalize row y
            m[y][x] /= c
    x_res =[] 
    for i in range(len(m[0])-1):
        x_res.append(m[i][-1])
    x_resf = ["%.2f" % member for member in x_res]
    print("Output:")
    print(x_resf)
    return True

t = Timer('Total time')

inp = open(inp_file, 'r')
A = []
n = inp.readline()

for line in inp:
    A.append(str_to_row(line))

a = np.array(A, dtype=np.float64)
print("Seq. Gauss-J input array:")
print(a)

gauss_jordan(a)

t.finish()