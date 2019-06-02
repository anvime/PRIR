## Jacobi
For sequential jacobi:
python3 jacobi_sequential.pl {input.txt}

If input_file is not provided then the script automatically looks for "input.txt"


For parallel jacobi:
mpiexec --hostfile hostfile -n {no_of_proccess} python3 jacobi_mpi.py

This ver of the program only takes "input.txt" as input file.
note that in parallel ver. the size of matrix must be divisible by no of procs.

### Example:

#### Sequential
```
python3 jacobi_sequential.py                 
A:
array([[4., 1.],
       [5., 7.]])
b:
array([11., 13.])
x:
array([ 2.7826087 , -0.13043478])
execution time :
0.0019011497497558594
```

#### MPI

```
mpiexec --hostfile hostfile -n 2 python3 jacobi_mpi.py
A:
array([[4., 1.],
       [5., 7.]])
b:
array([11., 13.])
x:
array([ 2.7826087 , -0.13043478])
execution time :
0.003748178482055664
```
