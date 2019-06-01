#!/bin/bash
#python gen.py "$1" "$2"
mpirun -np "$3" python solver.py
