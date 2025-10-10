#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np
import sys
import pyopencl as cl
from mpi4py import MPI
import math

idgg = int(sys.argv[1]);
nstepsgg = int(sys.argv[2]);
dxgg = 1/nstepsgg;

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
f = open("/content/pi2.cl","r")
string = f.read()
f.close()
prg = cl.Program(ctx, string).build()

# number of integration steps
nsteps = 10000000
# step size
dx = 1.0 / nsteps * dxgg
startgg = int(nsteps*dxgg*idgg)

if rank == 0:
    # determine the size of each sub-task
    ave, res = divmod(nsteps, nprocs)
    counts = [ave + 1 if p < res else ave for p in range(nprocs)]

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(nprocs)]
    ends = [sum(counts[:p+1]) for p in range(nprocs)]

    # save the starting and ending indices in data  
    data = [(starts[p], ends[p]) for p in range(nprocs)]
else:
    data = None

data = comm.scatter(data, root=0)

# compute partial contribution to pi on each process
res_vec = np.empty(data[1]-data[0]).astype(np.float32);
res_vec_cl = cl.Buffer(ctx, mf.WRITE_ONLY, res_vec.nbytes)

prg.pi_cl(queue, res_vec.shape, None, res_vec_cl, np.float32(dx),np.int32(startgg+data[0]))

res_np = np.empty_like(res_vec).astype(np.float32)
cl.enqueue_copy(queue, res_np, res_vec_cl)

partial_pi = sum(res_np)

pi = comm.reduce(partial_pi, op=MPI.SUM, root=0)

if rank == 0:
    print('{},'.format(pi))