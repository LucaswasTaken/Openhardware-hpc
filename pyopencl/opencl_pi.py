#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
f = open("/content/Openhardware-hpc/pyopencl/pi.cl","r")
string = f.read()
f.close()
prg = cl.Program(ctx, string).build()

n_steps = 1000;
dx = 1/n_steps;
res_vec = np.empty(n_steps).astype(np.float32);
res_vec_cl = cl.Buffer(ctx, mf.WRITE_ONLY, res_vec.nbytes)

prg.pi_cl(queue, res_vec.shape, None, res_vec_cl, np.float32(dx))

# res_np = np.empty_like(res_vec).astype(np.float32)
cl.enqueue_copy(queue, res_vec, res_vec_cl)
print(sum(res_vec))

