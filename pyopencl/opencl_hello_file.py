#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
f = open("/content/Openhardware-hpc/Opencl/src_opencl/hello.cl","r")
string = f.read()
f.close()
prg = cl.Program(ctx,string).build()

prg.hello(queue, np.zeros(10).shape, None)

