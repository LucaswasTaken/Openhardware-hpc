#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = cl.Program(ctx, """
__kernel void hello()
{
  int gid = get_global_id(0);
  printf("Hello from process: %d,  ",gid);
}
""").build()

prg.hello(queue, np.zeros(10).shape, None)

