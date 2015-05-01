#!/usr/bin/env python

"""
Demonstrates multiplication of two matrices on the GPU.
"""

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.tools as tools
from pycuda.compiler import SourceModule

import numpy as np

import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc
import string

from ctypes import *
cdll.LoadLibrary("C:/Program Files/kmeans-gpu/lib/libCudaKernelLibrary.so")
kmeansLib = CDLL("C:/Program Files/kmeans-gpu/lib/libCudaKernelLibrary.so")

culinalg.init()

# Double precision is only supported by devices with compute
# capability >= 1.3:
demo_types = [np.float32]
if cumisc.get_compute_capability(pycuda.autoinit.device) >= 5.0:
    demo_types.extend([np.float64])

for t in demo_types:
    np.random.seed(seed=42)
    m = 899946
    n = 129
    k = 1024

    print 'Testing matrix multiplication for type ' + str(np.dtype(t))
    a = np.asarray(np.random.rand(m,n), t)
    b = np.asarray(np.random.rand(n,k), t)
    
    start = cuda.Event()
    stop = cuda.Event()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = gpuarray.zeros((m,k),t,order="C")
    start.record()    
    culinalg.dot(a_gpu, b_gpu, out=c_gpu)
    stop.record()
    stop.synchronize()
    dtCuda = .001*stop.time_since(start)

    print 'Success status: ', np.allclose(np.dot(a, b), c_gpu.get())
    print " dt =",dtCuda

    aptr = cast(np.intp(a_gpu.gpudata), c_void_p)
    bptr = cast(np.intp(b_gpu.gpudata), c_void_p)
    cptr = cast(np.intp(c_gpu.gpudata), c_void_p)

    start.record()    
    kmeansLib.MatMatMultF(c_int(m),c_int(n),aptr,c_int(k),bptr,cptr)

    stop.record()
    stop.synchronize()
    dtMe = .001*stop.time_since(start)

    print 'Success status: ', np.allclose(np.dot(a, b), c_gpu.get())
    print " dt =",dtMe
    print " ratio =",dtMe/dtCuda
