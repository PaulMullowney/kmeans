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
cdll.LoadLibrary("/home/paul/projects/kmeans-gpu/lib/libCudaKernelLibrary.so")
kmeansLib = CDLL("/home/paul/projects/kmeans-gpu/lib/libCudaKernelLibrary.so")

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
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)

    # row norm of a
    p = np.linalg.norm(a, axis=1)
    anorm = p**2
    anorm = anorm.reshape(m,1)
    anorm_gpu = gpuarray.to_gpu(anorm)

    # column norm of b
    p = np.linalg.norm(b, axis=0)
    bnorm = p**2
    bnorm_gpu = gpuarray.to_gpu(bnorm)

    # timers
    start = cuda.Event()
    stop = cuda.Event()

    c_gpu = gpuarray.zeros((m,k),t,order="C")
    i_gpu = gpuarray.zeros((m),'int32')

    anormptr = cast(np.intp(anorm_gpu.gpudata), c_void_p)
    bnormptr = cast(np.intp(bnorm_gpu.gpudata), c_void_p)
    cptr = cast(np.intp(c_gpu.gpudata), c_void_p)
    iptr = cast(np.intp(i_gpu.gpudata), c_void_p)

    start.record()    
    culinalg.dot(a_gpu, b_gpu, out=c_gpu)
    kmeansLib.rowTransformMinimumF(c_int(m),c_int(k),anormptr,bnormptr,cptr,iptr)
    stop.record()
    stop.synchronize()
    dtCuda = .001*stop.time_since(start)

    # due to floating point differences, use the same MatMatMult for comparison
    p = (anorm + bnorm) - 2.0*c_gpu.get()
    i = np.argmin(p, axis=1)

    ig = i_gpu.get()
    print 'Success status: ', np.allclose(i, ig)
    print " dt =",dtCuda

    i_gpu.fill(0)

    aptr = cast(np.intp(a_gpu.gpudata), c_void_p)
    bptr = cast(np.intp(b_gpu.gpudata), c_void_p)

    nColsC = (k+64-1)/64
    c_gpu = gpuarray.zeros((m,nColsC),t,order="C")
    c_indices_gpu = gpuarray.zeros((m,nColsC),'int32',order="C")

    cptr = cast(np.intp(c_gpu.gpudata), c_void_p)
    c_indicesptr = cast(np.intp(c_indices_gpu.gpudata), c_void_p)

    start.record()    
    kmeansLib.ClosestCentersF(c_int(m),c_int(n),aptr,c_int(k),bptr,anormptr,bnormptr,
                              c_int(nColsC),cptr,c_indicesptr,iptr)
    stop.record()
    stop.synchronize()
    dtMe = .001*stop.time_since(start)

    ig = i_gpu.get()
    nErrors = 0
    for j in range(m):
        if (i[j]!=ig[j]):
            if(nErrors<100):
                print j+1," : ",i[j]," ",ig[j]
            nErrors+=1
    print 'Success status: ', np.allclose(i, ig)
    print " dt =",dtMe
    print " ratio =",dtMe/dtCuda
