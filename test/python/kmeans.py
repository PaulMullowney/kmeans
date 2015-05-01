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

import string

from ctypes import *
cdll.LoadLibrary("/home/paul/project/kmeans-gpu/lib/libCudaKernelLibrary.so")
cudaLib = CDLL("/home/paul/project/kmeans-gpu/lib/libCudaKernelLibrary.so")

cdll.LoadLibrary("/home/paul/project/kmeans-gpu/lib/libkmeans.so")
kmeansLib = CDLL("/home/paul/project/kmeans-gpu/lib/libkmeans.so")

# Double precision is only supported by devices with compute
# capability >= 1.3:

def kmeans(file, m, n, k, maxIters, useCUBLAS=0, numRetries=1):
    print 'Loading array from file ',file
    data = np.fromfile(file)

    # timers
    start = cuda.Event()
    stop = cuda.Event()

    centers = np.zeros((n,k),'float32',order="C")

    dataPtr = cast(np.intp(data.ctypes.data), c_void_p)
    centersPtr = cast(np.intp(centers.ctypes.data), c_void_p)

    print 'Running Kmeans'
    start.record()    
    err = kmeansLib.computeKmeansF(dataPtr, c_int(m), c_int(n), c_int(k), c_float(1.e-5),
                                   c_int(maxIters), c_int(numRetries), c_int(0), c_int(useCUBLAS), 
                                   centersPtr)
 
    stop.record()
    stop.synchronize()
    dtKmeans = .001*stop.time_since(start)

    print " dt =",dtKmeans, " error =",err
