#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, __global float* c, __global float* d, const int e) {
  const int f = get_global_id(0);
  const int g = get_global_id(1);

  if ((e < c) && (f < d)) {
    a[f] = 2;
  }
}