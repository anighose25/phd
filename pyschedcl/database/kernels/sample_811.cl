#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, __global float* c, const int d) {
  int e = get_global_id(0);

  if (e < c) {
    c[e] += 1;
    barrier(1);
    if (e < d)
      b[e] += b[e - 1];
  }

  b[e] = a[e];
}