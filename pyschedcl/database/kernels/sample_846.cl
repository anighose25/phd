#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, __global float* c, const int d) {
  int e = get_global_id(0);
  int f = get_global_id(1);

  if (e < c && f < d) {
    c[e] = a[e] + b[e];
  }
}