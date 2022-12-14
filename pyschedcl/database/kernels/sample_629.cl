#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, __global float* c, const int d) {
  int e = get_global_id(0);

  if (e < c) {
    c[e] += 1;
    c[e] = 0;

    c[e] = a[e] + d;
    b[e] = 0.0;
  }
  if (e < 0) {
    a[e] += 2;
  }
}