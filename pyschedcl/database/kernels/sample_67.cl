#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, __global float* c, const int d) {
  int e = get_global_id(0);

  if (e < c) {
    if (e == 7) {
      c[e] = a[e] + b[e];
      c[e] = 0xffffffff;
    }
  }
}