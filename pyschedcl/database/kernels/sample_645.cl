#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, __global float* c, const int d) {
  int e = get_global_id(0);
  if (e < d) {
    float f = b[0];
    float g = a[1] + a[e];

    a[e] = -1;
  }
}