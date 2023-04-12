#ifndef PRECISION
  #define PRECISION 32      // Data-types: half, single or double precision, complex or regular
#endif

#if PRECISION == 16
  #pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

//common vars
#ifndef FLT_MAX
#define FLT_MAX 0
#endif

#ifndef FLT_MIN
#define FLT_MIN 0
#endif

// Half-precision
#if PRECISION == 16
  typedef half real;
  typedef half2 real2;
  typedef half4 real4;
  typedef half8 real8;
  typedef half16 real16;
  #define ZERO 0
  #define ONE 1
  #define SMALLEST -1.0e14

// Single-precision
#elif PRECISION == 32
  typedef float real;
  typedef float2 real2;
  typedef float4 real4;
  typedef float8 real8;
  typedef float16 real16;
  #define ZERO 0.0f
  #define ONE 1.0f
  #define SMALLEST -1.0e37f
#endif

#ifndef VWM
    #define VWM 4
#endif

#ifdef VWM
    #if VWM == 1
        typedef real realM;
        #define vloadM(X) vload(0, &(X));
        #define dotM(X,Y) ((X) * (Y))
    #elif VWM == 2
        typedef real2 realM;
        #define vloadM(X) vload2(0, &(X));
        #define dotM(X,Y) dot((X),(Y))
    #elif VWM == 4
        typedef real4 realM;
        #define vloadM(X) vload4(0, &(X));
        #define dotM(X,Y) dot((X),(Y))
    #elif VWM == 8
        typedef real8 realM;
        #define vloadM(X) vload8(0, &(X));
        #define dotM(X,Y) (dot((X.s0123),(Y.s0123)) + dot((X.s4567),(Y.s4567)))
    #elif VWM == 16
        typedef real16 realM;
        #define vloadM(X) vload16(0, &(X));
        #define dotM(X,Y) (dot((X.s0123),(Y.s0123)) + dot((X.s4567),(Y.s4567)) + dot((X.s89ab),(Y.s89ab)) + dot((X.scdef),(Y.scdef)))
    #endif
#endif

#ifndef USE_CL_MAD
  #define USE_CL_MAD 0
#endif

static inline int getIndexFrom3D(int d1, int d2, int d3, int i1, int i2, int i3) {
	return i1 * (d2 * d3) + i2 * d3 + i3;
}

static inline int getIndexFrom4D(int d1, int d2, int d3, int d4, int i1, int i2, int i3, int i4) {
	return i1 * (d2 * d3 * d4) + i2 * (d3 * d4) + i3 * d4 + i4;
}

#if PRECISION == 16
__kernel void convertFloatToHalf(
    __global const float *input,
    __global half *output) {
    int idx = get_global_id(0);
    vstore_half(input[idx], 0, &output[idx]);
}

__kernel void convertHalfToFloat(
    __global const half *input,
    __global float *output) {
    int idx = get_global_id(0);
    output[idx] = convert_float(input[idx]);
    //output[idx] = (float)input[idx];
}
#endif

__kernel void memcpy(
        __global const real *input,
        __global real *output) {
    int idx = get_global_id(0);
    output[idx] = input[idx];
}


__kernel void caffe_maxpool(
    const int nthreads,
    __global const real* bottom_data,
    const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    __global real* top_data) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, (int)0);
    wstart = max(wstart, (int)0);
    real maxval = -FLT_MAX;
    int maxidx = -1;
    __global const real* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
  }
}

__kernel void caffe_avepool(
    const int nthreads, __global const real* const bottom_data, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, __global real* top_data) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    {
      const int pw = index % pooled_width;
      const int ph = (index / pooled_width) % pooled_height;
      const int c = (index / pooled_width / pooled_height) % channels;
      const int n = index / pooled_width / pooled_height / channels;
      int hstart = ph * stride_h - pad_h;
      int wstart = pw * stride_w - pad_w;
      int hend = min(hstart + kernel_h, height + pad_h);
      int wend = min(wstart + kernel_w, width + pad_w);
      const int pool_size = (hend - hstart) * (wend - wstart);
      hstart = max(hstart, (int)0);
      wstart = max(wstart, (int)0);
      hend = min(hend, height);
      wend = min(wend, width);
      real aveval = 0;
      __global const real* bottom_slice = bottom_data
          + (n * channels + c) * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          aveval += bottom_slice[h * width + w];
        }
      }
      top_data[index] = aveval / pool_size;
    }
  }
}

__kernel void dm_maxpool(
    __global const real *input_frame,
    const int input_w,
    const int input_h,
    const int num_channels,
    const int filter_w,
    const int filter_h,
    const int stride_w,
    const int stride_h,
    const int pad_w,
    const int pad_h,
    __global real *output_frame,
    const int output_w,
    const int output_h,
    const int batches) {

    int thrId_i = get_global_id(0);
    int thrId_j = get_global_id(1);
    int thrId_k = get_global_id(2);

    int max_i = get_global_size(0);
    int max_j = get_global_size(1);
    int max_k = get_global_size(2);

    int i,j,k;
    int x,y;
    for(k = thrId_k ; k < batches * num_channels ; k += max_k) {
        for(i = thrId_i ; i < output_w ; i += max_i) {
            for(j = thrId_j ; j < output_h ; j += max_j) {
                real max_value = -9999.9f;
                for(x = 0 ; x < filter_w ; x++) {
                    for(y = 0 ; y < filter_h ; y++) {
                        int x_ = i * stride_w + x - pad_w;
                        int y_ = j * stride_h + y - pad_h;
                        int valid = (x_ >= 0 && x_ < input_w && y_ >= 0 && y_ < input_h);
                        real val = (valid != 0) ? input_frame[getIndexFrom3D(input_h, input_w, num_channels, y_, x_, k)] : 0.0f;
                        max_value   = (val > max_value) ? val   : max_value;
                    }
                }
                output_frame[getIndexFrom3D(output_h, output_w, num_channels, j, i, k)] = max_value;
            }
        }
    }
}

__kernel void dm_avepool(
    __global const real *input_frame,
    const int input_w,
    const int input_h,
    const int num_channels,
    const int filter_w,
    const int filter_h,
    const int stride_w,
    const int stride_h,
    const int pad_w,
    const int pad_h,
    __global real *output_frame,
    const int output_w,
    const int output_h,
    const int batches) {

    int thrId_i = get_global_id(0);
    int thrId_j = get_global_id(1);
    int thrId_k = get_global_id(2);

    int max_i = get_global_size(0);
    int max_j = get_global_size(1);
    int max_k = get_global_size(2);

    int i,j,k;
    int x,y;

    for(k = thrId_k ; k < batches * num_channels ; k += max_k) {
        for(i = thrId_i ; i < output_w ; i += max_i) {
            for(j = thrId_j ; j < output_h ; j += max_j) {
                real avg = 0;
                for(x = 0 ; x < filter_w ; x++) {
                    for(y = 0 ; y < filter_h ; y++) {
                        int x_ = i * stride_w + x - pad_w;
                        int y_ = j * stride_h + y - pad_h;
                        int valid = (x_ >= 0 && x_ < input_w && y_ >= 0 && y_ < input_h);
                        avg += (valid != 0) ? input_frame[getIndexFrom3D(input_h, input_w, num_channels, y_, x_, k)] : 0.0f;
                    }
                }
                output_frame[getIndexFrom3D(output_h, output_w, num_channels, j, i, k)] = avg / (filter_w * filter_h);
            }
        }
    }
}