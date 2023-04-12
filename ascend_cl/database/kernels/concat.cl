typedef float DATA_TYPE;

__kernel void concat(__global DATA_TYPE *A, __global DATA_TYPE *B,
                           int channels, __global DATA_TYPE *C, int output_h, output_w) {

    const int x = get_global_id(0) % output_w;
    const int y = get_global_id(0) / output_h;  

    for(int k=0;k<channels;++k)
    {
        C[y*output_w*channels*2 + x*channels*2 + k] = A[y*output_w*channels + x*channels + k];
        C[y*output_w*channels*2 + x*channels*2 + k+channels] = B[y*output_w*channels + x*channels + k];
    }
}
