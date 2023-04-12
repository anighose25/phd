#define vloadM(X) vload4(0, &(X));
#define vstoreM(vx,X) vstore4(vx, 0, &(X));
#define dotM(X,Y) dot((X),(Y))

__kernel void mm(__global float* A,
__global float* B,
__global float* C, int m,int n,int k) 
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	int nv4 = n >> 2;
	float4 accum = (float4)(0.0f);
	for (int k = 0; k < nv4; ++k) {
	   float4 a  = vloadM(A[i*nv4+k]);
	   float4 b0 = vloadM(B[(4*k)*nv4+k]);
	   float4 b1 = vloadM(B[(4*k+1)*nv4+k]);
	   float4 b2 = vloadM(B[(4*k+2)*nv4+k]);
	   float4 b3 = vloadM(B[(4*k+3)*nv4+k]);
	   accum += a.s0*b0 +a.s1*b1+a.s2*b2+a.s3*b3;
	}
	vstoreM(accum,C[i*nv4+j]);
}
