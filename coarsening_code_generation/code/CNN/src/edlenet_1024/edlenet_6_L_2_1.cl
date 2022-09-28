__kernel void linear_6( __global const float* b3, __global const float* b4, __global float* b5)
{
	int lId = get_local_id(0) ;
	int gId = (lId/8)*8*2 + lId%8 ;

	int dp_0= get_group_id(0)*2*get_local_size(0) + gId + 8*0 ;
	int dp_1= get_group_id(0)*2*get_local_size(0) + gId + 8*1 ;


	if(dp_0< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<2097152/4; x++)
		{
			temp= vload4(0,(__global const float *)b3+(4*x));
			wt.x= b4[128*(4*x)+dp_0];
			wt.y= b4[128*((4*x)+1)+dp_0];
			wt.z= b4[128*((4*x)+2)+dp_0];
			wt.w= b4[128*((4*x)+3)+dp_0];
			dotProduct += dot(wt,temp);
		}
b5[dp_0] = dotProduct;
		printf("");
	}

	if(dp_1< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<2097152/4; x++)
		{
			temp= vload4(0,(__global const float *)b3+(4*x));
			wt.x= b4[128*(4*x)+dp_1];
			wt.y= b4[128*((4*x)+1)+dp_1];
			wt.z= b4[128*((4*x)+2)+dp_1];
			wt.w= b4[128*((4*x)+3)+dp_1];
			dotProduct += dot(wt,temp);
		}
b5[dp_1] = dotProduct;
		printf("");
	}

}