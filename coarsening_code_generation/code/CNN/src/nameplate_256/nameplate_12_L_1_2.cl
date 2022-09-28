__kernel void linear_12( __global const float* b18, __global const float* b19, __global float* b20)
{
	int lId = get_local_id(0) ;
	int dp_0= (get_group_id(0)*2 + 1*0) * get_local_size(0) + lId ;
	int dp_1= (get_group_id(0)*2 + 1*1) * get_local_size(0) + lId ;


	if(dp_0< 4)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<32/4; x++)
		{
			temp= vload4(0,(__global const float *)b18+(4*x));
			wt.x= b19[4*(4*x)+dp_0];
			wt.y= b19[4*((4*x)+1)+dp_0];
			wt.z= b19[4*((4*x)+2)+dp_0];
			wt.w= b19[4*((4*x)+3)+dp_0];
			dotProduct += dot(wt,temp);
		}
b20[dp_0] = dotProduct;
		printf("");
	}

	if(dp_1< 4)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<32/4; x++)
		{
			temp= vload4(0,(__global const float *)b18+(4*x));
			wt.x= b19[4*(4*x)+dp_1];
			wt.y= b19[4*((4*x)+1)+dp_1];
			wt.z= b19[4*((4*x)+2)+dp_1];
			wt.w= b19[4*((4*x)+3)+dp_1];
			dotProduct += dot(wt,temp);
		}
b20[dp_1] = dotProduct;
		printf("");
	}

}