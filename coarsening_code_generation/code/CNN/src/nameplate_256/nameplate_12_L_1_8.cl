__kernel void linear_12( __global const float* b24, __global const float* b25, __global float* b26)
{
	int lId = get_local_id(0) ;
	int dp_0= (get_group_id(0)*8 + 1*0) * get_local_size(0) + lId ;
	int dp_1= (get_group_id(0)*8 + 1*1) * get_local_size(0) + lId ;
	int dp_2= (get_group_id(0)*8 + 1*2) * get_local_size(0) + lId ;
	int dp_3= (get_group_id(0)*8 + 1*3) * get_local_size(0) + lId ;
	int dp_4= (get_group_id(0)*8 + 1*4) * get_local_size(0) + lId ;
	int dp_5= (get_group_id(0)*8 + 1*5) * get_local_size(0) + lId ;
	int dp_6= (get_group_id(0)*8 + 1*6) * get_local_size(0) + lId ;
	int dp_7= (get_group_id(0)*8 + 1*7) * get_local_size(0) + lId ;


	if(dp_0< 4)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<32/4; x++)
		{
			temp= vload4(0,(__global const float *)b24+(4*x));
			wt.x= b25[4*(4*x)+dp_0];
			wt.y= b25[4*((4*x)+1)+dp_0];
			wt.z= b25[4*((4*x)+2)+dp_0];
			wt.w= b25[4*((4*x)+3)+dp_0];
			dotProduct += dot(wt,temp);
		}
b26[dp_0] = dotProduct;
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
			temp= vload4(0,(__global const float *)b24+(4*x));
			wt.x= b25[4*(4*x)+dp_1];
			wt.y= b25[4*((4*x)+1)+dp_1];
			wt.z= b25[4*((4*x)+2)+dp_1];
			wt.w= b25[4*((4*x)+3)+dp_1];
			dotProduct += dot(wt,temp);
		}
b26[dp_1] = dotProduct;
		printf("");
	}

	if(dp_2< 4)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<32/4; x++)
		{
			temp= vload4(0,(__global const float *)b24+(4*x));
			wt.x= b25[4*(4*x)+dp_2];
			wt.y= b25[4*((4*x)+1)+dp_2];
			wt.z= b25[4*((4*x)+2)+dp_2];
			wt.w= b25[4*((4*x)+3)+dp_2];
			dotProduct += dot(wt,temp);
		}
b26[dp_2] = dotProduct;
		printf("");
	}

	if(dp_3< 4)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<32/4; x++)
		{
			temp= vload4(0,(__global const float *)b24+(4*x));
			wt.x= b25[4*(4*x)+dp_3];
			wt.y= b25[4*((4*x)+1)+dp_3];
			wt.z= b25[4*((4*x)+2)+dp_3];
			wt.w= b25[4*((4*x)+3)+dp_3];
			dotProduct += dot(wt,temp);
		}
b26[dp_3] = dotProduct;
		printf("");
	}

	if(dp_4< 4)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<32/4; x++)
		{
			temp= vload4(0,(__global const float *)b24+(4*x));
			wt.x= b25[4*(4*x)+dp_4];
			wt.y= b25[4*((4*x)+1)+dp_4];
			wt.z= b25[4*((4*x)+2)+dp_4];
			wt.w= b25[4*((4*x)+3)+dp_4];
			dotProduct += dot(wt,temp);
		}
b26[dp_4] = dotProduct;
		printf("");
	}

	if(dp_5< 4)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<32/4; x++)
		{
			temp= vload4(0,(__global const float *)b24+(4*x));
			wt.x= b25[4*(4*x)+dp_5];
			wt.y= b25[4*((4*x)+1)+dp_5];
			wt.z= b25[4*((4*x)+2)+dp_5];
			wt.w= b25[4*((4*x)+3)+dp_5];
			dotProduct += dot(wt,temp);
		}
b26[dp_5] = dotProduct;
		printf("");
	}

	if(dp_6< 4)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<32/4; x++)
		{
			temp= vload4(0,(__global const float *)b24+(4*x));
			wt.x= b25[4*(4*x)+dp_6];
			wt.y= b25[4*((4*x)+1)+dp_6];
			wt.z= b25[4*((4*x)+2)+dp_6];
			wt.w= b25[4*((4*x)+3)+dp_6];
			dotProduct += dot(wt,temp);
		}
b26[dp_6] = dotProduct;
		printf("");
	}

	if(dp_7< 4)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<32/4; x++)
		{
			temp= vload4(0,(__global const float *)b24+(4*x));
			wt.x= b25[4*(4*x)+dp_7];
			wt.y= b25[4*((4*x)+1)+dp_7];
			wt.z= b25[4*((4*x)+2)+dp_7];
			wt.w= b25[4*((4*x)+3)+dp_7];
			dotProduct += dot(wt,temp);
		}
b26[dp_7] = dotProduct;
		printf("");
	}

}