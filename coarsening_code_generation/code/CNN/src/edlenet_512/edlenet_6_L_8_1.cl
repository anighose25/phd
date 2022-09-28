__kernel void linear_6( __global const float* b9, __global const float* b10, __global float* b11)
{
	int lId = get_local_id(0) ;
	int gId = (lId/8)*8*8 + lId%8 ;

	int dp_0= get_group_id(0)*8*get_local_size(0) + gId + 8*0 ;
	int dp_1= get_group_id(0)*8*get_local_size(0) + gId + 8*1 ;
	int dp_2= get_group_id(0)*8*get_local_size(0) + gId + 8*2 ;
	int dp_3= get_group_id(0)*8*get_local_size(0) + gId + 8*3 ;
	int dp_4= get_group_id(0)*8*get_local_size(0) + gId + 8*4 ;
	int dp_5= get_group_id(0)*8*get_local_size(0) + gId + 8*5 ;
	int dp_6= get_group_id(0)*8*get_local_size(0) + gId + 8*6 ;
	int dp_7= get_group_id(0)*8*get_local_size(0) + gId + 8*7 ;


	if(dp_0< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<524288/4; x++)
		{
			temp= vload4(0,(__global const float *)b9+(4*x));
			wt.x= b10[128*(4*x)+dp_0];
			wt.y= b10[128*((4*x)+1)+dp_0];
			wt.z= b10[128*((4*x)+2)+dp_0];
			wt.w= b10[128*((4*x)+3)+dp_0];
			dotProduct += dot(wt,temp);
		}
b11[dp_0] = dotProduct;
		printf("");
	}

	if(dp_1< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<524288/4; x++)
		{
			temp= vload4(0,(__global const float *)b9+(4*x));
			wt.x= b10[128*(4*x)+dp_1];
			wt.y= b10[128*((4*x)+1)+dp_1];
			wt.z= b10[128*((4*x)+2)+dp_1];
			wt.w= b10[128*((4*x)+3)+dp_1];
			dotProduct += dot(wt,temp);
		}
b11[dp_1] = dotProduct;
		printf("");
	}

	if(dp_2< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<524288/4; x++)
		{
			temp= vload4(0,(__global const float *)b9+(4*x));
			wt.x= b10[128*(4*x)+dp_2];
			wt.y= b10[128*((4*x)+1)+dp_2];
			wt.z= b10[128*((4*x)+2)+dp_2];
			wt.w= b10[128*((4*x)+3)+dp_2];
			dotProduct += dot(wt,temp);
		}
b11[dp_2] = dotProduct;
		printf("");
	}

	if(dp_3< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<524288/4; x++)
		{
			temp= vload4(0,(__global const float *)b9+(4*x));
			wt.x= b10[128*(4*x)+dp_3];
			wt.y= b10[128*((4*x)+1)+dp_3];
			wt.z= b10[128*((4*x)+2)+dp_3];
			wt.w= b10[128*((4*x)+3)+dp_3];
			dotProduct += dot(wt,temp);
		}
b11[dp_3] = dotProduct;
		printf("");
	}

	if(dp_4< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<524288/4; x++)
		{
			temp= vload4(0,(__global const float *)b9+(4*x));
			wt.x= b10[128*(4*x)+dp_4];
			wt.y= b10[128*((4*x)+1)+dp_4];
			wt.z= b10[128*((4*x)+2)+dp_4];
			wt.w= b10[128*((4*x)+3)+dp_4];
			dotProduct += dot(wt,temp);
		}
b11[dp_4] = dotProduct;
		printf("");
	}

	if(dp_5< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<524288/4; x++)
		{
			temp= vload4(0,(__global const float *)b9+(4*x));
			wt.x= b10[128*(4*x)+dp_5];
			wt.y= b10[128*((4*x)+1)+dp_5];
			wt.z= b10[128*((4*x)+2)+dp_5];
			wt.w= b10[128*((4*x)+3)+dp_5];
			dotProduct += dot(wt,temp);
		}
b11[dp_5] = dotProduct;
		printf("");
	}

	if(dp_6< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<524288/4; x++)
		{
			temp= vload4(0,(__global const float *)b9+(4*x));
			wt.x= b10[128*(4*x)+dp_6];
			wt.y= b10[128*((4*x)+1)+dp_6];
			wt.z= b10[128*((4*x)+2)+dp_6];
			wt.w= b10[128*((4*x)+3)+dp_6];
			dotProduct += dot(wt,temp);
		}
b11[dp_6] = dotProduct;
		printf("");
	}

	if(dp_7< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<524288/4; x++)
		{
			temp= vload4(0,(__global const float *)b9+(4*x));
			wt.x= b10[128*(4*x)+dp_7];
			wt.y= b10[128*((4*x)+1)+dp_7];
			wt.z= b10[128*((4*x)+2)+dp_7];
			wt.w= b10[128*((4*x)+3)+dp_7];
			dotProduct += dot(wt,temp);
		}
b11[dp_7] = dotProduct;
		printf("");
	}

}