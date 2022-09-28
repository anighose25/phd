__kernel void linear_6( __global const float* b27, __global const float* b28, __global float* b29)
{
	int lId = get_local_id(0) ;
	int dp_0= (get_group_id(0)*16 + 1*0) * get_local_size(0) + lId ;
	int dp_1= (get_group_id(0)*16 + 1*1) * get_local_size(0) + lId ;
	int dp_2= (get_group_id(0)*16 + 1*2) * get_local_size(0) + lId ;
	int dp_3= (get_group_id(0)*16 + 1*3) * get_local_size(0) + lId ;
	int dp_4= (get_group_id(0)*16 + 1*4) * get_local_size(0) + lId ;
	int dp_5= (get_group_id(0)*16 + 1*5) * get_local_size(0) + lId ;
	int dp_6= (get_group_id(0)*16 + 1*6) * get_local_size(0) + lId ;
	int dp_7= (get_group_id(0)*16 + 1*7) * get_local_size(0) + lId ;
	int dp_8= (get_group_id(0)*16 + 1*8) * get_local_size(0) + lId ;
	int dp_9= (get_group_id(0)*16 + 1*9) * get_local_size(0) + lId ;
	int dp_10= (get_group_id(0)*16 + 1*10) * get_local_size(0) + lId ;
	int dp_11= (get_group_id(0)*16 + 1*11) * get_local_size(0) + lId ;
	int dp_12= (get_group_id(0)*16 + 1*12) * get_local_size(0) + lId ;
	int dp_13= (get_group_id(0)*16 + 1*13) * get_local_size(0) + lId ;
	int dp_14= (get_group_id(0)*16 + 1*14) * get_local_size(0) + lId ;
	int dp_15= (get_group_id(0)*16 + 1*15) * get_local_size(0) + lId ;


	if(dp_0< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_0];
			wt.y= b28[128*((4*x)+1)+dp_0];
			wt.z= b28[128*((4*x)+2)+dp_0];
			wt.w= b28[128*((4*x)+3)+dp_0];
			dotProduct += dot(wt,temp);
		}
b29[dp_0] = dotProduct;
		printf("");
	}

	if(dp_1< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_1];
			wt.y= b28[128*((4*x)+1)+dp_1];
			wt.z= b28[128*((4*x)+2)+dp_1];
			wt.w= b28[128*((4*x)+3)+dp_1];
			dotProduct += dot(wt,temp);
		}
b29[dp_1] = dotProduct;
		printf("");
	}

	if(dp_2< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_2];
			wt.y= b28[128*((4*x)+1)+dp_2];
			wt.z= b28[128*((4*x)+2)+dp_2];
			wt.w= b28[128*((4*x)+3)+dp_2];
			dotProduct += dot(wt,temp);
		}
b29[dp_2] = dotProduct;
		printf("");
	}

	if(dp_3< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_3];
			wt.y= b28[128*((4*x)+1)+dp_3];
			wt.z= b28[128*((4*x)+2)+dp_3];
			wt.w= b28[128*((4*x)+3)+dp_3];
			dotProduct += dot(wt,temp);
		}
b29[dp_3] = dotProduct;
		printf("");
	}

	if(dp_4< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_4];
			wt.y= b28[128*((4*x)+1)+dp_4];
			wt.z= b28[128*((4*x)+2)+dp_4];
			wt.w= b28[128*((4*x)+3)+dp_4];
			dotProduct += dot(wt,temp);
		}
b29[dp_4] = dotProduct;
		printf("");
	}

	if(dp_5< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_5];
			wt.y= b28[128*((4*x)+1)+dp_5];
			wt.z= b28[128*((4*x)+2)+dp_5];
			wt.w= b28[128*((4*x)+3)+dp_5];
			dotProduct += dot(wt,temp);
		}
b29[dp_5] = dotProduct;
		printf("");
	}

	if(dp_6< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_6];
			wt.y= b28[128*((4*x)+1)+dp_6];
			wt.z= b28[128*((4*x)+2)+dp_6];
			wt.w= b28[128*((4*x)+3)+dp_6];
			dotProduct += dot(wt,temp);
		}
b29[dp_6] = dotProduct;
		printf("");
	}

	if(dp_7< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_7];
			wt.y= b28[128*((4*x)+1)+dp_7];
			wt.z= b28[128*((4*x)+2)+dp_7];
			wt.w= b28[128*((4*x)+3)+dp_7];
			dotProduct += dot(wt,temp);
		}
b29[dp_7] = dotProduct;
		printf("");
	}

	if(dp_8< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_8];
			wt.y= b28[128*((4*x)+1)+dp_8];
			wt.z= b28[128*((4*x)+2)+dp_8];
			wt.w= b28[128*((4*x)+3)+dp_8];
			dotProduct += dot(wt,temp);
		}
b29[dp_8] = dotProduct;
		printf("");
	}

	if(dp_9< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_9];
			wt.y= b28[128*((4*x)+1)+dp_9];
			wt.z= b28[128*((4*x)+2)+dp_9];
			wt.w= b28[128*((4*x)+3)+dp_9];
			dotProduct += dot(wt,temp);
		}
b29[dp_9] = dotProduct;
		printf("");
	}

	if(dp_10< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_10];
			wt.y= b28[128*((4*x)+1)+dp_10];
			wt.z= b28[128*((4*x)+2)+dp_10];
			wt.w= b28[128*((4*x)+3)+dp_10];
			dotProduct += dot(wt,temp);
		}
b29[dp_10] = dotProduct;
		printf("");
	}

	if(dp_11< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_11];
			wt.y= b28[128*((4*x)+1)+dp_11];
			wt.z= b28[128*((4*x)+2)+dp_11];
			wt.w= b28[128*((4*x)+3)+dp_11];
			dotProduct += dot(wt,temp);
		}
b29[dp_11] = dotProduct;
		printf("");
	}

	if(dp_12< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_12];
			wt.y= b28[128*((4*x)+1)+dp_12];
			wt.z= b28[128*((4*x)+2)+dp_12];
			wt.w= b28[128*((4*x)+3)+dp_12];
			dotProduct += dot(wt,temp);
		}
b29[dp_12] = dotProduct;
		printf("");
	}

	if(dp_13< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_13];
			wt.y= b28[128*((4*x)+1)+dp_13];
			wt.z= b28[128*((4*x)+2)+dp_13];
			wt.w= b28[128*((4*x)+3)+dp_13];
			dotProduct += dot(wt,temp);
		}
b29[dp_13] = dotProduct;
		printf("");
	}

	if(dp_14< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_14];
			wt.y= b28[128*((4*x)+1)+dp_14];
			wt.z= b28[128*((4*x)+2)+dp_14];
			wt.w= b28[128*((4*x)+3)+dp_14];
			dotProduct += dot(wt,temp);
		}
b29[dp_14] = dotProduct;
		printf("");
	}

	if(dp_15< 128)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<8192/4; x++)
		{
			temp= vload4(0,(__global const float *)b27+(4*x));
			wt.x= b28[128*(4*x)+dp_15];
			wt.y= b28[128*((4*x)+1)+dp_15];
			wt.z= b28[128*((4*x)+2)+dp_15];
			wt.w= b28[128*((4*x)+3)+dp_15];
			dotProduct += dot(wt,temp);
		}
b29[dp_15] = dotProduct;
		printf("");
	}

}