__kernel void linear_7( __global const float* b12, __global const float* b13, __global float* b14)
{
	int lId = get_local_id(0) ;
	int gId = (lId/8)*8*16 + lId%8 ;

	int dp_0= get_group_id(0)*16*get_local_size(0) + gId + 8*0 ;
	int dp_1= get_group_id(0)*16*get_local_size(0) + gId + 8*1 ;
	int dp_2= get_group_id(0)*16*get_local_size(0) + gId + 8*2 ;
	int dp_3= get_group_id(0)*16*get_local_size(0) + gId + 8*3 ;
	int dp_4= get_group_id(0)*16*get_local_size(0) + gId + 8*4 ;
	int dp_5= get_group_id(0)*16*get_local_size(0) + gId + 8*5 ;
	int dp_6= get_group_id(0)*16*get_local_size(0) + gId + 8*6 ;
	int dp_7= get_group_id(0)*16*get_local_size(0) + gId + 8*7 ;
	int dp_8= get_group_id(0)*16*get_local_size(0) + gId + 8*8 ;
	int dp_9= get_group_id(0)*16*get_local_size(0) + gId + 8*9 ;
	int dp_10= get_group_id(0)*16*get_local_size(0) + gId + 8*10 ;
	int dp_11= get_group_id(0)*16*get_local_size(0) + gId + 8*11 ;
	int dp_12= get_group_id(0)*16*get_local_size(0) + gId + 8*12 ;
	int dp_13= get_group_id(0)*16*get_local_size(0) + gId + 8*13 ;
	int dp_14= get_group_id(0)*16*get_local_size(0) + gId + 8*14 ;
	int dp_15= get_group_id(0)*16*get_local_size(0) + gId + 8*15 ;


	if(dp_0< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_0];
			wt.y= b13[16*((4*x)+1)+dp_0];
			wt.z= b13[16*((4*x)+2)+dp_0];
			wt.w= b13[16*((4*x)+3)+dp_0];
			dotProduct += dot(wt,temp);
		}
b14[dp_0] = dotProduct;
		printf("");
	}

	if(dp_1< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_1];
			wt.y= b13[16*((4*x)+1)+dp_1];
			wt.z= b13[16*((4*x)+2)+dp_1];
			wt.w= b13[16*((4*x)+3)+dp_1];
			dotProduct += dot(wt,temp);
		}
b14[dp_1] = dotProduct;
		printf("");
	}

	if(dp_2< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_2];
			wt.y= b13[16*((4*x)+1)+dp_2];
			wt.z= b13[16*((4*x)+2)+dp_2];
			wt.w= b13[16*((4*x)+3)+dp_2];
			dotProduct += dot(wt,temp);
		}
b14[dp_2] = dotProduct;
		printf("");
	}

	if(dp_3< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_3];
			wt.y= b13[16*((4*x)+1)+dp_3];
			wt.z= b13[16*((4*x)+2)+dp_3];
			wt.w= b13[16*((4*x)+3)+dp_3];
			dotProduct += dot(wt,temp);
		}
b14[dp_3] = dotProduct;
		printf("");
	}

	if(dp_4< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_4];
			wt.y= b13[16*((4*x)+1)+dp_4];
			wt.z= b13[16*((4*x)+2)+dp_4];
			wt.w= b13[16*((4*x)+3)+dp_4];
			dotProduct += dot(wt,temp);
		}
b14[dp_4] = dotProduct;
		printf("");
	}

	if(dp_5< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_5];
			wt.y= b13[16*((4*x)+1)+dp_5];
			wt.z= b13[16*((4*x)+2)+dp_5];
			wt.w= b13[16*((4*x)+3)+dp_5];
			dotProduct += dot(wt,temp);
		}
b14[dp_5] = dotProduct;
		printf("");
	}

	if(dp_6< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_6];
			wt.y= b13[16*((4*x)+1)+dp_6];
			wt.z= b13[16*((4*x)+2)+dp_6];
			wt.w= b13[16*((4*x)+3)+dp_6];
			dotProduct += dot(wt,temp);
		}
b14[dp_6] = dotProduct;
		printf("");
	}

	if(dp_7< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_7];
			wt.y= b13[16*((4*x)+1)+dp_7];
			wt.z= b13[16*((4*x)+2)+dp_7];
			wt.w= b13[16*((4*x)+3)+dp_7];
			dotProduct += dot(wt,temp);
		}
b14[dp_7] = dotProduct;
		printf("");
	}

	if(dp_8< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_8];
			wt.y= b13[16*((4*x)+1)+dp_8];
			wt.z= b13[16*((4*x)+2)+dp_8];
			wt.w= b13[16*((4*x)+3)+dp_8];
			dotProduct += dot(wt,temp);
		}
b14[dp_8] = dotProduct;
		printf("");
	}

	if(dp_9< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_9];
			wt.y= b13[16*((4*x)+1)+dp_9];
			wt.z= b13[16*((4*x)+2)+dp_9];
			wt.w= b13[16*((4*x)+3)+dp_9];
			dotProduct += dot(wt,temp);
		}
b14[dp_9] = dotProduct;
		printf("");
	}

	if(dp_10< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_10];
			wt.y= b13[16*((4*x)+1)+dp_10];
			wt.z= b13[16*((4*x)+2)+dp_10];
			wt.w= b13[16*((4*x)+3)+dp_10];
			dotProduct += dot(wt,temp);
		}
b14[dp_10] = dotProduct;
		printf("");
	}

	if(dp_11< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_11];
			wt.y= b13[16*((4*x)+1)+dp_11];
			wt.z= b13[16*((4*x)+2)+dp_11];
			wt.w= b13[16*((4*x)+3)+dp_11];
			dotProduct += dot(wt,temp);
		}
b14[dp_11] = dotProduct;
		printf("");
	}

	if(dp_12< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_12];
			wt.y= b13[16*((4*x)+1)+dp_12];
			wt.z= b13[16*((4*x)+2)+dp_12];
			wt.w= b13[16*((4*x)+3)+dp_12];
			dotProduct += dot(wt,temp);
		}
b14[dp_12] = dotProduct;
		printf("");
	}

	if(dp_13< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_13];
			wt.y= b13[16*((4*x)+1)+dp_13];
			wt.z= b13[16*((4*x)+2)+dp_13];
			wt.w= b13[16*((4*x)+3)+dp_13];
			dotProduct += dot(wt,temp);
		}
b14[dp_13] = dotProduct;
		printf("");
	}

	if(dp_14< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_14];
			wt.y= b13[16*((4*x)+1)+dp_14];
			wt.z= b13[16*((4*x)+2)+dp_14];
			wt.w= b13[16*((4*x)+3)+dp_14];
			dotProduct += dot(wt,temp);
		}
b14[dp_14] = dotProduct;
		printf("");
	}

	if(dp_15< 16)
	{
		typedef float4 floatX;
		floatX wt,temp;
		float dotProduct;

		dotProduct=0.0;
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b12+(4*x));
			wt.x= b13[16*(4*x)+dp_15];
			wt.y= b13[16*((4*x)+1)+dp_15];
			wt.z= b13[16*((4*x)+2)+dp_15];
			wt.w= b13[16*((4*x)+3)+dp_15];
			dotProduct += dot(wt,temp);
		}
b14[dp_15] = dotProduct;
		printf("");
	}

}