__kernel void softmax_8( __global const float* b8, __global float* b9)
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


	if(dp_0 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_1 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_2 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_3 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_4 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_5 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_6 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_7 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_8 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_9 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_10 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_11 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_12 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_13 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_14 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

	if(dp_15 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b8[0];

		for(int i=0;i<class;i++)
			max = (max > b8[i]) ? max : b8[i];

		for(int i=0;i<class;i++)
			b9[i] = exp((b8[i] - max));

		for(int i=0;i<class;i++)
			sum+=b9[i];

		for(int i=0;i<class;i++)
			b9[i] = b9[i]/sum;

		printf("");
	}

}