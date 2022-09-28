__kernel void softmax_8( __global const float* b6, __global float* b7)
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


	if(dp_0 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b6[0];

		for(int i=0;i<class;i++)
			max = (max > b6[i]) ? max : b6[i];

		for(int i=0;i<class;i++)
			b7[i] = exp((b6[i] - max));

		for(int i=0;i<class;i++)
			sum+=b7[i];

		for(int i=0;i<class;i++)
			b7[i] = b7[i]/sum;

		printf("");
	}

	if(dp_1 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b6[0];

		for(int i=0;i<class;i++)
			max = (max > b6[i]) ? max : b6[i];

		for(int i=0;i<class;i++)
			b7[i] = exp((b6[i] - max));

		for(int i=0;i<class;i++)
			sum+=b7[i];

		for(int i=0;i<class;i++)
			b7[i] = b7[i]/sum;

		printf("");
	}

	if(dp_2 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b6[0];

		for(int i=0;i<class;i++)
			max = (max > b6[i]) ? max : b6[i];

		for(int i=0;i<class;i++)
			b7[i] = exp((b6[i] - max));

		for(int i=0;i<class;i++)
			sum+=b7[i];

		for(int i=0;i<class;i++)
			b7[i] = b7[i]/sum;

		printf("");
	}

	if(dp_3 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b6[0];

		for(int i=0;i<class;i++)
			max = (max > b6[i]) ? max : b6[i];

		for(int i=0;i<class;i++)
			b7[i] = exp((b6[i] - max));

		for(int i=0;i<class;i++)
			sum+=b7[i];

		for(int i=0;i<class;i++)
			b7[i] = b7[i]/sum;

		printf("");
	}

	if(dp_4 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b6[0];

		for(int i=0;i<class;i++)
			max = (max > b6[i]) ? max : b6[i];

		for(int i=0;i<class;i++)
			b7[i] = exp((b6[i] - max));

		for(int i=0;i<class;i++)
			sum+=b7[i];

		for(int i=0;i<class;i++)
			b7[i] = b7[i]/sum;

		printf("");
	}

	if(dp_5 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b6[0];

		for(int i=0;i<class;i++)
			max = (max > b6[i]) ? max : b6[i];

		for(int i=0;i<class;i++)
			b7[i] = exp((b6[i] - max));

		for(int i=0;i<class;i++)
			sum+=b7[i];

		for(int i=0;i<class;i++)
			b7[i] = b7[i]/sum;

		printf("");
	}

	if(dp_6 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b6[0];

		for(int i=0;i<class;i++)
			max = (max > b6[i]) ? max : b6[i];

		for(int i=0;i<class;i++)
			b7[i] = exp((b6[i] - max));

		for(int i=0;i<class;i++)
			sum+=b7[i];

		for(int i=0;i<class;i++)
			b7[i] = b7[i]/sum;

		printf("");
	}

	if(dp_7 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b6[0];

		for(int i=0;i<class;i++)
			max = (max > b6[i]) ? max : b6[i];

		for(int i=0;i<class;i++)
			b7[i] = exp((b6[i] - max));

		for(int i=0;i<class;i++)
			sum+=b7[i];

		for(int i=0;i<class;i++)
			b7[i] = b7[i]/sum;

		printf("");
	}

}