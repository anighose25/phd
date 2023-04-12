__kernel void softmax_8( __global const float* b16, __global float* b17)
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


	if(dp_0 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b16[0];

		for(int i=0;i<class;i++)
			max = (max > b16[i]) ? max : b16[i];

		for(int i=0;i<class;i++)
			b17[i] = exp((b16[i] - max));

		for(int i=0;i<class;i++)
			sum+=b17[i];

		for(int i=0;i<class;i++)
			b17[i] = b17[i]/sum;

		printf("");
	}

	if(dp_1 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b16[0];

		for(int i=0;i<class;i++)
			max = (max > b16[i]) ? max : b16[i];

		for(int i=0;i<class;i++)
			b17[i] = exp((b16[i] - max));

		for(int i=0;i<class;i++)
			sum+=b17[i];

		for(int i=0;i<class;i++)
			b17[i] = b17[i]/sum;

		printf("");
	}

	if(dp_2 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b16[0];

		for(int i=0;i<class;i++)
			max = (max > b16[i]) ? max : b16[i];

		for(int i=0;i<class;i++)
			b17[i] = exp((b16[i] - max));

		for(int i=0;i<class;i++)
			sum+=b17[i];

		for(int i=0;i<class;i++)
			b17[i] = b17[i]/sum;

		printf("");
	}

	if(dp_3 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b16[0];

		for(int i=0;i<class;i++)
			max = (max > b16[i]) ? max : b16[i];

		for(int i=0;i<class;i++)
			b17[i] = exp((b16[i] - max));

		for(int i=0;i<class;i++)
			sum+=b17[i];

		for(int i=0;i<class;i++)
			b17[i] = b17[i]/sum;

		printf("");
	}

	if(dp_4 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b16[0];

		for(int i=0;i<class;i++)
			max = (max > b16[i]) ? max : b16[i];

		for(int i=0;i<class;i++)
			b17[i] = exp((b16[i] - max));

		for(int i=0;i<class;i++)
			sum+=b17[i];

		for(int i=0;i<class;i++)
			b17[i] = b17[i]/sum;

		printf("");
	}

	if(dp_5 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b16[0];

		for(int i=0;i<class;i++)
			max = (max > b16[i]) ? max : b16[i];

		for(int i=0;i<class;i++)
			b17[i] = exp((b16[i] - max));

		for(int i=0;i<class;i++)
			sum+=b17[i];

		for(int i=0;i<class;i++)
			b17[i] = b17[i]/sum;

		printf("");
	}

	if(dp_6 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b16[0];

		for(int i=0;i<class;i++)
			max = (max > b16[i]) ? max : b16[i];

		for(int i=0;i<class;i++)
			b17[i] = exp((b16[i] - max));

		for(int i=0;i<class;i++)
			sum+=b17[i];

		for(int i=0;i<class;i++)
			b17[i] = b17[i]/sum;

		printf("");
	}

	if(dp_7 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b16[0];

		for(int i=0;i<class;i++)
			max = (max > b16[i]) ? max : b16[i];

		for(int i=0;i<class;i++)
			b17[i] = exp((b16[i] - max));

		for(int i=0;i<class;i++)
			sum+=b17[i];

		for(int i=0;i<class;i++)
			b17[i] = b17[i]/sum;

		printf("");
	}

}