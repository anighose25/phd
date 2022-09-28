__kernel void softmax_8( __global const float* b18, __global float* b19)
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


	if(dp_0 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_1 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_2 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_3 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_4 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_5 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_6 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_7 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_8 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_9 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_10 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_11 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_12 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_13 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_14 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

	if(dp_15 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b18[0];

		for(int i=0;i<class;i++)
			max = (max > b18[i]) ? max : b18[i];

		for(int i=0;i<class;i++)
			b19[i] = exp((b18[i] - max));

		for(int i=0;i<class;i++)
			sum+=b19[i];

		for(int i=0;i<class;i++)
			b19[i] = b19[i]/sum;

		printf("");
	}

}