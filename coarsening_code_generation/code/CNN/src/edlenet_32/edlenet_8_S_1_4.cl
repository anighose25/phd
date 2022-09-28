__kernel void softmax_8( __global const float* b14, __global float* b15)
{
	int lId = get_local_id(0) ;
	int dp_0= (get_group_id(0)*4 + 1*0) * get_local_size(0) + lId ;
	int dp_1= (get_group_id(0)*4 + 1*1) * get_local_size(0) + lId ;
	int dp_2= (get_group_id(0)*4 + 1*2) * get_local_size(0) + lId ;
	int dp_3= (get_group_id(0)*4 + 1*3) * get_local_size(0) + lId ;


	if(dp_0 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b14[0];

		for(int i=0;i<class;i++)
			max = (max > b14[i]) ? max : b14[i];

		for(int i=0;i<class;i++)
			b15[i] = exp((b14[i] - max));

		for(int i=0;i<class;i++)
			sum+=b15[i];

		for(int i=0;i<class;i++)
			b15[i] = b15[i]/sum;

		printf("");
	}

	if(dp_1 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b14[0];

		for(int i=0;i<class;i++)
			max = (max > b14[i]) ? max : b14[i];

		for(int i=0;i<class;i++)
			b15[i] = exp((b14[i] - max));

		for(int i=0;i<class;i++)
			sum+=b15[i];

		for(int i=0;i<class;i++)
			b15[i] = b15[i]/sum;

		printf("");
	}

	if(dp_2 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b14[0];

		for(int i=0;i<class;i++)
			max = (max > b14[i]) ? max : b14[i];

		for(int i=0;i<class;i++)
			b15[i] = exp((b14[i] - max));

		for(int i=0;i<class;i++)
			sum+=b15[i];

		for(int i=0;i<class;i++)
			b15[i] = b15[i]/sum;

		printf("");
	}

	if(dp_3 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b14[0];

		for(int i=0;i<class;i++)
			max = (max > b14[i]) ? max : b14[i];

		for(int i=0;i<class;i++)
			b15[i] = exp((b14[i] - max));

		for(int i=0;i<class;i++)
			sum+=b15[i];

		for(int i=0;i<class;i++)
			b15[i] = b15[i]/sum;

		printf("");
	}

}