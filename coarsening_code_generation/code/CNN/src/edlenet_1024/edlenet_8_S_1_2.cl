__kernel void softmax_8( __global const float* b12, __global float* b13)
{
	int lId = get_local_id(0) ;
	int dp_0= (get_group_id(0)*2 + 1*0) * get_local_size(0) + lId ;
	int dp_1= (get_group_id(0)*2 + 1*1) * get_local_size(0) + lId ;


	if(dp_0 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b12[0];

		for(int i=0;i<class;i++)
			max = (max > b12[i]) ? max : b12[i];

		for(int i=0;i<class;i++)
			b13[i] = exp((b12[i] - max));

		for(int i=0;i<class;i++)
			sum+=b13[i];

		for(int i=0;i<class;i++)
			b13[i] = b13[i]/sum;

		printf("");
	}

	if(dp_1 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b12[0];

		for(int i=0;i<class;i++)
			max = (max > b12[i]) ? max : b12[i];

		for(int i=0;i<class;i++)
			b13[i] = exp((b12[i] - max));

		for(int i=0;i<class;i++)
			sum+=b13[i];

		for(int i=0;i<class;i++)
			b13[i] = b13[i]/sum;

		printf("");
	}

}