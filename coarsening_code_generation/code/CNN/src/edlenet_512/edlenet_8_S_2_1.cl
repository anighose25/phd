__kernel void softmax_8( __global const float* b2, __global float* b3)
{
	int lId = get_local_id(0) ;
	int gId = (lId/8)*8*2 + lId%8 ;

	int dp_0= get_group_id(0)*2*get_local_size(0) + gId + 8*0 ;
	int dp_1= get_group_id(0)*2*get_local_size(0) + gId + 8*1 ;


	if(dp_0 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b2[0];

		for(int i=0;i<class;i++)
			max = (max > b2[i]) ? max : b2[i];

		for(int i=0;i<class;i++)
			b3[i] = exp((b2[i] - max));

		for(int i=0;i<class;i++)
			sum+=b3[i];

		for(int i=0;i<class;i++)
			b3[i] = b3[i]/sum;

		printf("");
	}

	if(dp_1 == 0)
	{
		int class;
		float sum, max;

		class=16;
		sum = 0.0;
		max = b2[0];

		for(int i=0;i<class;i++)
			max = (max > b2[i]) ? max : b2[i];

		for(int i=0;i<class;i++)
			b3[i] = exp((b2[i] - max));

		for(int i=0;i<class;i++)
			sum+=b3[i];

		for(int i=0;i<class;i++)
			b3[i] = b3[i]/sum;

		printf("");
	}

}