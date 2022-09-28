__kernel void pooling_7( __global const float* b12, __global float* b13)
{
	int lId = get_local_id(0) ;
	int dp_0= (get_group_id(0)*2 + 1*0) * get_local_size(0) + lId ;
	int dp_1= (get_group_id(0)*2 + 1*1) * get_local_size(0) + lId ;


	if(dp_0< 128*32*32)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_0 % (32*32);
		outputRow_P = localId_P / 32;
		outputCol_P = localId_P % 32;
		image2dIdx_P = dp_0 / (32*32);
		plane_P = image2dIdx_P % 128;
		n_P = image2dIdx_P / 128;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*128 + plane_P)*64*64;
		poolInputOffset = inputImageOffset + inputRow_P * 64 + inputCol_P;
		maxValue = b12[ poolInputOffset ];

		process = (inputRow_P + 1 < 64) && (inputCol_P + 1 < 64);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b12+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b12+poolInputOffset+64);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<32 && outputCol_P<32)
			b13[dp_0] = maxValue;

		printf("");
	}
	if(dp_1< 128*32*32)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_1 % (32*32);
		outputRow_P = localId_P / 32;
		outputCol_P = localId_P % 32;
		image2dIdx_P = dp_1 / (32*32);
		plane_P = image2dIdx_P % 128;
		n_P = image2dIdx_P / 128;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*128 + plane_P)*64*64;
		poolInputOffset = inputImageOffset + inputRow_P * 64 + inputCol_P;
		maxValue = b12[ poolInputOffset ];

		process = (inputRow_P + 1 < 64) && (inputCol_P + 1 < 64);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b12+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b12+poolInputOffset+64);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<32 && outputCol_P<32)
			b13[dp_1] = maxValue;

		printf("");
	}
}