__kernel void pooling_9( __global const float* b10, __global float* b11)
{
	int lId = get_local_id(0) ;
	int dp_0= (get_group_id(0)*1 + 1*0) * get_local_size(0) + lId ;


	if(dp_0< 128*8*4)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_0 % (8*4);
		outputRow_P = localId_P / 8;
		outputCol_P = localId_P % 4;
		image2dIdx_P = dp_0 / (8*4);
		plane_P = image2dIdx_P % 128;
		n_P = image2dIdx_P / 128;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*128 + plane_P)*16*8;
		poolInputOffset = inputImageOffset + inputRow_P * 16 + inputCol_P;
		maxValue = b10[ poolInputOffset ];

		process = (inputRow_P + 1 < 16) && (inputCol_P + 1 < 8);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b10+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b10+poolInputOffset+16);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<8 && outputCol_P<4)
			b11[dp_0] = maxValue;

		printf("");
	}
}