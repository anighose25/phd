__kernel void pooling_1( __global const float* b10, __global float* b11)
{
	int lId = get_local_id(0) ;
	int dp_0= (get_group_id(0)*1 + 1*0) * get_local_size(0) + lId ;


	if(dp_0< 16*512*512)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_0 % (512*512);
		outputRow_P = localId_P / 512;
		outputCol_P = localId_P % 512;
		image2dIdx_P = dp_0 / (512*512);
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*1024*1024;
		poolInputOffset = inputImageOffset + inputRow_P * 1024 + inputCol_P;
		maxValue = b10[ poolInputOffset ];

		process = (inputRow_P + 1 < 1024) && (inputCol_P + 1 < 1024);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b10+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b10+poolInputOffset+1024);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<512 && outputCol_P<512)
			b11[dp_0] = maxValue;

		printf("");
	}
}