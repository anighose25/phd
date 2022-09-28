__kernel void pooling_5( __global const float* b6, __global float* b7)
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


	if(dp_0< 64*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_0 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_0 / (128*128);
		plane_P = image2dIdx_P % 64;
		n_P = image2dIdx_P / 64;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*64 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b6[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b6+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b6+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b7[dp_0] = maxValue;

		printf("");
	}
	if(dp_1< 64*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_1 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_1 / (128*128);
		plane_P = image2dIdx_P % 64;
		n_P = image2dIdx_P / 64;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*64 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b6[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b6+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b6+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b7[dp_1] = maxValue;

		printf("");
	}
	if(dp_2< 64*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_2 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_2 / (128*128);
		plane_P = image2dIdx_P % 64;
		n_P = image2dIdx_P / 64;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*64 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b6[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b6+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b6+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b7[dp_2] = maxValue;

		printf("");
	}
	if(dp_3< 64*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_3 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_3 / (128*128);
		plane_P = image2dIdx_P % 64;
		n_P = image2dIdx_P / 64;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*64 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b6[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b6+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b6+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b7[dp_3] = maxValue;

		printf("");
	}
	if(dp_4< 64*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_4 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_4 / (128*128);
		plane_P = image2dIdx_P % 64;
		n_P = image2dIdx_P / 64;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*64 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b6[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b6+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b6+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b7[dp_4] = maxValue;

		printf("");
	}
	if(dp_5< 64*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_5 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_5 / (128*128);
		plane_P = image2dIdx_P % 64;
		n_P = image2dIdx_P / 64;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*64 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b6[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b6+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b6+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b7[dp_5] = maxValue;

		printf("");
	}
	if(dp_6< 64*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_6 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_6 / (128*128);
		plane_P = image2dIdx_P % 64;
		n_P = image2dIdx_P / 64;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*64 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b6[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b6+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b6+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b7[dp_6] = maxValue;

		printf("");
	}
	if(dp_7< 64*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_7 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_7 / (128*128);
		plane_P = image2dIdx_P % 64;
		n_P = image2dIdx_P / 64;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*64 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b6[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b6+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b6+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b7[dp_7] = maxValue;

		printf("");
	}
}