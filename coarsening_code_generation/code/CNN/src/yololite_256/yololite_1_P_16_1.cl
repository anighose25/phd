__kernel void pooling_1( __global const float* b8, __global float* b9)
{
	int lId = get_local_id(0) ;
	int gId = (lId/8)*8*16 + lId%8 ;

	int dp_0= get_group_id(0)*16*get_local_size(0) + gId + 8*0 ;
	int dp_1= get_group_id(0)*16*get_local_size(0) + gId + 8*1 ;
	int dp_2= get_group_id(0)*16*get_local_size(0) + gId + 8*2 ;
	int dp_3= get_group_id(0)*16*get_local_size(0) + gId + 8*3 ;
	int dp_4= get_group_id(0)*16*get_local_size(0) + gId + 8*4 ;
	int dp_5= get_group_id(0)*16*get_local_size(0) + gId + 8*5 ;
	int dp_6= get_group_id(0)*16*get_local_size(0) + gId + 8*6 ;
	int dp_7= get_group_id(0)*16*get_local_size(0) + gId + 8*7 ;
	int dp_8= get_group_id(0)*16*get_local_size(0) + gId + 8*8 ;
	int dp_9= get_group_id(0)*16*get_local_size(0) + gId + 8*9 ;
	int dp_10= get_group_id(0)*16*get_local_size(0) + gId + 8*10 ;
	int dp_11= get_group_id(0)*16*get_local_size(0) + gId + 8*11 ;
	int dp_12= get_group_id(0)*16*get_local_size(0) + gId + 8*12 ;
	int dp_13= get_group_id(0)*16*get_local_size(0) + gId + 8*13 ;
	int dp_14= get_group_id(0)*16*get_local_size(0) + gId + 8*14 ;
	int dp_15= get_group_id(0)*16*get_local_size(0) + gId + 8*15 ;


	if(dp_0< 16*128*128)
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
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_0] = maxValue;

		printf("");
	}
	if(dp_1< 16*128*128)
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
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_1] = maxValue;

		printf("");
	}
	if(dp_2< 16*128*128)
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
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_2] = maxValue;

		printf("");
	}
	if(dp_3< 16*128*128)
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
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_3] = maxValue;

		printf("");
	}
	if(dp_4< 16*128*128)
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
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_4] = maxValue;

		printf("");
	}
	if(dp_5< 16*128*128)
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
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_5] = maxValue;

		printf("");
	}
	if(dp_6< 16*128*128)
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
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_6] = maxValue;

		printf("");
	}
	if(dp_7< 16*128*128)
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
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_7] = maxValue;

		printf("");
	}
	if(dp_8< 16*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_8 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_8 / (128*128);
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_8] = maxValue;

		printf("");
	}
	if(dp_9< 16*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_9 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_9 / (128*128);
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_9] = maxValue;

		printf("");
	}
	if(dp_10< 16*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_10 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_10 / (128*128);
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_10] = maxValue;

		printf("");
	}
	if(dp_11< 16*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_11 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_11 / (128*128);
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_11] = maxValue;

		printf("");
	}
	if(dp_12< 16*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_12 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_12 / (128*128);
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_12] = maxValue;

		printf("");
	}
	if(dp_13< 16*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_13 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_13 / (128*128);
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_13] = maxValue;

		printf("");
	}
	if(dp_14< 16*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_14 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_14 / (128*128);
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_14] = maxValue;

		printf("");
	}
	if(dp_15< 16*128*128)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_15 % (128*128);
		outputRow_P = localId_P / 128;
		outputCol_P = localId_P % 128;
		image2dIdx_P = dp_15 / (128*128);
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*256*256;
		poolInputOffset = inputImageOffset + inputRow_P * 256 + inputCol_P;
		maxValue = b8[ poolInputOffset ];

		process = (inputRow_P + 1 < 256) && (inputCol_P + 1 < 256);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b8+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b8+poolInputOffset+256);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<128 && outputCol_P<128)
			b9[dp_15] = maxValue;

		printf("");
	}
}