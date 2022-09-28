__kernel void pooling_11( __global const float* b16, __global float* b17)
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


	if(dp_0< 4*0*0)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_0 % (0*0);
		outputRow_P = localId_P / 0;
		outputCol_P = localId_P % 0;
		image2dIdx_P = dp_0 / (0*0);
		plane_P = image2dIdx_P % 4;
		n_P = image2dIdx_P / 4;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*4 + plane_P)*1*1;
		poolInputOffset = inputImageOffset + inputRow_P * 1 + inputCol_P;
		maxValue = b16[ poolInputOffset ];

		process = (inputRow_P + 1 < 1) && (inputCol_P + 1 < 1);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b16+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b16+poolInputOffset+1);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<0 && outputCol_P<0)
			b17[dp_0] = maxValue;

		printf("");
	}
	if(dp_1< 4*0*0)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_1 % (0*0);
		outputRow_P = localId_P / 0;
		outputCol_P = localId_P % 0;
		image2dIdx_P = dp_1 / (0*0);
		plane_P = image2dIdx_P % 4;
		n_P = image2dIdx_P / 4;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*4 + plane_P)*1*1;
		poolInputOffset = inputImageOffset + inputRow_P * 1 + inputCol_P;
		maxValue = b16[ poolInputOffset ];

		process = (inputRow_P + 1 < 1) && (inputCol_P + 1 < 1);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b16+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b16+poolInputOffset+1);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<0 && outputCol_P<0)
			b17[dp_1] = maxValue;

		printf("");
	}
	if(dp_2< 4*0*0)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_2 % (0*0);
		outputRow_P = localId_P / 0;
		outputCol_P = localId_P % 0;
		image2dIdx_P = dp_2 / (0*0);
		plane_P = image2dIdx_P % 4;
		n_P = image2dIdx_P / 4;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*4 + plane_P)*1*1;
		poolInputOffset = inputImageOffset + inputRow_P * 1 + inputCol_P;
		maxValue = b16[ poolInputOffset ];

		process = (inputRow_P + 1 < 1) && (inputCol_P + 1 < 1);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b16+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b16+poolInputOffset+1);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<0 && outputCol_P<0)
			b17[dp_2] = maxValue;

		printf("");
	}
	if(dp_3< 4*0*0)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_3 % (0*0);
		outputRow_P = localId_P / 0;
		outputCol_P = localId_P % 0;
		image2dIdx_P = dp_3 / (0*0);
		plane_P = image2dIdx_P % 4;
		n_P = image2dIdx_P / 4;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*4 + plane_P)*1*1;
		poolInputOffset = inputImageOffset + inputRow_P * 1 + inputCol_P;
		maxValue = b16[ poolInputOffset ];

		process = (inputRow_P + 1 < 1) && (inputCol_P + 1 < 1);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b16+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b16+poolInputOffset+1);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<0 && outputCol_P<0)
			b17[dp_3] = maxValue;

		printf("");
	}
	if(dp_4< 4*0*0)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_4 % (0*0);
		outputRow_P = localId_P / 0;
		outputCol_P = localId_P % 0;
		image2dIdx_P = dp_4 / (0*0);
		plane_P = image2dIdx_P % 4;
		n_P = image2dIdx_P / 4;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*4 + plane_P)*1*1;
		poolInputOffset = inputImageOffset + inputRow_P * 1 + inputCol_P;
		maxValue = b16[ poolInputOffset ];

		process = (inputRow_P + 1 < 1) && (inputCol_P + 1 < 1);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b16+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b16+poolInputOffset+1);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<0 && outputCol_P<0)
			b17[dp_4] = maxValue;

		printf("");
	}
	if(dp_5< 4*0*0)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_5 % (0*0);
		outputRow_P = localId_P / 0;
		outputCol_P = localId_P % 0;
		image2dIdx_P = dp_5 / (0*0);
		plane_P = image2dIdx_P % 4;
		n_P = image2dIdx_P / 4;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*4 + plane_P)*1*1;
		poolInputOffset = inputImageOffset + inputRow_P * 1 + inputCol_P;
		maxValue = b16[ poolInputOffset ];

		process = (inputRow_P + 1 < 1) && (inputCol_P + 1 < 1);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b16+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b16+poolInputOffset+1);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<0 && outputCol_P<0)
			b17[dp_5] = maxValue;

		printf("");
	}
	if(dp_6< 4*0*0)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_6 % (0*0);
		outputRow_P = localId_P / 0;
		outputCol_P = localId_P % 0;
		image2dIdx_P = dp_6 / (0*0);
		plane_P = image2dIdx_P % 4;
		n_P = image2dIdx_P / 4;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*4 + plane_P)*1*1;
		poolInputOffset = inputImageOffset + inputRow_P * 1 + inputCol_P;
		maxValue = b16[ poolInputOffset ];

		process = (inputRow_P + 1 < 1) && (inputCol_P + 1 < 1);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b16+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b16+poolInputOffset+1);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<0 && outputCol_P<0)
			b17[dp_6] = maxValue;

		printf("");
	}
	if(dp_7< 4*0*0)
	{
		typedef float2 floatP;
		floatP data0_P,data1_P;

		int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
		float maxValue;
		bool process;

		localId_P = dp_7 % (0*0);
		outputRow_P = localId_P / 0;
		outputCol_P = localId_P % 0;
		image2dIdx_P = dp_7 / (0*0);
		plane_P = image2dIdx_P % 4;
		n_P = image2dIdx_P / 4;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*4 + plane_P)*1*1;
		poolInputOffset = inputImageOffset + inputRow_P * 1 + inputCol_P;
		maxValue = b16[ poolInputOffset ];

		process = (inputRow_P + 1 < 1) && (inputCol_P + 1 < 1);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b16+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b16+poolInputOffset+1);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<0 && outputCol_P<0)
			b17[dp_7] = maxValue;

		printf("");
	}
}