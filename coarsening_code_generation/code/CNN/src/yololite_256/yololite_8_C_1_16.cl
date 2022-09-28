__kernel void convolution_8( __global const float* b36, __global const float* b37, __global const float* b38, __global float* b39)
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


	if(dp_0< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_0%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_0 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_1< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_1%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_1 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_2< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_2%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_2 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_3< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_3%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_3 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_4< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_4%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_4 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_5< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_5%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_5 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_6< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_6%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_6 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_7< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_7%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_7 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_8< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_8%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_8 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_9< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_9%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_9 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_10< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_10%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_10 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_11< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_11%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_11 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_12< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_12%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_12 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_13< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_13%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_13 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_14< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_14%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_14 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

	if(dp_15< 128*16*16)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 16;
		h2 = 16;

		localId_C = dp_15%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_15 / (h2*w2);
		plane_C = image2dIdx_C % 128;
		n_C = image2dIdx_C / 128;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=16)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b36[0];
						data0_C.z = b36[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=16)
				{
					if(inputRow_C+i==16 && j==2)
					{
						data0_C.x = b36[(j*16*16)+(inputRow_C+i)*16+2];
						data0_C.y = b36[(j*16*16)+(inputRow_C+i)*16+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b36+(j*16*16)+(inputRow_C+i)*16+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b37+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<16 && outputCol_C<16)
			b39[plane_C*16*16+outputRow_C*16+outputCol_C] = dots + b38[plane_C];
	}

}