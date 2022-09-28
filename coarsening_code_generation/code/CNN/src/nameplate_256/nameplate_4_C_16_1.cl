__kernel void convolution_4( __global const float* b16, __global const float* b17, __global const float* b18, __global float* b19)
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


	if(dp_0< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_0%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_0 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_1< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_1%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_1 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_2< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_2%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_2 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_3< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_3%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_3 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_4< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_4%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_4 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_5< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_5%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_5 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_6< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_6%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_6 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_7< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_7%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_7 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_8< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_8%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_8 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_9< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_9%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_9 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_10< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_10%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_10 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_11< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_11%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_11 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_12< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_12%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_12 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_13< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_13%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_13 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_14< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_14%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_14 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

	if(dp_15< 64*64*32)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 64;
		h2 = 32;

		localId_C = dp_15%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_15 / (h2*w2);
		plane_C = image2dIdx_C % 64;
		n_C = image2dIdx_C / 64;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<32;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=64)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b16[0];
						data0_C.z = b16[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=32)
				{
					if(inputRow_C+i==32 && j==2)
					{
						data0_C.x = b16[(j*64*32)+(inputRow_C+i)*64+2];
						data0_C.y = b16[(j*64*32)+(inputRow_C+i)*64+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b16+(j*64*32)+(inputRow_C+i)*64+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b17+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<32 && outputCol_C<32)
			b19[plane_C*32*64+outputRow_C*64+outputCol_C] = dots + b18[plane_C];
	}

}