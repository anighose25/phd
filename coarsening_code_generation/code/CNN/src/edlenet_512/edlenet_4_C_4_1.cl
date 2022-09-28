__kernel void convolution_4( __global const float* b8, __global const float* b9, __global const float* b10, __global float* b11)
{
	int lId = get_local_id(0) ;
	int gId = (lId/8)*8*4 + lId%8 ;

	int dp_0= get_group_id(0)*4*get_local_size(0) + gId + 8*0 ;
	int dp_1= get_group_id(0)*4*get_local_size(0) + gId + 8*1 ;
	int dp_2= get_group_id(0)*4*get_local_size(0) + gId + 8*2 ;
	int dp_3= get_group_id(0)*4*get_local_size(0) + gId + 8*3 ;


	if(dp_0< 128*128*128)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 128;
		h2 = 128;

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

		for(int j=0;j<64;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=128)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b8[0];
						data0_C.z = b8[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b8+(j*128*128)+(inputRow_C+i)*128+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=128)
				{
					if(inputRow_C+i==128 && j==2)
					{
						data0_C.x = b8[(j*128*128)+(inputRow_C+i)*128+2];
						data0_C.y = b8[(j*128*128)+(inputRow_C+i)*128+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b8+(j*128*128)+(inputRow_C+i)*128+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b8+(j*128*128)+(inputRow_C+i)*128+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b9+(plane_C*3*3*64)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<128 && outputCol_C<128)
			b11[plane_C*128*128+outputRow_C*128+outputCol_C] = dots + b10[plane_C];
	}

	if(dp_1< 128*128*128)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 128;
		h2 = 128;

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

		for(int j=0;j<64;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=128)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b8[0];
						data0_C.z = b8[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b8+(j*128*128)+(inputRow_C+i)*128+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=128)
				{
					if(inputRow_C+i==128 && j==2)
					{
						data0_C.x = b8[(j*128*128)+(inputRow_C+i)*128+2];
						data0_C.y = b8[(j*128*128)+(inputRow_C+i)*128+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b8+(j*128*128)+(inputRow_C+i)*128+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b8+(j*128*128)+(inputRow_C+i)*128+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b9+(plane_C*3*3*64)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<128 && outputCol_C<128)
			b11[plane_C*128*128+outputRow_C*128+outputCol_C] = dots + b10[plane_C];
	}

	if(dp_2< 128*128*128)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 128;
		h2 = 128;

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

		for(int j=0;j<64;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=128)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b8[0];
						data0_C.z = b8[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b8+(j*128*128)+(inputRow_C+i)*128+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=128)
				{
					if(inputRow_C+i==128 && j==2)
					{
						data0_C.x = b8[(j*128*128)+(inputRow_C+i)*128+2];
						data0_C.y = b8[(j*128*128)+(inputRow_C+i)*128+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b8+(j*128*128)+(inputRow_C+i)*128+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b8+(j*128*128)+(inputRow_C+i)*128+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b9+(plane_C*3*3*64)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<128 && outputCol_C<128)
			b11[plane_C*128*128+outputRow_C*128+outputCol_C] = dots + b10[plane_C];
	}

	if(dp_3< 128*128*128)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 128;
		h2 = 128;

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

		for(int j=0;j<64;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=128)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b8[0];
						data0_C.z = b8[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b8+(j*128*128)+(inputRow_C+i)*128+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=128)
				{
					if(inputRow_C+i==128 && j==2)
					{
						data0_C.x = b8[(j*128*128)+(inputRow_C+i)*128+2];
						data0_C.y = b8[(j*128*128)+(inputRow_C+i)*128+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b8+(j*128*128)+(inputRow_C+i)*128+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b8+(j*128*128)+(inputRow_C+i)*128+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b9+(plane_C*3*3*64)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<128 && outputCol_C<128)
			b11[plane_C*128*128+outputRow_C*128+outputCol_C] = dots + b10[plane_C];
	}

}