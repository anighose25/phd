__kernel void convolution_2( __global const float* b28, __global const float* b29, __global const float* b30, __global float* b31)
{
	int lId = get_local_id(0) ;
	int dp_0= (get_group_id(0)*4 + 1*0) * get_local_size(0) + lId ;
	int dp_1= (get_group_id(0)*4 + 1*1) * get_local_size(0) + lId ;
	int dp_2= (get_group_id(0)*4 + 1*2) * get_local_size(0) + lId ;
	int dp_3= (get_group_id(0)*4 + 1*3) * get_local_size(0) + lId ;


	if(dp_0< 32*512*512)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 512;
		h2 = 512;

		localId_C = dp_0%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_0 / (h2*w2);
		plane_C = image2dIdx_C % 32;
		n_C = image2dIdx_C / 32;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<16;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=512)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b28[0];
						data0_C.z = b28[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b28+(j*512*512)+(inputRow_C+i)*512+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=512)
				{
					if(inputRow_C+i==512 && j==2)
					{
						data0_C.x = b28[(j*512*512)+(inputRow_C+i)*512+2];
						data0_C.y = b28[(j*512*512)+(inputRow_C+i)*512+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b28+(j*512*512)+(inputRow_C+i)*512+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b28+(j*512*512)+(inputRow_C+i)*512+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b29+(plane_C*3*3*16)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<512 && outputCol_C<512)
			b31[plane_C*512*512+outputRow_C*512+outputCol_C] = dots + b30[plane_C];
	}

	if(dp_1< 32*512*512)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 512;
		h2 = 512;

		localId_C = dp_1%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_1 / (h2*w2);
		plane_C = image2dIdx_C % 32;
		n_C = image2dIdx_C / 32;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<16;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=512)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b28[0];
						data0_C.z = b28[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b28+(j*512*512)+(inputRow_C+i)*512+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=512)
				{
					if(inputRow_C+i==512 && j==2)
					{
						data0_C.x = b28[(j*512*512)+(inputRow_C+i)*512+2];
						data0_C.y = b28[(j*512*512)+(inputRow_C+i)*512+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b28+(j*512*512)+(inputRow_C+i)*512+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b28+(j*512*512)+(inputRow_C+i)*512+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b29+(plane_C*3*3*16)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<512 && outputCol_C<512)
			b31[plane_C*512*512+outputRow_C*512+outputCol_C] = dots + b30[plane_C];
	}

	if(dp_2< 32*512*512)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 512;
		h2 = 512;

		localId_C = dp_2%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_2 / (h2*w2);
		plane_C = image2dIdx_C % 32;
		n_C = image2dIdx_C / 32;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<16;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=512)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b28[0];
						data0_C.z = b28[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b28+(j*512*512)+(inputRow_C+i)*512+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=512)
				{
					if(inputRow_C+i==512 && j==2)
					{
						data0_C.x = b28[(j*512*512)+(inputRow_C+i)*512+2];
						data0_C.y = b28[(j*512*512)+(inputRow_C+i)*512+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b28+(j*512*512)+(inputRow_C+i)*512+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b28+(j*512*512)+(inputRow_C+i)*512+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b29+(plane_C*3*3*16)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<512 && outputCol_C<512)
			b31[plane_C*512*512+outputRow_C*512+outputCol_C] = dots + b30[plane_C];
	}

	if(dp_3< 32*512*512)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 512;
		h2 = 512;

		localId_C = dp_3%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_3 / (h2*w2);
		plane_C = image2dIdx_C % 32;
		n_C = image2dIdx_C / 32;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<16;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=512)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b28[0];
						data0_C.z = b28[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b28+(j*512*512)+(inputRow_C+i)*512+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=512)
				{
					if(inputRow_C+i==512 && j==2)
					{
						data0_C.x = b28[(j*512*512)+(inputRow_C+i)*512+2];
						data0_C.y = b28[(j*512*512)+(inputRow_C+i)*512+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b28+(j*512*512)+(inputRow_C+i)*512+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b28+(j*512*512)+(inputRow_C+i)*512+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b29+(plane_C*3*3*16)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<512 && outputCol_C<512)
			b31[plane_C*512*512+outputRow_C*512+outputCol_C] = dots + b30[plane_C];
	}

}