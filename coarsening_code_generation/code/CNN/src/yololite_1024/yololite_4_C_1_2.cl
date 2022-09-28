__kernel void convolution_4( __global const float* b24, __global const float* b25, __global const float* b26, __global float* b27)
{
	int lId = get_local_id(0) ;
	int dp_0= (get_group_id(0)*2 + 1*0) * get_local_size(0) + lId ;
	int dp_1= (get_group_id(0)*2 + 1*1) * get_local_size(0) + lId ;


	if(dp_0< 64*256*256)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 256;
		h2 = 256;

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
				if(inputRow_C+i<0 || inputRow_C+i>=256)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b24[0];
						data0_C.z = b24[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b24+(j*256*256)+(inputRow_C+i)*256+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=256)
				{
					if(inputRow_C+i==256 && j==2)
					{
						data0_C.x = b24[(j*256*256)+(inputRow_C+i)*256+2];
						data0_C.y = b24[(j*256*256)+(inputRow_C+i)*256+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b24+(j*256*256)+(inputRow_C+i)*256+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b24+(j*256*256)+(inputRow_C+i)*256+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b25+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<256 && outputCol_C<256)
			b27[plane_C*256*256+outputRow_C*256+outputCol_C] = dots + b26[plane_C];
	}

	if(dp_1< 64*256*256)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 256;
		h2 = 256;

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
				if(inputRow_C+i<0 || inputRow_C+i>=256)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b24[0];
						data0_C.z = b24[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b24+(j*256*256)+(inputRow_C+i)*256+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=256)
				{
					if(inputRow_C+i==256 && j==2)
					{
						data0_C.x = b24[(j*256*256)+(inputRow_C+i)*256+2];
						data0_C.y = b24[(j*256*256)+(inputRow_C+i)*256+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b24+(j*256*256)+(inputRow_C+i)*256+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b24+(j*256*256)+(inputRow_C+i)*256+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b25+(plane_C*3*3*32)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<256 && outputCol_C<256)
			b27[plane_C*256*256+outputRow_C*256+outputCol_C] = dots + b26[plane_C];
	}

}