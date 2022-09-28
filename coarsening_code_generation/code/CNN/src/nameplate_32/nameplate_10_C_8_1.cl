__kernel void convolution_10( __global const float* b12, __global const float* b13, __global const float* b14, __global float* b15)
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


	if(dp_0< 4*1*1)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 1;
		h2 = 1;

		localId_C = dp_0%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_0 / (h2*w2);
		plane_C = image2dIdx_C % 4;
		n_C = image2dIdx_C / 4;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=1)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b12[0];
						data0_C.z = b12[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=1)
				{
					if(inputRow_C+i==1 && j==2)
					{
						data0_C.x = b12[(j*1*1)+(inputRow_C+i)*1+2];
						data0_C.y = b12[(j*1*1)+(inputRow_C+i)*1+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b13+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<1 && outputCol_C<1)
			b15[plane_C*1*1+outputRow_C*1+outputCol_C] = dots + b14[plane_C];
	}

	if(dp_1< 4*1*1)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 1;
		h2 = 1;

		localId_C = dp_1%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_1 / (h2*w2);
		plane_C = image2dIdx_C % 4;
		n_C = image2dIdx_C / 4;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=1)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b12[0];
						data0_C.z = b12[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=1)
				{
					if(inputRow_C+i==1 && j==2)
					{
						data0_C.x = b12[(j*1*1)+(inputRow_C+i)*1+2];
						data0_C.y = b12[(j*1*1)+(inputRow_C+i)*1+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b13+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<1 && outputCol_C<1)
			b15[plane_C*1*1+outputRow_C*1+outputCol_C] = dots + b14[plane_C];
	}

	if(dp_2< 4*1*1)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 1;
		h2 = 1;

		localId_C = dp_2%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_2 / (h2*w2);
		plane_C = image2dIdx_C % 4;
		n_C = image2dIdx_C / 4;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=1)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b12[0];
						data0_C.z = b12[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=1)
				{
					if(inputRow_C+i==1 && j==2)
					{
						data0_C.x = b12[(j*1*1)+(inputRow_C+i)*1+2];
						data0_C.y = b12[(j*1*1)+(inputRow_C+i)*1+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b13+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<1 && outputCol_C<1)
			b15[plane_C*1*1+outputRow_C*1+outputCol_C] = dots + b14[plane_C];
	}

	if(dp_3< 4*1*1)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 1;
		h2 = 1;

		localId_C = dp_3%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_3 / (h2*w2);
		plane_C = image2dIdx_C % 4;
		n_C = image2dIdx_C / 4;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=1)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b12[0];
						data0_C.z = b12[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=1)
				{
					if(inputRow_C+i==1 && j==2)
					{
						data0_C.x = b12[(j*1*1)+(inputRow_C+i)*1+2];
						data0_C.y = b12[(j*1*1)+(inputRow_C+i)*1+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b13+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<1 && outputCol_C<1)
			b15[plane_C*1*1+outputRow_C*1+outputCol_C] = dots + b14[plane_C];
	}

	if(dp_4< 4*1*1)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 1;
		h2 = 1;

		localId_C = dp_4%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_4 / (h2*w2);
		plane_C = image2dIdx_C % 4;
		n_C = image2dIdx_C / 4;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=1)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b12[0];
						data0_C.z = b12[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=1)
				{
					if(inputRow_C+i==1 && j==2)
					{
						data0_C.x = b12[(j*1*1)+(inputRow_C+i)*1+2];
						data0_C.y = b12[(j*1*1)+(inputRow_C+i)*1+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b13+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<1 && outputCol_C<1)
			b15[plane_C*1*1+outputRow_C*1+outputCol_C] = dots + b14[plane_C];
	}

	if(dp_5< 4*1*1)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 1;
		h2 = 1;

		localId_C = dp_5%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_5 / (h2*w2);
		plane_C = image2dIdx_C % 4;
		n_C = image2dIdx_C / 4;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=1)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b12[0];
						data0_C.z = b12[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=1)
				{
					if(inputRow_C+i==1 && j==2)
					{
						data0_C.x = b12[(j*1*1)+(inputRow_C+i)*1+2];
						data0_C.y = b12[(j*1*1)+(inputRow_C+i)*1+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b13+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<1 && outputCol_C<1)
			b15[plane_C*1*1+outputRow_C*1+outputCol_C] = dots + b14[plane_C];
	}

	if(dp_6< 4*1*1)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 1;
		h2 = 1;

		localId_C = dp_6%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_6 / (h2*w2);
		plane_C = image2dIdx_C % 4;
		n_C = image2dIdx_C / 4;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=1)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b12[0];
						data0_C.z = b12[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=1)
				{
					if(inputRow_C+i==1 && j==2)
					{
						data0_C.x = b12[(j*1*1)+(inputRow_C+i)*1+2];
						data0_C.y = b12[(j*1*1)+(inputRow_C+i)*1+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b13+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<1 && outputCol_C<1)
			b15[plane_C*1*1+outputRow_C*1+outputCol_C] = dots + b14[plane_C];
	}

	if(dp_7< 4*1*1)
	{
		typedef float4 floatC;
		floatC data0_C,data1_C;
		int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
		float dots;

		w2 = 1;
		h2 = 1;

		localId_C = dp_7%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = dp_7 / (h2*w2);
		plane_C = image2dIdx_C % 4;
		n_C = image2dIdx_C / 4;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

		for(int j=0;j<128;j++)
		{
			for(int i=0;i<3;i++)
			{
				if(inputRow_C+i<0 || inputRow_C+i>=1)
					data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
				else if(inputCol_C<0)
				{
					if(inputRow_C+i==0 && j==0)
					{
						data0_C.y = b12[0];
						data0_C.z = b12[1];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.x = 0;
				}
				else if(inputCol_C+3-1>=1)
				{
					if(inputRow_C+i==1 && j==2)
					{
						data0_C.x = b12[(j*1*1)+(inputRow_C+i)*1+2];
						data0_C.y = b12[(j*1*1)+(inputRow_C+i)*1+3];
					}
					else
						data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
					data0_C.z = 0;
				}
				else
				{
					data0_C.xyz = vload3(0,(__global const float *)b12+(j*1*1)+(inputRow_C+i)*1+inputCol_C);
				}
				data0_C.w = 0.0;
				data1_C.xyz = vload3(0,(__global const float *)b13+(plane_C*3*3*128)+(j*3*3)+i*3);
				dots += dot(data0_C,data1_C);
			}
		}
		if(outputRow_C<1 && outputCol_C<1)
			b15[plane_C*1*1+outputRow_C*1+outputCol_C] = dots + b14[plane_C];
	}

}