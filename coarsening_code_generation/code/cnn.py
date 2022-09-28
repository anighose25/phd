from coarsening_template import *


class CNN(Kernel):
    def __init__(
        self,
        uid,
        dag_id,
        start_node,
        depth,
        cfg,
        buffer_index,
        variable_index,
        datatype,
		
    ):

        super(CNN, self).__init__(
            uid=uid,
            dag_id=dag_id,
            start_node=start_node,
            depth=depth,
            cfg=cfg,
            buffer_index=buffer_index,
            variable_index=variable_index,
            
			
        )
        # self.header = "#if defined(cl_khr_fp64) \n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n#elif defined(cl_amd_fp64)  // AMD extension available?\n#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n#endif\n"

        layer_type,layer_info = cfg[start_node]
        if layer_type=='C':
            layer_name="convolution"
        elif layer_type=='P':
            layer_name="pooling"
        elif layer_type=='L':
            layer_name="linear"
        elif layer_type=='S':
            layer_name="softmax"
        self.name = layer_name+"_" + str(uid)
        self.datatype="float"
        self.work_dimension=1
        # print configuration
    
    def initialise_kernel_info(self):
        self.set_kernel_info()
        work_group_size = 0
        num_work_groups = 0
        global_work_size = 0
        
        for layer_index in range(self.start_node,self.start_node+self.depth):
            # print layer_index
            # print self.cfg
            layer = self.cfg[layer_index]
            layer_type,layer_info = layer
            if layer_type == 'C':
                ip,filt,b,op = layer_info
                work_group_size = max(work_group_size,op[1]*op[2])
                num_work_groups = max(num_work_groups,op[0])
            
            if layer_type =='P':
                ip,op = layer_info
                work_group_size = max(work_group_size,op[1]*op[2])
                num_work_groups = max(num_work_groups,op[0])

            if layer_type =='L':
                ip,w,op = layer_info
                global_work_size =max(global_work_size,op)

            if layer_type == 'S':
                ip,op = layer_info
                global_work_size =max(global_work_size,ip)

            
        global_work_size = max(global_work_size,work_group_size*num_work_groups)

        # print global_work_size, work_group_size
        self.global_work_size = [global_work_size, 1, 1]  
        self.local_work_size = [work_group_size, 1, 1] 


    def get_buffer_sizes(self):
        all_buffer_size=[]
        s_node=self.start_node
        for layer_index in range(self.start_node,self.start_node+self.depth):
            layer = self.cfg[layer_index]
            layer_type,layer_info = layer
            ipbuff_size=[]
            opbuff_size = []
            
            if layer_type == 'C':
                
                ip,filt,b,op = layer_info
                ipsize = ip[0]*ip[1]*ip[2]
                filtsize = filt[0]* filt[1]* filt[2]* filt[3]
                bias = b
                opsize = op[0]*op[1]*op[2]
                if layer_index == s_node:
                    ipbuff_size.append(ipsize)
                ipbuff_size.append(filtsize)
                ipbuff_size.append(bias)
                opbuff_size.append(opsize)
                

            
            elif layer_type =='P':
                ip,op = layer_info
                ipsize = ip[0]*ip[1]*ip[2]
                opsize = op[0]*op[1]*op[2]
                if layer_index == s_node:
                    ipbuff_size.append(ipsize)                
                opbuff_size.append(opsize)
            
            
            elif layer_type =='L':
                ip,w,op = layer_info
                ipsize = ip
                wsize = w[0]*w[1]
                opsize = op
                if layer_index == s_node:
                    ipbuff_size.append(ipsize)
                ipbuff_size.append(wsize)
                opbuff_size.append(opsize)

            elif layer_type == 'S':
                ip,op = layer_info
                ipsize = ip
                opsize = op
                if layer_index == s_node:
                    ipbuff_size.append(ipsize)
                opbuff_size.append(opsize)

            # print layer_type, len(ipbuff_size), len(opbuff_size)
            
            all_buffer_size.append((ipbuff_size,opbuff_size))
                
        return all_buffer_size
            
   
    def set_kernel_info(self):
        arg_index = 0
        persistence = 1
        all_buffer_sizes =self.get_buffer_sizes()

        # print all_buffer_sizes
        for l in range(self.start_node,self.start_node+self.depth):
            # print l,self.start_node, l-self.start_node
            ipbuffer_sizes, opbuffer_sizes = all_buffer_sizes[l-self.start_node]

            ip_info=[]
            counter = 0
            for i in range(len(ipbuffer_sizes)):     
                # print "length", len(ipbuffer_sizes)   
                if arg_index==0:
                	persistence=0    
                else:
                	persistence=1                      
                ip_info.append(
                    (
                        self.datatype,
                        ipbuffer_sizes[counter],
                        arg_index,
                        persistence,
                        "b" + str(self.buffer_index),
                        

                    )
                )
                arg_index += 1
                self.buffer_index += 1
                counter += 1
            # print "ip", l, counter, ip_info
            self.num_input_buffers=self.num_input_buffers+counter
            self.ipbuffer_info.update( {l : ip_info} )
            # print self.ipbuffer_info

            op_info=[]
            counter = 0
            persistence = 0
            for i in range(len(opbuffer_sizes)):                
                op_info.append(
                    (
                        self.datatype,
                        opbuffer_sizes[counter],
                        arg_index,
                        persistence,
                        "b" + str(self.buffer_index),
                        
                    )
                )
                arg_index += 1
                self.buffer_index += 1
                counter += 1
            # print "op", l, self.num_output_buffers, counter
            self.num_output_buffers=self.num_output_buffers+counter
            self.opbuffer_info.update( {l : op_info} )

    def function_signature(self):
        bi = []
        bo = []
        v = []
        function_sig = ""
        # print self.num_input_buffers,self.num_output_buffers
        # for i in range(self.num_input_buffers):
        #     bi.append('b'+str(i))
        # for i in range(self.num_output_buffers):
        #     bo.append('b'+str(self.num_input_buffers+i))
            # for i in range(self.num_variables):
            #     v.append(self.variable_info[i][-1])
        for l in range(self.start_node,self.start_node+self.depth):
            for i in range(len(self.ipbuffer_info[l])):
                bi.append(self.ipbuffer_info[l][i][-1])
            for i in range(len(self.opbuffer_info[l])):
                bo.append(self.opbuffer_info[l][i][-1])
        datatype = self.datatype
        # print bi
        function_sig += "__kernel void " + self.name + "("
        function_sig += " __global const " + datatype + "* " + bi[0]
        for i in range(len(bi) - 1):
            function_sig += ", __global const " + datatype + "* " + bi[i + 1]
        for i in range(len(bo)):
            function_sig += ", __global " + datatype + "* " + bo[i]
        # for i in range(len(v)):
        # 	String1+= " ,int " + v[i]
        function_sig += ")\n{\n"
        # print function_sig
        return function_sig

    def global_ids(self):
        String1 = ""
        # String1 += self.thread_ids(self.work_dimension)
        String1 +="\tint globalId = get_global_id(0);\n\n"
        # \tint localId =  get_local_id(0);\n\n"
        # print String1
        
        return String1

    def generate_global_ids_thread_coarsening(self,coarsening_factor,stride):
        String1 = ""
        # String1 += self.thread_ids(self.work_dimension)
        String1 +="\tint lId = get_local_id(0) ;\n"
        String1 +="\tint gId = (lId/"+str(stride)+")*"+str(stride)+"*"+str(coarsening_factor)+" + lId%"+str(stride)+" ;\n\n"
        for c in range(coarsening_factor):
            String1 +="\tint dp_"+str(c)+"= get_group_id(0)*"+str(coarsening_factor)+"*get_local_size(0) + gId + "+str(stride)+"*"+str(c)+" ;\n"
        String1 +="\n"
           
        return String1

    def generate_global_ids_block_coarsening(self,coarsening_factor,stride):
        String1 = ""
        # String1 += self.thread_ids(self.work_dimension)
        String1 +="\tint lId = get_local_id(0) ;\n"
        for c in range(coarsening_factor):
            String1 +="\tint dp_"+str(c)+"= (get_group_id(0)*"+str(coarsening_factor)+" + "+str(stride)+"*"+str(c)+") * get_local_size(0) + lId ;\n"
        String1 +="\n"
            
        return String1

    def coarsening_start_conv(self,index,layer_info):
        ip,filt,b,op = layer_info
        code = ""
        code += "\n\tif(dp_"+str(index) +"< " + str(op[0])+"*"+str(op[1])+"*"+str(op[2])+ ")\n\t{\n"
        
        return code

    def coarsening_start_pool(self,index,layer_info):
        ip,op = layer_info
        code = ""
        code += "\n\tif(dp_"+str(index) +"< " + str(op[0])+"*"+str(op[1])+"*"+str(op[2])+ ")\n\t{\n"
        
        return code

    def coarsening_start_linear(self,index,layer_info):
        ip,w,op = layer_info
        code = ""
        code += "\n\tif(dp_"+str(index) +"< "  + str(w[1]) + ")\n\t{\n"

        return code

    def coarsening_start_smax(self,index):
        code = ""
        code+="\n\tif(dp_"+str(index) +" == 0)\n\t{\n"  
        
        return code

    def init_private_variables_conv(self):
        code = ""
        code += "\t\ttypedef float4 floatC;\n"
        code += "\t\tfloatC data0_C,data1_C;\n"
        code += "\t\tint localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;\n"
        code += "\t\tfloat dots;\n\n"
        return code

    def init_private_variables_pool(self):        
        code = ""
        code += "\t\ttypedef float2 floatP;\n"
        code += "\t\tfloatP data0_P,data1_P;\n\n"
        code += "\t\tint localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;\n"
        code += "\t\tfloat maxValue;\n"
        code +="\t\tbool process;\n\n"
        return code;

    def init_private_variables_linear(self):
        code = ""
        code += "\t\ttypedef float4 floatX;\n"
        code += "\t\tfloatX wt,temp;\n"
        code += "\t\tfloat dotProduct;\n\n"
        return code

    def init_private_variables_sfmax(self):
        code = ""
        code+="\t\tint class;\n\t\tfloat sum, max;\n\n"
        return code

    def load(self):
        pass

    def load_compute_store_conv(self,depth,cf_index,layer_info):

        layer_index=self.start_node
        ip,filt,b,op = layer_info
        
        if depth==0:
            b0=self.ipbuffer_info[layer_index][0][-1]
            b1=self.ipbuffer_info[layer_index][1][-1]
            b2=self.ipbuffer_info[layer_index][2][-1]
            str_type="(__global const float *)"
        else:
            b0=self.opbuffer_info[layer_index-1][0][-1]
            b1=self.ipbuffer_info[layer_index][0][-1]
            b2=self.ipbuffer_info[layer_index][1][-1] 
            str_type="(__global float *)"
        b3=self.opbuffer_info[layer_index][0][-1]
        
        code = ""
        code += "\t\tw2 = "+str(ip[1])+";\n"
        code += "\t\th2 = "+str(ip[2])+";\n\n"
        code += "\t\tlocalId_C = dp_"+str(cf_index)+"%(w2*h2);\n"
        code += "\t\toutputRow_C = localId_C / w2;\n"
        code += "\t\toutputCol_C = localId_C % w2;\n\n"
        code += "\t\timage2dIdx_C = dp_"+str(cf_index)+" / (h2*w2);\n"
        code += "\t\tplane_C = image2dIdx_C % "+str(filt[0])+";\n"
        code += "\t\tn_C = image2dIdx_C / "+str(filt[0])+";\n"
        code += "\t\tdots = 0.0;\n"
        code += "\t\tstride = 1;\n\t\tpad=1;\n"
        code += "\t\tinputRow_C = outputRow_C * stride - pad;\n"
        code += "\t\tinputCol_C = outputCol_C * stride - pad;\n\n"
        
        #code +="\t\tif(localId_C <"+str(ip[1]*ip[2])+")\n\t\t{\n"
        code +="\t\tfor(int j=0;j<"+str(ip[0])+";j++)\n\t\t{\n"
        code +="\t\t\tfor(int i=0;i<"+str(filt[1])+";i++)\n\t\t\t{\n"
        #code +="\t\t\t\t\tdata0_C.xyz = (float3)(115.0f,212.0f,76.0f);\n"
        code +="\t\t\t\tif(inputRow_C+i<0 || inputRow_C+i>="+str(ip[1])+")\n"
        code +="\t\t\t\t\tdata0_C.xyz = (float3)(0.0f,0.0f,0.0f);\n"
        code +="\t\t\t\telse if(inputCol_C<0)\n\t\t\t\t{\n"
        code +="\t\t\t\t\tif(inputRow_C+i==0 && j==0)\n\t\t\t\t\t{\n" 
        code +="\t\t\t\t\t\tdata0_C.y = "+b0+"[0];\n"
        code +="\t\t\t\t\t\tdata0_C.z = "+b0+"[1];\n\t\t\t\t\t}\n"
        code +="\t\t\t\t\telse\n"
        code +="\t\t\t\t\t\tdata0_C.xyz = vload3(0,"+str_type+b0+"+(j*"+str(ip[1])+"*"+str(ip[2])+")+(inputRow_C+i)*"+str(ip[1])+"+inputCol_C);\n"
        code +="\t\t\t\t\tdata0_C.x = 0;\n"
        code +="\t\t\t\t}\n"
        code +="\t\t\t\telse if(inputCol_C+"+str(filt[1])+"-1>="+str(ip[2])+")\n\t\t\t\t{\n"
        code +="\t\t\t\t\tif(inputRow_C+i=="+str(ip[2])+" && j==2)\n\t\t\t\t\t{\n" 
        code +="\t\t\t\t\t\tdata0_C.x = "+b0+"[(j*"+str(ip[1])+"*"+str(ip[2])+")+(inputRow_C+i)*"+str(ip[1])+"+2];\n"
        code +="\t\t\t\t\t\tdata0_C.y = "+b0+"[(j*"+str(ip[1])+"*"+str(ip[2])+")+(inputRow_C+i)*"+str(ip[1])+"+3];\n\t\t\t\t\t}\n"
        code +="\t\t\t\t\telse\n"
        code +="\t\t\t\t\t\tdata0_C.xyz = vload3(0,"+str_type+b0+"+(j*"+str(ip[1])+"*"+str(ip[2])+")+(inputRow_C+i)*"+str(ip[1])+"+inputCol_C);\n"
        code +="\t\t\t\t\tdata0_C.z = 0;\n"
        code +="\t\t\t\t}\n"
        code +="\t\t\t\telse\n\t\t\t\t{\n"
        code +="\t\t\t\t\tdata0_C.xyz = vload3(0,"+str_type+b0+"+(j*"+str(ip[1])+"*"+str(ip[2])+")+(inputRow_C+i)*"+str(ip[1])+"+inputCol_C);\n"
        code +="\t\t\t\t}\n"
        code +="\t\t\t\tdata0_C.w = 0.0;\n"
        code +="\t\t\t\tdata1_C.xyz = vload3(0,(__global const float *)"+b1+"+(plane_C*"+str(filt[1])+"*"+str(filt[1])+"*"+str(ip[0])+")+(j*"+str(filt[1])+"*"+str(filt[1])+")+i*"+str(filt[1])+");\n"
        # code +="\t\t\t\t\tdots += data0_C.x*data1_C.x + data0_C.y*data1_C.y + data0_C.z*data1_C.z;\n"
        code +="\t\t\t\tdots += dot(data0_C,data1_C);\n"
        # code +="\t\t\t\tprintf(%f ,dots);\n"
        code +="\t\t\t}\n\t\t}\n"
        code +="\t\tif(outputRow_C<"+str(ip[2]) +" && outputCol_C<"+str(ip[2])+")\n"
        code +="\t\t\t"+b3+"[plane_C*"+str(ip[2])+"*"+str(ip[1])+"+outputRow_C*"+str(ip[1]) + "+outputCol_C] = dots + "+b2+"[plane_C];\n"
        code +="\t}\n"
        #code +="\t\tprintf(\"\");\n\t}"
        return code;

    def load_compute_store_pool(self,depth,cf_index,layer_index):
        layer_index=self.start_node
        layer = self.cfg[layer_index]
        layer_type,layer_info = layer
        ip,op = layer_info
        # print self.ipbuffer_info
        # print layer_index

        if depth==0:
            b0=self.ipbuffer_info[layer_index][0][-1]
        else:
            b0=self.opbuffer_info[layer_index-1][0][-1]
        b1=self.opbuffer_info[layer_index][0][-1]

        code = ""
        code += "\t\tlocalId_P = dp_"+str(cf_index)+" % ("+str(op[1])+"*"+str(op[2])+");\n"
        code += "\t\toutputRow_P = localId_P / "+str(op[1])+";\n"
        code += "\t\toutputCol_P = localId_P % "+str(op[2])+";\n"
        code += "\t\timage2dIdx_P = dp_"+str(cf_index)+" / ("+str(op[1])+"*"+str(op[2])+");\n"
        code += "\t\tplane_P = image2dIdx_P % "+str(op[0])+";\n"
        code += "\t\tn_P = image2dIdx_P / "+str(op[0])+";\n"
        code += "\t\tif(n_P > 1) \n"
        code += "\t\t\treturn;\n"
        code += "\t\t\n"
        code += "\t\tpool_size = 2;\n"
        code += "\t\tinputRow_P = outputRow_P * pool_size;\n"
        code += "\t\tinputCol_P = outputCol_P * pool_size;\n"
        code += "\t\tinputImageOffset  = (n_P*"+str(op[0])+" + plane_P)*"+str(ip[1])+"*"+str(ip[2])+";\n"
        code += "\t\tpoolInputOffset = inputImageOffset + inputRow_P * "+str(ip[1])+" + inputCol_P;\n"
        code += "\t\tmaxValue = "+b0+"[ poolInputOffset ];\n\n"
        code +="\t\tprocess = (inputRow_P + 1 < "+str(ip[1])+") && (inputCol_P + 1 < "+str(ip[2])+");\n\n"
        code +="\t\tif(process)\n\t\t{\n"
        if depth==0:
            code +="\t\t\tdata0_P = vload2(0,(__global const float *)"+b0+"+poolInputOffset);\n"
            code +="\t\t\tdata1_P = vload2(0,(__global const float *)"+b0+"+poolInputOffset+"+str(ip[1])+");\n\n"
        else:
            code +="\t\t\tdata0_P = vload2(0,(__global float *)"+b0+"+poolInputOffset);\n"
            code +="\t\t\tdata1_P = vload2(0,(__global float *)"+b0+"+poolInputOffset+"+str(ip[1])+");\n\n"
        code +="\t\t\tdata0_P = fmax(data0_P,data1_P);\n"
        code +="\t\t\tmaxValue = fmax(data0_P.s0,data0_P.s1);\n"
        code +="\t\t}\n\n"
        code +="\t\tif(outputRow_P<"+str(op[1])+" && outputCol_P<"+str(op[2])+")\n"
        code +="\t\t\t"+b1+"[dp_"+str(cf_index)+"] = maxValue;\n\n"
        code +="\t\tprintf(\"\");\n\t}"

        return code

    def load_compute_store_linear(self,depth,cf_index,layer_index):
        layer_index=self.start_node
        layer = self.cfg[layer_index]
        layer_type,layer_info = layer
        ip,w,op = layer_info

        if depth==0:
            b0=self.ipbuffer_info[layer_index][0][-1]
            b1=self.ipbuffer_info[layer_index][1][-1]
        else:
            b0=self.opbuffer_info[layer_index-1][0][-1]
            b1=self.ipbuffer_info[layer_index][0][-1]
        b2=self.opbuffer_info[layer_index][0][-1]

        code = ""
        code += "\t\tdotProduct=0.0;\n"
        code += "\t\tfor(int x=0; x<" + str(w[0]) + "/4; x++)\n\t\t{\n"    
        if depth==0:
            code += "\t\t\ttemp= vload4(0,(__global const float *)" + b0 + "+(4*x));\n"
        else:
            code += "\t\t\ttemp= vload4(0,(__global float *)" + b0 + "+(4*x));\n"
        code += "\t\t\twt.x= "+ b1+"["+ str(w[1]) +"*(4*x)+dp_"+str(cf_index)+"];\n"
        code += "\t\t\twt.y= "+ b1+"["+ str(w[1]) +"*((4*x)+1)+dp_"+str(cf_index)+"];\n"
        code += "\t\t\twt.z= "+ b1+"["+ str(w[1]) +"*((4*x)+2)+dp_"+str(cf_index)+"];\n"
        code += "\t\t\twt.w= "+ b1+"["+ str(w[1]) +"*((4*x)+3)+dp_"+str(cf_index)+"];\n"
        code += "\t\t\tdotProduct += dot(wt,temp);\n\t\t}\n"
        code += b2+"[dp_"+str(cf_index)+"] = dotProduct;\n\t\tprintf(\"\");\n\t}\n"
           
        return code

    def load_compute_store_softmax(self,depth,cf_index,layer_index):
        layer_index=self.start_node
        layer = self.cfg[layer_index]
        layer_type,layer_info = layer
        ip,op = layer_info
        # print layer_index
        # print self.ipbuffer_info
        if depth==0:
            b0=self.ipbuffer_info[layer_index][0][-1]
        else:
            b0=self.opbuffer_info[layer_index-1][0][-1]
        b1=self.opbuffer_info[layer_index][0][-1]


        code = ""
        code+="\t\tclass="+str(ip)+";\n\t\tsum = 0.0;\n\t\tmax = "+ b0+"[0];\n\n"          
        code+="\t\tfor(int i=0;i<class;i++)\n"
        code+="\t\t\tmax = (max > "+ b0+"[i]) ? max : "+ b0+"[i];\n\n"
        code+="\t\tfor(int i=0;i<class;i++)\n"
        code+="\t\t\t"+ b1+"[i] = exp(("+ b0+"[i] - max));\n\n"
        code+="\t\tfor(int i=0;i<class;i++)\n"
        code+="\t\t\tsum+="+ b1+"[i];\n\n"
        code+="\t\tfor(int i=0;i<class;i++)\n"
        code+="\t\t\t"+ b1+"[i] = "+ b1+"[i]/sum;\n\n"
        code+="\t\tprintf(\"\");\n\t}\n"
        return code

    def synchronization(self):
		code = ""
		code += "\n\n\tbarrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n\n"
		return code

    def store(self):
		pass

    def generate_source_code(self):
		# self.source_code += self.header
		#self.source_code += self.define_variables()
		self.source_code += self.function_signature()
		self.source_code += self.global_ids()
		self.source_code += self.init_private_variables()
		self.source_code += self.load_compute_store()
		self.source_code += "\n}"


    def generate_coarsened_source_code(self,coarsening_type,coarsening_factor,stride):
        # self.source_code += self.define_variables()
        self.source_code += self.function_signature()

        if coarsening_type == 'T':
            self.source_code += self.generate_global_ids_thread_coarsening(coarsening_factor,stride)
        elif coarsening_type == 'B':
            self.source_code += self.generate_global_ids_block_coarsening(coarsening_factor,stride)
        
        layer_type,layer_info = self.cfg[self.start_node]
        
        for cf_index in range(coarsening_factor):

            if layer_type == 'C':
                self.source_code += self.coarsening_start_conv(cf_index,layer_info)
                self.source_code +=self.init_private_variables_conv()
                self.source_code += self.load_compute_store_conv(0,cf_index,layer_info)
                # self.source_code += "\n\t}" 

            elif layer_type == 'P':
                self.source_code += self.coarsening_start_pool(cf_index,layer_info)
                self.source_code +=self.init_private_variables_pool()
                self.source_code += self.load_compute_store_pool(0,cf_index,layer_info)
                # self.source_code += "\n\t}" 


            elif layer_type == 'L':
                self.source_code += self.coarsening_start_linear(cf_index,layer_info)
                self.source_code +=self.init_private_variables_linear()
                self.source_code += self.load_compute_store_linear(0,cf_index,layer_info)
                # self.source_code += "\n\t}" 


            elif layer_type == 'S':
                self.source_code += self.coarsening_start_smax(cf_index)
                self.source_code +=self.init_private_variables_sfmax()
                self.source_code += self.load_compute_store_softmax(0,cf_index,layer_info)
                #   self.source_code += "\n\t}" 

        self.source_code += "\n}"

def generate_configuration_for_network(network_name,height,width):

    configuration = []

    if network_name=="synthetic":
        # synthetic CNNinference dag
        configuration.append(('C',((3,height,width),(16,3,3,3),(16),(16,height,width)))) #(ic*h*w),(oc*f*f*ic),(oc),(oc*h'*w')
        configuration.append(('P',((16,height,width),(16,height/2,width/2))))  #(ic,h,w),(ic,h',w')
        configuration.append(('C',((16,height/2,width/2),(32,3,3,16),(32),(32,height/2,width/2))))
        configuration.append(('P',((32,height/2,width/2),(32,height/4,width/4))))
        configuration.append(('C',((32,height/4,width/4),(64,3,3,32),(64),(64,height/4,width/4))))
        configuration.append(('P',((64,height/4,width/4),(64,height/8,width/8))))
        configuration.append(('L',((64*(height/8)*(width/8)),(64*(height/8)*(width/8),512),(512))))
        configuration.append(('L',((512),(512,128),(128)))) #(flattened version in 1d of previous output:l, bias:b, (l)(l,b))(b))
        configuration.append(('L',((128),(128,16),(16))))
        configuration.append(('S',((16),(16))))


    if network_name=="edlenet":
        # adelnet CNN inferencing
        configuration.append(('C',((1,height,width),(32,3,3,1),(32),(32,height,width)))) 
        configuration.append(('P',((32,height,width),(32,height/2,width/2)))) 
        configuration.append(('C',((32,height/2,width/2),(64,3,3,32),(64),(64,height/2,width/2))))
        configuration.append(('P',((64,height/2,width/2),(64,height/4,width/4))))
        configuration.append(('C',((64,height/4,width/4),(128,3,3,64),(128),(128,height/4,width/4))))
        configuration.append(('P',((128,height/4,width/4),(128,height/8,width/8))))
        configuration.append(('L',((128*(height/8)*(width/8)),(128*(height/8)*(width/8),128),(128)))) 
        configuration.append(('L',((128),(128,16),(16))))
        configuration.append(('S',((16),(16))))


    if network_name=="yololite":
        # adelnet CNN inferencing
        configuration.append(('C',((1,height,width),(16,3,3,1),(16),(16,height,width)))) #(ic*h*w),(oc*f*f*ic),(oc),(oc*h'*w')
        configuration.append(('P',((16,height,width),(16,height/2,width/2)))) #(ic,h,w),(ic,h',w')
        configuration.append(('C',((16,height/2,width/2),(32,3,3,16),(32),(32,height/2,width/2))))
        configuration.append(('P',((32,height/2,width/2),(32,height/4,width/4))))
        configuration.append(('C',((32,height/4,width/4),(64,3,3,32),(64),(64,height/4,width/4))))
        configuration.append(('P',((64,height/4,width/4),(64,height/8,width/8))))
        configuration.append(('C',((64,height/8,width/8),(128,3,3,64),(128),(128,height/8,width/8))))
        configuration.append(('P',((128,height/8,width/8),(128,height/16,width/16))))
        configuration.append(('C',((128,height/16,width/16),(128,3,3,128),(128),(128,height/16,width/16))))
        configuration.append(('P',((128,height/16,width/16),(128,height/32,width/32))))
        configuration.append(('C',((128,height/32,width/32),(256,3,3,128),(256),(256,height/32,width/32))))
        configuration.append(('C',((256,height/32,width/32),(125,3,3,256),(125),(125,height/32,width/32))))
        
        
    if network_name=="nameplate":
        
        configuration.append(('C',((1,height,width),(16,3,3,1),(16),(16,height,width)))) #(ic*h*w),(oc*f*f*ic),(oc),(oc*h'*w')
        configuration.append(('P',((16,height,width),(16,height/2,width/2))))  #(ic,h,w),(ic,h',w')
        configuration.append(('C',((16,height/2,width/2),(32,3,3,16),(32),(32,height/2,width/2))))
        configuration.append(('P',((32,height/2,width/2),(32,height/4,width/4))))
        configuration.append(('C',((32,height/4,width/4),(64,3,3,32),(64),(64,height/4,width/4))))
        configuration.append(('P',((64,height/4,width/4),(64,height/8,width/8))))
        configuration.append(('C',((64,height/8,width/8),(128,3,3,64),(128),(128,height/8,width/8))))
        configuration.append(('P',((128,height/8,width/8),(128,height/16,width/16))))
        configuration.append(('C',((128,height/16,width/16),(128,3,3,128),(128),(128,height/16,width/16))))
        configuration.append(('P',((128,height/16,width/16),(128,height/32,width/32))))
        configuration.append(('C',((128,height/32,width/32),(4,3,3,128),(4),(4,height/32,width/32))))
        configuration.append(('P',((4,height/32,width/32),(4,height/64,width/64))))
        configuration.append(('L',((32),(32,4),(4))))
        
    if network_name=="nameplatemini":
        
        configuration.append(('C',((1,height,width),(16,3,3,1),(16),(16,height,width)))) #(ic*h*w),(oc*f*f*ic),(oc),(oc*h'*w')
        configuration.append(('P',((16,height,width),(16,height/2,width/2))))  #(ic,h,w),(ic,h',w')
        configuration.append(('C',((16,height/2,width/2),(32,3,3,16),(32),(32,height/2,width/2))))
        configuration.append(('P',((32,height/2,width/2),(32,height/4,width/4))))
        configuration.append(('C',((32,height/4,width/4),(64,3,3,32),(64),(64,height/4,width/4))))
        configuration.append(('P',((64,height/4,width/4),(64,height/8,width/8))))
        configuration.append(('C',((64,height/8,width/8),(128,3,3,64),(128),(128,height/8,width/8))))
        configuration.append(('P',((128,height/8,width/8),(128,height/16,width/16))))
        configuration.append(('L',((512),(512,32),(32))))
        configuration.append(('L',((32),(32,4),(4))))
        
               
    return configuration


def populate_cnn_info(configuration):

    cnn = []
    for start_node in range(len(configuration)): 
        
        uid = ""
        tid = ""
        # variables=[]
        cfg=[]
        uid += str(start_node)
        tid += str(start_node)
        cfg.append(configuration[start_node])
        # print cfg

        dag = 1
        depth = 1
        ipbuffsize = 0
        opbuffsize = 0

        # ipbuffsize = 2 * depth + 1
        # opbuffsize = 1 * depth
        varsize = 2 * depth
        buf_index = 0
        var_index = 0
        datatype = "float"
        cnn.append(
            CNN(
                uid,
                dag,
                start_node,
                depth,
                configuration,
                buf_index,
                var_index,
                datatype,
            )
        )

        cnn[-1].initialise_kernel_info()

    return cnn


