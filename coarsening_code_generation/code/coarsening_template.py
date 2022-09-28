class Kernel(object):
    def __init__(
        self, uid, dag_id, start_node, depth, cfg, buffer_index, variable_index
    ):
        self.id = uid
        self.dag_id = dag_id
        self.start_node = start_node
        self.num_input_buffers = 0
        self.num_output_buffers = 0
        self.cfg = cfg
        # self.num_variables = num_variables
        self.source_code = ""
        self.depth = depth
        self.input_buffer_names = []
        self.output_buffer_names = []
        self.ipbuffer_info = {}
        self.opbuffer_info = {}
        self.variable_info = []
        self.variable_names = []
        self.variable_values = []
        self.buffer_index = buffer_index
        self.variable_index = variable_index
        self.header = "#if defined(cl_khr_fp64) \n #pragma OPENCL EXTENSION cl_khr_fp64 : enable\n #elif defined(cl_amd_fp64)  // AMD extension available?\n #pragma OPENCL EXTENSION cl_amd_fp64 : enable\n #endif\n #define TS 4\n #define pool_size 2\n"
        self.global_work_size = []
        self.local_work_size = []
        self.work_dimension = 1

    def thread_ids(self, kernel_dimension):
        if kernel_dimension == 1:
            return " int tx = get_local_id(0);\n int bx = get_group_id(0);\n"
        if kernel_dimension == 2:
            return " int tx = get_local_id(0); \n int ty = get_local_id(1); \n	int bx = get_group_id(0);\n	int by = get_group_id(1)\n;"

    def load(self):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def store(self):
        raise NotImplementedError

    def dump_tinfo(self, name, source, dimension, filename,coarsening_factor):
        kernelName = "KernelName=" + name + "\n"
        kernelSource = "KernelSource=" + source + "\n"
        workDimension = "workDimension=" + str(dimension) + "\n"
        globalWorkSize = (
            "globalWorkSize=" + ",".join([str(g/(coarsening_factor if g>1   else 1 )) for g in self.global_work_size]) + "\n"
        )
        localWorkSize = (
            "localWorkSize=" + ",".join([str(g) for g in self.local_work_size]) + "\n"
        )

        inputBuffers = ""
        # print self.start_node, self.start_node, self.depth
        # print self.ipbuffer_info
        for l in range(self.start_node, self.start_node + self.depth):
            for input_buffer in self.ipbuffer_info[l]:
                inputBuffers = inputBuffers + ",".join(
                    [str(b) for b in input_buffer[:-1]]
                )
                inputBuffers += ","
        inputBuffers = "inputBuffers=" + inputBuffers[:-1] + "\n"

        outputBuffers = ""
        for l in range(self.start_node, self.start_node + self.depth):
            for output_buffer in self.opbuffer_info[l]:
                outputBuffers = outputBuffers + ",".join(
                    [str(b) for b in output_buffer[:-1]]
                )
                outputBuffers += ","
        outputBuffers = "outputBuffers=" + outputBuffers[:-1] + "\n"

        varArguments = ""
        for variables in self.variable_info:
            varArguments = varArguments + ",".join([str(v) for v in variables[:-1]])
            varArguments += ","
        varArguments = "varArguments=" + varArguments[:-1] + "\n"
        # print self.opbuffer_info
        # print self.depth - 1
        outflow = ""
        outflow += (
            "data_outflow="
            + str(int(self.name[-1:], 10) + 1)
            + ",0,"
            + str(self.opbuffer_info[self.start_node + self.depth - 1][-1][-2])
        )
        # print outflow

        f = open(filename, "w")
        f.write(kernelName)
        f.write(kernelSource)
        f.write(workDimension)
        f.write(globalWorkSize)
        # f.write(localWorkSize)
        f.write(inputBuffers)
        f.write(outputBuffers)
        # f.write(varArguments)
        if int(self.name[-1:], 10) + 1 < 6:
            f.write(outflow)
        f.close()
     


    def dump_json(self, name, source, dimension, filename,coarsening_factor):
        kernelName = " \"name\": \"" + name + "\",\n"
        kernelSource =  " \"src\": \"" + source + "\",\n" 
        workDimension = " \"workDimension\": " + str(dimension) + ",\n"
        globalWorkSize = (
            " \"globalWorkSize\": \"[" + ",".join([str(g/(coarsening_factor if g>1   else 1 )) for g in self.global_work_size]) + "]\",\n"
        )
        localWorkSize = (
            " \"localWorkSize\": \"[" + ",".join([str(g) for g in self.local_work_size]) + "]\",\n"
        )


        inputBuffers = " \"inputBuffers\": [\n"
        # print self.start_node, self.start_node, self.depth
        # print self.ipbuffer_info
        for l in range(self.start_node, self.start_node + self.depth):
            ib_count=1
            for input_buffer in self.ipbuffer_info[l]:
            	# print input_buffer[:-1]
                inputBuffers = inputBuffers +  "  {\n"  
                inputBuffers = inputBuffers + "    \"break\": 0,\n"
                inputBuffers = inputBuffers + "    \"type\": \""+ input_buffer[:-1][0] + "\",\n"
                inputBuffers = inputBuffers + "    \"size\": \""+ str(input_buffer[:-1][1]) + "\",\n"
                inputBuffers = inputBuffers + "    \"pos\": "+ str(input_buffer[:-1][2]) + ",\n"      
                inputBuffers = inputBuffers + "    \"persistence\": "+ str(input_buffer[:-1][3]) + "\n"      
                
                if ib_count==len(self.ipbuffer_info[l]):
                    inputBuffers = inputBuffers +  "  }\n"
                else:
                    inputBuffers = inputBuffers +  "  },\n"
                ib_count += 1

        inputBuffers = inputBuffers +  " ],\n"

        outputBuffers = " \"outputBuffers\": [\n  {\n"
        # print self.start_node, self.start_node, self.depth
        # print self.ipbuffer_info
        for l in range(self.start_node, self.start_node + self.depth):
            ob_count=1
            for output_buffer in self.opbuffer_info[l]:
                outputBuffers = outputBuffers + "    \"break\": 0,\n"
                outputBuffers = outputBuffers + "    \"type\": \""+ output_buffer[:-1][0] + "\",\n"
                outputBuffers = outputBuffers + "    \"size\": \""+ str(output_buffer[:-1][1]) + "\",\n"
                outputBuffers = outputBuffers + "    \"pos\": "+ str(output_buffer[:-1][2]) + ",\n"
                outputBuffers = outputBuffers + "    \"persistence\": "+ str(output_buffer[:-1][3]) + "\n" 

                if ob_count==len(self.opbuffer_info[l]):
                    outputBuffers = outputBuffers +  "  }\n"
                else:
                    outputBuffers = outputBuffers +  "  },\n"
                ob_count += 1

        outputBuffers = outputBuffers +  " ]\n"

        # varArguments = " \"varArguments\": [\n  {\n"
        # # print self.start_node, self.start_node, self.depth
        # # print self.ipbuffer_info
        # for l in range(self.start_node, self.start_node + self.depth):
        #     va_count=1
        #     for varArguments in self.variable_info[l]:
        #         varArguments = varArguments + "    \"type\": \""+ variables[:-1][0] + "\",\n"
        #         varArguments = varArguments + "    \"pos\": "+ str(variables[:-1][1]) + ",\n"
        #         varArguments = varArguments + "    \"size\": "+ str(variables[:-1][2]) + ",\n"      
        #         if va_count==len(self.opbuffer_info[l]):
        #             varArguments = varArguments +  "  }\n"
        #         else:
        #             varArguments = varArguments +  "  },\n"
        #         va_count += 1

        # varArguments = varArguments +  " ],\n"

       
        # print self.opbuffer_info
        # print self.depth - 1
        # outflow = ""
        # outflow += (
        #     "data_outflow="
        #     + str(int(self.name[-1:], 10) + 1)
        #     + ",0,"
        #     + str(self.opbuffer_info[self.start_node + self.depth - 1][-1][-2])
        # )
        # print outflow

        f = open(filename, "w")
        f.write("{\n")
        f.write(kernelName)
        f.write(kernelSource)
        f.write(workDimension)
        f.write(globalWorkSize)
        # f.write(localWorkSize)
        f.write(inputBuffers)
        f.write(outputBuffers)
        # f.write(varArguments)
        # if int(self.name[-1:], 10) + 1 < 6:
        #     f.write(outflow)
        f.write("}\n")
        f.close()

