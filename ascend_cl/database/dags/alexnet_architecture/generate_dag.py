import sys
import os
from os import listdir
from os.path import join
import json
from copy import deepcopy   
from decimal import *
import convert_json
from collections import OrderedDict


def create_dag_file(filename):

    dag_file = open("dag.graph",'w')

    with open(filename,'r') as g:
        architecture = json.loads(g.read(),object_pairs_hook=OrderedDict)

    input_height = 227
    input_width = 227

    if filename == './architecture_cifar10.json':
        input_height = 32
        input_width = 32

    input_channels = 3
    

    for i, key in enumerate(architecture):
        if "conv" in key:
            if key[len(key)-1] != "0":
                input_height = output_dim
                input_width = output_dim
                input_channels = architecture[key][0]["input_channels"] 

            output_dim = 1 + (input_height - architecture[key][0]["kernel_size"] + 2*architecture[key][0]["padding"])//architecture[key][0]["stride"]
            output_channels = architecture[key][0]["output_channels"]
            local_wgs = 128
            global_wgs = ((output_dim ** 2) // local_wgs) * local_wgs
            if (output_dim ** 2) % local_wgs:
                global_wgs += local_wgs

            dag_file.write("{} conv{}.json ".format(i, key[len(key)-1]))
            dag_file.write("{")
            dag_file.write("\"I_height\":{},\"I_width\":{},\"I_channel\":{},".format(input_height, input_width, input_channels))
            dag_file.write("\"K_height\":{},\"K_width\":{},\"K_number\":{},".format(architecture[key][0]["kernel_size"], architecture[key][0]["kernel_size"], architecture[key][0]["output_channels"]))
            dag_file.write("\"O_height\":{},\"O_width\":{},".format(output_dim, output_dim))
            dag_file.write("\"stride_w\":{},\"stride_h\":{},".format(architecture[key][0]["stride"],architecture[key][0]["stride"]))
            dag_file.write("\"pad_w\":{},\"pad_h\":{},".format(architecture[key][0]["padding"],architecture[key][0]["padding"]))
            dag_file.write("\"local_wgs\":{},\"global_wgs\":{}".format(local_wgs,global_wgs))

            dag_file.write("}\n")

        elif "relu_fc" in key:
            dag_file.write("{} relu_fc{}.json ".format(i, key[len(key)-1]))
            dag_file.write("{")
            dag_file.write("\"input\":{}".format(output_dim))
            dag_file.write("}\n")

        elif "relu" in key:

            dag_file.write("{} relu{}.json ".format(i, key[len(key)-1]))
            dag_file.write("{")
            dag_file.write("\"I_height\":{},\"I_width\":{},\"I_channel\":{}".format(output_dim, output_dim, output_channels))
            dag_file.write("}\n")

        elif "maxpool" in key:
            input_height = output_dim
            input_width = output_dim
            input_channels = output_channels

            output_dim = 1 + (input_height - architecture[key][0]["kernel_size"])//architecture[key][0]["stride"]

            dag_file.write("{} maxpool{}.json ".format(i, key[len(key)-1]))
            dag_file.write("{")
            dag_file.write("\"I_height\":{},\"I_width\":{},\"I_channel\":{},".format(input_height, input_width, output_channels))
            dag_file.write("\"K_height\":{},\"K_width\":{},\"K_number\":{},".format(architecture[key][0]["kernel_size"], architecture[key][0]["kernel_size"], input_channels))
            dag_file.write("\"O_height\":{},\"O_width\":{},".format(output_dim, output_dim))
            dag_file.write("\"stride_w\":{},\"stride_h\":{},".format(architecture[key][0]["stride"],architecture[key][0]["stride"]))
            dag_file.write("\"pad_w\":{},\"pad_h\":{}".format(architecture[key][0]["padding"],architecture[key][0]["padding"]))
            dag_file.write("}\n")    

        elif "lr_norm" in key:
            
            object_ = architecture[key][0] 

            dag_file.write("{} lr_norm{}.json ".format(i, key[len(key)-1]))
            dag_file.write("{")
            dag_file.write("\"I_height\":{},\"I_width\":{},\"I_channel\":{},".format(output_dim, output_dim, output_channels))
            dag_file.write("\"alpha\":{},\"beta\":{},\"N\":{},\"K\":{}".format(object_["alpha"], object_["beta"], object_["N"], object_["K"]))
            dag_file.write("}\n")

        elif "fc" in key:

            object_ = architecture[key][0] 

            dag_file.write("{} fc{}.json ".format(i, key[len(key)-1]))
            dag_file.write("{")
            dag_file.write("\"input\":{},\"output\":{}".format(object_["input"], object_["output"]))
            dag_file.write("}\n")

            output_dim = object_["output"]

        elif "softmax" in key:

            object_ = architecture[key][0] 
            
            dag_file.write("{} softmax{}.json ".format(i, key[len(key)-1]))
            dag_file.write("{")
            dag_file.write("\"I_size\":{},\"num_classes\":{}".format(object_["input"], object_["num_classes"]))
            dag_file.write("}\n")

            output_dim = object_["num_classes"]

    dag_file.write("---\n")

    positions = {}
    with open('buffer_positions.json', 'r') as file:
        positions = json.loads(file.read())

    input_buf = 0
    output_buf = 0
    idx = 0

    for i, key in enumerate(architecture):
        if i > 0:
            input_buf = positions[key[0:-1]]["input"]
            dag_file.write("{} {}-{} {}\n".format(i - 1, output_buf, i, input_buf))
            output_buf = positions[key[0:-1]]["output"]
        else:
            output_buf = positions[key[0:-1]]["output"]

    dag_file.write("---\n")
if __name__=="__main__":

    architecture_file = sys.argv[1]
    
    create_dag_file(architecture_file)
    convert_json.create_dag("info", "dag.graph")