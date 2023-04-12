import sys
import os
from os import listdir
from os.path import join
import json
from copy import deepcopy   
from decimal import *
import convert_json
import re

def get_name_number(key, S):
    return key.replace(S, "")

def create_dag_file(filename):

    dag_file = open("dag.graph",'w')

    with open(filename,'r') as g:
        architecture = json.loads(g.read())

    input_height = 32
    input_width = 32
    input_channels = 3
    

    for i, key in enumerate(architecture):
        if "conv" in key:
            if re.findall(r'\d+', key)[0] != "0":
                input_height = output_dim
                input_width = output_dim
                input_channels = architecture[key][0]["input_channels"] 

            output_dim = 1 + (input_height - architecture[key][0]["kernel_size"] + 2*architecture[key][0]["padding"])//architecture[key][0]["stride"]
            output_channels = architecture[key][0]["output_channels"]

            dag_file.write("{} conv{}.json ".format(i, get_name_number(key, "conv")))
            dag_file.write("{")
            dag_file.write("\"I_height\":{},\"I_width\":{},\"I_channel\":{},".format(input_height, input_width, input_channels))
            dag_file.write("\"K_height\":{},\"K_width\":{},\"K_number\":{},".format(architecture[key][0]["kernel_size"], architecture[key][0]["kernel_size"], architecture[key][0]["output_channels"]))
            dag_file.write("\"O_height\":{},\"O_width\":{},".format(output_dim, output_dim))
            dag_file.write("\"stride_w\":{},\"stride_h\":{},".format(architecture[key][0]["stride"],architecture[key][0]["stride"]))
            dag_file.write("\"pad_w\":{},\"pad_h\":{}".format(architecture[key][0]["padding"],architecture[key][0]["padding"]))
            dag_file.write("}\n")

        elif "relu" in key:

            dag_file.write("{} relu{}.json ".format(i, get_name_number(key, "relu")))
            dag_file.write("{")
            dag_file.write("\"I_height\":{},\"I_width\":{},\"I_channel\":{},".format(output_dim, output_dim, output_channels))
            dag_file.write("\"slope\":{}".format(architecture[key][0]["slope"]))
            dag_file.write("}\n")

        elif "maxpool" in key or "avgpool" in key:
            input_height = output_dim
            input_width = output_dim
            input_channels = output_channels

            output_dim = 1 + (input_height - architecture[key][0]["kernel_size"])//architecture[key][0]["stride"]

            if "maxpool" in key:
                dag_file.write("{} maxpool{}.json ".format(i, get_name_number(key, "maxpool")))
            else:
                dag_file.write("{} avgpool{}.json ".format(i, get_name_number(key, "avgpool")))    
            dag_file.write("{")
            dag_file.write("\"I_height\":{},\"I_width\":{},\"I_channel\":{},".format(input_height, input_width, output_channels))
            dag_file.write("\"K_height\":{},\"K_width\":{},\"K_number\":{},".format(architecture[key][0]["kernel_size"], architecture[key][0]["kernel_size"], input_channels))
            dag_file.write("\"O_height\":{},\"O_width\":{},".format(output_dim, output_dim))
            dag_file.write("\"stride_w\":{},\"stride_h\":{},".format(architecture[key][0]["stride"],architecture[key][0]["stride"]))
            dag_file.write("\"pad_w\":{},\"pad_h\":{}".format(architecture[key][0]["padding"],architecture[key][0]["padding"]))
            dag_file.write("}\n") 

        elif "concat" in key:
            dag_file.write("{} concat{}.json ".format(i, get_name_number(key, "concat")))
            dag_file.write("{")
            dag_file.write("\"I_height\":{},\"I_width\":{},\"I_channel\":{}".format(output_dim, output_dim, output_channels))
            dag_file.write("}\n")
            
            output_channels *= 2

        # elif "fire" in key:
        #     for K in architecture[key]:
        #         if "conv" in K:

        #         elif K == "relu":

        #         elif K == "convA":

        #         elif K == "convB":

        #         elif K == "concatAB":                
        #         dag_file.write("{} relu{}.json ".format(i, key[len(key)-1]))
        #         dag_file.write("{")
        #         dag_file.write("\"I_height\":{},\"I_width\":{},\"I_channel\":{},".format(output_dim, output_dim, output_channels))
        #         dag_file.write("\"slope\":{}".format(architecture[key][0]["slope"]))
        #         dag_file.write("}\n")       

    dag_file.write("---\n")

if __name__=="__main__":

    architecture_file = sys.argv[1]
    
    create_dag_file(architecture_file)
    convert_json.create_dag("info", "dag.graph")