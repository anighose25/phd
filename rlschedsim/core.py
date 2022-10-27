import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import cPickle as pickle
import json
import sys
import networkx as nx
import numpy as np
import random
import Queue as Q
import collections
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.plotly as py
import datetime
import heapq
from copy import deepcopy
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models.graphs import from_networkx
from sklearn import linear_model
from math import sqrt, log, exp


class Kernel(object):
    """
    Class to handle all operations perfomed on kernel. Same as PySchedCL
    """

    def __init__(self, src, dataset=1024, partition=None, identifier=None):
        self.dataset = dataset
        if 'id' in src:
            self.id = src['id']
        else:
            self.id = generate_unique_id()
        if identifier is not None:
            self.id = identifier
        if 'ecos' in src and str(dataset) in src['ecos']:
            self.eco = src['ecos'][str(dataset)]
        elif 'eco' in src:
            self.eco = src['eco']
        else:
            self.eco = 1
        self.name = src['name']
        self.src = src['src']
        self.partition = src['partition']
        if partition is not None:
            self.partition = partition
        else:
            partition = self.partition
        self.work_dimension = src['workDimension']
        self.global_work_size = src['globalWorkSize']
        if type(self.global_work_size) in [str, unicode]:
            self.global_work_size = eval(self.global_work_size)
        if type(self.global_work_size) is int:
            self.global_work_size = [self.global_work_size]
        if 'localWorkSize' in src:
            self.local_work_size = src['localWorkSize']
        else:
            self.local_work_size = []
        if type(self.local_work_size) in [str, unicode]:
            self.local_work_size = eval(self.local_work_size)
        elif type(self.local_work_size) is int:
            self.local_work_size = [self.local_work_size]
        self.buffer_info = dict()
        if 'inputBuffers' in src:
            self.buffer_info['input'] = src['inputBuffers']
        else:
            self.buffer_info['input'] = []
        if 'outputBuffers' in src:
            self.buffer_info['output'] = src['outputBuffers']
        else:
            self.buffer_info['output'] = []
        if 'ioBuffers' in src:
            self.buffer_info['io'] = src['ioBuffers']
        else:
            self.buffer_info['io'] = []
        self.input_buffers = {'gpu': dict(), 'cpu': dict()}
        self.output_buffers = {'gpu': dict(), 'cpu': dict()}
        self.io_buffers = {'gpu': dict(), 'cpu': dict()}
        self.data = {}
        self.buffer_deps = {}
        if 'varArguments' in src:
            self.variable_args = deepcopy(src['varArguments'])
            self.vargs = src['varArguments']
        else:
            self.variable_args = []
            self.vargs = []
        if 'cpuArguments' in src:
            self.cpu_args = src['cpuArguments']
            print "Ignoring CPU Arguments"
        if 'gpuArguments' in src:
            self.gpu_args = src['gpuArguments']
            print "Ignoring GPU Arguments"
        if 'localArguments' in src:
            self.local_args = src['localArguments']
            for i in range(len(self.local_args)):
                self.local_args[i]['size'] = eval(self.local_args[i]['size'])
        else:
            self.local_args = []
            # self.buffer_info['local'] = deepcopy(self.local_args)
        self.kernel_objects = dict()
        for btype in ['input', 'output', 'io']:
            for i in range(len(self.buffer_info[btype])):
                if type(self.buffer_info[btype][i]['size']) in [str, unicode]:
                    self.buffer_info[btype][i]['size'] = eval(self.buffer_info[btype][i]['size'])
                if 'chunk' in self.buffer_info[btype][i] and type(self.buffer_info[btype][i]['chunk']) in [str,
                                                                                                           unicode]:
                    self.buffer_info[btype][i]['chunk'] = eval(self.buffer_info[btype][i]['chunk'])
                self.buffer_info[btype][i]['create'] = True
                self.buffer_info[btype][i]['enq_write'] = True
                self.buffer_info[btype][i]['enq_read'] = True
                if 'from' in self.buffer_info[btype][i]:
                    self.buffer_deps[self.buffer_info[btype][i]['pos']] = (self.buffer_info[btype][i]['from']['kernel'],
                                                                           self.buffer_info[btype][i]['from']['pos'])



# Auxillary functions and variables
###########################################################################################
# py.sign_in('anighose25', 'nrJZ4ZwpuHlTRV2zlAD1')
CPU_FLOPS = 86.4
# CPU_FLOPS = 179.2
GPU_FLOPS = 515.0
BW = 144.0

dispatch_step = 0

ALL_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)',
              'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)',
              'rgb(217,217,217)', 'rgb(240,2,127)', 'rgb(253,205,172)', 'rgb(179,205,227)', 'rgb(166,86,40)',
              'rgb(51,160,44)', 'rgb(247,129,191)', 'rgb(253,191,111)', 'rgb(190,186,218)', 'rgb(231,41,138)',
              'rgb(166,216,84)', 'rgb(153,153,153)', 'rgb(166,118,29)', 'rgb(230,245,201)', 'rgb(255,255,204)',
              'rgb(102,102,102)', 'rgb(77,175,74)', 'rgb(228,26,28)', 'rgb(217,95,2)', 'rgb(255,255,179)',
              'rgb(178,223,138)', 'rgb(190,174,212)', 'rgb(253,180,98)', 'rgb(255,217,47)', 'rgb(31,120,180)',
              'rgb(56,108,176)', 'rgb(229,216,189)', 'rgb(251,154,153)', 'rgb(222,203,228)', 'rgb(203,213,232)',
              'rgb(188,128,189)', 'rgb(55,126,184)', 'rgb(231,138,195)', 'rgb(244,202,228)', 'rgb(191,91,23)',
              'rgb(128,177,211)', 'rgb(27,158,119)', 'rgb(229,196,148)', 'rgb(253,218,236)', 'rgb(102,166,30)',
              'rgb(241,226,204)', 'rgb(255,127,0)', 'rgb(252,141,98)', 'rgb(227,26,28)', 'rgb(254,217,166)',
              'rgb(141,160,203)', 'rgb(204,235,197)', 'rgb(117,112,179)', 'rgb(152,78,163)', 'rgb(202,178,214)',
              'rgb(141,211,199)', 'rgb(106,61,154)', 'rgb(253,192,134)', 'rgb(255,255,51)', 'rgb(179,226,205)',
              'rgb(127,201,127)', 'rgb(251,128,114)', 'rgb(255,242,174)', 'rgb(230,171,2)', 'rgb(102,194,165)',
              'rgb(255,255,153)', 'rgb(179,179,179)', 'rgb(179,222,105)', 'rgb(252,205,229)', 'rgb(204,204,204)',
              'rgb(242,242,242)', 'rgb(166,206,227)', 'rgb(251,180,174)']

AC = ALL_COLORS

def lineno():
    import inspect
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

def get_random_integer_around(x, r):
    """
    Returns an integer selected randomly between r*x and (2-r)*x where r takes a value between 0 and 1

    Args:
        x (int): The integer around which we want to generate a random integer 
        r (float): Controls the degree to which we want to

    """

    a = r * x
    b = (2.0 - r) * x
    # print r, x, a, b
    return max(1,np.random.randint(a, b))


def load_regression_model(filename):
    """
    Loads and returns the regression model for predicting execution times for different buffer sizes

    Args:
        filename (str): Name of file containing pickle file of regression model generated by scipy

    Returns:
        pickle object: Pickle object containing the regression model
    """
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

# regression_model = load_regression_model("Stats/buffer_regression_model.sav")


def extract_dict_from_pickle(dictionary):
    """
    Create dictionary pertaining to some statistics from a pickle file
    Eg: Map between kernel names and execution times
    """
    with open(dictionary, 'rb') as handle:
        test_map = pickle.load(handle)
        return test_map


def make_dict(lines):
    x = dict()
    for line in lines:
        kvs = line.split(' ')
        x[kvs[0]] = kvs[1].strip("\n")
    return x


# print os.path.join(os.path.dirname(__file__), "/Stats/buffer_times.txt")
buffer_times = make_dict(open(os.path.join(os.path.dirname(__file__), 'Stats/buffer_times.txt')).readlines())


def create_feat_dict(key, global_map):
    """
    Returns a dictionary of feature names and feature values for a kernel and worksize pair
    """
    feature_source = "Stats/PART/" + global_map[key][:-6] + "_core2.stats"
    feature_source = os.path.join(os.path.dirname(__file__), feature_source)
    feature_list = open(feature_source, "r").readlines()
    feature_dict = make_dict(feature_list)
    return feature_dict


def get_sizeof(type_name):
    """

    :param type_name:
    :type str:
    :return: size in bytes
    :rtype:
    """
    if type_name == "int":
        return 2.0
    if type_name == "float":
        return 4.0
    if type_name == "double":
        return 8.0


def value_int(partition):
    """
    Returns integer equivalent of partition class value
    """
    if partition == "ZERO":
        return 0
    if partition == "ONE":
        return 1
    if partition == "TWO":
        return 2
    if partition == "THREE":
        return 3
    if partition == "FOUR":
        return 4
    if partition == "FIVE":
        return 5
    if partition == "SIX":
        return 6
    if partition == "SEVEN":
        return 7
    if partition == "EIGHT":
        return 8
    if partition == "NINE":
        return 9
    if partition == "TEN":
        return 10


def partition_class_value(class_string):
    if (value_int(class_string) > 5):
        return "gpu"
    else:
        return "cpu"



def partition_class_absolute(task):
    return value_int(task.Class)

def partition_class(task):
    """
    Returns partition class value of task
    """
    # print task.Class
    if (value_int(task.Class) > 5):
        return "gpu"
    else:
        return "cpu"


def all_processed(dags):
    """
    Checks whether all dags have finished processing
    """
    flag = True
    for dag in dags:
        # print "Checking", dag.dag_id
        flag = flag & dag.is_processed()
    return flag


def generate_unique_id():
    """
    Generates and returns a unique id string.
    """
    import uuid
    return str(uuid.uuid1())


def obtain_kernel_info(key):
    """
    Returns kernel object for a kernel name and worksize pair
    """
    # print key
    kernel_info = key.split("_")
    kernelName = kernel_info[0]
    worksize = kernel_info[1].strip("\n")
    if kernelName == 'uncoalesced' or kernelName == 'shared':
        kernelName = kernelName + "_copy"
        worksize = kernel_info[2].strip("\n")
    if kernelName == 'transpose':
        kernelName = kernelName + "_naive"
        worksize = kernel_info[2].strip("\n")
    dataset = int(worksize)
    kernel_info_source = "info/" + kernelName + ".json"
    kernel_info_source = os.path.join(os.path.dirname(__file__), kernel_info_source)
    info = json.loads(open(kernel_info_source).read())
    return Kernel(info, dataset)

def obtain_kernel_dimension(key):
    k = obtain_kernel_info(key)
    return k.work_dimension


def obtain_SimTask_object(key, global_map, ex_map):

    ex_cpu, ex_gpu = ex_map
    feat_dict = create_feat_dict(key, global_map)
    extime = 0.0
    if value_int(feat_dict['Class']) > 5:
        extime = ex_cpu[key]
    else:
        extime = ex_gpu[key]
    simtask = SimTask(key, 0, 0, feat_dict, extime)
    return simtask

def parse_configuration(configuration_file):
    file_contents = open(configuration_file,'r').readlines()
    layer_config = 0
    values = []
    for line in file_contents:
        line_contents = line.strip('\n').split("=")
        
        if line_contents[0]=="layer":
            layer_config = eval(line_contents[1])
        elif line_contents[0] in ["BATCH_SIZE", "GAMMA","num_states","num_actions","replay_size"]:
            values.append(int(line_contents[1]))
        else:
            values.append(float(line_contents[1]))
    
    return layer_config, values
    
        

#################################################################################################

