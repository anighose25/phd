#!/bin/bash

mpiexec -H localhost -npernode 12 python parallel_train_final.py
