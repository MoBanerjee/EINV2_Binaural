#!/bin/bash

GPU_ID=0,1,2

CUDA_VISIBLE_DEVICES=$GPU_ID python3 code/main.py train --port=12361