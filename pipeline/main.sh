#!/bin/bash

#first print the parameter name
echo "Parameters:" $1

echo "--------------------------"
python3 model_result_extraction.py $1
echo "--------------------------"
python3 tracking.py $1
echo "--------------------------"
python3 behaviors.py $1
echo "--------------------------"
python3 plotting.py $1
echo "--------------------------"
echo "Pipeline complete."