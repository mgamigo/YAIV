#!/bin/bash
# Usefull script to create test_enviroments and check installation
DIR=`pwd`
cd /home/martin/Software/enviroments
jupyter kernelspec uninstall test_env
rm -r test_env
#virtualenv --python=/usr/bin/python3.8 test_env
virtualenv test_env
source test_env/bin/activate
pip install --upgrade pip
pip install ipykernel
#pip install ipykernel==6.17.1
python -m ipykernel install --user --name=test_env
jupyter kernelspec list
cd $DIR
#pip install -e ./
