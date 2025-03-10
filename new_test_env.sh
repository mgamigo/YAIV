#!/bin/bash
# Usefull script to create test_enviroments and check installation
DIR=`pwd`
cd ~/Software/enviroments
#jupyter kernelspec uninstall test_env
rm -r test_env
/usr/bin/python3.10 -m venv test_env
source test_env/bin/activate
pip install --upgrade pip
#pip install ipykernel==6.17.1
pip install ipykernel
python -m ipykernel install --user --name=test_env
#jupyter kernelspec list
cd $DIR
#Autoinstall YAIV
pip install -e ./
#pip install .
