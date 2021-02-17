#!/bin/bash
echo "Make sure you called 'source load_modules.sh' before!"
echo "Creating python environment.."
python3 -m venv venv

# source ./load_modules.sh
source venv/bin/activate
python --version
which python
which pip
pip install --upgrade pip
pip -V

# pip install wheel # error then installing bih 
pip install pyyaml attrs numpy ruamel.yaml matplotlib h5py
pip install -e bgem
pip install -e MLMC

#pip freeze
deactivate

