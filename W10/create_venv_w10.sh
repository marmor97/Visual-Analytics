#!/usr/bin/env bash

VENVNAME=marmor_visw10

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter
pip install pandas 
pip install argparse
pip install opencv-python
pip install numpy 
pip install matplotlib
pip install pathlib 
pip install sklearn
pip install tensorflow

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install requirements.txt

deactivate
echo "build $VENVNAME"
