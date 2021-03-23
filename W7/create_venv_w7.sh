#!/usr/bin/env bash

VENVNAME=marmor_visw7

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter
pip install pandas
pip install numpy
pip install sys
pip install pathlib
pip install argparse
pip install os
pip install sys
pip install sklearn
pip install utils

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install requirements.txt

deactivate
echo "build $VENVNAME"
