#!/usr/bin/env bash

VENVNAME = marmor_vis

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter

python -m ipykernel install --user --name=$VENVNAME
test -f requirements.txt && xargs -n1 pip3 install < requirements.txt
# pip install -r requirements.txt

deactivate
echo "build $VENVNAME"