#!/usr/bin/env bash

VENVNAME=marmor_visw10
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME