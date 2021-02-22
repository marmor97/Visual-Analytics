#!/usr/bin/env bash

VENVNAME=marmor_vis
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME