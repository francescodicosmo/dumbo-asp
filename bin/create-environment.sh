#!/bin/sh

echo "Specify the environment name, or type ENTER to use dumbo-asp"
read name
if [ -z "$name" ]; then
    name="dumbo-asp"
fi

conda create --yes --name "$name" python=3.10

conda install --yes --name "$name" -c conda-forge poetry
conda install --yes --name "$name" -c conda-forge chardet
conda install --yes --name "$name" -c potassco clingo
conda update --all --yes --name "$name"

echo "Activate the environment (conda activate $name) and run \"poetry install\""
