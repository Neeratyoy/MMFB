#!/bin/bash

url=$1
path=$2
name=$3

echo $path/$name

echo "Downloading..."
wget --content-disposition $url -P $path
echo -e "\n"
echo "Unzipping..."
unzip $path$name.zip -d $path/$name
ls $path/$name