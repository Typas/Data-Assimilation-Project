#!/usr/bin/env bash

cd data
rm -f y_o.txt
ln -s y_o_sync_miss.txt y_o.txt
cd ..
