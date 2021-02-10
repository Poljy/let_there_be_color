#!/bin/bash

n_nodes=$1
offset=$2
echo "Killing $n_nodes nodes"
rank=0
for i in $(cat computers_name | tail -n +$offset | head -n $n_nodes)
do
    ssh -oStrictHostKeyChecking=no $i "tmux kill-session -t imagnum" &
    echo Killed node $i with rank $rank
    ((rank++))
done
