#!/bin/bash

n_nodes=$1
offset=$2
echo "Launching $n_nodes nodes"
rank=0
mkdir -p logs/launch_stdout
mkdir -p logs/launch_stderr
master=$(cat ./computers_name | tail -n +$offset | head -n 1 | cut -d @ -f 2)
port=$(shuf -i 2000-65000 -n 1)
echo "Master is $master"
echo "Master port is $port"
for i in $(cat ./computers_name | tail -n +$offset)
do
    if (($rank >= $n_nodes)); then
        break
    fi
    ssh -oStrictHostKeyChecking=no $i "who > ~/timothee/tmp"
    if (($(cat ~/timothee/tmp | wc -l) > 0)); then
        continue
    fi
    if (($rank == 0)); then
        master=$(echo $i | cut -d @ -f 2)
    fi
    rm -f logs/launch_stderr/log.$i
    rm -f logs/launch_stdout/log.$i
    ssh -oStrictHostKeyChecking=no $i "tmux new-session -d -s imagnum \"cd ~/timothee/imagnum/let_there_be_color && ./launch_worker.sh $port $master $n_nodes $rank >logs/launch_stdout/log.$i 2>logs/launch_stderr/log.$i\"" &
    echo Launched node $i with rank $rank
    ((rank++))
done
