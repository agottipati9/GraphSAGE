#!/bin/bash

# Error Check
if [ "$#" -ne 2 ]; then
    echo "Please Run: run_tda_eval.sh [number of subgraphs] [number of nodes per subgraph]"
    exit 1
fi

iter=$(($1+0))
n_nodes=$(($2+0))
echo "Generating ${iter} subgraphs with ${n_nodes} node embeddings..."
for ((i = 0; i < iter; i++));
do
	python -m graphsage.unsupervised_train --train_prefix ./example_data/toy-ppi --name "val${i}" --num_nodes $2 --model graphsage_mean --max_total_steps 1000 --validate_iter 10
done

echo "Generating subgraph embeddings..."
python gen_subgraph_embeds.py

echo "Evaluating subgraph embeddings without TDA features..."
python eval_scripts/tda_eval.py no > results_no_tda.txt

echo "Evaluating subgraph embeddings with TDA features..."
python eval_scripts/tda_eval.py yes > results_tda.txt
