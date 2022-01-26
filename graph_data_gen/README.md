
# Deprecated

To generate the uniformly distributed random samples from the search space. This will simulate ac and dc characteristic of the circuit and measure the output voltages and currents and stores them in hdf5 file.
```
NGSPICE_TMP_DIR=/store/nosnap/results/ngspice python graph_data_gen/gen_data.py --path graph_data_gen/config/
```

To convert the generated raw data into an intermediary graph reperesentation:

```
python graph_data_gen/collect_train_json.py --raw_path /store/nosnap/results/ngspice/designs_two_stage_graph --dst_path /store/nosnap/results/ngspice/two_stage_graph/train/raw
```

For biased pmos circuit do:

```
python graph_data_gen/collect_train_json.py --raw_path /store/nosnap/results/ngspice/designs_two_stage_biased_pmos/ --dst_path /store/nosnap/results/ngspice/two_stage_biased_pmos/train/raw --biased_pmos
```

To create a digestable dataset for GNN architectures:
```
python graph_data_gen/graph_data.py
```
This will create a processed folder and caches the final representation. Data will still needs to be transformed and normalized for proper usage.

To generate the test data:

```
NGSPICE_TMP_DIR=/tmp/ngspice python blackbox_eval_engine/run_scripts/generate_rand_db.py -db /store/nosnap/results/ngspice/test_two_stage_graph.pkl -n 1000
```

To convert the generated pickle file to the folder structure consistent with the graph data:

```
python graph_data_gen/collect_test_json.py --raw_file /store/nosnap/results/ngspice/test_two_stage_graph.pkl --dst_path /store/nosnap/results/ngspice/two_stage_graph/test/raw
```
