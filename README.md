
# Introduction
This repository includes the code for generating the circuit datasets used in [this](link) paper that requires simulation backend (ngspice).
We have separated the data generation code from the machine learning code to allow users to focus on what they percieve as important.

# Setup 

Clone the repo and update the submodules.

```
git clone
cd repo
git submodule update --init --recursive
```

You should now have three submodules (`bb_envs`, `blackbox_eval_engine`, and `utils`) that are required for running ngspice simulations from python environments.

# NGSpice installation (for circuit simulations)
NGspice 2.7 needs to be installed separately, via this [installation link](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/27/). Page 607 of the pdf manual on the website has instructions on how to install. Note that you might need to remove some of the flags to get it to install correctly for your machine.

We have also provided a [docker file](https://github.com/kouroshHakha/circuit-fewshot/blob/master/docker/DockerFile) for setting up the dependencies in a separate isolated environment. 



# Code Structure
- `blackbox_eval_engine` is the submodule wrapping around a general purpose evaluation engine 
(Don't worry about it)
- `bb_envs` is the submodule instantiating different black boxes (e.g. ngspice circuit env instances, 
or simple optimization benchmark functions)
- `utils`: This is a utils submodule re-using most common ML/python convenience methods. 
- `graph_data_gen`: This folder includes the scripts / class implementations for creating the graph datasets used in the paper (and more).

# Data generation instruction

## Circuit setup
The first step for generating a dataset using this flow is to make sure the simulator engine for that particular circuit is setup properly. 
This includes setting up the netlist template, the simulation flow classes (i.e. `NgspiceWrapper` and `NgspiceFlowManager`). 
This part requires a bit of domain knowledge about circuits and how the simulation should be setup. 
Inheritance from `NgspiceWrapper` and `NgspiceFlowManager` should take care of the majority of the engineering required for calling ngspice processes with the correct parameters. 
Example implementations of these files are included in the `bb_envs` submodule for a couple of circuits includeing `weathstoneb`, `two_stage_graph`, and `two_stage_biased_pmos` which re-uses implementations from `two_stage_graph`.

Once these components are available and tested (via running `python bb_env/run_scripts/test_random_sampling.py` for example) we can go ahead and create a config file for data generation procedure. This config file will describe how we should sweep the parameters of the circuit. For example you can specify that two parameters should always be matched to each other or other constraints in form of a random decision tree (see examples in [graph_data_gen/configs/biased_pmos/conf.py](https://github.com/kouroshHakha/circuit-fewshot/blob/master/graph_data_gen/configs/biased_pmos/conf.py)).

## Raw data generation
To generate the raw data (simulation hdf5 files) by randomly sweeping the parameters defined in the config file we do:

```
NGSPICE_TMP_DIR=<TMP_DIR> python graph_data_gen/gen_data.py --path <config_folder_path>
```

Example for two_stage_graph data:
```
NGSPICE_TMP_DIR=/store/nosnap/results/ngspice python graph_data_gen/gen_data.py --path graph_data_gen/configs/two_stage_graph
```

This is the most time-consuming part of the data generation as it requires generating the netlists and calling the simulator on thos netlists and post-processing the results in python. 

## Converting the raw data to a raw graph data 
This part includes running a script that describes the graph of that particular circuit and clearly defines nodes and edges along with the simulation data. This is the most error-prone part of the data generation process. Make sure all nodes are specified correctly and the results are getting aggregated without any silent issues. This will bite you later during machine learning training.

We have created three separate instantiations of this part. For Weathstone bridge and other resistor network circuits we consider a separate, more ad-hoc, script (i.e. `create_resistive_test_networks.py`), and for transistor-based circuits we call a different script (i.e. `collect_train_json.py`):

1) Two stage Graph: this dataset generates the circuit instances of a two stage opamp with current mirror self bias pmos. Each instance is simulated and both the DC and AC simulation results are stored in the graph. 

```
python graph_data_gen/collect_train_json.py --raw_path /store/nosnap/results/ngspice/designs_two_stage_graph --dst_path /store/nosnap/results/ngspice/two_stage_graph/train/raw
```

The `<raw_path>` is where the simulation results are stored. The `dst_path` is where the raw graph results should be stored.

2) Two stage graph with the biased pmos: In this script we modify the previous graph slightly by removing the self bias connections and adding a new voltage source. 

To get this graph results:
```
python graph_data_gen/collect_train_json.py --raw_path /store/nosnap/results/ngspice/designs_two_stage_biased_pmos/ --dst_path /store/nosnap/results/ngspice/two_stage_biased_pmos/train/raw --biased_pmos
```

The `--biased_pmos` flag informs the script to change the graph topology.

3) Wheatstone bridge: Since the manual computation of voltages for a wheatstone bridge is difficult to hard-code we opted to use the ngspice simulator instead. 

You can simply run the `create_resistive_test_networks.py` by commenting out a few lines. Please refer to the script for more information.

The resulting raw folder is the raw data that is required by the custom `pytorch_geometric` datasets (located in the [algorithm repository](link)). We have also uploaded the datasets to cloud for easier accessability.
