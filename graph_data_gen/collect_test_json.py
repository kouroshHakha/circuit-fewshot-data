"""Converts the collect raw graph data to a compressed, easy-to-work-with json file
The data is assumed to be training data that is collected by simulating a single netlist per instance.
"""


from graph_data_gen.collect_train_json import get_netlist_dict
import torch
from torch_geometric.utils import from_networkx


import argparse
from pathlib import Path
import json
from tqdm import tqdm
from utils.file import read_pickle, write_hdf5
from copy import deepcopy

from graph_data_gen.circuits_data import Netlist

from utils.pdb import register_pdb_hook
register_pdb_hook()


def _parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_file', help='The database pkl file with all the Design objects')
    parser.add_argument('--dst_path', help='The path for the destination folder')

    return parser.parse_args()

def get_netlist_dicts(design):
    """ For test set each design is mapped to 9 graphs (w/ different node features) for this
    particular problem. This is because at test time we only want to consider graph-level 
    outputs that are not inputs consistent with graph's structure. In other words, we are 
    seeking input agnostic graph embeddings for each netlist. Therefore, we have to manually 
    update the node features to reflect different input scenarios and have identical outputs 
    for all these variations.

    Also for this test data dc sources and CLOAD are fixed.
    """

    params = design.value_dict

    skeleton  = dict(
        mn1=dict(type='M', params=dict(is_nmos=True, w=params['mn1']), terminals=dict(D='net4', G='net2', S='net3', B='vss')),
        mn2=dict(type='M', params=dict(is_nmos=True, w=params['mn1']), terminals=dict(D='net5', G='net1', S='net3', B='vss')),
        mn3=dict(type='M', params=dict(is_nmos=True, w=params['mn3']), terminals=dict(D='net3', G='net7', S='vss', B='vss')),
        mn4=dict(type='M', params=dict(is_nmos=True, w=params['mn4']), terminals=dict(D='net7', G='net7', S='vss', B='vss')),
        mn5=dict(type='M', params=dict(is_nmos=True, w=params['mn5']), terminals=dict(D='net6', G='net7', S='vss', B='vss')),
        mp1=dict(type='M', params=dict(is_nmos=False, w=params['mp1']), terminals=dict(D='net4', G='net4', S='vdd', B='vdd')),
        mp2=dict(type='M', params=dict(is_nmos=False, w=params['mp1']), terminals=dict(D='net5', G='net4', S='vdd', B='vdd')),
        mp3=dict(type='M', params=dict(is_nmos=False, w=params['mp3']), terminals=dict(D='net6', G='net5', S='vdd', B='vdd')),
        rz=dict(type='R', params=dict(value=params['rz']), terminals=dict(PLUS='net8', MINUS='net6')),
        cc=dict(type='C', params=dict(value=params['cc']), terminals=dict(PLUS='net5', MINUS='net8')),
        CL=dict(type='C', params=dict(value=10e-12), terminals=dict(PLUS='net6', MINUS='0')),
    )

    input_cases = [
        dict(ibias_ac_mag=1, vvdd_ac_mag=0, vvss_ac_mag=0, vin1_ac_mag=0, vin2_ac_mag=0, vin1_ac_ph=0, vin2_ac_ph=0),
        dict(ibias_ac_mag=0, vvdd_ac_mag=1, vvss_ac_mag=0, vin1_ac_mag=0, vin2_ac_mag=0, vin1_ac_ph=0, vin2_ac_ph=0),
        dict(ibias_ac_mag=0, vvdd_ac_mag=0, vvss_ac_mag=1, vin1_ac_mag=0, vin2_ac_mag=0, vin1_ac_ph=0, vin2_ac_ph=0),
        dict(ibias_ac_mag=0, vvdd_ac_mag=0, vvss_ac_mag=0, vin1_ac_mag=1, vin2_ac_mag=0, vin1_ac_ph=0, vin2_ac_ph=0),
        dict(ibias_ac_mag=0, vvdd_ac_mag=0, vvss_ac_mag=0, vin1_ac_mag=0, vin2_ac_mag=1, vin1_ac_ph=0, vin2_ac_ph=0),
        dict(ibias_ac_mag=0, vvdd_ac_mag=0, vvss_ac_mag=0, vin1_ac_mag=1, vin2_ac_mag=1, vin1_ac_ph=0, vin2_ac_ph=0),
        dict(ibias_ac_mag=0, vvdd_ac_mag=0, vvss_ac_mag=0, vin1_ac_mag=1, vin2_ac_mag=1, vin1_ac_ph=180, vin2_ac_ph=180),
        dict(ibias_ac_mag=0, vvdd_ac_mag=0, vvss_ac_mag=0, vin1_ac_mag=1, vin2_ac_mag=1, vin1_ac_ph=0, vin2_ac_ph=180),
        dict(ibias_ac_mag=0, vvdd_ac_mag=0, vvss_ac_mag=0, vin1_ac_mag=1, vin2_ac_mag=1, vin1_ac_ph=180, vin2_ac_ph=0),
    ]

    netlist_dict_list = []
    for case in input_cases:
        netlist_dict = deepcopy(skeleton)
        netlist_dict.update(
            ibias=dict(type='I', params=dict(dc=30e-6, ac_mag=case['ibias_ac_mag'], ac_ph=0), terminals=dict(PLUS='vdd', MINUS='net7')),
            vvdd=dict(type='V', params=dict(dc=1.2, ac_mag=case['vvdd_ac_mag'], ac_ph=0), terminals=dict(PLUS='vdd', MINUS='0')),
            vvss=dict(type='V', params=dict(dc=0, ac_mag=case['vvss_ac_mag'], ac_ph=0), terminals=dict(PLUS='vss', MINUS='0')),
            vin1=dict(type='V', params=dict(dc=0.6, ac_mag=case['vin1_ac_mag'], ac_ph=case['vin1_ac_ph']), terminals=dict(PLUS='net1', MINUS='0')),
            vin2=dict(type='V', params=dict(dc=0.6, ac_mag=case['vin2_ac_mag'], ac_ph=case['vin2_ac_ph']), terminals=dict(PLUS='net2', MINUS='0')),
        )
        netlist_dict_list.append(netlist_dict)

    return netlist_dict_list

def get_id(design):

    ac_path = design.specs['ac']
    name = str(Path(ac_path).parent.stem)
    dsn_id = name.split('_')[-1]
    return dsn_id

def main(pargs):
    raw_file = Path(pargs.raw_file)
    dst_path = Path(pargs.dst_path)

    if (dst_path / 'test.json').exists():
        raise FileExistsError("File exists: '/store/nosnap/results/ngspice/two_stage_graph/raw'")
        
    dst_path.mkdir(exist_ok=True, parents=True)

    data = []

    designs = read_pickle(raw_file)
    
    for dsn in tqdm(designs):
        netlist_dicts = get_netlist_dicts(dsn)
        
        dsn_id = get_id(dsn)
        for i, netlist_dict in enumerate(netlist_dicts):
            netlist = Netlist(netlist_dict)
            graph = netlist.graph

            # nodes
            nodes = [dict(name=node, type=graph.nodes[node]['type'], 
            props = graph.nodes[node].get('props', {})) for node in graph]

            # edges
            edges = list(graph.edges())
            
            data.append(
                dict(
                    id=f'{dsn_id}_{i}',
                    nodes=nodes,
                    edges=edges,
                    gout_fpath=f'gout_{dsn_id}.hdf5',
                )
            )


        gout = dict(
            gain=dsn.specs['gain'],
            ugbw=dsn.specs['ugbw'],
            pm=dsn.specs['pm'],
            ibias=dsn.specs['ibias'].item(),
            psrr=dsn.specs['psrr'],
            offset_sys=dsn.specs['offset_sys'],
            tset=dsn.specs['tset'],
            cost=dsn.specs['cost'],
        )
        write_hdf5(gout, dst_path / f'gout_{dsn_id}.hdf5')

    with open(dst_path / 'test.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    main(_parse_args())