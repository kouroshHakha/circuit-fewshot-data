"""Converts the collect raw graph data to a compressed, easy-to-work-with json file
The data is assumed to be training data that is collected by simulating a single netlist per instance.
"""


from asyncore import read
import torch
from torch_geometric.utils import from_networkx


import argparse
from pathlib import Path
import json
import shutil
from tqdm import tqdm

from graph_data_gen.circuits_data import Netlist

from utils.pdb import register_pdb_hook
register_pdb_hook()


def _parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', help='The folder with all the simulation results')
    parser.add_argument('--dst_path', help='The path for the destination folder')
    parser.add_argument('--biased_pmos', action='store_true')

    return parser.parse_args()


def get_netlist_dict(json_path, biased_pmos=False):
    with open(json_path, 'r') as f:
        params = json.load(f)

    ans = dict(
        mn1=dict(type='M', params=dict(is_nmos=True, w=int(params['mn1'])), terminals=dict(D='net4', G='net2', S='net3', B='vss')),
        mn2=dict(type='M', params=dict(is_nmos=True, w=int(params['mn1'])), terminals=dict(D='net5', G='net1', S='net3', B='vss')),
        mn3=dict(type='M', params=dict(is_nmos=True, w=int(params['mn3'])), terminals=dict(D='net3', G='net7', S='vss', B='vss')),
        mn4=dict(type='M', params=dict(is_nmos=True, w=int(params['mn4'])), terminals=dict(D='net7', G='net7', S='vss', B='vss')),
        mn5=dict(type='M', params=dict(is_nmos=True, w=int(params['mn5'])), terminals=dict(D='net6', G='net7', S='vss', B='vss')),
        mp1=dict(type='M', params=dict(is_nmos=False, w=int(params['mp1'])), terminals=dict(D='net4', G='net4', S='vdd', B='vdd')),
        mp2=dict(type='M', params=dict(is_nmos=False, w=int(params['mp1'])), terminals=dict(D='net5', G='net4', S='vdd', B='vdd')),
        mp3=dict(type='M', params=dict(is_nmos=False, w=int(params['mp3'])), terminals=dict(D='net6', G='net5', S='vdd', B='vdd')),
        ibias=dict(type='I', params=dict(dc=params['ibias_dc'], ac_mag=params['ibias_mag'], ac_ph=params['ibias_ph']), terminals=dict(PLUS='vdd', MINUS='net7')),
        vvdd=dict(type='V', params=dict(dc=params['vdd_dc'], ac_mag=params['vdd_mag'], ac_ph=params['vdd_ph']), terminals=dict(PLUS='vdd', MINUS='0')),
        vvss=dict(type='V', params=dict(dc=params['vss_dc'], ac_mag=params['vss_mag'], ac_ph=params['vss_ph']), terminals=dict(PLUS='vss', MINUS='0')),
        vin1=dict(type='V', params=dict(dc=params['vin1_dc'], ac_mag=params['vin1_mag'], ac_ph=params['vin1_ph']), terminals=dict(PLUS='net1', MINUS='0')),
        vin2=dict(type='V', params=dict(dc=params['vin2_dc'], ac_mag=params['vin2_mag'], ac_ph=params['vin2_ph']), terminals=dict(PLUS='net2', MINUS='0')),
        rz=dict(type='R', params=dict(value=params['rz']), terminals=dict(PLUS='net8', MINUS='net6')),
        cc=dict(type='C', params=dict(value=params['cc']), terminals=dict(PLUS='net5', MINUS='net8')),
        CL=dict(type='C', params=dict(value=params['cload']), terminals=dict(PLUS='net6', MINUS='0')),
    )

    if biased_pmos:
        ans.update(
            **dict(
                mp1=dict(type='M', params=dict(is_nmos=False, w=int(params['mp1'])), terminals=dict(D='net4', G='netb', S='vdd', B='vdd')),
                mp2=dict(type='M', params=dict(is_nmos=False, w=int(params['mp1'])), terminals=dict(D='net5', G='netb', S='vdd', B='vdd')),
                vbias=dict(type='V', params=dict(dc=params['vbias_dc'], ac_mag=0, ac_ph=0), terminals=dict(PLUS='vdd', MINUS='netb')),
            )
        )
    
    return ans

def main(pargs):
    raw_path = Path(pargs.raw_path)
    dst_path = Path(pargs.dst_path)

    if (dst_path / 'train.json').exists():
        raise FileExistsError(f"File exists: {dst_path / 'train.json'}")
        
    dst_path.mkdir(exist_ok=True, parents=True)

    data = []

    with tqdm(total=len(list(raw_path.iterdir()))) as progress_bar:
        for rpath in raw_path.iterdir():
            if rpath.is_dir():
                dsn_id = rpath.stem.split('_')[-1]
                try:
                    netlist_dict = get_netlist_dict(rpath / 'params.json', biased_pmos=pargs.biased_pmos)
                except FileNotFoundError:
                    continue
                netlist = Netlist(netlist_dict)
                graph = netlist.graph

                # nodes
                nodes = [dict(name=node, type=graph.nodes[node]['type'], 
                props=graph.nodes[node].get('props', {})) for node in graph]

                # edges
                edges = list(graph.edges())
                
                # sim 
                sim_fpath = f'sim_{dsn_id}.hdf5'

                shutil.copy(rpath / 'sim.hdf5', dst_path / sim_fpath)
                data.append(
                    dict(
                        id=dsn_id,
                        nodes=nodes,
                        edges=edges,
                        sim_fpath=sim_fpath,
                    )
                )

            progress_bar.update(1)
        
    with open(dst_path / 'train.json', 'w') as f:
        json.dump(data, f)
    

if __name__ == '__main__':
    main(_parse_args())

            


