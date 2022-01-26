from logging import warning
from typing import Tuple, Dict

import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from pathlib import Path
from torch_geometric.data import Dataset
import json
import networkx as nx
import numpy as np
from utils.file import read_hdf5
from tqdm import tqdm
import warnings

from utils.pdb import register_pdb_hook
register_pdb_hook()

class CircuitGraphDataset(Dataset):
    node_values = [
        'C_value',
        'R_value',
        'I_dc',
        'I_ac_mag',
        'I_ac_ph',
        'V_dc',
        'V_ac_mag',
        'V_ac_ph',
        'M_w',
    ]

    node_types = [
        'I_PLUS', 'I_MINUS', 
        'V_PLUS', 'V_MINUS', 
        'C_PLUS', 'C_MINUS', 
        'R_PLUS', 'R_MINUS', 
        'M_nmos_D', 'M_nmos_G', 'M_nmos_S', 'M_nmos_B',
        'M_pmos_D', 'M_pmos_G', 'M_pmos_S', 'M_pmos_B',
        'VNode', # non-terminal node
        'GND', # reference node
    ]

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, mode='train'):

        self.feat_dim = len(self.node_values) + len(self.node_types)
        self.mode = mode
        super().__init__(
            root=str(Path(root) / mode), 
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter
        )

        # in-memory attributes to map back to the original graph 
        info = torch.load(Path(self.processed_dir) / self.processed_file_names[0])
        self.graph_ids = info['graph_ids']
        self.graph_nodes = info['graph_nodes']

    @property
    def raw_file_names(self):
        # A list of files in the raw_dir which needs to be found in order to skip the download.
        return [f'{self.mode}.json']

    @property
    def processed_file_names(self):
        return ['graphs.pt']

    def download(self):
        # Downloads raw data into raw_dir
        raise ValueError(f'Did not find {self.mode}.json, make sure your folder has raw/{self.mode}.json in it')

    def process(self):
        # Processes raw data and saves it into the processed_dir
        with open(Path(self.raw_dir) / f'{self.mode}.json', 'r') as f:
            print('Reading the json file ...')
            content = json.load(f)
            print('Read successful!')

        graph_ids = []
        graph_nodes = {}
        i = 0
        for graph_dict in tqdm(content):
            # try:
            node_map, data = self.graph_to_data(graph_dict)
            # except:
            #     continue
            graph_ids.append(graph_dict['id'])
            graph_nodes[graph_dict['id']] = node_map

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, Path(self.processed_dir) / f'data_{i}.pt')
            i += 1

        if len(graph_ids) < len(content):
            warnings.warn(f'{len(content) - len(graph_ids)} graphs were not created.')

        if graph_ids:
            torch.save(dict(graph_ids=graph_ids, graph_nodes=graph_nodes), Path(self.processed_dir) / 'graphs.pt')


    def graph_to_data(self, graph_dict) -> Tuple[Dict[str, int], Data]:
        """ 
        1. Create nodes using integer indices + encode their features
        2. Connect nodes using their edges
        3. Construct output arrays for nodes + output node masks (if available)
        4. Construct output arrays for edges + output edge masks (if available)
        """
        node_map = {}
        G = nx.Graph()
        
        # during infernce/test time sim_fpath is not available and should be predicted
        include_output = 'sim_fpath' in graph_dict
        if include_output:
            sim_data = read_hdf5(Path(self.raw_dir) / graph_dict['sim_fpath'])
            vnodes = sim_data['nodes']
            i_vsrcs = sim_data['vsrcs']
        else:
            vnodes, i_vsrcs = {}, {}
        
        output_node_mask = torch.zeros(len(graph_dict['nodes']), dtype=torch.long)
        v_freq, vac_r, vac_i, vdc = [], [], [], []
        v_device_nodes = []
        for i, node in enumerate(graph_dict['nodes']):
            node_x = np.zeros(self.feat_dim, dtype=np.float32)
            
            # encoding node properties into discretized feature vectors
            props = node['props']
            key_params = {}
            if node['type'] == 'V':
                is_gnd = props['is_gnd']
                node_type_str = 'GND' if is_gnd else 'VNode'

                if vnodes and not is_gnd:                
                    vnode_key = node['name'].split('_')[-1].lower()
                    node_sims = vnodes[vnode_key] 
                    
                    v_freq.append(node_sims['freq'])
                    vac_r.append(node_sims['real'])
                    vac_i.append(node_sims['imag'])
                    vdc.append(node_sims['dc'])
                
                if not is_gnd:
                    output_node_mask[i] = 1

            else:
                # terminal node
                dev_class = props['device_class']
                term_type = props['terminal_type']
                if dev_class == 'M':
                    mos_type = 'n' if props['is_nmos'] else 'p'
                    node_type_str = f'M_{mos_type}mos_{term_type}'
                    key_params['M_w'] = props['w']
                else:
                    node_type_str = f'{dev_class}_{term_type}'

                    if dev_class in ['I', 'V']:
                        key_params[f'{dev_class}_dc'] = props['dc']
                        key_params[f'{dev_class}_ac_mag'] = props['ac_mag']
                        key_params[f'{dev_class}_ac_ph'] = props['ac_ph']

                        if dev_class == 'V':
                            v_device_nodes.append(node['name'])
                    else:
                        key_params[f'{dev_class}_value'] = props['value']

            for key, value in key_params.items():
                node_x[len(self.node_types) + self.node_values.index(key)] = value

            node_enc = np.array(one_hot(self.node_types.index(node_type_str), ncodes=len(self.node_types)))
            node_x[:len(self.node_types)] = node_enc
        
            G.add_node(i, x=node_x)
            node_map[node['name']] = i

        i_freq, iac_r, iac_i, idc = [], [], [], []
        output_current_mask = torch.zeros(len(graph_dict['nodes']), dtype=torch.long)
        for i, edge in enumerate(graph_dict['edges']):
            u, v = edge
            G.add_edge(node_map[u], node_map[v])

            if include_output and u in v_device_nodes and v in v_device_nodes:
                vsrc_key = u.split('_')[1]
                vsrc_sim = i_vsrcs[vsrc_key] 

                i_freq.append(vsrc_sim['freq'])
                iac_r.append(vsrc_sim['real'])
                iac_i.append(vsrc_sim['imag'])
                idc.append(vsrc_sim['dc'])

                output_current_mask[node_map[u]] = 1
                output_current_mask[node_map[v]] = 1
        
        data = from_networkx(G)

        if include_output:
            data.v_freq = torch.tensor(np.stack(v_freq, 0))
            data.vac_real = torch.tensor(np.stack(vac_r, 0))
            data.vac_imag = torch.tensor(np.stack(vac_i, 0))
            data.vdc = torch.tensor(np.stack(vdc, 0)[:, None])

            data.i_freq = torch.tensor(np.stack(i_freq, 0)).repeat_interleave(2, 0)
            data.iac_real = torch.tensor(np.stack(iac_r, 0)).repeat_interleave(2, 0)
            data.iac_imag = torch.tensor(np.stack(iac_i, 0)).repeat_interleave(2, 0)
            data.idc = torch.tensor(np.stack(idc, 0)[:, None]).repeat_interleave(2, 0)

            data.output_node_mask = output_node_mask.bool()
            data.output_current_mask = output_current_mask.bool()

        if 'gout_fpath' in graph_dict:
            gout = read_hdf5(Path(self.raw_dir) / graph_dict['gout_fpath'])

            for k, v in gout.items():
                data.__setattr__(k, torch.tensor(v)[None])

        return node_map, data

    def len(self):
        return len(self.graph_ids)

    def get(self, idx):
        data = torch.load(Path(self.processed_dir) / f'data_{idx}.pt')
        return data


def one_hot(p: int, ncodes: int):
    if p < 0:
        raise ValueError('p should be positive')
    p_str = format(1 << p, f'0{ncodes+2}b')[2:]
    return list(map(int, list(p_str)))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--mode', default='train')
    _args = parser.parse_args()

    dataset = CircuitGraphDataset(_args.root, mode=_args.mode)
    breakpoint()