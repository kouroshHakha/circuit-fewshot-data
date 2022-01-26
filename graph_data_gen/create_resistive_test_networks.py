
import torch
import numpy as np
from tqdm import tqdm
import hashlib
from pathlib import Path

from torch_geometric.data import Data

# from utils.pdb import register_pdb_hook
# register_pdb_hook()

from utils.file import read_hdf5

import sys

def _info(etype, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(etype, value, tb)
    else:
        import pdb
        import traceback
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(etype, value, tb)
        print()
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)


def register_pdb_hook():
    sys.excepthook = _info

register_pdb_hook()

def one_hot(p: int, ncodes: int):
    if p < 0:
        raise ValueError('p should be positive')
    p_str = format(1 << p, f'0{ncodes+2}b')[2:]
    return list(map(int, list(p_str)))

def par(rx, ry):
    if rx == float('inf'):
        return ry
    if ry == float('inf'):
        return rx
    return rx * ry / (rx + ry)

def gen_par_res_data():
    """
    Genereates Vin + R1||R2 + R3 + V1 Circuit 
    """

    param_dict=dict(
        vin=np.random.rand(),
        rp1=np.random.rand(),
        rp2=np.random.rand(),
        rv=np.random.rand(),
        v2=np.random.rand(),
    )

    nodes = {
        0: ('gnd', {}),
        1: ('vsrcm', {'value': param_dict['vin']}),
        2: ('vsrcp', {'value': param_dict['vin']}),
        3: ('vnode_i', {}),
        4: ('rt', {'value': param_dict[f'rp1']}),
        5: ('rt', {'value': param_dict[f'rp1']}),
        6: ('rt', {'value': param_dict[f'rp2']}),
        7: ('rt', {'value': param_dict[f'rp2']}),
        8: ('vnode_o', {}),
        9: ('rt', {'value': param_dict[f'rv']}),
        10: ('rt', {'value': param_dict[f'rv']}),
        11: ('vnode_i', {}),
        12: ('vp', {'value': param_dict[f'v2']}),
        13: ('vn', {'value': param_dict[f'v2']}),
        
    }

    edge_list = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (3, 6),
        (4, 5),
        (6, 7),
        (5, 8),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (13, 0),
    ]

    rpar = par(param_dict['rp1'], param_dict['rp2'])
    rv = param_dict['rv']
    tot_r = rpar + rv
    vdc = rv / tot_r * param_dict['vin'] + rpar / tot_r * param_dict['v2']
    vdc = torch.tensor([[vdc]]).float()

    return nodes, edge_list, vdc


def gen_ser_res_data():
    """
    Genereates Vin + (R1+R2) + R3 + V1 Circuit 
    """

    param_dict=dict(
        vin=np.random.rand(),
        rs1=np.random.rand(),
        rs2=np.random.rand(),
        rv=np.random.rand(),
        v2=np.random.rand(),
    )

    nodes = {
        0: ('gnd', {}),
        1: ('vsrcm', {'value': param_dict['vin']}),
        2: ('vsrcp', {'value': param_dict['vin']}),
        3: ('vnode_i', {}),
        4: ('rt', {'value': param_dict[f'rs1']}),
        5: ('rt', {'value': param_dict[f'rs1']}),
        6: ('vnode_o', {}),
        7: ('rt', {'value': param_dict[f'rs2']}),
        8: ('rt', {'value': param_dict[f'rs2']}),
        9: ('vnode_o', {}),
        10: ('rt', {'value': param_dict[f'rv']}),
        11: ('rt', {'value': param_dict[f'rv']}),
        12: ('vnode_i', {}),
        13: ('vp', {'value': param_dict[f'v2']}),
        14: ('vn', {'value': param_dict[f'v2']}),
    }

    edge_list = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (13, 14),
        (14, 0),
    ]

    tot_r = param_dict['rs1'] + param_dict['rs2'] + param_dict['rv']
    v1 = (param_dict['rs2'] + param_dict['rv']) / tot_r * param_dict['vin'] + param_dict['rs1'] / tot_r * param_dict['v2']
    v2 = param_dict['rv'] / tot_r * param_dict['vin'] + (param_dict['rs1'] + param_dict['rs2']) / tot_r * param_dict['v2']
    vdc = torch.tensor([[v1], [v2]]).float()

    return nodes, edge_list, vdc



def gen_rl_div_res_data():
    """
    Genereates Vin + Rh + (Rv + V1) || (RL) Circuit 
    """

    param_dict=dict(
        vin=np.random.rand(),
        rh=np.random.rand(),
        rv=np.random.rand(),
        rl=np.random.rand(),
        v2=np.random.rand(),
    )

    nodes = {
        0: ('gnd', {}),
        1: ('vsrcm', {'value': param_dict['vin']}),
        2: ('vsrcp', {'value': param_dict['vin']}),
        3: ('vnode_i', {}),
        4: ('rt', {'value': param_dict[f'rh']}),
        5: ('rt', {'value': param_dict[f'rh']}),
        6: ('vnode_o', {}),
        7: ('rt', {'value': param_dict[f'rv']}),
        8: ('rt', {'value': param_dict[f'rv']}),
        9: ('vnode_i', {}),
        10: ('vp', {'value': param_dict[f'v2']}),
        11: ('vn', {'value': param_dict[f'v2']}),
        12: ('rt', {'value': param_dict[f'rl']}),
        13: ('rt', {'value': param_dict[f'rl']}),
    }

    edge_list = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 0),
        (6, 12),
        (12, 13),
        (13, 0),
    ]

    rh = param_dict['rh']
    rv = param_dict['rv']
    rl = param_dict['rl']
    vin_factor = par(rl, rv) / (par(rl, rv) + rh)
    v2_factor = par(rl, rh) / (par(rl, rh) + rv)
    v1 = vin_factor * param_dict['vin'] + v2_factor * param_dict['v2']
    vdc = torch.tensor([[v1]]).float()

    return nodes, edge_list, vdc

def gen_wheatstone_bridge_data():

    try:
        from bb_envs.ngspice.wrappers.wheatstoneb import Wrapper as WBWrapper
    except ModuleNotFoundError:
        raise ModuleNotFoundError('You should include bb_env package in your python file with WheatstoneB Wrapper')

    netlist_manager = WBWrapper(
        num_process=100, 
        design_netlist='bb_envs/src/bb_envs/ngspice/templates/wheatstoneb/wb.cir',
    )

    params = dict(
        v1=np.random.rand(),
        v2=np.random.rand(),
        r1=np.random.rand(),
        r2=np.random.rand(),
        rp1=np.random.rand(),
        rp2=np.random.rand(),
        rp3=np.random.rand(),
        rp4=np.random.rand(),
        rload=np.random.rand(),
    )

    hashable_params = repr(tuple(sorted(params.items()))).encode('utf-8')
    params.update(id=hashlib.sha1(hashable_params).hexdigest()[:10])
    params['dc'] = str(netlist_manager.get_design_folder(params['id']) / f'dc.csv')
    params.update(include=str(Path('bb_envs/src/bb_envs/ngspice/models/45nm_bulk.txt').resolve()))
    
    with netlist_manager as netlister:
        netlister.run([params], verbose=False)

    sim_data = read_hdf5(Path(params['dc']).parent / 'sim.hdf5')
    vnodes = sim_data['nodes']
    
    nodes = {
        0: ('gnd', {}),
        1: ('vsrcm', {'value': params['v1']}),
        2: ('vsrcp', {'value': params['v1']}),
        3: ('vnode_i', {}),
        4: ('rt', {'value': params[f'r1']}),
        5: ('rt', {'value': params[f'r1']}),
        6: ('vnode_o', {}),
        7: ('rt', {'value': params[f'rp1']}),
        8: ('rt', {'value': params[f'rp1']}),
        9: ('vnode_o', {}),
        10: ('rt', {'value': params[f'rp2']}),
        11: ('rt', {'value': params[f'rp2']}),
        12: ('vnode_o', {}),
        13: ('rt', {'value': params[f'rp3']}),
        14: ('rt', {'value': params[f'rp3']}),
        15: ('vnode_o', {}),
        16: ('rt', {'value': params[f'rp4']}),
        17: ('rt', {'value': params[f'rp4']}),
        18: ('rt', {'value': params[f'rload']}),
        19: ('rt', {'value': params[f'rload']}),
        20: ('rt', {'value': params[f'r2']}),
        21: ('rt', {'value': params[f'r2']}),
        22: ('vnode_i', {}),
        23: ('vp', {'value': params[f'v2']}),
        24: ('vn', {'value': params[f'v2']}),
    }

    edge_list = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (6, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 12),
        (15, 18),
        (18, 19),
        (19, 9),
        (12, 20),
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
        (24, 0),
    ]

    # rh = param_dict['rh']
    # rv = param_dict['rv']
    # rl = param_dict['rl']
    # vin_factor = par(rl, rv) / (par(rl, rv) + rh)
    # v2_factor = par(rl, rh) / (par(rl, rh) + rv)
    # v1 = vin_factor * param_dict['vin'] + v2_factor * param_dict['v2']

    v6 = float(vnodes['2']['dc'])
    v9 = float(vnodes['3']['dc'])
    v12 = float(vnodes['4']['dc'])
    v15 = float(vnodes['5']['dc'])
    vdc = torch.tensor([[v6, v9, v12, v15]]).T.float()

    return nodes, edge_list, vdc


def gen_data(circuit_name: str):

    if circuit_name == 'rpar':
        ret = gen_par_res_data()
    elif circuit_name == 'rser':
        ret = gen_ser_res_data()
    elif circuit_name == 'rl_div':
        ret = gen_rl_div_res_data()
    elif circuit_name == 'wheatstone_b':
        ret = gen_wheatstone_bridge_data()
    
    nodes, edge_list, vdc = ret
    
    edge_list += [(j, i) for i, j in edge_list]
    
    # label node types as integers 0, 1, ..., n - 1 by sorting them alphabetically
    node_sets = sorted(list({val[0] for val in nodes.values()}))
    n_ntypes = len(node_sets)
    ntype_map = dict(zip(node_sets, range(n_ntypes)))
    
    x = torch.tensor([node_prop[1].get('value', 0) for node_prop in nodes.values()]).float().unsqueeze(-1)
    x_type = torch.tensor([one_hot(ntype_map[node_prop[0]], n_ntypes) for node_prop in nodes.values()]).float()
    output_node_mask = torch.tensor([node_prop[0] == 'vnode_o' for node_prop in nodes.values()]).bool()

    data = Data(edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(), x=x)
    data.type_tens = x_type
    data.output_node_mask = output_node_mask
    data.vdc = vdc


    # debugging the graph conversion ...
    from torch_geometric.utils import to_networkx
    import networkx as nx
    import matplotlib.pyplot as plt
    g = to_networkx(data, to_undirected=True)

    nx.draw_circular(g, with_labels=True)
    plt.savefig('graph.png', dpi=250)
    breakpoint()

    return data

def get_dataset(circuit_name, n=1000, seed=10):

    # we have a total of 15 stimulous types, for each one create n random cases
    data_list = []
    
    np.random.seed(seed)
    for _ in tqdm(range(n)):
        data = gen_data(circuit_name)
        data_list.append(data)

    return data_list


if __name__ == '__main__':
    # # downstream train
    # test_set = get_dataset('rpar', 1000, seed=10)
    # torch.save(test_set, 'rdiv/rpar_train.pt')
    # test_set = get_dataset('rpar', 1000, seed=100)
    # torch.save(test_set, 'rdiv/rpar_test.pt')

    # # downstream train
    # test_set = get_dataset('rser', 1000, seed=10)
    # torch.save(test_set, 'rdiv/rser_train.pt')
    # test_set = get_dataset('rser', 1000, seed=100)
    # torch.save(test_set, 'rdiv/rser_test.pt')

    # # downstream train
    # test_set = get_dataset('rl_div', 1000, seed=10)
    # torch.save(test_set, 'rdiv/rl_div_train.pt')
    # test_set = get_dataset('rl_div', 1000, seed=100)
    # torch.save(test_set, 'rdiv/rl_div_test.pt')

    # downstream train
    test_set = get_dataset('wheatstone_b', 1000, seed=10)
    torch.save(test_set, 'wheatstone_b_train.pt')
    test_set = get_dataset('wheatstone_b', 1000, seed=100)
    torch.save(test_set, 'wheatstone_b_test.pt')
