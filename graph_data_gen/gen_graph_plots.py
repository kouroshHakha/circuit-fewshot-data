import torch

from utils.pdb import register_pdb_hook
register_pdb_hook()

from torch_geometric.utils import to_networkx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate as interp
import scipy.optimize as sciopt
from graph_data_gen.graph_data import CircuitGraphDataset
import math


def get_vout_freq(data, output_node_idx):
    output_node = torch.arange(data.num_nodes)[data.output_node_mask] == output_node_idx
    vout_real = data.vac_real[output_node].squeeze(0)
    vout_imag = data.vac_imag[output_node].squeeze(0)
    vout = (vout_real + 1j * vout_imag)
    freq = data.v_freq[output_node].squeeze(0)
    return vout, freq

def compute_dc_gain(data, output_node_idx):
    vout, _ = get_vout_freq(data, output_node_idx)
    gain = vout.abs()[0].item()
    return gain

def find_ugbw(data, output_node_idx):
    vout, freq = get_vout_freq(data, output_node_idx)

    gain = vout.abs()
    ugbw, valid = _get_best_crossing(freq.numpy(), gain.numpy(), val=1)
    if valid:
        return ugbw
    else:
        return freq[0]


def _get_best_crossing(xvec, yvec, val):
    interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

    def fzero(x):
        return interp_fun(x) - val

    xstart, xstop = xvec[0], xvec[-1]
    try:
        return sciopt.brentq(fzero, xstart, xstop), True
    except ValueError:
        # avoid no solution
        # if abs(fzero(xstart)) < abs(fzero(xstop)):
        #     return xstart
        return xstop, False

if __name__ == '__main__':


    dataset = CircuitGraphDataset(root='/store/nosnap/results/ngspice/two_stage_graph')
    dataset.shuffle()

    node_lut = next(iter(dataset.graph_nodes.values()))
    inverted_node_lut = {v:k for k, v in node_lut.items()}

    vsrcs = ['vin1, vin2, vvss, vvdd']

    ac_input_cats = dict(
        vdd=[],
        vss=[],
        ibias=[],
        vin1=[],
        vin2=[],
        vin_cm_0=[],
        vin_cm_180=[],
        vin_diff_0=[],
        vin_diff_180=[],
    )

    anchor_idx = len(dataset.node_types)

    df = pd.DataFrame()
    df_data = []
    for data in dataset[:1000]:
        df_dict = {
            'ac_input': None,
            'input_matched': None,
            'mn1': None,
            'mn2': None,
            'mn3': None,
            'mn4': None,
            'mn5': None,
            'mp1': None,
            'mp2': None,
            'mp3': None,
            'rz': None,
            'cc': None,
            'cload': None,
            'vdd_dc': None,
            'vss_dc': None,
            'vin2_dc': None,
            'vin1_dc': None,
            'ibias_dc': None,
        }

        feat = data.x

        ######## Check graph degrees
        # graph = to_networkx(data)
        # for n in graph:
        #     neighbors = [inverted_node_lut[i] for i in graph.neighbors(n)]
        #     print(f'{inverted_node_lut[n]} : {",".join(neighbors)}')

        ######## Check distribution of graph categories based on ac_inputs / outputs
        vdd_feat = feat[node_lut['T_vvdd_PLUS']][anchor_idx:][5:8]
        vss_feat = feat[node_lut['T_vvss_PLUS']][anchor_idx:][5:8]
        ibias_feat = feat[node_lut['T_ibias_PLUS']][anchor_idx:][2:5]
        vin2_feat = feat[node_lut['T_vin2_PLUS']][anchor_idx:][5:8]
        vin1_feat = feat[node_lut['T_vin1_PLUS']][anchor_idx:][5:8]

        ac_input = None
        if vdd_feat[1]:
            # ac_input_cats['vdd'].append(data)
            ac_input = 'vdd'
        elif vss_feat[1]:
            # ac_input_cats['vss'].append(data)
            ac_input = 'vss'
        elif ibias_feat[1]:
            # ac_input_cats['ibias'].append(data)
            ac_input = 'ibias'
        elif vin1_feat[1] and not vin2_feat[1]:
            # ac_input_cats['vin1'].append(data)
            ac_input = 'vin1'
        elif not vin1_feat[1] and vin2_feat[1]:
            # ac_input_cats['vin2'].append(data)
            ac_input = 'vin2'
        elif vin1_feat[1] and vin2_feat[1]:
            if vin1_feat[-1] == 0 and vin2_feat[-1] == 0:
                # ac_input_cats['vin_cm_0'].append(data)
                ac_input = 'vin_cm_0'
            elif vin1_feat[-1] == 180 and vin2_feat[-1] == 180:
                # ac_input_cats['vin_cm_180'].append(data)
                ac_input = 'vin_cm_180'
            elif vin1_feat[-1] == 180 and vin2_feat[-1] == 0:
                # ac_input_cats['vin_diff_0'].append(data)
                ac_input = 'vin_diff_0'
                df_dict['dc_gain'] = compute_dc_gain(data, node_lut['V_net6'])
                df_dict['log_fu'] = math.log(find_ugbw(data, node_lut['V_net6']))
            elif vin1_feat[-1] == 0 and vin2_feat[-1] == 180:
                # ac_input_cats['vin_diff_180'].append(data)
                ac_input = 'vin_diff_180'
                df_dict['dc_gain'] = compute_dc_gain(data, node_lut['V_net6'])
                df_dict['log_fu'] = math.log(find_ugbw(data, node_lut['V_net6']))

        df_dict['ac_input'] = ac_input
        df_dict['input_matched'] = (vin1_feat[0] == vin2_feat[0]).item()
        df_dict.update({
            'mn1': feat[node_lut['T_mn1_D']][-1].item(),
            'mn2': feat[node_lut['T_mn2_D']][-1].item(),
            'mn3': feat[node_lut['T_mn3_D']][-1].item(),
            'mn4': feat[node_lut['T_mn4_D']][-1].item(),
            'mn5': feat[node_lut['T_mn5_D']][-1].item(),
            'mp1': feat[node_lut['T_mp1_D']][-1].item(),
            'mp2': feat[node_lut['T_mp2_D']][-1].item(),
            'mp3': feat[node_lut['T_mp3_D']][-1].item(),
            'cc':  feat[node_lut['T_cc_PLUS']][anchor_idx].item(),
            'cload': feat[node_lut['T_CL_PLUS']][anchor_idx].item(),
            'rz':  feat[node_lut['T_rz_PLUS']][anchor_idx + 1].item(),
            'vdd_dc': vdd_feat[0].item(),
            'vss_dc': vss_feat[0].item(),
            'vin2_dc': vin2_feat[0].item(),
            'vin1_dc': vin1_feat[0].item(),
            'ibias_dc': ibias_feat[0].item(),
        })

        df_data.append(df_dict)
    
    df = df.append(df_data)
    
    plt.close()
    sns.histplot(data=df, x='dc_gain', bins=20)
    plt.savefig('dc_gain.png')

    plt.close()
    sns.histplot(data=df, x='log_fu', bins=20)
    plt.savefig('log_fu.png')

    plt.close()
    ax = sns.displot(df, x='ac_input', shrink=0.8, hue='input_matched', multiple='dodge')
    ax.set_xticklabels(rotation=45)
    plt.savefig('hist_ac_input.png')

    hist_key_list = ['mn1','mn2','mn3','mn4','mn5','mp1','mp2','mp3','rz' ,'cc' , 'cload', 'vdd_dc', 'vss_dc', 'vin2_dc', 'vin1_dc', 'ibias_dc']

    plt.close()
    _, axes = plt.subplots(4,4, figsize=(16, 16))

    for ax, title in zip(axes.flatten(), hist_key_list):
        sns.histplot(data=df, x=title, ax=ax, bins=20)
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig('hist_circuit_values.png')


