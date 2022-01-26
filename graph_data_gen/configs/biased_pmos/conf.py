
from hyperopt import hp
import hyperopt.pyll.stochastic as sampler
from hyperopt.pyll import scope
from copy import copy

from bb_envs.ngspice.wrappers.two_stage_biased_pmos import TwoStageOpenLoop

NSAMPLES = int(3e3)

# low, high, quantization
common_m_range = [1, 100, 1]
common_cap_range = [0.1e-12, 10.0e-12, 0.1e-12]
common_res_range = [0, 1000, 50]

vin12_cm = hp.uniform('vin12_cm',  0.4, 0.8)

default_input = dict(
    ibias_mag=0, 
    ibias_ph=0,
    vin1_mag=0, 
    vin1_ph=0,
    vin2_mag=0,
    vin2_ph=0,
    vss_mag=0, 
    vss_ph=0,
    vdd_mag=0, 
    vdd_ph=0,
)


def update_and_return(input_dict, **kwargs):
    cdict = copy(input_dict)
    cdict.update(**kwargs)
    return cdict

raw_space = {
    'mp1': hp.quniform('mp1', *common_m_range),
    'mn1': hp.quniform('mn1', *common_m_range),
    'mp3': hp.quniform('mp3', *common_m_range),
    'mn3': hp.quniform('mn3', *common_m_range),
    'mn4': hp.quniform('mn4', *common_m_range),
    'mn5': hp.quniform('mn5', *common_m_range),
    'cc': hp.uniform('cc',  0.1e-12, 10.0e-12),
    'rz': hp.uniform('rz',  0, 1000),
    'cload': hp.uniform('cload',  0.1e-12, 10.0e-12),
    'ibias_dc': hp.uniform('ibias_dc',  10e-6, 50e-6),
    'vbias_dc': hp.uniform('vbias_dc',  0.2, 0.8),
    'vss_dc': hp.uniform('vss_dc',  0, 0.2),
    'vdd_dc': hp.uniform('vdd_dc',  1, 1.2),
    'vin12_dc': hp.pchoice('vin12_cm', [
        (0.5, {'vin1_dc': vin12_cm, 'vin2_dc': vin12_cm}), # matched choice
        (0.5, {'vin1_dc': hp.uniform('vin1_dc',  0.4, 0.8), 'vin2_dc': hp.uniform('vin2_dc',  0.4, 0.8)}), # independent choice 
    ]),
    'ac_input': hp.choice('ac_input', [
        update_and_return(default_input, vdd_mag=1.0),
        update_and_return(default_input, vss_mag=1.0),
        update_and_return(default_input, ibias_mag=1.0),
        update_and_return(default_input, vin1_mag=1.0),
        update_and_return(default_input, vin2_mag=1.0),
        update_and_return(default_input, vin1_mag=1.0, vin2_mag=1.0, vin2_ph=0, vin1_ph=0),
        update_and_return(default_input, vin1_mag=1.0, vin2_mag=1.0, vin2_ph=0, vin1_ph=180),
        update_and_return(default_input, vin1_mag=1.0, vin2_mag=1.0, vin2_ph=180, vin1_ph=180),
        update_and_return(default_input, vin1_mag=1.0, vin2_mag=1.0, vin2_ph=180, vin1_ph=0),
    ])
}


@scope.define
def process(hp_space):
    vin12_dc_inst = hp_space.pop('vin12_dc')
    hp_space.update(**vin12_dc_inst)

    ac_input_inst = hp_space.pop('ac_input')
    hp_space.update(**ac_input_inst)
    
    return hp_space

space = scope.process(raw_space)


# netlist manager
sim_model = 'bb_envs/src/bb_envs/ngspice/models/45nm_bulk.txt'
netlist_manager = TwoStageOpenLoop(
    num_process=100, 
    design_netlist='bb_envs/src/bb_envs/ngspice/templates/two_stage_biased_pmos/two_stage_biased_pmos.cir',
)

# netlist graph collector
netlist_skeleton = dict(
    mn1=dict(type='M', params=dict(is_nmos=True, w=10), terminals=dict(D='net4', G='net2', S='net3', B='VSS')),
    mn2=dict(type='M', params=dict(is_nmos=True, w=10), terminals=dict(D='net5', G='net1', S='net3', B='VSS')),
    mn3=dict(type='M', params=dict(is_nmos=True, w=10), terminals=dict(D='net3', G='net7', S='VSS', B='VSS')),
    mn4=dict(type='M', params=dict(is_nmos=True, w=5), terminals=dict(D='net7', G='net7', S='VSS', B='VSS')),
    mn5=dict(type='M', params=dict(is_nmos=True, w=10), terminals=dict(D='net6', G='net7', S='VSS', B='VSS')),
    mp1=dict(type='M', params=dict(is_nmos=False, w=20), terminals=dict(D='net4', G='netb', S='VDD', B='VDD')),
    mp2=dict(type='M', params=dict(is_nmos=False, w=20), terminals=dict(D='net5', G='netb', S='VDD', B='VDD')),
    mp3=dict(type='M', params=dict(is_nmos=False, w=20), terminals=dict(D='net6', G='net5', S='VDD', B='VDD')),
    ibias=dict(type='I', params=dict(dc=10e-6, ac_mag=0, ac_ph=0), terminals=dict(PLUS='VDD', MINUS='net7')),
    vbias=dict(type='V', params=dict(dc=0.6, ac_mag=0, ac_ph=0), terminals=dict(PLUS='netb', MINUS='VDD')),
    vvdd=dict(type='V', params=dict(dc=1, ac_mag=0, ac_ph=0), terminals=dict(PLUS='VDD', MINUS='0')),
    vvss=dict(type='V', params=dict(dc=0, ac_mag=0, ac_ph=0), terminals=dict(PLUS='VSS', MINUS='0')),
    vin1=dict(type='V', params=dict(dc=0.6, ac_mag=1, ac_ph=0), terminals=dict(PLUS='net1', MINUS='0')),
    vin2=dict(type='V', params=dict(dc=0.6, ac_mag=1, ac_ph=180), terminals=dict(PLUS='net2', MINUS='0')),
    rz=dict(type='R', params=dict(value=1000), terminals=dict(PLUS='net8', MINUS='net6')),
    cc=dict(type='C', params=dict(value=10e-12), terminals=dict(PLUS='net5', MINUS='net8')),
    CL=dict(type='C', params=dict(value=10e-12), terminals=dict(PLUS='net6', MINUS='0')),
)

if __name__ == '__main__':
    foo = sampler.sample(space)
    print(foo)
    breakpoint()
