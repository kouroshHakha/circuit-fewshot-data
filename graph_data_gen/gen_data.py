""" Generate the training data based on randomly sampling the search space. 
Output path is setup by specifying NGSPICE_TMP_DIR.
"""
import argparse
from pathlib import Path
import importlib
import numpy as np
import hashlib
import tqdm

def _parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Input conf path of a file in format of conf.py')
    parser.add_argument('--seed', default=0, help='Random Seed')

    return parser.parse_args()

def get_conf_mod(fpath):
    loader = importlib.machinery.SourceFileLoader('conf', str(Path(fpath) / 'conf.py'))
    mod = loader.load_module()
    return mod

def get_design_params(conf, rseed):
    sampler = conf.sampler
    space = conf.space
    sim_model = conf.sim_model
    netlist_manager = conf.netlist_manager
    rng = np.random.RandomState(rseed)

    params = sampler.sample(space, rng)

    hashable_params = repr(tuple(sorted(params.items()))).encode('utf-8')
    params.update(id=hashlib.sha1(hashable_params).hexdigest()[:10])
    for k in ['ac', 'dc']:
            params[k] = str(netlist_manager.get_design_folder(params['id']) / f'{k}.csv')
    params.update(include=str(Path(sim_model).resolve()))
    
    return params

def main(pargs):
    conf = get_conf_mod(pargs.path)
    nsamples = conf.NSAMPLES
    seed = pargs.seed
    netlist_manager = conf.netlist_manager
    
    designs = []
    for i in tqdm.tqdm(range(nsamples)):
        designs.append(get_design_params(conf, i + seed))

    with netlist_manager as netlister:
        netlister.run(designs, verbose=True)


if __name__ == '__main__':

    pargs = _parse_args()
    main(pargs)
