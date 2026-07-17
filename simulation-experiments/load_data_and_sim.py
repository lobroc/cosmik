import MRzeroCore as mr0
import numpy as np
import torch

from sys import path
from pathlib import Path
from types import MethodType
from argparse import ArgumentParser

parser = ArgumentParser('Load disk phantom and simulate')
parser.add_argument('-c', "--change-t2s", action='store_true', help='Whether to modify phantom T2* or not. Defaults to false.')
args = parser.parse_args()

CHANGE_T2S = args.change_t2s

path.insert(0, str(Path.cwd().resolve().parent / 'simulations-mr0'))
from src.time_curve import TimeCurve
from create_mr0_phantom import MovingVoxelGridPhantom

ph_rawdata = np.load('ph_for_mrzero.mrzphantom.npz')

if CHANGE_T2S:
    print('Changing T2* values')
    ph_rawdata = dict(ph_rawdata)
    ph_rawdata['T2dash_map'] = ph_rawdata['T2_map'] / 50
else:
    print('NOT changing T2* values')

obj = mr0.CustomVoxelPhantom(
    pos=(np.stack((ph_rawdata['x'], ph_rawdata['y'], ph_rawdata['z']), axis=1)),
    PD=ph_rawdata['PD_map'],
    T1=ph_rawdata['T1_map'],
    T2=ph_rawdata['T2_map'],
    T2dash=ph_rawdata['T2dash_map'],
    D=np.zeros_like(ph_rawdata['PD_map']),
    B0=np.zeros_like(ph_rawdata['PD_map']),
    B1=np.ones_like(ph_rawdata['PD_map'])[None, ...],
    voxel_size=torch.tensor([4e-2/50]*3, device='cuda')
)

obj_simdata = obj.build()

# seq = mr0.Sequence.import_file("/home/brock/biospig4-drive/simulator-data/sequences/aztek_radial_v4_compat_120_180_180_px_10s_runtime_2s_dummy_part_spoke.seq")
seq = mr0.Sequence.import_file('radial_2d.seq', backend='pydisseqt')

graph = mr0.compute_graph(seq, obj_simdata, 5_000, 1e-4)

with torch.no_grad():
    signal_no_motion = mr0.execute_graph(graph, seq.cuda(), obj_simdata.cuda(), min_emitted_signal=1e-4, min_latent_signal=1e-4, print_progress=True)

np.save('sig_mr0_no_motion.npy' if not CHANGE_T2S else 'sig_mod_mr0_no_motion.npy', signal_no_motion.cpu().numpy())

obj = mr0.CustomVoxelPhantom(
    pos=(np.stack((ph_rawdata['x'], ph_rawdata['y'], ph_rawdata['z']), axis=1) / 1000), # Not sure why, but need to downscale spatial dims with customvoxelphantom when using motion.
    PD=ph_rawdata['PD_map'],
    T1=ph_rawdata['T1_map'],
    T2=ph_rawdata['T2_map'],
    T2dash=ph_rawdata['T2dash_map'],
    D=np.zeros_like(ph_rawdata['PD_map']),
    B0=np.zeros_like(ph_rawdata['PD_map']),
    B1=np.ones_like(ph_rawdata['PD_map'])[None, ...],
    voxel_size=torch.tensor([4e-2/50]*3, device='cuda')
)

obj_simdata = obj.build()

obj_simdata.get_voxel_motion = MethodType(MovingVoxelGridPhantom.get_voxel_motion, obj_simdata) # Bind to this object
obj_simdata.activate_motion = MethodType(MovingVoxelGridPhantom.activate_motion, obj_simdata)
obj_simdata.deactivate_motion = MethodType(MovingVoxelGridPhantom.deactivate_motion, obj_simdata)
obj_simdata.set_deformations = MethodType(MovingVoxelGridPhantom.set_deformations, obj_simdata)
obj_simdata.cubic_coefs = None
obj_simdata.cubic_motion = True
obj_time_curve = TimeCurve(t=np.stack([ph_rawdata['time_scale_min'], ph_rawdata['time_scale_max']]), t_unit=[0, 1], periodic=True)
obj_simdata.set_deformations(ph_rawdata['deformations'], obj_time_curve)
obj_simdata.cubic_coefs = tuple(x.cuda() for x in obj_simdata.cubic_coefs)
obj_simdata.time_curve = obj_simdata.time_curve.cuda()
graph = mr0.compute_graph(seq, obj_simdata, 5_000, 1e-4)

with torch.no_grad():
    signal = mr0.execute_graph(graph, seq.cuda(), obj_simdata.cuda(), min_emitted_signal=1e-4, min_latent_signal=1e-4, print_progress=True)

np.save('sig_mr0.npy' if not CHANGE_T2S else 'sig_mod_mr0.npy', signal.cpu().numpy())
