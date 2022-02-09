# Dress SMPL mesh into body
# based on Multi-Garment-Network project code
# https://github.com/cr00z/virtual_tryon


import numpy as np
import pickle as pkl
from psbody.mesh import Mesh, MeshViewers
from utils.smpl_paths import get_hres_smpl_model_data
from lib.ch_smpl import Smpl
from os.path import join


def get_garment_mesh(
        garment_path, garment_type, betas=np.zeros(10), pose=np.zeros(72)):
    """
    Get garment mesh with SMPL parameters
    :param garment_path: path to garment dir in Multi-Garment dataset
    :param garment_type: Pants, TShirtNoCoat, etc...
    :param betas: SMPL beta
    :param pose: SMPL pose (pheta)
    :return:
    """
    # smpl model in t-pose with beta
    smpl_tgt = Smpl(get_hres_smpl_model_data())
    #print(smpl_tgt.betas)
    #print(betas.device)
    smpl_tgt.betas[:] = betas.cpu()

    # smpl model in t-pose with garment beta
    dat = pkl.load(
        open(join(garment_path, 'registration.pkl'), 'rb'), encoding='latin1'
    )
    smpl_src = Smpl(get_hres_smpl_model_data())
    smpl_src.betas[:] = dat['betas']
    body_src = Mesh(smpl_src.v, smpl_src.f)

    # garment in t-pose with beta
    garment = Mesh(filename=join(garment_path, garment_type + '.obj'))

    # this file contains correspondances between garment vertices and smpl body
    fts_file = 'assets/garment_fts.pkl'
    vert_indices, _ = pkl.load(open(fts_file, 'rb'), encoding='latin1')
    vert_inds = vert_indices[garment_type]

    # retarget
    verts, _ = body_src.closest_vertices(garment.v)
    verts = np.array(verts)
    tgt_garment = garment.v - body_src.v[verts] + smpl_tgt.r[verts]

    # repose
    offsets = np.zeros_like(smpl_tgt.r)
    offsets[vert_inds] = tgt_garment - smpl_tgt.r[vert_inds]
    smpl_tgt.v_personal[:] = offsets
    smpl_tgt.pose[:] = pose.cpu()
    garment_ret_posed = Mesh(smpl_tgt.r, smpl_tgt.f).keep_vertices(vert_inds)

    # texture
    garment_ret_posed.vt, garment_ret_posed.ft = garment.vt, garment.ft
    garment_texture = join(garment_path, 'multi_tex.jpg')
    garment_ret_posed.set_texture_image(garment_texture)
    # smpl_tgt_mesh = Mesh(smpl_tgt.r, smpl_tgt.f)
    return garment_ret_posed


if __name__ == '__main__':
    betas = (np.random.rand(10) - 0.5) * 2.5
    pose = np.random.rand(72) - 0.5
    garment_ret_posed = get_garment_mesh(11, 11, betas, pose)
    mvs = MeshViewers((1, 1))
    mvs[0][0].set_static_meshes([garment_ret_posed])
