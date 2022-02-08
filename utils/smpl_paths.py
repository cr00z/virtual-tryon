import numpy as np
import pickle as pkl
import scipy.sparse as sp
from utils.geometry import get_hres


# set your paths here
smpl_path = './assets/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'


def get_hres_smpl_model_data():
    dd = pkl.load(open(smpl_path, 'rb'), encoding='latin1')
    hv, hf, mapping = get_hres(dd['v_template'], dd['f'])

    num_betas = dd['shapedirs'].shape[-1]
    J_reg = dd['J_regressor'].asformat('csr')

    model = {
        'v_template': hv,
        'weights': np.hstack([
            np.expand_dims(
                np.mean(
                    mapping.dot(
                        np.repeat(np.expand_dims(dd['weights'][:, i], -1),
                                  3)).reshape(-1, 3)
                    , axis=1),
                axis=-1)
            for i in range(24)
        ]),
        'posedirs': mapping.dot(dd['posedirs'].reshape((-1, 207))).reshape(-1,
                                                                           3,
                                                                           207),
        'shapedirs': mapping.dot(
            dd['shapedirs'].reshape((-1, num_betas))).reshape(-1, 3, num_betas),
        'J_regressor': sp.csr_matrix((J_reg.data, J_reg.indices, J_reg.indptr),
                                     shape=(24, hv.shape[0])),
        'kintree_table': dd['kintree_table'],
        'bs_type': dd['bs_type'],
        'bs_style': dd['bs_style'],
        'J': dd['J'],
        'f': hf,
    }

    return model