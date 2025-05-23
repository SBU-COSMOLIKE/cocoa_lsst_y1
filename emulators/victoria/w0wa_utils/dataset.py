import numpy as np
import logging
from pandas import DataFrame, Series
from os.path import join
from .config import base_path, PARAM_COLS
from .utils import load_cola, format_z, lcdm_projection

def boosts_equal(a, b):
    if isinstance(a, (DataFrame, Series)) and isinstance(b, (DataFrame, Series)):
        return a.equals(b)
    else:
        return np.array_equal(np.asarray(a), np.asarray(b))


class ColaDataset:
    def __init__(self, input_type, z=0.0, use_ref=False, verbose=False):
        self.input_type = input_type
        self.z = format_z(z)
        self.use_ref = use_ref
        self.verbose = verbose
        
        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        logging.info(f"Initializing ColaData with input_type={input_type}, z={self.z}")
        
        
        self.params, self.ks, self.pk_lin, self.pk_nl = self.load()
        
        if self.use_ref: self._split_reference()

    def load(self):
        '''Load data from output files'''
        path = join(base_path, self.input_type)
        return load_cola(path, self.z, use_ref=self.use_ref)

    def _split_reference(self):
        '''If reference cosmology is included in files, separate out'''
        self.pk_lin_ref = self.pk_lin[-1]
        self.pk_nl_ref = self.pk_nl[-1]
        self.pk_lin = self.pk_lin[:-1, :]
        self.pk_nl = self.pk_nl[:-1, :]
        self.boosts_ref = self.pk_nl_ref/self.pk_lin_ref
    
    @property
    def boosts(self):
        '''Calculate boosts from P(k) and P(k)lin'''
        return self.pk_nl / self.pk_lin

#     @property
#     def logboosts(self):
#         '''Return natural log of boosts'''
#         return np.log(self.boosts)

    def boosts_to_df(self, boosts = None):
        '''Return clean pandas dataframe with boosts'''
        boosts = self.boosts if boosts is None else boosts
        boosts_df = DataFrame(boosts).T
        boosts_df.index = self.ks
        boosts_df.index.name = "k"
        boosts_df.columns = [f"Boost {i}" for i in range(len(boosts))]
        # Add "Boost Ref" if use_ref is True
        if self.use_ref and boosts_equal(boosts, self.boosts): boosts_df["Boost Ref"] = self.boosts_ref

        return boosts_df
    
    def params_to_df(self, params = None):
        '''Return clean pandas dataframe with cosmo params'''
        params = params or self.params
        return DataFrame(params, columns=PARAM_COLS)

