from joblib import dump, load
from numpy import log
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from .config import *
from .utils import format_z
from .dataset import ColaDataset


class ColaPreprocessor:
    def __init__(self, dataset: ColaDataset, n_components=NUM_PCS, update=False):
        self.dataset = dataset
        self.num_pcs = n_components
        self.update = update

    def normalize_params(self, params, z, update=None):
        """
        Prepare normalized cosmological parameters using MinMaxScaler.

        Returns:
            DataFrame: Normalized parameters with named columns.
        """
        if update is None:
            update = self.update
        logging.info("Normalizing cosmological parameters")
        norm_params = self._transform(
            MinMaxScaler,
            ColaPreprocessor._params_scaler_path(z),
            params,
            update=update
        )
        return DataFrame(norm_params, columns=PARAM_COLS)

    def pca_boosts(self, boosts, z, num_pcs):
        """
        Prepare Principal Components of normalized logboosts.

        Returns:
            DataFrame: PCS with named columns.
        """
        logboosts = log(boosts)
        
        logging.info(f"Preprocessing log-boosts with shape {logboosts.shape}")
        
        norm_logboosts = self._transform(MinMaxScaler, 
                                         ColaPreprocessor._boost_scaler_path(z), 
                                         logboosts)
        
        pca_components = self._transform(PCA, 
                                         ColaPreprocessor._pca_path(z), 
                                         norm_logboosts,
                                         n_components=self.num_pcs)
        
        y = DataFrame(pca_components, columns = [f"PC {i}" for i in range(self.num_pcs)])
        y.index = [f"Log Boost {i}" for i in range(len(y))]
        
        logging.info("PCA transformation complete")
        return y
    
    def prepare_data(self):   
        """
        Pre-process cosmological parameters and boosts.

        Returns:
            DataFrame: Normalized parameters with named columns.
            DataFrame: Boost PCs with named columns.
        """
        return self.normalize_params(self.dataset.params, self.dataset.z), self.pca_boosts(self.dataset.boosts, self.dataset.z, self.num_pcs)

    def _transform(self, model_class, path, data, update=None, **kwargs):
        """Load prepared PCA/MinMaxScaler from file or update existing"""
        if update is None:
            update = self.update
        if update:
            logging.info(f"Computing and saving new {model_class} for z={self.dataset.z}")
            model = model_class(**kwargs)
            model.fit(data)
            dump(model, path)
        else:
            logging.info(f"Loading existing {model_class} for z={self.dataset.z}")
            model = load(path)
        return model.transform(data)

    @staticmethod
    def _boost_scaler_path(z): return f"{BOOST_DIR}/scaler_v{VER}_z{format_z(z)}.pkl"
    @staticmethod
    def _pca_path(z): return f"{BOOST_DIR}/pca_v{VER}_z{format_z(z)}.pkl"
    @staticmethod
    def _params_scaler_path(z): return f"{PARAM_DIR}/scaler_v{VER}_z{format_z(z)}.pkl"
