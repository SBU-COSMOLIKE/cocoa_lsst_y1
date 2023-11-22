import os
import sys
from itertools import product
import numpy as np
from tqdm import trange, tqdm
import scipy 
from scipy.signal import savgol_filter
current_dir =  os.path.dirname(__file__)
sys.path.append(f'{current_dir}')
import pce_
 


  




class emu_cons_proto(object):
 
    r"""
    PCE for (log)-Bcase_smeared
    Attributes:
        parameters (list):
            model parameters, sorted in the desired order
        modes (numpy.ndarray):
            multipoles or k-values in the (log)-spectra
        n_pcas (int):
            number of PCA components
        parameters_filenames (list [str]):
            list of .npz filenames for parameters
        features_filenames (list [str]):
            list of .npz filenames for (log)-spectra
        verbose (bool):
            whether to print messages at intermediate steps or not
    """
        
 
   
    
    def __init__(self):
        r"""
        Constructor
        """ 
        # attributes
        current_dir =  os.path.dirname(__file__) 
        self.ks_cola = np.loadtxt(f'{current_dir}/k_cola_high.txt')
        
        self.redshift_default_ = ([0.000, 0.020, 0.041, 0.062,
           0.085, 0.109, 0.133, 0.159,
           0.186, 0.214, 0.244, 0.275, 
           0.308, 0.342, 0.378, 0.417,
           0.457, 0.500, 0.543, 0.588, 
           0.636, 0.688, 0.742, 0.800, 
           0.862, 0.929, 1.000, 1.087,
           1.182, 1.286, 1.400, 1.526,
           1.667, 1.824, 2.000, 2.158,
           2.333, 2.529, 2.750, 3.000])

        
        self.z_interp_2D = np.linspace(0,2.0,95)
        self.z_interp_2D = np.concatenate((self.z_interp_2D, np.linspace(3.0,10,5)),axis=0)
        self.z_interp_2D[0] = 0       
        self.z_interp_2D=self.z_interp_2D[:96]
        
        self.lhs_ =np.loadtxt(f'{current_dir}/lhs_cola_lcdm.txt')  
        self.lhs_max= self.lhs_.max(axis=0)   
        self.lhs_min= self.lhs_.min(axis=0)   
        
        self.max_=1
        self.min_= -1 
        self.scale_ =  (self.max_ - self.min_) / (self.lhs_max - self.lhs_min)      
        
        self.values = self.lhs_[0]
        self.d = self.lhs_.shape[1]
        self.order_max =7
        self.numbers = np.arange(self.order_max + 1)
        self.combinations = np.array(list(product(self.numbers, repeat=self.d)), dtype=np.float64)
        self.max_values = np.array([7,7,5,5,5])
        self.combinations= np.array([row for row in self.combinations if all(row[i] <= self.max_values[i] for i in range(self.d))])
        self.filtered_combs = pce_.filter_q0_norm(self.combinations, threshold=4) #5         
        self.norm = self.filtered_combs/self.max_values          
        self.allowcomb= pce_.filter_combinations( self.norm,  p=1 + 1e-12,  threshold=10 + 1e-12  )*self.max_values         


     
        self.A_34=[]
        self.b_34 = [] 
        self.Wtrans_34=[]
        for Init in tqdm(range(35)):
            Init =Init+0

            redshift=format(self.redshift_default_[Init], '.3f')

           # print('redshift=',redshift)

            self.A_34.append(np.loadtxt(f'{current_dir}/LCDM_PNN_HIGH/PCE_LCDM_z'+str(redshift)+'_A.txt') )
            self.b_34.append(np.loadtxt(f'{current_dir}/LCDM_PNN_HIGH/PCE_LCDM_z'+str(redshift)+'_b.txt') )
            self.Wtrans_34.append(np.loadtxt(f'{current_dir}/LCDM_PNN_HIGH/PCE_LCDM_z'+str(redshift)+'_WTRANS.txt') )
 

        self.A_34=np.array(self.A_34)
        self.b_34=np.array(self.b_34)
        self.Wtrans_34=np.array(self.Wtrans_34)

        self.A_6=[]
        self.b_6 = [] 
        self.Wtrans_6=[]
        for Init in tqdm(range(5)):
            Init =Init+35

            redshift=format(self.redshift_default_[Init], '.3f')

           # print('redshift=',redshift)

            self.A_6.append(np.loadtxt(f'{current_dir}/LCDM_PNN_HIGH/PCE_LCDM_z'+str(redshift)+'_A.txt') )
            self.b_6.append(np.loadtxt(f'{current_dir}/LCDM_PNN_HIGH/PCE_LCDM_z'+str(redshift)+'_b.txt') )
            self.Wtrans_6.append(np.loadtxt(f'{current_dir}/LCDM_PNN_HIGH/PCE_LCDM_z'+str(redshift)+'_WTRANS.txt') )


        self.A_6=np.array(self.A_6)
        self.b_6=np.array(self.b_6)
        self.Wtrans_6=np.array(self.Wtrans_6)  
        
    def scaler_trans(self, params_):
        X_std = (params_ - self.lhs_min) / ( self.lhs_max - self.lhs_min)


        return np.clip(np.array(X_std * (self.max_ - self.min_) + self.min_) , -1, 1)

    def scaler_invtrans(self, scaled_params_):
 
        min_par = self.min_ - self.lhs_.min(axis=0) *  self.scale_
        return   (scaled_params_ - min_par)/self.scale_
 

    def dict_to_ordered_arr_np(self, input_dict):

        return np.stack([input_dict[k] for k in input_dict]).T   
    
    def get_boost_cocoa(self, cosmo_dict, log10k_interp_2D ):
        pars= self.scaler_trans(self.dict_to_ordered_arr_np(cosmo_dict))                 
        pce_expansion=pce_.calculate_power_expansion( self.allowcomb, pars)
        cosmos_34 = pce_.predict_one_cosmo( pce_expansion,  self.Wtrans_34,   self.A_34,   self.b_34)[:,0,:]
        cosmos_6 = pce_.predict_one_cosmo( pce_expansion,  self.Wtrans_6,   self.A_6,   self.b_6)[:,0,:]

        
        logkbt34,logkbt6 =  np.log10(self.ks_cola),np.log10(self.ks_cola[:256])
        
        num_of_points= 10  #25
        num_of_points2= 10  #25
            
        interp = scipy.interpolate.interp1d(logkbt6, 
          np.exp(np.concatenate((cosmos_6[:,:256-num_of_points], savgol_filter(cosmos_6, num_of_points2, 1)[:,256-num_of_points: ] ), axis=1)), 
          kind = 'linear', 
          fill_value = 'extrapolate', 
          assume_sorted = True)
          
        log_tmp_bt6 = np.log(interp(log10k_interp_2D))
        
        num_of_points= 25#30ou 60ou 50 ou 70 
        num_of_points2= 25  #30 ou60 ou 50 ou 70
        interp = scipy.interpolate.interp1d(logkbt34, 
          np.exp(np.concatenate((cosmos_34[:,:512-num_of_points], savgol_filter(cosmos_34, num_of_points2, 1)[:,512-num_of_points: ] ), axis=1)), 
          kind = 'linear', 
          fill_value = 'extrapolate', 
          assume_sorted = True)
          
        log_tmp_bt34 = np.log(interp(log10k_interp_2D))
        log_tmp_bt34[:,log10k_interp_2D < np.log10(self.ks_cola[0])] = 0  
        log_tmp_bt6[:,log10k_interp_2D < np.log10(self.ks_cola[0])] = 0  






        
        return  self.ks_cola, log_tmp_bt34, log_tmp_bt6






















