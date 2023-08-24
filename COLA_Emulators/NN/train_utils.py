# Auxiliary functions for Power Spectrum Emulation
# Author: João Victor Silva Rebouças, May 2022
# import camb
# from camb import model
import numpy as np
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy.fftpack import dst, idst
from scipy.integrate import simps
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers
from keras.regularizers import l1_l2
from keras.callbacks import Callback
#------------------------------------------------------------------------------------------------------------
# Parameter space
params = ['h', 'Omegab', 'Omegam', 'As', 'ns', 'w']

# Parameter limits for training
lims = {}
lims['h'] = [0.61, 0.73]
lims['Omegab'] = [0.04, 0.06]
lims['Omegam'] = [0.24, 0.4]
lims['As'] = [1.7e-9, 2.5e-9]
lims['ns'] = [0.92, 1]
lims['w'] = [-1.3, -0.7]

# Reference values
ref = {}
ref['h'] = 0.67
ref['Omegab'] = 0.049
ref['Omegam'] = 0.319
ref['As'] = 2.1e-9
ref['ns'] = 0.96
ref['w'] = -1

# Define k, z bins
redshifts = np.array([9.182, 7.690, 6.579, 5.720, 5.036, 4.478, 4.015, 3.624, 3.289, 3.000, 2.750, 2.529, 2.333, 2.158, 2.000, 1.824, 1.667, 1.526, 1.400, 1.286, 1.182, 1.087, 1.000, 0.929, 0.862, 0.800, 0.742, 0.688, 0.636, 0.588, 0.543, 0.500, 0.457, 0.417, 0.378, 0.342, 0.308, 0.275, 0.244, 0.214, 0.186, 0.159, 0.133, 0.109, 0.085, 0.062, 0.041, 0.020, 0.000])
redshifts = np.flip(redshifts)
path_to_train = "/home/grads/data/jonathan/cola_projects/COLA_output/LCDM/400_1"
ks_cola_default = np.loadtxt(f"{path_to_train}/a/output/ref/pofk_ref_total_z0.000.txt", unpack=True, usecols=(0))
ks_cola_default = ks_cola_default[:512] # Need to restrain to k = pi
#------------------------------------------------------------------------------------------------------------
def is_cosmo_inside_ee2_box(params):
    """
    Returns `True` if the cosmology is inside the EE2 box.
    """
    Omega_m = params[0]
    Omega_b = params[1]
    ns = params[2]
    As = params[3]
    h = params[4]
    result = (
        Omega_m > lims['Omegam'][0] and Omega_m < lims['Omegam'][1] and
        Omega_b > lims['Omegab'][0] and Omega_b < lims['Omegab'][1] and
        ns > lims['ns'][0] and ns < lims['ns'][1] and
        As > lims['As'][0] and As < lims['As'][1] and
        h > lims['h'][0] and h < lims['h'][1]
    )
    if len(params) == 6:
        w = params[5]
        result = result and (w > lims['w'][0] and w < lims['w'][1])
    return result
    

def get_pk(h, Omegab, Omegam, As10to9, ns, w, redshifts = np.linspace(3, 0, 101), tau = 0.078):
    '''
    Returns [k, Pk_lin, Pk_nonlin], the scales and power spectrum for the given cosmology.
    k: array of scales in h/Mpc.
    Pk_lin: array of shape (len(redshifts), len(k)) of linear matter power spectra.
    Pk_nonlin: array of shape (len(redshifts), len(k)) of nonlinear matter power spectra.
    In the Pk arrays, latest redshift comes first.
    Set to 250 k-bins from k = 1e-4 to k = 10 (h/Mpc).
    '''
    Omegac = Omegam - Omegab
    As = As10to9 * 10**-9
    cosmology = camb.set_params(
        # Background
        H0 = 100*h,
        ombh2=Omegab*h**2,
        omch2=Omegac*h**2,
        TCMB = 2.7255,
        # Dark Energy
        dark_energy_model='fluid', w = w,
        # Neutrinos
        nnu=3.046,
        mnu = 0.058,
        # Initial Power Spectrum
        As = As,
        ns = ns,
        tau = tau,
        YHe = 0.246,
        WantTransfer=True
    )
    cosmology.set_matter_power(redshifts=redshifts, kmax=100.0)
    cosmology.NonLinear = model.NonLinear_none
    results = camb.get_results(cosmology)
    
    # Calculating Linear Pk
    ks, _, pk_lin = results.get_matter_power_spectrum(minkh=1e-3, maxkh=100, npoints = 1200)
    
    # Recalculating Pk with Nonlinear
    cosmology.NonLinear = model.NonLinear_both
    cosmology.NonLinearModel.set_params(halofit_version='takahashi')
    results.calc_power_spectra(cosmology)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-3, maxkh=100, npoints = 1200)   
    return ks, pk_lin, pk_nonlin
#------------------------------------------------------------------------------------------------------------
def normalize_array(array):
    '''
    Returns the Min-Max normalized array: (array - min(array))/(max(array) - min(array))
    '''
    return (array - np.amin(array))/(np.amax(array) - np.amin(array))

#------------------------------------------------------------------------------------------------------------
def unnormalize_array(norm_array, original_array):
    '''
    Returns `norm_array` scaled by `original_array`: norm_array * (max(array) - min(array)) + min(array)
    '''
    return norm_array * (np.amax(original_array) - np.amin(original_array)) + np.amin(original_array)
#------------------------------------------------------------------------------------------------------------
def normalize_params(params):
    '''
    Takes a params = (h, Omegam, Omegab, As10to9, ns, wde) array and returns normalized parameters.
    Normalization is given by normalized_param = (param - param_min)(param_max - param_min).
    '''
    if len(params) == 6:
        Omegam, Omegab, ns, As10to9, h, wde = params
        normalized_params = [
            (Omegam-lims['Omegam'][0])/(lims['Omegam'][1] - lims['Omegam'][0]),
            (Omegab-lims['Omegab'][0])/(lims['Omegab'][1] - lims['Omegab'][0]),
            (ns-lims['ns'][0])/(lims['ns'][1] - lims['ns'][0]),
            (As10to9-lims['As'][0])/(lims['As'][1] - lims['As'][0]),
            (h-lims['h'][0])/(lims['h'][1] - lims['h'][0]),
            (wde-lims['w'][0])/(lims['w'][1] - lims['w'][0])
        ]
    if len(params) == 5:
        Omegam, Omegab, ns, As10to9, h = params
        normalized_params = [
            (Omegam-lims['Omegam'][0])/(lims['Omegam'][1] - lims['Omegam'][0]),
            (Omegab-lims['Omegab'][0])/(lims['Omegab'][1] - lims['Omegab'][0]),
            (ns-lims['ns'][0])/(lims['ns'][1] - lims['ns'][0]),
            (As10to9-lims['As'][0])/(lims['As'][1] - lims['As'][0]), 
            (h-lims['h'][0])/(lims['h'][1] - lims['h'][0])
        ]

    normalized_params = np.array(normalized_params)
    return normalized_params
#------------------------------------------------------------------------------------------------------------
def pca_reduction(num_pcs, norm_log_boosts, redshifts=redshifts):
    '''
    Fits a PCA under all redshifts and reduces the log boosts to their principal components
    Returns `pcas`, the set of PCAs for each redshift, and `reduced_norm_log_boosts`, the PCs of our input data.
    Also works on qs since it's the same process!
    '''
    pcas = []
    reduced_norm_log_boosts = np.zeros((len(norm_log_boosts), len(redshifts), num_pcs))
    for i in range(len(redshifts)):
        pca = PCA(n_components=num_pcs)
        reduced_norm_log_boosts_thisz = pca.fit_transform(norm_log_boosts[:,i,:])
        pcas.append(pca)
        reduced_norm_log_boosts[:,i,:] = reduced_norm_log_boosts_thisz
    return pcas, reduced_norm_log_boosts   
#------------------------------------------------------------------------------------------------------------
def pca_reconstruction_analysis(num_pcs, norm_qs, qs, sample, save=False, imgName='pc_analysis.pdf'):
    '''
    Plots the reconstruction errors for a given number of PCs
    The boosts can be replaced by Qs
    '''
    pcas = []
    reduced_norm_qs = np.zeros((len(sample), len(redshifts), num_pcs))
    reconstructed_boosts = np.zeros((len(sample), len(redshifts), len(ks)))
    for i in range(len(redshifts)):
        pca = PCA(n_components=num_pcs)
        reduced_norm_qs_thisz = pca.fit_transform(norm_qs[:,i,:])
        reconstructed_norm_qs_thisz = pca.inverse_transform(reduced_norm_qs_thisz)
        pcas.append(pca)
        reduced_norm_qs[:,i,:] = reduced_norm_qs_thisz
        reconstructed_qs_thisz = unnormalize_array(reconstructed_norm_qs_thisz, qs)
        reconstructed_boost_thisz = np.exp(reconstructed_qs_thisz)
        reconstructed_boosts[:,i,:] = reconstructed_boost_thisz
    # Plot reconstructed data
    for i, point in enumerate(sample):
        for j, redshift in enumerate(redshifts):
            plt.semilogx(ks, reconstructed_boosts[i,j,:]/np.exp(qs)[i,j,:] - 1, c='gray',alpha=0.5)
    plt.fill_between(ks, -0.01,0.01, color='gray',alpha=0.2)
    plt.grid()
    plt.xlabel(r'$k$ (h/Mpc)')
    which_quantity = input('Which is the emulated quantity (B or Q)? ')
    if which_quantity=='B':
        plt.ylabel(r'$B^{PCA}/B^{original} - 1$')
    if which_quantity=='Q':
        plt.ylabel(r'$B^{PCA}/B^{original} - 1$')
    title = input('Enter a title for this plot: ')
    plt.title(title)
    if save:
        plt.savefig(imgName)
    plt.show()
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
def smooth_bao(ks, pk):
    
 
    
    n = 15
    dst_ks = np.linspace(1e-4, 5, 2**n) #10
    logks = np.log(dst_ks)
    
    
    spline_loglog_pk_2 = interpolate.interp1d(np.log(ks), np.log(pk), kind='linear', fill_value='extrapolate')
    spline_loglog_pk2 = spline_loglog_pk_2(np.log(np.linspace(1e-4, 5, 2**n)))
    
    spline_loglog_pk = interpolate.splrep(np.log(np.linspace(1e-4,5, 2**n)), spline_loglog_pk2, s=0)    
        
    logkpk = np.log10(dst_ks * np.exp(interpolate.splev(np.log(dst_ks), spline_loglog_pk, der=0, ext=0)))
    sine_transf_logkpk = dst(logkpk, type=2)# dst(logkpk, type=2, norm='ortho')
    odds = [] # odd entries
    evens = [] # even entries
    even_is = [] # odd indices
    odd_is = [] # even indices
    all_is = [] # all indices
    for i, entry in enumerate(sine_transf_logkpk):
        all_is.append(i)
        if i%2 == 0:
            even_is.append(i)
            evens.append(entry)
        else:
            odd_is.append(i)
            odds.append(entry)
    odd_is=np.array(odd_is)
    even_is=np.array(even_is)
    odds=np.array(odds)
    evens=np.array(evens)
 

    odd_is = np.array(odd_is)
    even_is = np.array(even_is)
    
    
    odds_interp = interpolate.splrep(odd_is, odds, s=0) 
    evens_interp = interpolate.splrep(even_is, evens, s=0) 
    

    
    
    d2_odds =interpolate.splev(odd_is, odds_interp, der=2, ext=0)    
    d2_evens =interpolate.splev(even_is, evens_interp, der=2, ext=0)
    
    
    
    d2_odds_1 =interpolate.splev(odd_is +2, odds_interp, der=2, ext=0) 
    d2_evens_1 =interpolate.splev(even_is +2, evens_interp, der=2, ext=0)
   


    d2_odds_2 =interpolate.splev(odd_is - 2, odds_interp, der=2, ext=0)
    d2_evens_2 =interpolate.splev(even_is - 2 , evens_interp, der=2, ext=0)
    
   
    
    
    
    d2_odds_avg = (d2_odds + d2_odds_2 + d2_odds_1)/3
  

    d2_evens_avg = (d2_evens + d2_evens_2 + d2_evens_1)/3 

    
    
    imin_even = 50+np.argmax(d2_evens_avg[50:150]) -9
    
    imax_even = 50+np.argmin(d2_evens_avg[50:150])+36

    imin_odd = 50+np.argmax(d2_odds_avg[50:150])-9

    imax_odd = 50+np.argmin(d2_odds_avg[50:150])+37

    
    
        
    
    even_is_removed_bumps = np.concatenate((even_is[:imin_even], even_is[imax_even:]))
    odd_is_removed_bumps = np.concatenate((odd_is[:imin_odd], odd_is[imax_odd:]))

    evens_removed_bumps = np.concatenate((evens[:imin_even], evens[imax_even:]))
    odds_removed_bumps = np.concatenate((odds[:imin_odd], odds[imax_odd:]))

    even_holed_cs = interpolate.splrep(even_is_removed_bumps, evens_removed_bumps * (even_is_removed_bumps+1)**2, s=0)
    odd_holed_cs = interpolate.splrep(odd_is_removed_bumps, odds_removed_bumps * (odd_is_removed_bumps+1)**2, s=0)
  
    
    evens_treated = interpolate.splev(even_is, even_holed_cs, der=0, ext=0) / (even_is + 1)**2
    odds_treated = interpolate.splev(odd_is, odd_holed_cs, der=0, ext=0) / (odd_is + 1)**2
    treated_transform = []
    for odd, even in zip(odds_treated, evens_treated):
        treated_transform.append(even)
        treated_transform.append(odd)
    treated_transform=np.array(treated_transform)    
    treated_logkpk =idst(treated_transform, type=2)/ (2 * len(treated_transform)) # idst(treated_transform, type=2, norm='ortho')
    pk_nw = 10**(treated_logkpk)/dst_ks
    
    k_highk = ks[ks > 4]
    p_highk = pk[ks > 4]

    k_extended = np.concatenate((dst_ks[dst_ks < 4], k_highk))
    
    p_extended = np.concatenate((pk_nw[dst_ks < 4], p_highk))
    

    pksmooth_cs = interpolate.splrep(np.log(k_extended), np.log(p_extended), s=0)
    pksmooth_interp = np.exp(interpolate.splev(np.log(ks), pksmooth_cs, der=0, ext=0))

    
    
        
   
    return pksmooth_interp#, d2_odds_avg, d2_evens_avg, odd_is,even_is, imin_odd,imax_odd, imin_even, imax_even, np.exp(interpolate.splev(np.log(dst_ks),spline_loglog_pk, der=0, ext=0)), dst_ks,np.argmax(d2_evens_avg[100:300]),np.argmin(d2_evens_avg[100:300]),np.argmax(d2_odds_avg[100:300]),np.argmin(d2_odds_avg[100:300]),spline_loglog_pk2
def smear_bao(ks, pk, pk_nw, par=0.5):
    from scipy.integrate import trapz

     
       
    integral = simps(pk,ks)#trapz(ks * pk, x=np.log(ks))  #simps(pk,ks)#
    k_star_inv = (1.0/(3.0 * np.pi**2)) * integral
    Gk = np.array([np.exp(-par*k_star_inv * (k_**2)) for k_ in ks])
    pk_smeared = pk*Gk + pk_nw*(1.0 - Gk)
    return pk_smeared
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
class CustomActivationLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomActivationLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.beta = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True, name="beta")
        self.gamma = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True, name="gamma")
        super(CustomActivationLayer, self).build(input_shape)

    def call(self, x):
        # See e.g. https://arxiv.org/pdf/1911.11778.pdf, Equation (8)
        func = tf.add(self.gamma, tf.multiply(tf.sigmoid(tf.multiply(self.beta, x)), tf.subtract(1.0, self.gamma)))
        return tf.multiply(func, x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

def save_model(model, file_path):
    model.save(file_path)
    
def generate_mlp(input_shape, output_shape, num_layers, num_neurons, activation="custom", alpha=0.01, l1_ratio=0.01, learning_rate=1e-3, optimizer='adam', loss='mse'):
    '''
    Generates an MLP model with `num_res_blocks` residual blocks.
    '''
    reg = l1_l2(l1=alpha*l1_ratio, l2=alpha*(1-l1_ratio)) if alpha != 0 else None
    
    # Define the input layer
    inputs = layers.Input(shape=(input_shape,))
    
    # Define the first hidden layer separately because it needs to connect with the input layer
    x = layers.Dense(num_neurons, kernel_regularizer=reg)(inputs)
    if activation == "custom":
        x = CustomActivationLayer(num_neurons)(x)
    elif activation == "relu":
        x = keras.activations.relu(x)
    else:
        raise Exception(f"Unexpected activation {activation}")
    
    # Add more hidden layers
    for _ in range(num_layers - 1): # subtract 1 because we've already added the first hidden layer
        x = layers.Dense(num_neurons, kernel_regularizer=reg)(x)
        if activation == "custom":
            x = CustomActivationLayer(num_neurons)(x)
        elif activation == "relu":
            x = keras.activations.relu(x)
        else:
            raise Exception(f"Unexpected activation {activation}")

    # Define the output layer
    outputs = layers.Dense(output_shape)(x)
    
    # Choose the optimizer
    if optimizer.lower() == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.99, nesterov=True)
    else:
        raise ValueError(f"Unhandled optimizer: {optimizer}")

    # Construct and compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.summary()
    # Compile the model
    model.compile(optimizer=opt, loss=loss)   # or any other suitable loss function

    return model
#------------------------------------------------------------------------------------------------------------
def generate_resnet(input_shape, output_shape, num_res_blocks=1, num_of_neurons=512, activation="relu", alpha=1e-5, l1_ratio=0.1, dropout=0.1):
    '''
    Generates a ResNet model with `num_res_blocks` residual blocks.
    '''
    nn_layers = []
    regularization_term = l1_l2(l1=alpha*l1_ratio, l2=alpha*(1-l1_ratio))
    
    # Adding layers
    input_layer = layers.Input(shape=input_shape)
    
    # Adding first residual block
    hid1 = layers.Dense(units=num_of_neurons,
         kernel_regularizer=regularization_term,
         bias_regularizer=regularization_term)(input_layer)
    act1 = CustomActivationLayer(num_of_neurons)(hid1)
    
    hid2 = layers.Dense(units=num_of_neurons,
         kernel_regularizer=regularization_term,
         bias_regularizer=regularization_term)(act1)
    act2 = CustomActivationLayer(num_of_neurons)(hid2)
    residual = layers.Add()([act1, act2])
    
    if num_res_blocks > 1:
        for i in range(num_res_blocks - 1):
            hid1 = layers.Dense(units=num_of_neurons,
                 kernel_regularizer=regularization_term,
                 bias_regularizer=regularization_term)(residual)
            act1 = CustomActivationLayer(num_of_neurons)(hid1)
            hid2 = layers.Dense(units=num_of_neurons,
                 kernel_regularizer=regularization_term,
                 bias_regularizer=regularization_term)(act1)
            act2 = CustomActivationLayer(num_of_neurons)(hid2)
            residual = layers.Add()([act1, act2])
    
    output_layer = layers.Dense(units=output_shape)(residual)
    
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    model.summary()
    
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.MeanAbsoluteError()
    )
    
    return model
#------------------------------------------------------------------------------------------------------------
def nn_model_train(model, epochs, input_data, truths, validation_features=None, validation_truths=None, decayevery=None, decayrate=None):
    '''
    Trains a neural network model that emulates the truths from the input_data
    Can program the number of epochs and a step-based learning rate decay
    '''
    if decayevery and decayrate: 
        def scheduler(epoch, learning_rate):
            # Halves the learning rate at some points during training
            if epoch != 0 and epoch % decayevery == 0:
                return learning_rate/decayrate
            else:
                return learning_rate
        learning_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
    else:
        learning_scheduler = keras.callbacks.LearningRateScheduler(lambda epoch, learning_rate: learning_rate)
    
    if validation_features and validation_truths:
        history = model.fit(
            input_data,
            truths,
            batch_size = 30,
            epochs = epochs,
            validation_data = (validation_features, validation_truths),
            callbacks=[learning_scheduler],
        )
    else:
        history = model.fit(
            input_data,
            truths,
            batch_size = 30,
            epochs = epochs,
            callbacks=[learning_scheduler],
        )
    
    last_loss = history.history['loss'][-1]
    return last_loss
#------------------------------------------------------------------------------------------------------------
def test_model(emulator, test_sample, test_data, pc_components, pca_z, log_quantity, save=False):
    '''
    Plots emulation errors for the test data and returns the max errors of the test data
    '''
    for i, test_quantity in enumerate(test_data):
        norm_test_params = normalize_params(test_sample[i])
        emulated_test_norm_pcs = emulator(np.array([norm_test_params]))
        emulated_test_pcs = unnormalize_array(emulated_test_norm_pcs, pc_components)
        emulated_test_norm_log_quantity = pca_z.inverse_transform(emulated_test_pcs)
        emulated_test_log_quantity = unnormalize_array(emulated_test_norm_log_quantity, log_quantity)
        emulated_test_quantity = np.exp(emulated_test_log_quantity)
        plt.semilogx(ks, emulated_test_quantity[0]/test_quantity[0] - 1, c='gray', alpha=0.5)
    
    plt.grid()
    plt.fill_between(ks, -0.01,0.01, color='gray',alpha=0.2)
    plt.xlabel('k (h/Mpc)')
    plt.ylabel('Emulation Relative Error')
    title = input('Define a title for the plot: ')
    plt.title(title)
    if save:
        imgName = input('Define the archive name: ')
        plt.savefig(imgName)
    plt.show()
#------------------------------------------------------------------------------------------------------------
def evaluate_test_points(emulator, test_sample, test_data, pc_components, pca_z, log_quantity):
    '''
    Returns max_emulation_errors, the maximum emulation errors for the test data
    '''
    max_emulation_errors = np.zeros(len(test_sample))
    for i, test_quantity in enumerate(test_data):
        norm_test_params = normalize_params(test_sample[i])
        emulated_test_norm_pcs = emulator(np.array([norm_test_params]))
        emulated_test_pcs = unnormalize_array(emulated_test_norm_pcs, pc_components)
        emulated_test_norm_log_quantity = pca_z.inverse_transform(emulated_test_pcs)
        emulated_test_log_quantity = unnormalize_array(emulated_test_norm_log_quantity, log_quantity)
        emulated_test_quantity = np.exp(emulated_test_log_quantity)
        max_emulation_errors[i] = np.amax(emulated_test_quantity[0]/test_quantity[0] - 1)
    return max_emulation_errors
#------------------------------------------------------------------------------------------------------------
def iterative_training(emulator, initial_training_sample, initial_training_data, initial_test_sample, initial_test_data):
    assert False, 'Not implemented'
    # Initial training
    history = emulator_model.fit(
            initial_training_sample,
            initial_training_data,
            batch_size = 5,
            epochs = 2000,
            # validation_data = (validation_input_data, validation_truth_data)
            )
    emulation_errors = evaluate_test_points(emulator, initial_test_sample, initial_test_data, pc_components, pca_z, log_quantity)
    sorted_errors = np.sort(emulation_errors, order='descending')
    treshold = sorted_errors[round(len(emulation_errors)/5)] # The top 20% error
    outliers = []
    outliers_data = []
    for i, error in enumerate(emulation_errors):
        if error > treshold:
            outliers.append(initial_test_sample[i])
            outliers_data.append(initial_test_data[i])
    outliers = np.array(outliers)
    outliers_data = np.array(outliers_data)
    new_training_sample = np.concatenate((initial_training_sample, outliers))
    new_training_data = np.concatenate((initial_traiing_data, outliers_data))

#------------------------------------------------------------------------------------------------------------ 
def _nowiggles_pk(k_lin=None, pk_lin=None, k_emu=None):
    """De-wiggled linear prediction of the cold matter power spectrum

    The BAO feature is removed by identifying and removing its corresponding
    bump in real space, by means of a DST, and consequently transforming
    back to Fourier space.
    See:
    - Baumann et al 2018 (https://arxiv.org/pdf/1712.08067.pdf)
    - Giblin et al 2019 (https://arxiv.org/pdf/1906.02742.pdf)

    :param k_lin: a vector of wavemodes in h/Mpc, if None the wavemodes used by
              camb are returned, defaults to None
    :type k_lin: array_like, optional
    :param pk_lin: a vector of linear power spectrum computed at k_lin, if None
              camb will be called, defaults to None
    :type pk_lin: array_like, optional

    :param k_emu: a vector of wavemodes in h/Mpc, if None the wavemodes used by
              the emulator are returned, defaults to None
    :type k_emu: array_like, optional

    :return: dewiggled pk computed at k_emu
    :rtype: array_like
    """

    from scipy.fftpack import dst, idst

    nk = int(2**15)
    kmin = k_lin.min()
    kmax = 10
    klin = np.linspace(kmin, kmax, nk)

    pkcamb_cs = interpolate.splrep(np.log(k_lin), np.log(pk_lin), s=0)
    pklin = np.exp(interpolate.splev(np.log(klin), pkcamb_cs, der=0, ext=0))

    f = np.log10(klin * pklin)

    dstpk = dst(f, type=2)

    even = dstpk[0::2]
    odd = dstpk[1::2]

    i_even = np.arange(len(even)).astype(int)
    i_odd = np.arange(len(odd)).astype(int)

    even_cs = interpolate.splrep(i_even, even, s=0)
    odd_cs = interpolate.splrep(i_odd, odd, s=0)

    even_2nd_der = interpolate.splev(i_even, even_cs, der=2, ext=0)
    odd_2nd_der = interpolate.splev(i_odd, odd_cs, der=2, ext=0)

    # these indexes have been fudged for the k-range considered
    # [~1e-4, 10], any other choice would require visual inspection
    imin_even = i_even[100:300][np.argmax(even_2nd_der[100:300])] - 20
    imax_even = i_even[100:300][np.argmin(even_2nd_der[100:300])] + 70
    imin_odd = i_odd[100:300][np.argmax(odd_2nd_der[100:300])] - 20
    imax_odd = i_odd[100:300][np.argmin(odd_2nd_der[100:300])] + 75

    i_even_holed = np.concatenate((i_even[:imin_even], i_even[imax_even:]))
    i_odd_holed = np.concatenate((i_odd[:imin_odd], i_odd[imax_odd:]))

    even_holed = np.concatenate((even[:imin_even], even[imax_even:]))
    odd_holed = np.concatenate((odd[:imin_odd], odd[imax_odd:]))

    even_holed_cs = interpolate.splrep(i_even_holed, even_holed * (i_even_holed+1)**2, s=0)
    odd_holed_cs = interpolate.splrep(i_odd_holed, odd_holed * (i_odd_holed+1)**2, s=0)

    even_smooth = interpolate.splev(i_even, even_holed_cs, der=0, ext=0) / (i_even + 1)**2
    odd_smooth = interpolate.splev(i_odd, odd_holed_cs, der=0, ext=0) / (i_odd + 1)**2

    dstpk_smooth = []
    for ii in range(len(i_even)):
        dstpk_smooth.append(even_smooth[ii])
        dstpk_smooth.append(odd_smooth[ii])
    dstpk_smooth = np.array(dstpk_smooth)

    pksmooth = idst(dstpk_smooth, type=2) / (2 * len(dstpk_smooth))
    pksmooth = 10**(pksmooth) / klin

    k_highk = k_lin[k_lin > 5]
    p_highk = pk_lin[k_lin > 5]

    k_extended = np.concatenate((klin[klin < 5], k_highk))
    p_extended = np.concatenate((pksmooth[klin < 5], p_highk))

    pksmooth_cs = interpolate.splrep(np.log(k_extended), np.log(p_extended), s=0)
    pksmooth_interp = np.exp(interpolate.splev(np.log(k_emu), pksmooth_cs, der=0, ext=0))

    return pksmooth_interp

def _smeared_bao_pk(k_lin=None, pk_lin=None, k_emu=None, pk_lin_emu=None, pk_nw=None, grid=None):
    """Prediction of the cold matter power spectrum using a Boltzmann solver with smeared BAO feature

    :param k_lin: a vector of wavemodes in h/Mpc, if None the wavemodes used by
              camb are returned, defaults to None
    :type k_lin: array_like, optional
    :param pk_lin: a vector of linear power spectrum computed at k_lin, if None
              camb will be called, defaults to None
    :type pk_lin: array_like, optional

    :param k_emu: a vector of wavemodes in h/Mpc, if None the wavemodes used by
              the emulator are returned, defaults to None
    :type k_emu: array_like, optional
    :param pk_emu: a vector of linear power spectrum computed at k_emu, defaults to None
    :type pk_emu: array_like, optional
    :param pk_nw: a vector of no-wiggles power spectrum computed at k_emu, defaults to None
    :type pk_nw: array_like, optional
    :param grid: dictionary with parameter and vector of values where to evaluate the emulator, defaults to None
    :type grid: array_like, optional

    :return: smeared BAO pk computed at k_emu
    :rtype: array_like
    """
    from scipy.integrate import trapz

    if grid is None:
        sigma_star_2 = trapz(k_lin * pk_lin, x=np.log(k_lin)) / (3 * np.pi**2)
        k_star_2 = 1 / sigma_star_2
        G = np.exp(-0.5 * (k_emu**2 / k_star_2))
        if pk_nw is None:
            pk_nw = _nowiggles_pk(k_lin=k_lin, pk_lin=pk_lin, k_emu=k_emu)
    else:
        sigma_star_2 = trapz(k_lin[None,:] * pk_lin, x=np.log(k_lin[None:,]), axis=1) / (3 * np.pi**2)
        k_star_2 = 1 / sigma_star_2
        G = np.exp(-0.5 * (k_emu**2 / k_star_2[:,None]))
        if pk_nw is None:
            pk_nw = np.array([_nowiggles_pk(k_lin=k_lin, pk_lin=pkl, k_emu=k_emu) for pkl in pk_lin])
    return pk_lin_emu * G + pk_nw * (1 - G)
