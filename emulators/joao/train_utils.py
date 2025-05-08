# Auxiliary functions for Power Spectrum Emulation
# Author: João Victor Silva Rebouças, May 2022
import math
import pickle
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from keras import models, layers, optimizers
from keras.regularizers import l1_l2
from tqdm.notebook import tqdm

import euclidemu2 as ee2

#------------------------------------------------------------------------------------------------------------
# Parameter space
params = ['h', 'Omega_b', 'Omega_m', 'As', 'ns', 'w', 'wa']
params_latex = [r'$h$', r'$\Omega_b$', r'$\Omega_m$', r'$A_s$', r'$n_s$', r'$w_0$', r'$w_a$']

zs_cola = [
    0.000, 0.020, 0.041, 0.062, 0.085, 0.109, 0.133, 0.159, 0.186, 0.214, 0.244, 0.275, 0.308, 
    0.342, 0.378, 0.417, 0.457, 0.500, 0.543, 0.588, 0.636, 0.688, 0.742, 0.800, 0.862, 0.929, 
    1.000, 1.087, 1.182, 1.286, 1.400, 1.526, 1.667, 1.824, 2.000, 2.158, 2.333, 2.529, 2.750, 
    3.000
]

# Parameter limits for training, see Table 2 from https://arxiv.org/pdf/2010.11288
lims = {}
lims['h'] = [0.61, 0.73]
lims['Omega_b'] = [0.04, 0.06]
lims['Omega_m'] = [0.24, 0.4]
lims['As'] = [1.7e-9, 2.5e-9]
lims['ns'] = [0.92, 1]
lims['w'] = [-1.3, -0.7]
lims['wa']  = [-0.7, 0.5]

# Reference values
ref = {}
ref['h'] = 0.67
ref['Omega_b'] = 0.049
ref['Omega_m'] = 0.319
ref['As'] = 2.1e-9
ref['ns'] = 0.96
ref['w'] = -1
ref['wa'] = 0
params_ref = [ref[param] for param in params]

#------------------------------------------------------------------------------------------------------------

def load_set(path, z):
    lhs = np.loadtxt(f"{path}/lhs.txt")
    num_samples = len(lhs)
    
    pks_lin = []
    pks_nl  = []
    for i in range(num_samples):
        # NOTE: skipping first k-bin
        ks, pk_nl_a, pk_lin = np.loadtxt(f"{path}/output/a/{i}/pofk_run_{i}_cb_z{z:.3f}.txt", unpack=True, usecols=(0,1,2), skiprows=2)
        ks, pk_nl_b, pk_lin = np.loadtxt(f"{path}/output/b/{i}/pofk_run_{i}_cb_z{z:.3f}.txt", unpack=True, usecols=(0,1,2), skiprows=2)
        pk_nl = 0.5*(pk_nl_a + pk_nl_b)
        pks_lin.append(pk_lin)
        pks_nl.append(pk_nl)
    pks_lin = np.array(pks_lin)
    pks_nl  = np.array(pks_nl)
    
    return lhs, ks, pks_lin, pks_nl

class COLASet:
    def __init__(self, path, z):
        self.num_pcs = None
        self.z = z
        data = load_set(path, z)
        self.lhs, self.ks, self.pks_lin, self.pks_nl = data
        self.boosts = self.pks_nl/self.pks_lin
        self.logboosts = np.log(self.boosts)

    def change_ks(self, ks):
        boosts = []
        for boost in self.boosts:
            boost_interp = CubicSpline(self.ks, boost)
            new_boost = boost_interp(ks)
            boosts.append(new_boost)
        self.boosts = boosts
        self.logboosts = np.log(self.boosts)
        self.ks = ks
        if self.num_pcs is not None: self.prepare(self.num_pcs)

    def update(self, cosmos, boosts):
        self.lhs = np.vstack([self.lhs, cosmos])
        self.boosts = np.vstack([self.boosts, boosts])
        self.logboosts = np.log(self.boosts)
        if self.num_pcs is not None: self.prepare(self.num_pcs)

    def prepare(self, num_pcs):
        self.num_pcs = num_pcs
        self.param_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.lhs)
        self.lhs_norm = self.param_scaler.transform(self.lhs)
        self.boost_scaler = Scaler()
        self.boost_scaler.fit(self.logboosts)
        self.logboosts_norm = self.boost_scaler.transform(self.logboosts)    
        self.num_pcs = num_pcs
        self.pca = PCA(n_components=num_pcs)
        self.pcs = self.pca.fit_transform(self.logboosts_norm)

    def plot_boosts(self):
        for boost in self.boosts:
            plt.semilogx(self.ks, boost)
        plt.title(f"Halofit train boosts for z = {self.z:.3f}")
        plt.xlabel("k")
        plt.ylabel("Boost")
    
    def plot_logboosts(self):
        for boost in self.logboosts:
            plt.semilogx(self.ks, boost)
        plt.title(f"Halofit train boosts for z = {self.z:.3f}")
        plt.xlabel("k")
        plt.ylabel("Log(Boost)")

    def plot_logboosts_norm(self):
        for boost in self.logboosts_norm:
            plt.semilogx(self.ks, boost)
        plt.title(f"Halofit train boosts for z = {self.z:.3f}")
        plt.xlabel("k")
        plt.ylabel("Normalized Log(Boost)")

    def plot_lhs(self, model_to_map_errors=None):
        D = 7
        assert D == len(self.lhs[0])
        if model_to_map_errors is not None:
            model_predictions = model_to_map_errors.predict(self.lhs)
            errors = np.abs(model_predictions/self.boosts - 1)
            max_errors = []
            for error in errors: max_errors.append(np.log10(np.amax(error)))
        fig, axs = plt.subplots(D, D, figsize=(15, 15), gridspec_kw={"hspace": 0.1, "wspace": 0.1})
        for row in range(D):
            for col in range(D):
                ax = axs[row, col]
                if row == D-1: ax.set_xlabel(params_latex[col])
                else: ax.set_xticks([])
                if col == 0: ax.set_ylabel(params_latex[row])
                else: ax.set_yticks([])
                if col >= row: ax.remove()
                x_index = params.index(params[col])
                y_index = params.index(params[row])
                x_params = [sample[x_index] for sample in self.lhs]
                y_params = [sample[y_index] for sample in self.lhs]
                if model_to_map_errors is not None: im = ax.scatter(x_params, y_params, s=6, alpha=0.75, c=max_errors)
                else: im = ax.scatter(x_params, y_params, s=2, alpha=0.5)
        
        # See https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
        if model_to_map_errors is not None:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.75, 0.25, 0.025, 0.35])
            fig.colorbar(im, cax=cbar_ax, shrink=0.75, label="Log Error")
        return fig
        
class COLAModel():
    def __init__(self, trainSet):
        self.boost_scaler = trainSet.boost_scaler
        self.param_scaler = trainSet.param_scaler
        self.pca = trainSet.pca

    def fit(self, trainSet, num_epochs):
        raise NotImplementedError("ERROR: COLAModel must override `fit` method")

    def predict_pcs(self, x):
        raise NotImplementedError("ERROR: COLAModel must override `predict_pcs` method")

    def predict(self, x):
        pcs = self.predict_pcs(x)
        logboost_norm = self.pca.inverse_transform(pcs)
        logboost = self.boost_scaler.inverse_transform(logboost_norm)
        return np.exp(logboost)

    def plot_errors(self, testSet):
        preds = self.predict(testSet.lhs)
        fig, ax = plt.subplots()
        targets = testSet.boosts
        for pred, target in zip(preds, targets):
            error = pred/target - 1
            ax.semilogx(testSet.ks, error)
        ax.fill_between(testSet.ks, -0.0025, 0.0025, color="gray", alpha=0.75)
        ax.fill_between(testSet.ks, -0.005, 0.005, color="gray", alpha=0.5)
        ax.set_xlabel("k")
        ax.set_ylabel("Emulation Error")
        return fig, ax

    def get_outliers(self, testSet, log=False):
        boosts_pred = self.predict(testSet.lhs)
        cosmos = []
        boosts = []
        for boost_test, boost_pred, cosmo in zip(boosts_pred, testSet.boosts, testSet.lhs):
            error = np.abs(boost_pred/boost_test - 1)
            if np.any(error > 0.005):
                h, Omega_b, Omega_m, As, ns, w, w0pwa = cosmo
                if log: print(f"Outlier: [{h=}, {Omega_b=}, {Omega_m=}, {As=}, {ns=}, {w=}, {w0pwa=}] has error {np.amax(error)} ")
                cosmos.append(cosmo)
                boosts.append(boost_test)
        return cosmos, boosts

    def save(self, path):
        with open(path, "wb") as f: pickle.dump(self, f)

def load_model(path):
    with open(path, "rb") as f: model = pickle.load(f)
    return model

class COLA_NN_Keras(COLAModel):
    def __init__(self, trainSet, num_layers=3, num_neurons=2048):
        super().__init__(trainSet)

        self.mlp = generate_mlp(
            input_shape=len(trainSet.lhs[0]),
            output_shape=len(trainSet.pcs[0]),
            num_layers=num_layers,
            num_neurons=num_neurons,
            activation="custom",
            alpha=0,
            l1_ratio=0
        )
    
    def fit(self, trainSet, num_epochs, decayevery, decayrate):
        try: nn_model_train_keras(self.mlp, epochs=num_epochs, input_data=trainSet.lhs_norm, truths=trainSet.pcs, decayevery=decayevery, decayrate=decayrate)
        except KeyboardInterrupt:
            print("Training interrupted.")
            return
    
    def predict_pcs(self, x):
        x_norm = self.param_scaler.transform(x)
        return self.mlp(x_norm)

    def compile_to_c(self, path):
        names = [
            "weights1", "bias1", "gamma1", "beta1",
            "weights2", "bias2", "gamma2", "beta2",
            "weights3", "bias3", "gamma3", "beta3",
            "weights4", "bias4"
        ]
        for name, params in zip(names, self.mlp.get_weights()):
            p = params.flatten()
            with open(f"{path}/{name}.c", "w") as f:
                f.write(f"static double {name}[{len(p)}] = {'{'}")
                for param in p: f.write(f"{param}, ")
                f.write("};")

class COLA_PCE(COLAModel):
    def __init__(self, trainSet, max_order):
        super().__init__(trainSet)
        d = len(trainSet.lhs[0])
        max_values = np.array([max_order for _ in range(d)]) # FOI ESSE melhor com tudo 5
        numbers = np.arange(max_order + 1)
        combinations = np.array(list(product(numbers, repeat=d)), dtype=np.float64)
        filtered_data = pce_utils.filter_q0_norm(combinations, threshold=5)
        self.allowcomb = pce_utils.filter_combinations(filtered_data/max_values,  p=0.95 + 1e-12, threshold=1 + 1e-12) #best=6
        self.allowcomb = self.allowcomb*max_values
        X = pce_utils.calculate_power_expansion(self.allowcomb, np.clip(trainSet.lhs_norm, -1, 1))
        self.n_samples, self.n_features = X.shape
        self.n_targets = trainSet.pcs.shape[1]
        self.n_tasks = trainSet.pcs.shape[1]
        W = np.asfortranarray(
            np.zeros(
                (self.n_targets, self.n_features),
                dtype=X.dtype.type,
                order="F"
            )
        ) # W are the coefficients of the polynomial expansion
        
        R = np.zeros((self.n_samples, self.n_tasks), dtype=X.dtype.type, order='F')
        norm_cols_X = np.zeros(self.n_features, dtype=X.dtype.type)
        norm_cols_X = (np.asarray(X)**2).sum(axis=0)

        R = trainSet.pcs - np.dot(X, W.T)

        self.W = np.asfortranarray(W)
        self.X = np.asfortranarray(X) 
        self.R = np.asfortranarray(R) 
        self.norm_cols_X = np.asfortranarray(norm_cols_X)

    def fit(self, trainSet, num_epochs, alpha=1e-5, l1_ratio=0.05):
        # Bernardo recommended:
        # z = 0: alpha = 1e-5, l1_ratio = 0.05
        # else: alpha = 1e-6, l1_ratio = 0.9
        self.l1_reg =  alpha *  l1_ratio * self.n_samples
        self.l2_reg =  alpha * (1.0 -  l1_ratio) * self.n_samples
        for _ in range(num_epochs):
            ElNetFortran.fit(
                self.n_features,
                1,
                self.l1_reg,
                self.l2_reg,
                self.W,
                self.X,
                self.R,
                self.norm_cols_X
            )

    def predict_pcs(self, x):
        x_norm = self.param_scaler.transform(x)
        yteste = pce_utils.calculate_power_expansion(self.allowcomb, np.clip(x_norm, -1, 1))
        pcs_pred = yteste@self.W.T
        return pcs_pred

#------------------------------------------------------------------------------------------------------------

# Bernardo's code for KAN model in PyTorch
class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return (X* self.std) + self.mean

#------------------------------------------------------------------------------------------------------------

# Joao's code for NN in Keras, used in the COLA_NN_Keras class
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
    elif activation == "sigmoid":
        x = keras.activations.sigmoid(x)
    else:
        raise Exception(f"Unexpected activation {activation}")
    
    # Add more hidden layers
    for _ in range(num_layers - 1): # subtract 1 because we've already added the first hidden layer
        x = layers.Dense(num_neurons, kernel_regularizer=reg)(x)
        if activation == "custom":
            x = CustomActivationLayer(num_neurons)(x)
        elif activation == "relu":
            x = keras.activations.relu(x)
        elif activation == "sigmoid":
            x = keras.activations.sigmoid(x)
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
    model.compile(optimizer=opt, loss=loss)

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

def nn_model_train_keras(model, epochs, input_data, truths, validation_features=None, validation_truths=None, decayevery=None, decayrate=None):
    '''
    Trains a neural network model that emulates the truths from the input_data
    Can program the number of epochs and a step-based learning rate decay
    '''
    # See https://stackoverflow.com/questions/44931689/how-to-disable-printing-reports-after-each-epoch-in-keras
    class PrintCallback(tf.keras.callbacks.Callback):
        SHOW_NUMBER = 10
        epoch = 0

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch

        def on_epoch_end(self, batch, logs=None):
            print(f'Epoch: {self.epoch} => Loss = {logs["loss"]}', end="\r")
    
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
            callbacks=[learning_scheduler, PrintCallback()],
            verbose=0
        )
    else:
        history = model.fit(
            input_data,
            truths,
            batch_size = 30,
            epochs = epochs,
            callbacks=[learning_scheduler, PrintCallback()],
            verbose=0
        )
    
    last_loss = history.history['loss'][-1]
    return last_loss

#------------------------------------------------------------------------------------------------------------