import numpy as np
import logging
from keras.models import load_model
from keras import models, layers, optimizers
from tensorflow import keras
from tensorflow import add, multiply, subtract, sigmoid
import keras.regularizers as regularizers
import euclidemu2 as ee2
from time import perf_counter
from joblib import dump, load
from pandas import DataFrame
from .dataset import ColaDataset
from .preprocessor import ColaPreprocessor
from .utils import lcdm_projection, get_ee2
from .config import VER, MODELS_DIR, k_vals, k_maxs, zs_cola
from keras.saving import register_keras_serializable


class ColaTrainer:
    def __init__(self, dataset: ColaDataset, update_processing=False):
        self.dataset = dataset
        
        preprocessor = ColaPreprocessor(self.dataset, update=update_processing)
        self.num_pcs = preprocessor.num_pcs
        self.X, self.y = preprocessor.prepare_data()
        
        self.z = self.dataset.z
        
        z_idx = zs_cola.tolist().index(float(self.z))
        k_max = int(k_maxs[z_idx])
        self.ks = k_vals[:k_max]
        
        self.lcdm_params = list(map(lcdm_projection, self.dataset.params))
        self.X_lcdm = preprocessor.normalize_params(self.lcdm_params, self.z, update=False)
        
        #allow user to input z
        
    def _build_model(self):
        return generate_mlp(
            input_shape=len(self.X.columns),
            output_shape=self.num_pcs,
            num_layers=3,
            num_neurons=2048,
            activation="custom",
            alpha=0,
            l1_ratio=0.1,
            loss='mse',
            learning_rate=1e-3,
        )

    def load_model(self):
        return load_model(f"{MODELS_DIR}/NN_{self.num_pcs}_v{VER}_z{self.z}.keras")

    def train_model(self):
        mlp = self._build_model()
        start = perf_counter()
        last_loss = nn_model_train(mlp, 1600, self.X, self.y, decayevery=1500, decayrate=2)
        last_loss = nn_model_train(mlp, 1000, self.X, self.y, decayevery=200, decayrate=2)
        print(f"Training took {perf_counter() - start} seconds")
        mlp.save(f"{MODELS_DIR}/NN_{self.num_pcs}_v{VER}_z{self.z}.keras")
        del mlp
        pass

    #@staticmethod
    def predict_boost(self, X):
        mlp = self.load_model()
        pred_pcs = mlp.predict(X)
        pred_boosts = self.deprocess_boosts(pred_pcs, self.z)
        self.raw_predictions = pred_boosts

        # Infinite references
        lcdm_pred_pcs = mlp.predict(self.X_lcdm)
        lcdm_pred_boosts = self.deprocess_boosts(lcdm_pred_pcs, self.z)

        lcdm_ee2_booosts = [get_ee2(lcdms,  self.ks, z_val=float(self.z)) for lcdms in self.lcdm_params]

        boosts = np.array(lcdm_ee2_booosts) * (np.array(pred_boosts) / np.array(lcdm_pred_boosts))

        boosts_df = self.dataset.boosts_to_df(boosts)

        return boosts_df
            
            
    @staticmethod
    def deprocess_boosts(pcs, z):
        """
        De-process boost PCs by inverting PCA and scaling. 

        Returns:
            DataFrame: Raw boosts with ks as columns.
        """        
        # Load information and recover boosts
        logging.info(f"Loading PCA and scaler for z={z}")
        pca = load(ColaPreprocessor._pca_path(z))
        scaler = load(ColaPreprocessor._boost_scaler_path(z))

        logboosts = scaler.inverse_transform(pca.inverse_transform(pcs))
        boosts = np.exp(logboosts)
        
        z_idx = zs_cola.tolist().index(float(z))
        k_max = int(k_maxs[z_idx])
        ks = k_vals[:k_max]

        # Wrap boosts in dataframe
        boosts_df = DataFrame(boosts)
        boosts_df.index = [f"Boost {i}" for i in boosts_df.index]
        boosts_df.columns =  ks
        boosts_df.columns.name = "k"
        logging.info(f"De-processing boosts of shape {boosts_df.shape} complete")

        return boosts_df
    
class ColaEmu:
    def __init__(self, params):
        self.params = params
        _, ee2_proj = [get_ee2(self.params[i], k_vals, z_val=zs_cola, lcdm=True)
                       for i in range(len(params))]

@register_keras_serializable()
class CustomActivationLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomActivationLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.beta = self.add_weight(shape=(self.units,), initializer='random_normal', 
                                    trainable=True, name="beta")
        self.gamma = self.add_weight(shape=(self.units,), initializer='random_normal', 
                                     trainable=True, name="gamma")
        super(CustomActivationLayer, self).build(input_shape)

    def call(self, x):
        # See e.g. https://arxiv.org/pdf/1911.11778.pdf, Equation (8)
        #func = tf.add(self.gamma, tf.multiply(tf.sigmoid(tf.multiply(self.beta, x)), 
        #                                      tf.subtract(1.0, self.gamma)))
        #return tf.multiply(func, x)
        func = add(self.gamma, multiply(sigmoid(multiply(self.beta, x)), 
                                        subtract(1.0, self.gamma)))
        return multiply(func, x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    
def generate_mlp(input_shape, output_shape, num_layers, num_neurons, activation="custom", 
                 alpha=0.01, l1_ratio=0.01, learning_rate=1e-3, optimizer='adam', loss='mse'):
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