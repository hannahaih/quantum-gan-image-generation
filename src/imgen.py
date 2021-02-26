import copy
import numpy as np
import pandas as pd
import pennylane as qml
import tensorflow as tf
import plotly.express as px
from IPython.display import clear_output

from src.utils import blur_and_normalize, vec_bin_array, bool2int, plot_dist
from src.generator import generator_prob_circuit
from src.discriminator import discriminator_network


class ImageGenerator():
    """
    Class to perform image recognition using qGAN. Main processes involves: 
    1) load in image datasets
    2) create qGAN with quantum generator and classical discriminator
    3) Train discriminator on tensorflow and update variational quantum circuit
    """
    def __init__(self,
            num_qubits, num_layers,
            epoch_sample_size=1e4, batch_sample_size=1e3,
            generator_learning_rate=1e-3, discriminator_learning_rate=1e-3,
            discriminator_num_nodes_1=50, discriminator_num_nodes_2=10, discriminator_alpha_1=1e-2, discriminator_alpha_2=1e-2,
            enable_remapping=False, mapping_sample_size=1e6
        ):

        if num_qubits % 2 != 0:
            raise ValueError('The number of qubits must be even.')

        self.__num_qubits = num_qubits
        self.__num_layers = num_layers
        self.__image_dim = int(2 ** (self.__num_qubits / 2))
        self.__image_size = self.__image_dim ** 2
        self.__epoch_sample_size = epoch_sample_size
        self.__batch_sample_size = batch_sample_size
        self.__real_dist = None     # Output array from preprocessed image
        self.__mapping_arr = None   # Array prepared for remapping
        self.__reverse_mapping_arr = None    # Reverse the mapped array for right order

        self.__dev = qml.device('default.qubit', wires=self.__num_qubits, analytic=True, shots=1)
        self.__generator = qml.QNode(lambda params: generator_prob_circuit(params, self.__num_qubits), self.__dev, interface='tf')
        self.__discriminator_hyperparams = [discriminator_num_nodes_1, discriminator_num_nodes_2, discriminator_alpha_1, discriminator_alpha_2]
        # Using Adam algorithm for generator & discriminator optimization
        self.__generator_optimizer = tf.keras.optimizers.Adam(generator_learning_rate)
        self.__discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_learning_rate)

        self.__enable_remapping = enable_remapping
        self.__mapping_sample_size = mapping_sample_size

        # Using cross entropy to compute the losses for generator & discriminator
        self.__cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.__cross_entropy_list = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        # Generate fake output for discriminator training
        self.__fake_sample_arr = tf.cast(
            - (vec_bin_array(np.array(range(2**self.__num_qubits)), self.__num_qubits) * 2 - 1),
            dtype='float32'
        )

        self.reinit() # Initialize discriminator & generator


    def reinit(self):
        self._init_generator()
        self._init_discriminator()
        self._clear_history()
    

    def load_image(self, image, blur_sigma=0.0, show_figure=False, figure_width=500):
        """
        Load input image datasets, preprocess with gaussian filter, optionally implement remapping of the image
        array into monotone order for better efficiency.
        Args:
            image (2d arr): image array
            blur_sigma (float): std for the gaussian kernel
            show_figure (boolean): True for plotting the processed image
            figure_width (scalar)
        Returns:
            none
        """
        w, h = image.shape
        if w != self.__image_dim or h != self.__image_dim:
            raise ValueError('Input image must be a {dim} by {dim} floating point array.'.format(dim = self.__image_dim))
        self.__real_dist = blur_and_normalize(image, sigma=blur_sigma, show_figure=show_figure, figure_width=figure_width)

        if self.__enable_remapping:
            # if remapping enabled, remap the image array into monotone order
            self.__mapping_arr = self._generate_mapping(self.__real_dist, self.__mapping_sample_size)
        else:
            # otherwise generate monotone array according to image size
            self.__mapping_arr = np.array(range(self.__image_size), dtype=np.int)

        self.__reverse_mapping_arr = self._reverse_mapping(self.__mapping_arr)

        self.reinit() # Initialize discriminator & generator


    def make_dataset(self):
        # Draw and shuffle random datasets
        train_dataset_raw = self._discrete_sample(self.__real_dist, self.__epoch_sample_size, mapping_arr=self.__mapping_arr).reshape(self.__epoch_sample_size, self.__num_qubits).astype('float32')
        return tf.data.Dataset.from_tensor_slices(train_dataset_raw).shuffle(self.__epoch_sample_size).batch(self.__batch_sample_size)
    

    def get_generator_loss_history(self):
        # Record the generator loss through epoches
        return copy.deepcopy(self.__generator_loss_history)
    
    
    def get_discriminator_loss_history(self):
        # Record the discriminator loss through epoches
        return copy.deepcopy(self.__discriminator_loss_history)

    
    def get_output_distribution_history(self):
        # Record the output distribution through epoches
        return copy.deepcopy(self.__output_distribution_history)
    
    
    def get_params_historty(self):
        # Record the variational parameters through epoches
        return copy.deepcopy(self.__params_history)
    
    
    def get_loss_history_dataframe(self, gen_label='Generator', disc_label='Discriminator'):
        return pd.DataFrame({
            gen_label: np.array(self.get_generator_loss_history()).flatten(),
            disc_label: np.array(self.get_discriminator_loss_history()).flatten()
        })


    def plot_loss_history(self, gen_label='Generator', disc_label='Discriminator'):
        # Visualization of performance by plotting the loss history
        df = self.get_loss_history_dataframe(gen_label=gen_label, disc_label=disc_label)
        return px.line(df, labels={
            'index': 'Step',
            "value": 'Cross Entropy Loss',
            'variable': ''
        }, title='Generator and Discriminator Losses')


    def plot_output_distribution_history(self, figure_width=700):
        # Visualization of performance by plotting the output distribution history
        arr = self.get_output_distribution_history()
        return px.imshow(
            np.array(arr),
            animation_frame=0,
            width=figure_width, height=figure_width,
            color_continuous_scale='sunset',
            labels={
                'animation_frame': 'Epoch'
            },
            title='Probability Distribution of Circuit Output'
        )


    def get_final_output_distribution(self, avg_window=1):
        # Get final output from qGAN by averaging over output history
        return np.average(np.array(self.get_output_distribution_history()[-avg_window:]), axis=0)
    
    
    def plot_final_output_distribution(self, avg_window=1, figure_width=700):
        # Plot final output distribution
        dist = self.get_final_output_distribution(avg_window=avg_window)
        return plot_dist(dist, figure_width=figure_width)


    def get_final_params(self, avg_window=1):
        # Get final variational parameters by averaging over in history
        return np.average(np.array(self.get_params_historty()[-avg_window:]), axis=0)


    def train(self, dataset, num_epochs, show_progress=False, figure_width=500):
        """
        Overall training structure for the discriminator & generator
        Args:
            dataset (arr): complete input dataset
            num_epochs (int): training epoch
            show_progess (boolean): if True, show progress across epochs
            figure_width (scalar)
        Returns:
            none
        """
        for epoch in range(num_epochs):
            
            # Flip the output array for correct order
            output = np.flip(np.array(self.__generator(self.__generator_params)))
            remapped_output = np.zeros_like(output)
            # Inverse the remapped array
            for i in range(len(self.__reverse_mapping_arr)):
                remapped_output[self.__reverse_mapping_arr[i]] = output[i]

            distribution = np.reshape(remapped_output, (self.__image_dim, self.__image_dim))

            self.__output_distribution_history.append(distribution)
            
            # Show progress across epochs
            if show_progress:
                clear_output(wait=True)
                print('Training epoch {} of {}:'.format(epoch + 1, num_epochs))
                plot_dist(distribution, figure_width=figure_width).show()

            for image_batch in dataset:
                g, d, p = self._train_step(image_batch)
                self.__generator_loss_history.append(g)
                self.__discriminator_loss_history.append(d)
                self.__params_history.append(p)


    def _init_generator(self):
        # Initialize variational parameters for the generator
        self.__generator_params = tf.Variable(
            np.zeros(shape=(self.__num_layers, self.__num_qubits)),
            dtype='float32'
        )
    

    def _init_discriminator(self):
        # Initialize discriminator network
        self.__discriminator = discriminator_network(*self.__discriminator_hyperparams)
    

    def _clear_history(self):
        # In each initialization, clear the previously recorded history
        self.__generator_loss_history = []
        self.__discriminator_loss_history = []
        self.__output_distribution_history = []
        self.__params_history = []
    

    def _discriminator_loss(self, real_output, fake_output, fake_output_prob):
        # Compute the discriminator loss based on cross entropy by considering both loss from real and fake output
        real_loss = self.__cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = tf.tensordot(self.__cross_entropy_list(tf.zeros_like(fake_output), fake_output), fake_output_prob, 1)
        total_loss = (real_loss + fake_loss) / 2
        return total_loss
    

    def _generator_loss(self, fake_output, fake_output_prob):
        # Comput the generater loss based on cross entropy by considering loss from fake output
        return tf.tensordot(self.__cross_entropy_list(tf.ones_like(fake_output), fake_output), fake_output_prob, 1)


    def _discrete_sample(self, dist, num_samples, mapping_arr=None):
        """
        Randomly generate discrete sample in array of bit vectors
        Args:
            dist (arr): input arr from the image
            num_samples (int): sample size
            mapping_arr (Optional, arr)
        Returns:
            (arr): output arr of bit vectors
        """
        c = np.random.choice(dist.size, size=num_samples, replace=True, p=dist.flatten())
        if mapping_arr is not None:
            c = mapping_arr[c]
        return vec_bin_array(c, int(np.log2(dist.size))) * 2 - 1


    def _generate_mapping(self, dist, num_samples):
        """
        Randomly generate discrete sample in array of bit vectors according to the size of the input image. 
        Arrange the array in monotone order based on sampling frequency to improve efficiency of the QM circuit.
        Args:
            dist (array): input image arr
            num_samples (int): sample size
        Returns:
            map_arr (arr): rearangd array after mapping
        """
        ds = ((self._discrete_sample(self.__real_dist, 10000) + 1) / 2).astype(int)
        unp = np.array([bool2int(x[::-1]) for x in ds])
        u, c = np.unique(unp, return_counts=True)
        inds = c.argsort() # Return the indices for a sorted array
        c = c[inds]
        u = u[inds]
        diff = np.setdiff1d(np.array(range(self.__image_size)), u)
        
        # Initilize mapping array 
        map_arr = np.zeros(self.__image_size, dtype=np.int)
        
        # Update mapping array
        for i in range(len(diff)):
            map_arr[diff[i]] = i
        for i in range(len(u)):
            map_arr[u[i]] = i + len(diff)
                
        return map_arr
    

    # https://stackoverflow.com/questions/54153270/how-to-get-a-reverse-mapping-in-numpy-in-o1
    def _reverse_mapping(self, mapping_arr):
        # Reverse the remapped array for correct order
        t = np.zeros(np.max(mapping_arr) + 1, dtype=np.int)
        t[mapping_arr] = np.arange(0, mapping_arr.size)
        
        return t
    

    def _train_step(self, real_sample):
        """
        Defines one specific epoch during the training.
        Args:
            real_sample (arr): real input image arr
        Returns:
            gen_loss_arr (arr): updated record of generator loss history
            disc_loss_arr (arr): updated record of discriminator loss history
            params_arr (arr): update record of variational parameters history
        """
        gen_loss_arr = []
        disc_loss_arr = []
        params_arr = []

        # Compute gradient with gradient tape
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Calculate output after generator training
            # Use probability of states output by the generator for easier calculation
            fake_output_prob = tf.cast(self.__generator(self.__generator_params), dtype='float32')

            # Calculate the output after discriminator training
            real_output = self.__discriminator(real_sample, training=True)
            fake_output = self.__discriminator(self.__fake_sample_arr, training=True)

            # Compute losses and variational parameters
            gen_loss = self._generator_loss(fake_output, fake_output_prob)
            disc_loss = self._discriminator_loss(real_output, fake_output, fake_output_prob)
            gen_loss_arr.append(gen_loss.numpy())
            disc_loss_arr.append(disc_loss.numpy())
            params_arr.append(self.__generator_params.numpy())
        
        # Optimization with gradients
        gradients_of_generator = gen_tape.gradient(gen_loss, [self.__generator_params])
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.__discriminator.trainable_variables)

        self.__generator_optimizer.apply_gradients(zip(gradients_of_generator, [self.__generator_params]))
        self.__discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.__discriminator.trainable_variables))

        return gen_loss_arr, disc_loss_arr, params_arr