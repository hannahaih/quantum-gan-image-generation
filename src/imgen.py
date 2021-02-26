import copy
import numpy as np
import pennylane as qml
import tensorflow as tf
from IPython.display import clear_output

from src.utils import blur_and_normalize, vec_bin_array, bool2int, plot_dist
from src.generator import generator_prob_circuit
from src.discriminator import discriminator_network


class ImageGenerator():
    """
    Class to perform image recognition using qGAN. Main process involves: 
    1) load in image datasets
    2) create qGAN with quantum generator and classical discriminator
    3) Train discriminator on tensorflow and update variational quantum circuit
    """
    def __init__(self,
            num_qubits, num_layers,
            epoch_sample_size=1e4, batch_sample_size=1e3,
            generator_learning_rate=1e-3, discriminator_learning_rate=1e-3,
            discriminator_num_nodes_1=50, discriminator_num_nodes_2=10, discriminator_alpha_1=1e-2, discriminator_alpha_2=1e-2,
            enable_remapping=True, mapping_sample_size=1e6
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
        self.__mapping_arr = None   #
        self.__reverse_mapping_arr = None    # 

        self.__dev = qml.device('default.qubit', wires=self.__num_qubits, analytic=True, shots=1)
        self.__generator = qml.QNode(lambda params: generator_prob_circuit(params, self.__num_qubits), self.__dev, interface='tf')
        self.__discriminator_hyperparams = [discriminator_num_nodes_1, discriminator_num_nodes_2, discriminator_alpha_1, discriminator_alpha_2]
        self.__generator_optimizer = tf.keras.optimizers.Adam(generator_learning_rate)
        self.__discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_learning_rate)

        self.__enable_remapping = enable_remapping
        self.__mapping_sample_size = mapping_sample_size

        self.__cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.__cross_entropy_list = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        self.__fake_sample_arr = tf.cast(
            - (vec_bin_array(np.array(range(2**self.__num_qubits)), self.__num_qubits) * 2 - 1),
            dtype='float32'
        )

        self.reinit()


    def reinit(self):
        self._init_generator()
        self._init_discriminator()
        self._clear_history()
    

    def load_image(self, image, blur_sigma=0.0, show_figure=False):
        """
        Load in input image, preprocess with gaussian filter, optionally 
        """
        w, h = image.shape
        if w != self.__image_dim or h != self.__image_dim:
            raise ValueError('Input image must be a {dim} by {dim} floating point array.'.format(dim = self.__image_dim))
        self.__real_dist = blur_and_normalize(image, sigma=blur_sigma, show_figure=show_figure)

        if self.__enable_remapping:
            self.__mapping_arr = self._generate_mapping(self.__real_dist, self.__mapping_sample_size)
        else:
            self.__mapping_arr = np.array(range(self.__image_size), dtype=np.int)

        self.__reverse_mapping_arr = self._reverse_mapping(self.__mapping_arr)

        self.reinit()


    def make_dataset(self):
        train_dataset_raw = self._discrete_sample(self.__real_dist, self.__epoch_sample_size, mapping_arr=self.__mapping_arr).reshape(self.__epoch_sample_size, self.__num_qubits).astype('float32')
        return tf.data.Dataset.from_tensor_slices(train_dataset_raw).shuffle(self.__epoch_sample_size).batch(self.__batch_sample_size)
    

    def get_generator_loss_history(self):
        return copy.deepcopy(self.__generator_loss_history)
    
    
    def get_discriminator_loss_history(self):
        return copy.deepcopy(self.__discriminator_loss_history)

    
    def get_output_distribution_history(self):
        return copy.deepcopy(self.__output_distribution_history)
    

    def _init_generator(self):
        self.__generator_params = tf.Variable(
            np.zeros(shape=(self.__num_layers, self.__num_qubits)),
            dtype='float32'
        )
    

    def _init_discriminator(self):
        self.__discriminator = discriminator_network(*self.__discriminator_hyperparams)
    

    def _clear_history(self):
        self.__generator_loss_history = []
        self.__discriminator_loss_history = []
        self.__output_distribution_history = []
    

    def _discriminator_loss(self, real_output, fake_output, fake_output_prob):
        real_loss = self.__cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = tf.tensordot(self.__cross_entropy_list(tf.zeros_like(fake_output), fake_output), fake_output_prob, 1)
        total_loss = real_loss + fake_loss
        return total_loss
    

    def _generator_loss(self, fake_output, fake_output_prob):
        return tf.tensordot(self.__cross_entropy_list(tf.ones_like(fake_output), fake_output), fake_output_prob, 1)


    def _discrete_sample(self, dist, num_samples, mapping_arr=None):
        """
        Randomly generate discrete sample in arrays of bit vectors
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
        
        """
        ds = ((self._discrete_sample(self.__real_dist, 10000) + 1) / 2).astype(int)
        unp = np.array([bool2int(x[::-1]) for x in ds])
        u, c = np.unique(unp, return_counts=True)
        inds = c.argsort()
        c = c[inds]
        u = u[inds]
        diff = np.setdiff1d(np.array(range(self.__image_size)), u)
        
        map_arr = np.zeros(self.__image_size, dtype=np.int)
        
        for i in range(len(diff)):
            map_arr[diff[i]] = i
        for i in range(len(u)):
            map_arr[u[i]] = i + len(diff)
                
        return map_arr
    

    # https://stackoverflow.com/questions/54153270/how-to-get-a-reverse-mapping-in-numpy-in-o1
    def _reverse_mapping(self, mapping_arr):
        t = np.zeros(np.max(mapping_arr) + 1, dtype=np.int)
        t[mapping_arr] = np.arange(0, mapping_arr.size)
        
        return t
    

    def _train_step(self, real_sample):
        gen_loss_arr = []
        disc_loss_arr = []

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            fake_output_prob = tf.cast(self.__generator(self.__generator_params), dtype='float32')

            real_output = self.__discriminator(real_sample, training=True)
            fake_output = self.__discriminator(self.__fake_sample_arr, training=True)

            gen_loss = self._generator_loss(fake_output, fake_output_prob)
            disc_loss = self._discriminator_loss(real_output, fake_output, fake_output_prob)
            gen_loss_arr.append(gen_loss.numpy())
            disc_loss_arr.append(disc_loss.numpy())
        
        gradients_of_generator = gen_tape.gradient(gen_loss, [self.__generator_params])
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.__discriminator.trainable_variables)

        self.__generator_optimizer.apply_gradients(zip(gradients_of_generator, [self.__generator_params]))
        self.__discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.__discriminator.trainable_variables))

        return gen_loss_arr, disc_loss_arr


    def train(self, dataset, num_epochs, show_progress=False):
        for epoch in range(num_epochs):
            
            output = np.flip(np.array(self.__generator(self.__generator_params)))
            remapped_output = np.zeros_like(output)
            for i in range(len(self.__reverse_mapping_arr)):
                remapped_output[self.__reverse_mapping_arr[i]] = output[i]

            distribution = np.reshape(remapped_output, (self.__image_dim, self.__image_dim))

            self.__output_distribution_history.append(distribution)
            
            if show_progress:
                clear_output(wait=True)
                print('Training epoch {} of {}:'.format(epoch + 1, num_epochs))
                plot_dist(distribution).show()

            for image_batch in dataset:
                g, d = self._train_step(image_batch)
                self.__generator_loss_history.append(g)
                self.__discriminator_loss_history.append(d)