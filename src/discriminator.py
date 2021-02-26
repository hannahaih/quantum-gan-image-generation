import tensorflow as tf

def discriminator_network(num_nodes_1, num_nodes_2, alpha_1=0.01, alpha_2=0.01):
    """
    Constructs a three-layer classical discriminator in a sequential model using keras
    Args:
        num_nodes_1 (int): # of nodes in the first layer
        num_nodes_2 (int): # of nodes in the second layer
        alpha_1 (float): the slope for values lower than the threshold in the activation function
        alpha_2 (float): the slope for values lower than the threshold in the activation function
    Returns:
        a 
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes_1, activation=tf.keras.layers.LeakyReLU(alpha=alpha_1)),
        tf.keras.layers.Dense(num_nodes_2, activation=tf.keras.layers.LeakyReLU(alpha=alpha_2)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])