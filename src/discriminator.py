import tensorflow as tf

def discriminator_network(num_nodes_1, num_nodes_2, alpha_1=0.01, alpha_2=0.01):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes_1, activation=tf.keras.layers.LeakyReLU(alpha=alpha_1)),
        tf.keras.layers.Dense(num_nodes_2, activation=tf.keras.layers.LeakyReLU(alpha=alpha_2)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])