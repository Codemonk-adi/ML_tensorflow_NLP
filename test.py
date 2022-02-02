import tensorflow as tf 
import numpy as np 
# samples has shape [1, 5], where each value is either 0 or 1 with equal
# probability.
array = np.array([[0.2,0.6,0.2],[0.2,0.6,0.2],[0.2,0.6,0.2]])
samples = tf.random.categorical(tf.math.log(array), num_samples=1)
test = samples[1,0].numpy()
print(samples)
print(test)