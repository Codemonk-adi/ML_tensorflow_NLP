import tensorflow as tf
import numpy as np
import os
import time
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
# print(sys.stdout.encoding)
data_folder = Path("C:\\Users\\Aaditya\\.keras\\datasets")
file_path = data_folder / "IDGAH.txt"
file = open(file_path, encoding="utf-8")
text = file.read()
# print(len(text))
# print(text[:250])

vocab = sorted(set(text))

# print(len(vocab))

char2indx = {u:i for i,u in enumerate(vocab)}
indx2char = np.array(vocab)

text_as_int = np.array([char2indx[c] for c in text])
# print(char2indx['ौ'])
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# for i in char_dataset.take(5):
#   print(indx2char[i.numpy()])
  
sequences = char_dataset.batch(seq_length+1 , drop_remainder=True)
# for item in sequences.take(3):
#   print(repr(''.join(indx2char[item.numpy()])))
@tf.autograph.experimental.do_not_convert
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# for input_example, target_example in  dataset.take(1):
#   print ('Input data: ', repr(''.join(indx2char[input_example.numpy()])))
#   print ('Target data:', repr(''.join(indx2char[target_example.numpy()])))

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
for num, _ in enumerate(dataset):
    pass

print(f'Number of elements: {num}')
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
#   print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

# print(example_batch_predictions[0])

# sampled_indices = tf.random.categorical(example_batch_predictions[0],num_samples=1)
# sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

# print(repr(''.join(indx2char[sampled_indices])))

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# example_batch_loss = loss(target_example_batch, example_batch_predictions)
# print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
# print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 10

# history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2indx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  
  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 0.4

  # Here batch size == 1
  model.reset_states()
  for _ in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(indx2char[predicted_id])

  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"आज "))
# print(char2indx[model(4)])