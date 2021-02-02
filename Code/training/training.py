import tensorflow as tf
import numpy as np
import pandas as pd
import time

@tf.function
def train_step(img_tensor, target, decoder, encoder):
    loss = 0

    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['\t']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

def start_training(EPOCHS=10, dataset):
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 1 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss / int(target.shape[1])))

        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save() 
        print(num_steps)
        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
