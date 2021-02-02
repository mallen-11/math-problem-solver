import tensorflow as tf
import numpy as np

def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def label_tokenizer(y, top_k=45)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters=' ',
                                                  char_level=True)
    tokenizer.fit_on_texts(y)
    train_seqs = tokenizer.texts_to_sequences(y)


    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    train_seqs = tokenizer.texts_to_sequences(y)

    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    max_length = calc_max_length(train_seqs)

    return tokenizer, cap_vector

