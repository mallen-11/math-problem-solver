def configurations(embedding_dim=45, units=32, top_k=45, X_train, X_val, features_shape=2048, attention_features_shape):
    embedding_dim = embedding_dim
    units = units
    vocab_size = top_k + 1
    num_steps = len(X_train) // 1
    val_num_steps = len(X_val) // 1
    features_shape = features_shape
    attention_features_shape = attention_features_shape

    return embedding_dim, units, vocab_size, num_steps, val_num_steps, features_shape, attention_features_shape
