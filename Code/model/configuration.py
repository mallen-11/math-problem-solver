def configurations(embedding_dim=45, units=32, top_k=45, X_train, features_shape=2048, attention_features_shape):
    embedding_dim = embedding_dim
    units = units
    vocab_size = top_k + 1
    num_steps = len(img_train) // 1
    features_shape = features_shape
    attention_features_shape = attention_features_shape
