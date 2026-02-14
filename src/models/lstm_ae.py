from src import layers, keras, logger

def build_models(timesteps, n_features, latent_dim=16):
    logger.info(
        f"Construction des modèles - Timesteps: {timesteps}, Features: {n_features}, Latent dim: {latent_dim}"
    )

    # ========= ENCODER =========
    encoder_inputs = keras.Input(shape=(timesteps, n_features))
    x = layers.GaussianNoise(0.1)(encoder_inputs)

    x = layers.LSTM(32, return_sequences=True)(encoder_inputs)
    x = layers.GaussianDropout(0.2)(x)
    x = layers.LSTM(latent_dim)(x)

    latent = layers.Dense(latent_dim, name="latent_vector")(x)

    encoder = keras.Model(encoder_inputs, latent, name="encoder")
    logger.info(f"Encoder construit - Paramètres: {encoder.count_params():,}")

    # ========= DECODER =========
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.RepeatVector(timesteps)(latent_inputs)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    x = layers.GaussianDropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=True)(x)

    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    decoder = keras.Model(latent_inputs, outputs, name="decoder")
    logger.info(f"Decoder construit - Paramètres: {decoder.count_params():,}")

    # ========= AUTOENCODER =========
    autoencoder_outputs = decoder(encoder(encoder_inputs))

    autoencoder = keras.Model(encoder_inputs, autoencoder_outputs, name="autoencoder")

    autoencoder.compile(optimizer="adam", loss="mse")
    logger.info(
        f"Autoencoder compilé - Total paramètres: {autoencoder.count_params():,}"
    )

    return encoder, decoder, autoencoder
