from keras import regularizers, layers, models, callbacks
import keras
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_models(timesteps, n_features, latent_dim=16):

    logger.info(
        f"Building models - Timesteps: {timesteps}, Features: {n_features}, Latent dim: {latent_dim}"
    )

    # ========= ENCODER =========

    encoder_inputs = keras.Input(shape=(timesteps, n_features))

    x = layers.GaussianNoise(0.1)(encoder_inputs)

    # Feature extraction (temporal local patterns)
    x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(x)

    x = layers.GaussianDropout(0.2)(x)

    # Sequence modeling
    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.GRU(latent_dim)(x)

    latent = layers.Dense(latent_dim, name="latent_vector")(x)

    encoder = keras.Model(encoder_inputs, latent, name="encoder")

    logger.info(f"Encoder built - Params: {encoder.count_params():,}")

    # ========= DECODER =========

    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.RepeatVector(timesteps)(latent_inputs)

    x = layers.GRU(latent_dim, return_sequences=True)(x)
    x = layers.GaussianDropout(0.2)(x)
    x = layers.GRU(32, return_sequences=True)(x)

    # Reconstruction head
    x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(x)

    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    logger.info(f"Decoder built - Params: {decoder.count_params():,}")

    # ========= AUTOENCODER =========

    autoencoder_outputs = decoder(encoder(encoder_inputs))

    autoencoder = keras.Model(encoder_inputs, autoencoder_outputs, name="autoencoder")

    autoencoder.compile(optimizer="adam", loss="mse")

    logger.info(f"Autoencoder compiled - Total params: {autoencoder.count_params():,}")

    return encoder, decoder, autoencoder
