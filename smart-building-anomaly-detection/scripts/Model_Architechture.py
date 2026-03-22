import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from keras.optimizers import Adam

def build_lstm_autoencoder(window_size: int,
                           n_features: int,
                           lstm_units: list,
                           latent_dim: int,
                           decoder_units: list,
                           dropout_rate: float,
                           learning_rate: float) -> Model:
    
   # ── INPUT ────────────────────────────────────────────────────────────
    inputs = Input(shape=(window_size, n_features), name="encoder_input")

    # ── ENCODER ─────────────────────────────────────────────────────────
    # Layer 1: processes full sequence, hands every hidden state to layer 2
    x = LSTM(lstm_units[0], return_sequences=True,
             name="encoder_lstm_1")(inputs)
    x = Dropout(dropout_rate, name="encoder_drop_1")(x)

    # Layer 2: summarises the sequence into one fixed-size vector
    x = LSTM(lstm_units[1], return_sequences=False,
             name="encoder_lstm_2")(x)
    x = Dropout(dropout_rate, name="encoder_drop_2")(x)

    # Bottleneck: further compress to `latent_dim`
    latent = Dense(latent_dim, activation="relu", name="latent_vector")(x)

    # ── BRIDGE ───────────────────────────────────────────────────────────
    # RepeatVector expands the latent vector back into a sequence
    x = RepeatVector(window_size, name="repeat_vector")(latent)

    # ── DECODER ─────────────────────────────────────────────────────────
    # Layer 1: interprets the repeated latent context
    x = LSTM(decoder_units[0], return_sequences=True,
             name="decoder_lstm_1")(x)
    x = Dropout(dropout_rate, name="decoder_drop_1")(x)

    # Layer 2: refines the reconstruction
    x = LSTM(decoder_units[1], return_sequences=True,
             name="decoder_lstm_2")(x)
    x = Dropout(dropout_rate, name="decoder_drop_2")(x)

    # Output: project back to original feature space at each timestep
    outputs = TimeDistributed(Dense(n_features), name="output_layer")(x)

    # ── COMPILE ─────────────────────────────────────────────────────────
    model = Model(inputs=inputs, outputs=outputs, name="LSTM_Autoencoder")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    model.summary()
    print()
    return model