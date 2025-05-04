import ast
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tcn import TCN

import pandas as pd


def build_tcn_model(nb_filters, kernel_size, dilations, nb_stacks, dropout_rate):
    input_layer = Input(shape=(X.shape[1], X.shape[2]))
    tcn_layer = TCN(nb_filters=nb_filters, kernel_size=kernel_size, dilations=dilations,
                    nb_stacks=nb_stacks, dropout_rate=dropout_rate)(input_layer)
    output_layer = Dense(1, activation='linear')(tcn_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


train_files = [
    '/kaggle/input/dataset/TCN/train_features_w_150000_s_150000.npz',
    '/kaggle/input/dataset/TCN/train_features_w_15000_s_15000.npz',
    '/kaggle/input/dataset/TCN/train_features_w_15000_s_7500.npz',
    '/kaggle/input/dataset/TCN/train_features_w_1500_s_1500.npz',
    '/kaggle/input/dataset/TCN/train_features_w_1500_s_750.npz'
]


d = {}

for train_file in train_files:
    tmp = train_file.split('train_features_')[1].split('.')[0]
    w = tmp.split('_s_')[0][2:]
    s = tmp.split('_s_')[1]

    data = np.load(train_file)
    X, y = data['X'], data['y']
    print(X.shape, y.shape, train_file)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    param_grid = {
        'nb_filters': [32, 64],
        'kernel_size': [3],
        'dilations': [[1, 3], [1, 3, 9], [1, 3, 9, 27]],
        'nb_stacks': [1, 2],
        'dropout_rate': [0.2, 0.5],
    }

    d[(w, s)] = {}

    for nb_filters in param_grid['nb_filters']:
        for kernel_size in param_grid['kernel_size']:
            for dilations in param_grid['dilations']:
                for nb_stacks in param_grid['nb_stacks']:
                    for dropout_rate in param_grid['dropout_rate']:

                        print(
                            f"nb_filters={nb_filters}, kernel_size={kernel_size}, dilations={dilations}, nb_stacks={nb_stacks}, dropout={dropout_rate}, ", end='')
                        print(
                            f"Reception field = {1 + 2 * (kernel_size - 1) * nb_stacks * sum(dilations)} ", end='')

                        model_path = f"/kaggle/working/{w}_{s}_{nb_filters}_{kernel_size}_{dilations}_{nb_stacks}_{dropout_rate}.weights.h5"

                        checkpoint_callback = ModelCheckpoint(
                            model_path,
                            monitor="val_mae",
                            save_weights_only=True,
                            save_best_only=True,
                            mode="min",
                        )

                        model = build_tcn_model(
                            nb_filters, kernel_size, dilations, nb_stacks, dropout_rate)
                        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(
                            X_val, y_val), callbacks=[checkpoint_callback])

                        model.load_weights(model_path)
                        val_loss, val_mae = model.evaluate(
                            X_val, y_val, verbose=0)
                        print(f"Validation MAE: {val_mae:.4f}")

                        d[(w, s)][(nb_filters, kernel_size, str(dilations),
                                   nb_stacks, dropout_rate)] = (val_loss, val_mae)


for k, v in d.items():
    print(k)
    v = sorted(list(v.items()), key=lambda x: x[1])[:5]
    for line in v:
        print(line)

if not os.path.exists('/kaggle/working/predictions'):
    os.mkdir('/kaggle/working/predictions')


for (w, s), v in d.items():
    test_file = f'/kaggle/input/dataset/Transformer/test_features_w_{w}_s_{s}.npz'
    data = np.load(test_file)
    X, id = data['X'], data['seg_ids']
    num_features = X.shape[2]
    seq_len = X.shape[1]

    for k, _ in sorted(list(v.items()), key=lambda x: x[1])[:5]:
        nb_filters, kernel_size, dilations, nb_stacks, dropout_rate = k
        dilations = ast.literal_eval(dilations)

        model_path = f"/kaggle/working/{w}_{s}_{nb_filters}_{kernel_size}_{dilations}_{nb_stacks}_{dropout_rate}.weights.h5"

        model = build_tcn_model(nb_filters, kernel_size,
                                dilations, nb_stacks, dropout_rate)
        model.load_weights(model_path)

        y = model.predict(X, verbose=2)

        df = pd.DataFrame()
        df['seg_id'] = id
        df['time_to_failure'] = np.squeeze(y, 1)
        df.to_csv(
            f'/kaggle/working/predictions/{w}_{s}_{nb_filters}_{kernel_size}_{dilations}_{nb_stacks}_{dropout_rate}.csv', index=False)
