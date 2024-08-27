import tensorflow as tf
from tensorflow import keras
from keras_tuner import HyperModel
from keras_tuner.tuners import BayesianOptimization
from keras_tuner import Objective
import keras.backend as K
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

train_df = pd.read_csv("clean_df.csv")
df_x = train_df.drop(columns=['class_e', 'class_p'])
df_y = train_df['class_e']

train_x, val_x, train_y, val_y = train_test_split(df_x, df_y, test_size=0.01, random_state=16)


def mcc(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn) - (fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


class MyHyperModel(HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(train_x.shape[1],)))

        for i in range(hp.Int('num_layers', 2, 15)):
            units = hp.Choice(f'units_{i}', values=[2**n for n in range(5, 11)])
            model.add(keras.layers.Dense(units=units, activation=None))
            
            batch_norm = hp.Boolean(f'batch_norm_{i}')
            momentum = hp.Float(f'batch_norm_momentum_{i}', 0.85, 0.99, step=0.01)
            if batch_norm:
                model.add(keras.layers.BatchNormalization(momentum=momentum))
                
            model.add(keras.layers.Activation('relu'))

            dropout = hp.Boolean(f'dropout_{i}')
            dropout_rate = hp.Float(f'dropout_rate_{i}', min_value=0, max_value=0.5, step=0.05)
            if dropout:
                model.add(keras.layers.Dropout(rate=dropout_rate))

        model.add(keras.layers.Dense(1, activation='sigmoid'))

        learning_rate = hp.Float('learning_rate', min_value=0.00005, max_value=0.0015, sampling='log')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[mcc]
        )

        return model

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

tuner = BayesianOptimization(
    MyHyperModel(),
    objective=Objective("val_mcc", direction="max"),
    max_trials=20,
    executions_per_trial=2,
    directory='kaggle',
    project_name='s04e08'
)


with tf.device('/GPU:0'):
    tuner.search(
        train_x, train_y,
        epochs=50,
        validation_data=(val_x, val_y),
        batch_size=256,
        callbacks=[early_stopping]
    )

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Hyperparameters: {best_hps.values}")

best_model = tuner.get_best_models(num_models=1)[0]
best_model.save('best_model.h5')
best_model.save('tf_model')