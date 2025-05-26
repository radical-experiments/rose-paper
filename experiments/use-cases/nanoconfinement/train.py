import os
import random
import argparse
from datetime import datetime
from pprint import pprint
import numpy as np
import tensorflow as tf
import joblib
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

NUM_OF_BINS = 502

def preprocess_inputdata(all_data):
    input_data, output, errors, z_data = [], [], [], []
    for key, data in all_data.items():
        paras = key.split("_")
        input_names, input_paras = paras[0::2], paras[1::2]
        density_profiles = [data['pos'][:,1], data['neg'][:,1]]
        density_errors   = [data['pos'][:,2], data['neg'][:,2]]
        z_values         = [data['pos'][:,0], data['neg'][:,0]]
        input_data.append(input_paras)
        output.append(density_profiles)
        errors.append(density_errors)
        z_data.append(z_values)

    input_arr = np.array(input_data)
    output_arr = np.array(output).reshape(-1, NUM_OF_BINS*2)
    errors_arr = np.array(errors).reshape(-1, NUM_OF_BINS*2)
    z_arr      = np.array(z_data).reshape(-1, NUM_OF_BINS*2)

    print(f"Input data shape: {input_arr.shape}")
    print(f"Output data shape: {output_arr.shape}")
    print(f"Error data shape: {errors_arr.shape}")
    print(f"Bin center data shape: {z_arr.shape}")
    return input_arr, output_arr, errors_arr, z_arr

def main():
    now = datetime.now()
    print("Start training at ", now.strftime("%Y-%m-%d %H:%M:%S"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter",            required=True, type=int, help='The iteraion number of this execution (zero-index)')
    parser.add_argument("--instance",        required=True, type=int, help='The instance id of this training within each pipeline (zero-index)')
    parser.add_argument('--data_dir',        required=True, help='Directory containing .pk files')
    parser.add_argument('--pipeline_dir',    required=True, help='Directory of this specific pipeline')
    parser.add_argument('--split_fraction',  type=float, default=0.8, help='Train/validation split fraction')
    parser.add_argument('--seed',            type=int, default=1, help='Random seed')
    parser.add_argument('--batch_size',      type=int, default=32, help='Batch size')
    parser.add_argument('--epochs',          type=int, default=200, help='Number of epochs')
    args = parser.parse_args()
    print("Doing training with args = \n")
    pprint(vars(args))

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load data dumps
    train_pk = os.path.join(args.data_dir, 'data_dump_density_preprocessed_train.pk')
    test_pk  = os.path.join(args.data_dir, 'data_dump_density_preprocessed_test.pk')
    with open(train_pk, 'rb') as f:
        raw_train = pickle.load(f)
    with open(test_pk, 'rb') as f:
        raw_test  = pickle.load(f)

    index_file = os.path.join(args.pipeline_dir, "index.npy")
    include_idx = np.load(index_file).astype(int)
    print("include_idx = \n", include_idx)

    keys = list(raw_train.keys())
    train_data = { keys[i]: raw_train[keys[i]] for i in include_idx }

    # Preprocess
    x_all, y_all, err_all, z_all = preprocess_inputdata(train_data)
    x_test, y_test, err_test, z_test = preprocess_inputdata(raw_test)

    # Shuffle and split
    x_shuf, y_shuf, err_shuf, z_shuf = shuffle(x_all, y_all, err_all, z_all, random_state=args.seed)
    split_idx = int(x_shuf.shape[0] * args.split_fraction)
    x_train, x_val = x_shuf[:split_idx], x_shuf[split_idx:]
    y_train, y_val = y_shuf[:split_idx], y_shuf[split_idx:]

    print(f"Train input:  {x_train.shape}")
    print(f"Train output: {y_train.shape}")
    print(f"Val input:    {x_val.shape}")
    print(f"Val output:   {y_val.shape}")

    # Load and apply scaler
    scaler_path = os.path.join(args.data_dir, 'scaler_new.pkl')
    scaler = joblib.load(scaler_path)
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled   = scaler.transform(x_val)

    # Model hyperparameters
    input_features = x_train.shape[1]
    output_classes = NUM_OF_BINS * 2
    hidden1, hidden2 = 256, 512
    dropout_rate = 0.1
    lr, beta1, beta2, decay = 1e-4, 0.9, 0.999, 0.0

    # Build model
    initializer = tf.keras.initializers.GlorotNormal()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden1, activation='relu', kernel_initializer=initializer, input_shape=(input_features,)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(hidden2, activation='sigmoid', kernel_initializer=initializer),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(output_classes, activation='relu', kernel_initializer=initializer)
    ])

    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2, decay=decay)
    )

    # Train
    history = model.fit(
        x_train_scaled, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_val_scaled, y_val),
        shuffle=True,
        verbose=1
    )

    # Save model
    model_file = os.path.join(args.pipeline_dir, "model", f'my_model_iter_{args.iter}_instance_{args.instance}.h5')
    model.save(model_file)
    print(f"Model saved to {model_file}")

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.yscale('log')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    loss_file = os.path.join(args.pipeline_dir, "loss", f'loss_iter_{args.iter}_instance_{args.instance}.png')
    plt.savefig(loss_file)
    print(f"Loss curve saved to {loss_file}")

    # Load and show summary of saved model
    new_model = tf.keras.models.load_model(model_file, compile=False)
    new_model.summary()

    now = datetime.now()
    print("Ending training at ", now.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == '__main__':
    np.set_printoptions(threshold=5000)
    main()

