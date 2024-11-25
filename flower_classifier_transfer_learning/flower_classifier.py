import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def download_model():
    """
    Downloads a pretrained MobileNetV2 network from the Tensorflow library
    It also calls the freeze_layers function to freeze their weights

    @return:
        The pre-trained MobileNetV2 model with frozen layers
    """
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet')

    return freeze_layers(base_model)


def freeze_layers(model):
    """
    Freezes all layers in the given model to avoid changing the information they contain

    @param model:
        The model to be frozen

    @return:
        The given model with all layers frozen
    """
    
    for layer in model.layers: layer.trainable = False

    return model 
    

def replace_last_layer(model):
    """
    Adds new trainable layers on top of the base model

    @param model:
        The base model

    @return:
        A new model with additional layers
    """
    
    x = model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(5, activation='softmax')(x)
    new_model = tf.keras.models.Model(inputs=model.input, outputs=predictions)
    
    return new_model


def prepare_non_accelerated_datasets():
    """
    Prepares the train, validation, and test datasets for the non-accelerated version of transfer learning

    @return:
        A tuple containing the train, validation, and test datasets
    """
    
    data_dir = 'small_flower_dataset'
    batch_size = 32
    img_size = (224, 224)

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.36,  # 36% used as validation data
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    )

    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
                                                      
    return train_generator, validation_generator, test_generator


def prepare_accelerated_datasets(model):
    """
    Prepares the train, validation, and test datasets for the accelerated version of transfer learning

    @param model:
        The model to be used to predict the datasets

    @return:
        A tuple containing the train, validation, and test datasets along with their respective labels
    """
    
    train_generator, val_generator, test_generator = prepare_non_accelerated_datasets()

    return collect_predictions(train_generator, model), collect_predictions(val_generator, model), collect_predictions(test_generator, model)


def collect_predictions(generator, model):

    img_size = (224, 224)
    acc_dataset = []
    labels = []
    
    for _ in range(generator.__len__()):
        batch = next(generator)
        images, batch_labels = batch
        resized_images = tf.image.resize(images, img_size)
        processed_images = tf.keras.applications.mobilenet_v2.preprocess_input(resized_images)
        features = model.predict(processed_images)
        acc_dataset.append(features)
        labels.append(batch_labels)
    acc_dataset = np.concatenate(acc_dataset, axis=0)
    labels = np.concatenate(labels, axis=0)

    return acc_dataset, labels
    

def compile_and_train(model, train_data, val_data, train_labels=None, 
                        val_labels=None, learning_rate=0.01,
                        momentum=0.0, nesterov=False):
    """
    Compiles and trains the model on the given datasets

    @param model: 
        The model to be compiled and trained

    @param train_data: 
        The training data

    @param val_data: 
        The validation data

    @param train_labels: 
        The labels for the training dataset

    @param val_labels: 
        The labels for the validation dataset

    @param learning_rate:
        The learning rate for the optimizer

    @param momentum:
        The momentum for the optimizer

    @param nesterov:
        Whether to use Nesterov momentum or not

    @return: 
        A tuple containing the trained model and the training history
    """

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    epochs = 10
    batch_size = 32

    if (train_labels is None and val_labels is None): 
        history = model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data
        )
    else:
        history = model.fit(
            train_data, 
            train_labels,
            validation_data=(val_data, val_labels),
            batch_size=batch_size,
            epochs=epochs
    )   

    return model, history
    
    
def split_model(model):
    """
    Splits a model into a base model and a head model

    @param model:
        The model to be split

    @return:
        The base model and the head model
    """
    
    base_model = model.layers[-4].output
    base_model = tf.keras.models.Model(inputs=model.input, outputs=base_model)
    
    new_input = tf.keras.Input(shape=(7, 7, 1280))
    x = model.layers[-3](new_input)
    x = model.layers[-2](x)
    predictions = model.layers[-1](x)
    head_model = tf.keras.models.Model(inputs=new_input, outputs=predictions)
    
    return base_model, head_model


def plot_results(history):
    """
    Plot the training and validation accuracies and errors over the epochs

    @param history:
    the training history of the model
    """
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy and Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
    

def standard_learning(learning_rate=0.01, momentum=0.0):
    """
    Downloads a model, replace it's last layer, prepares non-accelerated datasets, 
    trains the new model on the training data, validates it on the validation data,
    and plots the training history.

    @param learning_rate (default 0.01):
    The learning rate used in training the model

    @param momentum (default 0.0):
    The momentum parameter used in training the model
    """

    model = download_model()
    new_model = replace_last_layer(model)
    train_data, val_data, test_data = prepare_non_accelerated_datasets()
    history = compile_and_train(new_model, train_data, val_data, learning_rate=learning_rate, momentum=momentum, nesterov=False)[1]
    plot_results(history)


def accelerated_learning(learning_rate=0.01, momentum=0.0):
    """
    Similar to standard_learning but uses accelerated datasets and 
    involves splitting the new model into a base model and a head model

    @param learning_rate (default 0.01):
    The learning rate used in training the model

    @param momentum (default 0.0):
    The momentum parameter used in training the model
    """
    
    model = download_model()
    new_model = replace_last_layer(model)
    base_model, head_model = split_model(new_model)
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = prepare_accelerated_datasets(base_model)
    history = compile_and_train(head_model, train_data, val_data, train_labels, val_labels, learning_rate, momentum, False)[1]
    plot_results(history) 
    

def testing_lr():
    """
    Test the standard_learning function with different learning rates 
    """

    learning_rates = [0.1, 0.001, 0.01]
    for lr in learning_rates:
        standard_learning(learning_rate=lr)
        

def testing_momentums():
    """
    Test the standard_learning function with different momentum values. 
    """

    momentums = [0.1, 0.5, 0.9]
    for momentum in momentums:
        standard_learning(learning_rate=0.1, momentum=momentum)


def testing_accelerated_momentums():
    """
    Test the accelerated_learning function with different momentum values. 
    """

    momentums = [0.1, 0.5, 0.9]
    for momentum in momentums:
        accelerated_learning(learning_rate=0.1, momentum=momentum)


if __name__ == "__main__":

    standard_learning()
    
    # Testing learning rates with non-accelerated transfer learning
    testing_lr()

    # Testing momentums with non-accelerated transfer learning
    testing_momentums()

    # Testing learning rates with accelerated transfer learning
    testing_accelerated_momentums()
    