# Face_Recognizer


Certainly! Here is the complete YAML code for your README.md file:

```yaml
# Face Recognition Project

## Overview

This project implements a Convolutional Neural Network (CNN) for face recognition using TensorFlow and Keras. The model is trained on a dataset located at `/kaggle/input/face-recognition`. The dataset is preprocessed and augmented using image data generators.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- scikit-learn

## Installation

```bash
pip install tensorflow numpy scikit-learn
```

## Usage

1. Set the path to your dataset:

    ```python
    dataset_path = "/kaggle/input/face-recognition"
    ```

2. Define image size and batch size:

    ```python
    img_size = (250, 250)
    batch_size = 32
    ```

3. Data preprocessing and augmentation:

    ```python
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    ```

4. Create train and test generators:

    ```python
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    test_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    ```

5. Define the CNN model:

    ```python
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 3)))
    # ... (add more layers as needed)
    ```

6. Compile the model:

    ```python
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ```

7. Train the model:

    ```python
    epochs = 1
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator
    )
    ```

8. Evaluate the model on the test set:

    ```python
    test_loss, test_acc = model.evaluate(test_generator, verbose=2)
    print(f"\nTest accuracy: {test_acc}")
    ```

9. Make predictions on the test set:

    ```python
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_labels = test_generator.classes
    ```

## Results

- Training accuracy: _(add training accuracy from history)_
- Validation accuracy: _(add validation accuracy from history)_
- Test accuracy: _(display test accuracy)_

## Author

- Your Name

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
```

Feel free to customize it further based on your specific project details.
