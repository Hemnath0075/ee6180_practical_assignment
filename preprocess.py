import tensorflow as tf
import os

def load_image(image_path, image_size):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1  # Normalize to [-1, 1]
    img = tf.image.resize(img, [image_size, image_size])
    return img

def load_image_pair(image_path, image_size):
    # Assume dataset has paired images side-by-side (input | target)
    img = load_image(image_path, image_size)
    w = tf.shape(img)[1]
    w = w // 2
    input_image = img[:, :w, :]
    target_image = img[:, w:, :]
    return input_image, target_image

def augment(input_image, target_image):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)
    return input_image, target_image

def load_dataset(dataset_path, batch_size, image_size):
    # Assume dataset_path contains images with paired input and target
    image_paths = tf.data.Dataset.list_files(os.path.join(dataset_path, '*.jpg'))
    
    def process_path(file_path):
        input_image, target_image = load_image_pair(file_path, image_size)
        input_image, target_image = augment(input_image, target_image)
        return input_image, target_image

    dataset = image_paths.map(
        process_path, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Split into train and test (e.g., 80% train, 20% test)
    dataset_size = len(list(image_paths))
    train_size = int(0.8 * dataset_size)
    train_dataset = dataset.take(train_size).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset