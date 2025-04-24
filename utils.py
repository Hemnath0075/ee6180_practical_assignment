import tensorflow as tf
import os
import matplotlib.pyplot as plt

def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def save_images(input_image, target_image, generated_image, output_dir, epoch):
    def denormalize(img):
        return (img + 1) * 127.5  # Convert [-1, 1] to [0, 255]

    plt.figure(figsize=(15, 5))
    
    images = [input_image[0], target_image[0], generated_image[0]]
    titles = ['Input', 'Target', 'Generated']
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, i + 1)
        img = denormalize(img)
        img = tf.clip_by_value(img, 0, 255)
        img = tf.cast(img, tf.uint8)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch}.png'))
    plt.close()