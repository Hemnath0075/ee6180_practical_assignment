import tensorflow as tf
import argparse
import os
from model import Pix2Pix
from preprocess import load_dataset
from utils import save_images, create_output_dir

def parse_args():
    parser = argparse.ArgumentParser(description='Train Pix2Pix model.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset directory.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--image_size', type=int, default=256, help='Image size (height and width).')
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='L1 loss weight.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory
    create_output_dir(args.output_dir)

    # Load and preprocess dataset
    train_dataset, test_dataset = load_dataset(
        args.dataset_path, args.batch_size, args.image_size
    )

    # Initialize model
    pix2pix = Pix2Pix(image_size=args.image_size, lambda_l1=args.lambda_l1)

    # Compile model
    pix2pix.compile()

    # Training loop
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        for step, (input_image, target) in enumerate(train_dataset):
            pix2pix.train_step(input_image, target)

            if step % 100 == 0:
                print(f'Step {step}')

        # Generate and save sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            for test_input, test_target in test_dataset.take(1):
                generated = pix2pix.generator(test_input, training=False)
                save_images(test_input, test_target, generated, args.output_dir, epoch + 1)

    # Save final model
    pix2pix.generator.save(os.path.join(args.output_dir, 'generator'))
    pix2pix.discriminator.save(os.path.join(args.output_dir, 'discriminator'))

if __name__ == '__main__':
    main()