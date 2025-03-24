# Image-to-Image Translation Training and Inference

This project allows training and testing various image-to-image translation models like CycleGAN, Pix2Pix, and Colorization.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt  # Install necessary packages
```

## Training the Model

To train a model, run the following command:

```bash
python train.py --dataroot ./datasets/<dataset_name> --name <experiment_name> --model <model_name>
```

#### Example to train the model

```
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --gpu_ids 0,1,2
```

### Training Examples:

* **Train a CycleGAN model:**
  ```bash
  python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
  ```
* **Train a Pix2Pix model:**
  ```bash
  python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
  ```

### Training Options:

* `--dataroot` : Path to dataset.
* `--name` : Name of the experiment (used for saving logs and models).
* `--model` : Choose from `cycle_gan`, `pix2pix`, etc.
* `--continue_train` : Resume training from the last checkpoint.

## Testing the Model

To test a trained model:

```bash
python test.py --dataroot ./datasets/<dataset_name> --name <experiment_name> --model <model_name>
```

Example:

```bash
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```

## Inference and Results

### Where to Find Trained Models:

Trained models are saved in:

```
checkpoints/<experiment_name>/
```

This folder contains:

* `latest_net_G.pth` : Latest generator model.
* `latest_net_D.pth` : Latest discriminator model (if applicable).
* Epoch-specific checkpoints.

### Where to Find Output Images:

After testing, generated images are saved in:

```
results/<experiment_name>/
```

This folder contains output images and their corresponding inputs.

## Notes

* Training takes significant time, depending on dataset size and model type.
* Use `--continue_train` to resume from the latest checkpoint.
* Customize training settings in `options/train_options.py` and `options/base_options.py`.

## References

For more details, check the official documentation:
[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

---

Thanks and regards,
Hemnath
