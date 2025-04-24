import tensorflow as tf
from tensorflow.keras import layers, Model

class Pix2Pix:
    def __init__(self, image_size=256, lambda_l1=100.0):
        self.image_size = image_size
        self.lambda_l1 = lambda_l1
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def build_generator(self):
        inputs = layers.Input(shape=[self.image_size, self.image_size, 3])

        # Encoder
        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            self.downsample(128, 4),  # (bs, 64, 64, 128)
            self.downsample(256, 4),  # (bs, 32, 32, 256)
            self.downsample(512, 4),  # (bs, 16, 16, 512)
            self.downsample(512, 4),  # (bs, 8, 8, 512)
            self.downsample(512, 4),  # (bs, 4, 4, 512)
            self.downsample(512, 4),  # (bs, 2, 2, 512)
            self.downsample(512, 4),  # (bs, 1, 1, 512)
        ]

        # Decoder
        up_stack = [
            self.upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 512)
            self.upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 512)
            self.upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 512)
            self.upsample(512, 4),  # (bs, 16, 16, 512)
            self.upsample(256, 4),  # (bs, 32, 32, 256)
            self.upsample(128, 4),  # (bs, 64, 64, 128)
            self.upsample(64, 4),   # (bs, 128, 128, 64)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(
            3, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh'
        )  # (bs, 256, 256, 3)

        x = inputs
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = layers.Concatenate()([x, skip])

        x = last(x)
        return Model(inputs=inputs, outputs=x)

    def build_discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)
        inp = layers.Input(shape=[self.image_size, self.image_size, 3], name='input_image')
        tar = layers.Input(shape=[self.image_size, self.image_size, 3], name='target_image')
        x = layers.Concatenate()([inp, tar])  # (bs, 256, 256, 6)

        x = self.downsample(64, 4, apply_batchnorm=False)(x)  # (bs, 128, 128, 64)
        x = self.downsample(128, 4)(x)  # (bs, 64, 64, 128)
        x = self.downsample(256, 4)(x)  # (bs, 32, 32, 256)

        x = layers.ZeroPadding2D()(x)  # (bs, 34, 34, 256)
        x = layers.Conv2D(
            512, 4, strides=1, kernel_initializer=initializer, use_bias=False
        )(x)  # (bs, 31, 31, 512)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.ZeroPadding2D()(x)  # (bs, 33, 33, 512)
        x = layers.Conv2D(
            1, 4, strides=1, kernel_initializer=initializer
        )(x)  # (bs, 30, 30, 1)

        return Model(inputs=[inp, tar], outputs=x)

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            layers.Conv2D(
                filters, size, strides=2, padding='same',
                kernel_initializer=initializer, use_bias=False
            )
        )
        if apply_batchnorm:
            result.add(layers.BatchNormalization())
        result.add(layers.LeakyReLU())
        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            layers.Conv2DTranspose(
                filters, size, strides=2, padding='same',
                kernel_initializer=initializer, use_bias=False
            )
        )
        result.add(layers.BatchNormalization())
        if apply_dropout:
            result.add(layers.Dropout(0.5))
        result.add(layers.ReLU())
        return result

    def compile(self):
        self.generator.compile()
        self.discriminator.compile()

    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
                disc_generated_output, gen_output, target
            )
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        gen_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables
        )
        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )

        return gen_total_loss, disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.bce_loss_fn(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.lambda_l1 * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.bce_loss_fn(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.bce_loss_fn(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss