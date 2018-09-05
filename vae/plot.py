import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import os
import matplotlib.cm as cm
import keras

class Plot(keras.callbacks.Callback):
    def __init__(self, test_data, test_target, encoder, decoder, model_name='AE', full_model=None, plots=True):
        self.test_target = test_target
        self.encoder = encoder
        self.decoder = decoder
        self.epoch = 0
        self.model_name = model_name
        self.plots = plots
        self.test_data = test_data
        if model_name == 'VAE':
            self.full_model = full_model
        elif self.model_name == 'COND':
            self.test_data , self.constraint = self.test_data 
        if len(self.test_target.shape) > 1:
            self.test_labels = np.argmax(self.test_target, axis=1)
        else:
            self.test_labels = self.test_target

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch

    def _plot_reconstruction(self, data, save_figure=False, batch=0):
        n_images = 6
        f, axs = plt.subplots(2, n_images, figsize=(16, 5), num=1)
        for i in range(n_images):
            axs[0, i].imshow(data[i].reshape(28, 28))
            axs[1, i].imshow(self.test_data[i].reshape(28, 28))
        if save_figure:
            plt.savefig('images/%sbatch-%d_ecpoch-%d_reconstruction.png' % (self.model_name, batch, self.epoch))
        
    def _plot_pca_ica(self, data, save_figure=False, batch=0):
        from sklearn.decomposition import PCA, FastICA

        pca = PCA(2)
        projected_PCA = pca.fit_transform(data)
        fig = plt.figure(num=2, figsize=(16, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(projected_PCA[:, 0], projected_PCA[:, 1],
                       c=self.test_labels, edgecolor='none', alpha=0.5,
                       cmap=plt.cm.get_cmap('nipy_spectral', 10))
        ax1.set_title('PCA Scatter Plot')

        ica = FastICA(2, max_iter=500)
        projected_ICA = ica.fit_transform(data)
        ax2 = fig.add_subplot(1, 2, 2)
        ax = ax2.scatter(projected_ICA[:, 0], projected_ICA[:, 1],
                    c=self.test_labels, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('nipy_spectral', 10))
        fig.colorbar(ax)
        ax2.set_title('ICA Scatter Plot')

        if save_figure:
            plt.savefig('images/%sbatch-%d_ecpoch-%d_PCA_ICA.png' % (self.model_name, batch, self.epoch))

    def _plot_lattent(self, data, save_figure=False, batch=0):
        if data.shape[-1] == 2:
            plt.figure(figsize = (17, 5), num=3)
            plt.scatter(data[:, 0], data[:, 1],
                        c=self.test_labels, edgecolor='none', alpha=0.5,
                        cmap=plt.cm.get_cmap('nipy_spectral', 10))
            
            plt.colorbar()
            plt.title('%s Scatter Plot' % self.model_name)
            if save_figure:
                plt.savefig('images/%sbatch-%d_ecpoch-%d_lattent.png' % (self.model_name, batch, self.epoch))

    
    def on_batch_end(self, batch, logs=None):
        PLOT_PERIOD = 30
        save_figure = True if batch % (2 * PLOT_PERIOD) == 0 else False
        # if batch % PLOT_PERIOD == 0 and self.plots:
        if False:
            clear_output(wait=True)
            
            if self.model_name == 'VAE':
                z_mean, _, _ = self.encoder.predict(self.test_data)
                decoded_imgs = self.full_model.predict(self.test_data[:100])
                self._plot_lattent(z_mean, save_figure=save_figure, batch=batch)
                self._plot_reconstruction(decoded_imgs, save_figure=save_figure, batch=batch)
            elif self.model_name == 'COND':
                encoded_imgs = self.encoder.predict([self.test_data, self.constraint])
                decoded_imgs = self.decoder.predict(np.concatenate((encoded_imgs, self.constraint), axis=-1))
                self._plot_lattent(encoded_imgs, save_figure=save_figure, batch=batch)
                self._plot_reconstruction(decoded_imgs, save_figure=save_figure, batch=batch)
            else:
                encoded_imgs = self.encoder.predict(self.test_data)
                decoded_imgs = self.decoder.predict(encoded_imgs)
                self._plot_lattent(encoded_imgs, save_figure=save_figure, batch=batch)
                self._plot_reconstruction(decoded_imgs, save_figure=save_figure, batch=batch)
            plt.show()

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(20, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 20
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()
