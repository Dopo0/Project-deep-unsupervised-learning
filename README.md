# Project deep unsupervised learning
This project is an evaluate generation images from noise of three different methods
....

# Table of contents
1. GMM
2. AE
3. GAN
   1. [First dense model](#First dense model)
   2. [Second dense model](#Second dense model)
   3. [CNN model](#CNN model)
   4. [CNN model for discriminator and dense model for the generator](#CNN model for discriminator and dense model for the generator)


## GAN


### First dense model

    

The first linear model has three hidden layers with a progressively increasing number of neurons: 256, 512, and 1024. It uses LeakyReLU activation function, which allows for a small, non-zero gradient when the unit is not active. This model does not employ dropout, which means it may be more prone to overfitting compared to a model that does. The output is generated through a single linear layer with Sigmoid activation, which is typical for binary classification. Due to the larger hidden layers, it has more parameters, making it more complex.

for more details , you can access the notebook named "GAN_G_Dense_D_Dense_1.ipynb"
<div align="center">
    <img src="GAN/results_GAN/First_dense_model_50_epochs.png" alt="Team Photo" width="150" height="160">
    <img src="GAN/results_GAN/First_dense_model_100_epochs.png" alt="Team Photo" width="150" height="160">
    <img src="GAN/results_GAN/First_dense_model_150_epochs.png" alt="Team Photo" width="150" height="160">
    <br>
    <em>Figure: Images show the results of the model when we train it for 50 epochs , 100 epochs and 150 epochs which will be our final generation.</em>
</div>


### Second dense model

The second linear model also has three hidden layers but with decreasing sizes: 256, 128, and 64. It uses the same LeakyReLU activation function but incorporates dropout with a probability of 50%, introducing regularization to help prevent overfitting. It similarly concludes with a single linear layer with Sigmoid activation for the output. This model is less complex due to smaller hidden layers, meaning it has fewer parameters to learn.

for more details , you can access the notebook named "GAN_G_Dense_D_Dense_2.ipynb"
<div align="center">
  <img src="GAN/results_GAN/Second_dense_model_150_epochs.png" alt="Results of Second Dense GAN Model after 150 Epochs" width="250" height="260">
  <br>
  <em>Figure: Results of the Second Dense GAN Model on Fashion MNIST dataset after 150 epochs</em>
</div>

### CNN model for discriminator and dense model for the generator

The generator is designed with a noise input dimension of 100, an initial dense layer, and a combination of upsampling and convolutional layers that expand the noise into a 2D structure. The use of ReLU and Tanh activation functions facilitates the learning of complex patterns. This model is more sophisticated due to convolutional and upsampling layers, allowing it to create finer details, which can be crucial for generating realistic Fashion MNIST images.

For the discriminator, the model processes input images through multiple convolutional layers with LeakyReLU activations and employs dropout to prevent overfitting. The complexity is higher due to more channels in convolutions, which might provide better feature extraction for classifying real versus generated images.

The model with a convolutional generator yielded better results due to a more balanced power dynamic between the discriminator and the generator. Both components of the GAN would be more evenly matched in their capacity to analyze and generate images, leading to more coherent and recognizable outputs. This highlights the importance of designing both the generator and discriminator in a GAN with compatible complexities to achieve high-quality generation of images.

for more details , you can access the notebook named "GAN_G_CNN_D_CNN.ipynb"
<div align="center">
  <img src="GAN/results_GAN/GAN_CNN.png" alt="Results of Second Dense GAN Model after 150 Epochs" width="250" height="260">
  <br>
  <em>Figure: Results of the CNN model for discriminator and dense model for the generator on Fashion MNIST dataset after 150 epochs</em>
</div>

### CNN model for discriminator and dense model for the generator 

The generator also starts with a noise input dimension of 100 and an initial linear layer, but it lacks the convolutional and upsampling complexity of the first, relying solely on linear layers. It uses LeakyReLU and Tanh activations, and its simplicity with only linear layers means it’s likely faster to train but may produce less detailed images.

The discriminator model has a similar structure but with fewer channels, making it less complex and possibly quicker to train. Reduced convolution channels might limit its feature detection capability but improve generalization due to a simpler model. Both use Sigmoid in the output layer for binary classification.

The results are evidently not desirable as the images consist of unrecognizable, noisy patterns. This indicates that the linear generator in the GAN was unable to capture and reproduce the complexities of the data it was trained on, which in this case would be Fashion MNIST images.

The poor performance could be due to the imbalance between the capabilities of the generator and the discriminator. The discriminator, being a convolutional neural network (CNN), likely has a far greater ability to distinguish between real and fake images than the linear generator has to generate plausible images. Thus, the generator fails to create convincing images to fool the discriminator, resulting in it not learning meaningful features.

for more details , you can access the notebook named "GAN_G_Dense_D_CNN.ipynb"

<div align="center">
  <img src="GAN/results_GAN/Second_CNN_Dense_model_150_epochs.png" alt="Results of Second Dense GAN Model after 150 Epochs" width="250" height="260">
  <br>
  <em>Figure: Results of the CNN model for discriminator and dense model for the generator on Fashion MNIST dataset after 150 epochs </em>
</div>



