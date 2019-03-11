## Follow me project

---

### 1- Introduction:

The model that is used for identifying the target person in the 
quadcopter's input images is a Fully Convolutional Network (FCN). 
An FCN is a type of Convolutional Neural Network (CNN) where the last 
fully connected layer is another 1x1 convolutional layer. This additional
layer has the benefit of adding locational information about the objects
in the scene which is instructive for the quadcopter.

The FCN process flow is understood diagramatically below in Image 1. 
The FCN is split into two parts (convolution or encoder) part which will 
extract features from the image and (deconvolution or decoder) part which
upscales the output of the encoder to the original image size.


#### Image 1- Fully Convolutional Network (FCN)
![Image 1](./Photos/FCN.png)


In summary an FCN is consisting of the following components:

* **Encoder blocks**: these essentially perform a variety of functions including object recognition and downsampling of the input data

* **1x1 Convolution block**: that will reduce the channels of an image and capture the global context of the scene whilst reducing computational cost down the line.

* **Decoder blocks**: this takes the encoded blocks as input and up-samples. By adding previous encoder blocks through skip connections it is possible to recover some of the lost information which improves resolution for segmentation.

* **Softmax activation**: is a  convolution layer takes outputs from the previous decoder block. This indicates the class and location of objects for the output image in a process called semantic segmentation.


### 2- the Network architecture:

![Image 2](./Photos/NetworkArchitecture.jpg)



The architecture of the project FCN is structured in the same manor as Image 1, with:

- 5 encoders. Each with 8^2n filters
  - Number of strides = 2

- A 1x1 convolutional layer with 512 filters. Here the 1x1 convolutionl layer has been optmised to double the number of filters which increases the overall final grade result.

- 5 decoders, each with 2 seperable convolutional layers

- A skip connection setup as in Image 2 above


## Project Code:

Following sections will list all used layers along with its python code:

#### Separable convolution layer:

Separable convolution layers in the encoder will each include padding. This includes batch normalization with the ReLU activation function as shown in below code:

```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

#### Regular Conv layer:

Regular convolution block is used for 1x1 convulation with batch normalization and Relu activation.

```python
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

#### Bilinear Upsampling layer

This is used in the upsampling stage. A value of 2 has been chosen for this project.

```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```

#### Encoder Blocks

5 Encoder blocks are used, each encoder block is consisting of one separable convolution layer that is having batch normalization and ReLU activation function.

```python
def encoder_block(input_layer, filters, strides):
    
    # Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```

#### Decoder Blocks

5 decoder blocks are used, each decoder block is consisting of Upsampler to collect input from a previous layer with smaller size, a concatenate function to add upsampled layer to the input of decoder then pass the resulting output to two layers of separable conv+batch normalization+ReLU activation function.

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # Upsample the small input layer using the bilinear_upsample() function.
    upsampled_small_ip_layer = bilinear_upsample(small_ip_layer)
    
    # Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([upsampled_small_ip_layer, large_ip_layer])
    
    # Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm( output_layer, filters, strides=1)
    output_layer = separable_conv2d_batchnorm( output_layer, filters, strides=1)
    
    return output_layer
```

#### Softmax activation

Last layer in FCN is regular convolution layer with softmax activation and same padding:

```python
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(layer07)
```
    
#### the FCN model:

Below is the FCN code which combines all functions above

```python
def fcn_model(inputs, num_classes):

    # TODO Add Encoder Blocks.
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    enc1 = encoder_block(inputs, 16, 2)
    enc2 = encoder_block(enc1, 32, 2)
    enc3 = encoder_block(enc2, 64, 2)
    enc4 = encoder_block(enc3, 128, 2)
    enc5 = encoder_block(enc4, 256, 2)
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    con1 = conv2d_batchnorm(enc5, 512, 1, 1)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    dec1 = decoder_block(con1, enc4, 256)
    dec2 = decoder_block(dec1, enc3, 128)
    dec3 = decoder_block(dec2, enc2, 64)
    dec4 = decoder_block(dec3, enc1, 32)
    x    = decoder_block(dec4, inputs, 16)

    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```


### 4- HyperParameter(HP) choosing:

Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

[link](https://www.coursera.org/learn/deep-neural-network)


"the code in the mentionned cells is merged together. below is the same as the default cell provided by udacity what I did is, I merged them in one cell and added some variable to decided which HPs is the best 


1- Choose a range for each of the hyper parameters

   generate_rand_hp(lr_a = -1, lr_b = -3, bs_a = 5, bs_b = 7, num_data=4131)


2- Generate random values for each of the parameters in the ranges defined.

   rand_hp = generate_rand_hp()

3- For the predefined number of iterations, run the parameter values and calculate the final grade
   score.

   # And the final grade score is 
   final_score = final_IoU * weight

4- If the score improves, then save the new parameters. If not continue to the next iteration.

    if final_score > best_final_score:
        best_final_score = final_score

        best_hp['learning_rate']    = learning_rate
        best_hp['batch_size']       = batch_size 
        best_hp['num_epochs']       = num_epochs
        best_hp['steps_per_epoch']  = steps_per_epoch
        best_hp['validation_steps'] = validation_steps
        best_hp['workers']          = workers
        
        print("a new better parameter : {}".format(best_hp))
        
        # Save your trained model weights
        weight_file_name = 'model_weights'
        model_tools.save_network(model, weight_file_name)



### 5- Data for training

I used the default data provided by udacity.

### 6- Final results

the video below show the final results 

[Final Results](https://www.youtube.com/watch?v=QlZK7eJRojE)

### 7-Future Enhacement

1- in the training curve the final epoch's traning loss = 0.0115 and validation loss = 0.0341 overfitting like ==> adding more data can improve this

2- trying one of the existed architecture (for example [LINK](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review) )

3- the video show that the drone follow the hero for the whole recording time but it takes sometimes >= 5 min to catch him, I think doing the previous two steps can improve this.

I don't think this model will works well for following small object (e.g. cats, dogs) specially from long distance and some angles, but it should works well (maybe not the same final score as this) for larger object (e.g. car, horses)

