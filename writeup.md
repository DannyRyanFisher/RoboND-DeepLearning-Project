## Follow me project

---

[Model Weights](https://github.com/JafarAbdi/RoboND-DeepLearning-Project/blob/master/data/weights/model_weights)

[Jupyter Notebook](https://github.com/JafarAbdi/RoboND-DeepLearning-Project/blob/master/code/model_training.ipynb)

[//]: # (Image References)

[NetworkImg]: ./img/Network.png
[Conv1x1Img]: ./img/con1x1.png

### 1- Introduction:

In this project we asked to build a convolutional neural network (CNN) model to enable the drone to follow a deired target (here the "hero")
given some data to train our model

### 2- the Network architecture:

![NetworkImg]

the architecture of my model is as follow ::

1- five encoders with # filters = 8*2^n where n is the nth encoder (16, 32, 64, 128, 256) and strides = 2

2- 1x1 Convolutional layer with 512 filters

3- five decoders each decoder has 2 separable convolution layers

4- add skip connection as shown in the Network architecture image above 

```python
def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
    
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    small_in_upsampled = bilinear_upsample(small_ip_layer)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output_layer1 = layers.concatenate([small_in_upsampled, large_ip_layer])
    # TODO Add some number of separable convolution layers
    output_layer2 = separable_conv2d_batchnorm(output_layer1, filters)
    output_layer3 = separable_conv2d_batchnorm(output_layer2, filters)
    return output_layer3
    
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
### 3- 1x1 Convolutional layer

![Conv1x1Img]

the 1x1 Convolutional layer is used to change the number of filters with kernel and stride = 1, 
I used the 1x1 Convolutional layer to increase the number of filters from 256 to 512 so the dimension of the layer become (256, 5, 5) ==> 
(512, 5, 5). (despite it's used in general to reduce the number of filters, I increased it because it gives better final results)



### 4- HyperParameter(HP) choosing:

I used the method described in the third week "Hyperparameter tuning" section of prof. Andrew Ng's new Deep Learning course [Link](https://www.youtube.com/watch?v=QlZK7eJRojE), 
the method is as follow:

1- choose the range for each HP

2- generate random value in this range using the appropriate scale (for learning_rate I used logarithmic scale)

```python
# to generate random Hyper-Parameters (random grid method)
def generate_rand_hp(lr_a = -1, lr_b = -3, bs_a = 5, bs_b = 7, num_data=4131):
    
    # generate random number between [a, b)
    r = (lr_b - lr_a) * np.random.rand() + lr_a
    
    # logarithmic scale 
    learning_rate = 10 ** r
    
    # batch_size
    batch_size = 2 ** np.random.randint(bs_a, bs_b)
    # the loss curve was saturating after 10-25 epochs
    num_epochs = 25
    steps_per_epoch = (4131 // batch_size) # * 3
    validation_steps = 50
    workers = 4
    
    return learning_rate, batch_size, num_epochs, steps_per_epoch, validation_steps, workers
```

3- see the final results and decide which HP to use

this's how I selected my HPs

"the code below is the same as the default cell provided by udacity what I did is, I merged them in one cell and added some variable to decided which HPs is the best 

```python

best_hp = {'learning_rate': 0., 
           'batch_size': 0, 
           'num_epochs': 0,
           'steps_per_epoch': 100, 
           'validation_steps': 50, 
           'workers': 4}

best_final_score = 0.

num_itr = 2

for itr in range(num_itr):
    rand_hp = generate_rand_hp()
    learning_rate, batch_size, num_epochs, steps_per_epoch, validation_steps, workers = rand_hp
    
    print("the randomly generated Hyper-Parameters : ", rand_hp)
    
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    # Define the Keras model and compile it for training
    model = models.Model(inputs=inputs, outputs=output_layer)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')

    # Data iterators for loading the training and validation data
    train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                   data_folder=os.path.join('..', 'data', 'train'),
                                                   image_shape=image_shape,
                                                   shift_aug=True)

    val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                 data_folder=os.path.join('..', 'data', 'validation'),
                                                 image_shape=image_shape)

    logger_cb = plotting_tools.LoggerPlotter()
    callbacks = [logger_cb]

    model.fit_generator(train_iter,
                        steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                        epochs = num_epochs, # the number of epochs to train for,
                        validation_data = val_iter, # validation iterator
                        validation_steps = validation_steps, # the number of batches to validate on
                        callbacks=callbacks,
                        workers = workers)
    
    run_num = 'run_' + str(itr)

    val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                            run_num,'patrol_with_targ', 'sample_evaluation_data') 

    val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model, 
                                            run_num,'patrol_non_targ', 'sample_evaluation_data') 

    val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                            run_num,'following_images', 'sample_evaluation_data')
    
    # images while following the target
    im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','following_images', run_num) 
    for i in range(3):
        im_tuple = plotting_tools.load_images(im_files[i])
        plotting_tools.show_images(im_tuple)
        
    # images while at patrol without target
    im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_non_targ', run_num) 
    for i in range(3):
        im_tuple = plotting_tools.load_images(im_files[i])
        plotting_tools.show_images(im_tuple)
        
       
    # images while at patrol with target
    im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_with_targ', run_num) 
    for i in range(3):
        im_tuple = plotting_tools.load_images(im_files[i])
        plotting_tools.show_images(im_tuple)

    # Scores for while the quad is following behind the target. 
    true_pos1, false_pos1, false_neg1, iou1 = scoring_utils.score_run_iou(val_following, pred_following)

    # Scores for images while the quad is on patrol and the target is not visable
    true_pos2, false_pos2, false_neg2, iou2 = scoring_utils.score_run_iou(val_no_targ, pred_no_targ)

    # This score measures how well the neural network can detect the target from far away
    true_pos3, false_pos3, false_neg3, iou3 = scoring_utils.score_run_iou(val_with_targ, pred_with_targ)

    # Sum all the true positives, etc from the three datasets to get a weight for the score
    true_pos = true_pos1 + true_pos2 + true_pos3
    false_pos = false_pos1 + false_pos2 + false_pos3
    false_neg = false_neg1 + false_neg2 + false_neg3

    weight = true_pos/(true_pos+false_neg+false_pos)
    print(weight)

    # The IoU for the dataset that never includes the hero is excluded from grading
    final_IoU = (iou1 + iou3)/2
    print(final_IoU)

    # And the final grade score is 
    final_score = final_IoU * weight
    print(final_score)
    
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
    
    print('#' * 25 + ', ' + str(itr) + ' iteration finished.')
```

### 5- Final results

the video below show the final results 

[Final Results](https://www.youtube.com/watch?v=QlZK7eJRojE)

### 6-Future Enhacement

1- in the training curve the final epoch's traning loss = 0.0115 and validation loss = 0.0341 overfitting like ==> adding more data can improve this

2- trying one of the existed architecture (for example [LINK](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review) )

3- the video show that the drone follow the hero for the whole recording time but it takes sometimes >= 5 min to catch him, I think doing the previous two steps can improve this.



