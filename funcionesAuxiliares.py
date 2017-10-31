import pickle
import numpy as np
from matplotlib import pyplot as plt
import math
"""

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

"""

def readData(file):

    with open(file, mode='rb') as f:
        data = pickle.load(f)

    X_file, y_file = data['features'], data['labels']

    return X_file, y_file

# display an image
def display(img):
    
    # (784) => (28x28)
    one_image = img.reshape(28,28)
    plt.axis('off')
    plt.imshow(one_image, cmap='binary')

# Strutified shuffle is used insted of simple shuffle in order to achieve sample balancing
    # or equal number of examples in each of 10 classes.
# Since there are different number of examples for each 10 classes in the MNIST data you may
    # also use simple shuffle.
def stratified_shuffle(labels, num_classes):
    ix = np.argsort(labels).reshape((num_classes,-1))
    for i in range(len(ix)):
        np.random.shuffle(ix[i])
    return ix.T.reshape((-1))


def activation_vector(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


#Function used to plot 16 images in a 4x4 grid, and writing the true and predicted classes below each image.
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 16
    img_shape = (28, 28)
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(4, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "REAL: {0}".format(cls_true[i])
        else:
            xlabel = "Des: {0}, Calc: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

#Function for plotting examples of images from the test-set that have been mis-classified.
def plot_example_errors(cls_pred, correct,images, labels_flat):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = labels_flat[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:16],
                cls_true=cls_true[0:16],
                cls_pred=cls_pred[0:16])

def plot_confusion_matrix(cls_pred,cls_true,num_classes):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted classifications for the test-set

    # cls_true is an array of the true classifications for the test-set
    
    
    # Get the confusion matrix using sklearn.   
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    #print(cm)

    # Plot the confusion matrix as an image.
    #plt.matshow(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusion")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes),rotation=45)
    plt.yticks(tick_marks, range(num_classes))
    

    thresh = cm.max() / 2.
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")


    plt.xlabel('Predecida')
    plt.ylabel('Deseada')

def plot_conv_weights(w, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.
    
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    #w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]
    xs = num_filters/8
    ys = 8
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(xs, ys)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            import itertools
            for ii, jj in itertools.product(range(w.shape[0]), range(w.shape[1])):
                ax.text(jj, ii, round(img[ii, jj],1),size=3, va='center', ha="center", color="blue")
            
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='sinc',#'nearest' | sinc
                      cmap='RdGy'# cmap = 'seismic | binary'
                      )
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
