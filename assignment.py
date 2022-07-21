from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data

class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying MNIST with
    batched learning. Please implement the TODOs for the entire
    model but do not change the method and constructor arguments.
    Make sure that your Model class works with multiple batch
    sizes. Additionally, please exclusively use NumPy and
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = 784 # Size of image vectors
        self.num_classes = 10 # Number of classes/possible labels
        self.batch_size = 100
        self.learning_rate = 0.5

        # TODO: Initialize weights and biases
        self.W = np.random.normal(size = (self.num_classes, self.input_size), scale = 0.01)
        self.b = np.random.normal(size = (self.num_classes, ), scale = 0.01)

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 784) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        # TODO: Calculate, then return, the probability for each class per image using the Softmax equation

        logits = np.matmul(inputs, np.transpose(self.W))
        
        logits = logits + self.b
        
        raised = np.exp(logits)    
        
        denom = np.sum(raised, axis = 1, keepdims = True)
        
        probabilities = np.divide(raised, denom)

        return probabilities






    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be decreasing with every training loop (step).
        NOTE: This function is not actually used for gradient descent
        in this assignment, but is a sanity check to make sure model
        is learning.
        :param probabilities: matrix that contains the probabilities
        per class per image of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        # TODO: calculate average cross entropy loss for a batch

        tot_loss = 0

        for image_num in range(len(probabilities)):
            image_vector = probabilities[image_num]
            tot_loss -= np.log(image_vector[labels[image_num]])

        return (tot_loss / self.batch_size)



    def back_propagation(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases
        after one forward pass and loss calculation. The learning
        algorithm for updating weights and biases mentioned in
        class works for one image, but because we are looking at
        batch_size number of images at each step, you should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss
        
        labelsArray = np.zeros((self.batch_size, self.num_classes))
        
        for im_num in range(self.batch_size):
            index = labels[im_num]
            labelsArray[im_num, int(index)] = 1
    
        prob_diff = labelsArray - probabilities
        
        transposed = np.transpose(prob_diff)

        grad = np.matmul(transposed, inputs)
        
        w_grad = grad  / self.batch_size
        
        b_grad = np.sum(prob_diff / self.batch_size, axis = 0)

        return w_grad, b_grad



    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        :param probabilities: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy

        totalRight = 0

        for x in range(len(probabilities)):
            output = np.argmax(probabilities[x])
            if output == labels[x]:
                totalRight += 1.0

        return totalRight / len(probabilities)


    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        # TODO: change the weights and biases of the model to descent the gradient
        self.W = self.W + self.learning_rate * gradW
        self.b = self.b + self.learning_rate * gradB


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''

    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    # TODO: For every batch, compute then descend the gradients for the model's weights
    batchCounter = 0
    numberBatches = 0
    num_inputs = len(train_inputs)
    bSize = model.batch_size
    
    
    if num_inputs % bSize == 0:
        numberBatches = num_inputs / bSize
    else:
        numberBatches = num_inputs / bSize + 1

    while batchCounter < numberBatches:
        batch = []
        labels = [] 
        if (batchCounter == numberBatches - 1):
            batch = train_inputs
            labels = train_labels
        else:
            batch = train_inputs[:bSize]
            labels = train_labels[:bSize]
            train_inputs = train_inputs[bSize:]
            train_labels = train_labels[bSize:]
            
        

        probabilities = model.call(batch)
        gradW, gradB = model.back_propagation(batch, probabilities, labels)
        model.gradient_descent(gradW, gradB)

        batchCounter += 1


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment,
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """

    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set

    probabilities = model.call(test_inputs)
    return model.accuracy(probabilities, test_labels)


def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def main():
    '''
    Read in MNIST data, initialize your model, and train and test your model
    for one epoch. The number of training steps should be your the number of
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    :return: None
    '''

    # TODO: load MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels
    trainInput, trainLabel = get_data("MNIST_data/train-images-idx3-ubyte.gz", "MNIST_data/train-labels-idx1-ubyte.gz", 60000)
    testInput, testLabel = get_data("MNIST_data/t10k-images-idx3-ubyte.gz", "MNIST_data/t10k-labels-idx1-ubyte.gz", 10000)
    

    # TODO: Create Model
    model = Model()

    # TODO: Train model by calling train() ONCE on all data
    train(model, trainInput, trainLabel)

    # TODO: Test the accuracy by calling test() after running train()
    test(model, testInput, testLabel)

    # TODO: Visualize the data by using visualize_results()
    visualize_results(testInput[:5], model.call(testInput[:5]), testLabel[:5])


if __name__ == '__main__':
    main()
