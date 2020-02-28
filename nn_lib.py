import numpy as np
import pickle
import math

def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)

class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative log-
    likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        self._cache_current = None


    def forward(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        temp = 1 / (1 + np.exp(-x))
        # the gradient of sigmoid(x)
        self._cache_current = np.multiply(temp, 1-temp)

        return temp

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z): 
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # actually, @param grad_z should be grad_loss_wrt_x
        assert grad_z.shape == self._cache_current.shape,\
             print('Wrong dimension in the sigmoid layer: grad_z is {}, \
                 grad_sigmoid is {}.'.format(grad_z.shape,self._cache_current.shape))

        # chain rule
        return np.multiply(self._cache_current,grad_z)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        self._cache_current = None


    def forward(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        x = np.maximum(0,x)

        # the gradient of RuLu(temp)
        u = np.zeros_like(x)
        u[x > 0] = 1
        self._cache_current = u

        return x

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z): # actually, @param grad_z should be grad_loss_wrt_x
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        assert grad_z.shape == self._cache_current.shape,\
             print('Wrong dimension in the relu layer: grad_z is {}, \
                 grad_relu is {}.'.format(grad_z.shape,self._cache_current.shape))

        # chain rule
        return np.multiply(self._cache_current,grad_z)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """Constructor.

        Arguments:
            n_in {int} -- Number (or dimension) of inputs.
            n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._W = xavier_init((n_in,n_out)) # n_in by n_out matrix
        self._b = xavier_init((1,self.n_out))  # bias term, shape of 1 by n_out

        # the gradients of z with respect to x, _W and _b
        self._cache_current = None
        # the gradients of loss with respect to _W and _b
        self._grad_W_current = None 
        self._grad_b_current = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns xW + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        assert x.shape[1] == self.n_in,\
             print('Wrong dimension for the linear layer. now the dim for the \
                 input is {}, n_in is {}'.format(x.shape[1],self.n_in))
        

        #batch_size = x.shape[0] # num of examples (size of the batch)

        # the affine transform z = x*_W + _b
        z = np.dot(x,self._W) + self._b # numpy array broadcasting

        # the gradients of z with respect to x, _W and _b
        # i.e. grad_z_wrt_x = _W, grad_z_wrt_W = x, grad_z_wrt_b = ones(n_out,n.out)
        self._cache_current = self._W, x, np.ones((self.n_out,self.n_out))

        return z

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # the gradients of z with respect to x, _W and _b
        # i.e. grad_z_wrt_x = _W, grad_z_wrt_W = x, grad_z_wrt_b = ones(n.in)
        grad_z_wrt_x, grad_z_wrt_W, grad_z_wrt_b = self._cache_current

        assert grad_z.shape[1] == grad_z_wrt_x.shape[1],\
             print('Wrong dimension in the lieaner layer: grad_z is {}, \
                 grad_z_wrt_x is {}.'.format(grad_z.shape,grad_z_wrt_x.shape))
        
        assert grad_z.shape[0] == grad_z_wrt_W.shape[0],\
             print('Wrong dimension in the lieaner layer: grad_z is {}, \
                 grad_z_wrt_W is {}.'.format(grad_z.shape,grad_z_wrt_W.shape))
        
        assert grad_z.shape[1] == grad_z_wrt_b.shape[0],\
             print('Wrong dimension in the lieaner layer: grad_z is {}, \
                 grad_z_wrt_b is {}.'.format(grad_z.shape,grad_z_wrt_b.shape))

        # chain rule
        self._grad_W_current = np.dot(np.transpose(grad_z_wrt_W),grad_z)
        # sum over different data points, remember that _b is a 1-d vector
        #self._grad_b_current = np.sum(np.dot(grad_z,grad_z_wrt_b),axis=0)
        self._grad_b_current = np.average(np.dot(grad_z,grad_z_wrt_b),axis=0)
        
        return np.dot(grad_z,np.transpose(grad_z_wrt_x))

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        self._W -= self._grad_W_current * learning_rate
        self._b -= self._grad_b_current * learning_rate

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """Constructor.

        Arguments:
            input_dim {int} -- Dimension of input (excluding batch dimension).
            neurons {list} -- Number of neurons in each layer represented as a
                list (the length of the list determines the number of layers).
            activations {list} -- List of the activation function to use for
                each layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # N.B. Conventinally a linear layer + a non-linear layer together forms
        # one single "layer", although the aforementioned code to some extent
        # treats them as seperate layers

        self._layers = []

        LEN = len(neurons)
        for i in range(LEN):
            # linear layer first
            if i == 0:
                # the input layer
                self._layers.append(LinearLayer(input_dim,neurons[0]))
            else:
                # hidden layer
                self._layers.append(LinearLayer(neurons[i-1],neurons[i]))
            # then the activation layer
            if activations[i] == "relu":
                self._layers.append(ReluLayer())
            elif activations[i] == "sigmoid":
                self._layers.append(SigmoidLayer())
            elif activations[i] == "identity":
                self._layers.append("identity") # dummpy layer


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for layer in self._layers:
            # the output of layer L becomes input of layer L+1
            if isinstance(layer,Layer):
                x = layer.forward(x)
            # elif isinstance(layer,str):
            #     continue
        return x

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (1,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # backprop starts from the output layer
        for layer in reversed(self._layers):
            # the gradient of loss wrt z or x in layer L becomes the 
            # input to layer L-1
            if isinstance(layer,Layer):
                grad_z = layer.backward(grad_z)

        return grad_z

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for layer in reversed(self._layers):
            if isinstance(layer,Layer): # an identity layer has no update_params()
                layer.update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """Constructor.

        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            batch_size {int} -- Training batch size.
            nb_epoch {int} -- Number of training epochs.
            learning_rate {float} -- SGD learning rate to be used in training.
            loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        if loss_fun == "mse":
            self._loss_layer = MSELossLayer()
        elif loss_fun == "cross_entropy":
            self._loss_layer = CrossEntropyLossLayer()
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).

        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # guide to shuffling via array indexing courtesy of below
        # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
        assert len(input_dataset) == len(target_dataset)
        pos = np.random.permutation(len(input_dataset))
        return input_dataset[pos], target_dataset[pos]

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, ).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        number_input_col = input_dataset.shape[0]
        numberOfBatches = math.ceil(number_input_col/self.batch_size)

        for epoch in range(self.nb_epoch):
            if self.shuffle_flag == True:
                input_dataset , target_dataset = self.shuffle(input_dataset, target_dataset)

            # Split dataset into selected size
            # Array split guide found here: https://numpy.org/doc/1.18/reference/generated/numpy.array_split.html
            input_dataset_batches = np.array_split(input_dataset, numberOfBatches, axis = 0)
            target_dataset_batches = np.array_split(target_dataset, numberOfBatches, axis = 0)

            for input_batch, output_batch in zip(input_dataset_batches, target_dataset_batches):
                fwdPredictedOutput = self.network.forward(input_batch)
                number = self._loss_layer.forward(fwdPredictedOutput, output_batch)
                gradLossWrtY = self._loss_layer.backward()
                # # Something about this, not sure right or not.
                self.network.backward(gradLossWrtY)
                self.network.update_params(self.learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, ).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        fwdPredictedOutput = self.network.forward(input_dataset)
        return self._loss_layer.forward(fwdPredictedOutput, target_dataset)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            - data {np.ndarray} dataset used to determined the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.maxValue = np.max(data, axis = 0)
        self.minValue = np.min(data, axis = 0)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            - data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        return (data - self.minValue) / (self.maxValue - self.minValue)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.

        Arguments:
            - data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        return (data * (self.maxValue - self.minValue)) + self.minValue

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]

    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")

    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)


    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()