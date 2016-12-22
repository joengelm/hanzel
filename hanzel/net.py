import tensorflow as tf

class Net():
    def __init__(self, config):
        """
        Instantiate a new Neural Net.

        Example usage:
            config = { ... }
            with Net() as net:
                net.train(train_X, train_Y)
                net.test(test_X, test_Y)

        Another example usage:
            config = { ... }
            net = Net(config)
            net.train(train_X, train_Y)
            net.test(test_X, test_Y)
            net.close()
        """
        self.config = Configuration(config)
        self.session = tf.Session()

        # Setup placeholders
        self.setup()

        # Build the logits
        self.logits = self.inference()

        # Build the loss computation
        self.loss_val = self.loss()

        # Build the training operation
        self.train_op = self.optimize()

        # Setup saver to persist model state when `save` is called (or to restore a saved state)
        self.saver = tf.train.Saver()

        # Initialize all variables for the operations described above
        self.session.run(tf.initialize_all_variables())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.session.close()

    def setup(self):
        """
        Instantiate the network variables for all layers.
        """
        raise NotImplementedError('Abstract method `setup` called. You must implement `setup` in subclass.')

    def inference(self):
        """
        Build the inference computation graph for the model, going from the input to the output logits.
        """
        raise NotImplementedError('Abstract method `inference` called. You must implement `inference` in subclass.')

    def loss(self):
        """
        Build the loss computation graph.
        """
        raise NotImplementedError('Abstract method `loss` called. You must implement `loss` in subclass.')

    def optimize(self):
        """
        Build the training operation, using the loss and an optimizer. Override this in subclass if desired.
        """
        return tf.train.AdamOptimizer(self.config['learning_rate']).minimize(self.loss_val)

    def attach_session(self, session):
        """
        Give this model a TensorFlow session so that it may train itself.
        """
        self.session = session

    def train(self, X, Y):
        """
        Train this model on some (batched) inputs and expected outputs.

        This model must have a session attached in order for this method to work.

        `X` and `Y` represent the inputs and expected outputs respectively.
        """
        raise NotImplementedError('Abstract method `train` called. You must implement `train` in subclass.')

    def test(self, x, y):
        """
        Test this model on some input and expected output.
        
        This model must have a session attached in order for this method to work.

        `x` and `y` represent the input and expected output to test.
        """
        raise NotImplementedError('Abstract method `test` called. You must implement `test` in subclass.')

    def sample(self, x):
        """
        Return a probability distribution with the probabilities for y in Y, given an x from X.
        """
        raise NotImplementedError('Abstract method `sample` called. You must implement `sample` in subclass.')

    def save(self, path):
        """
        Save this entire model in a checkpoint file, to be restored later.

        `path` is the location at which to save the checkpoint file 
        """
        return self.saver.save(self.session, path)

    def restore(self, path):
        """
        Restore a previously saved model from a checkpoint file.

        `path` is the location of the saved checkpoint file
        """
        self.saver.restore(self.session, path)

    def close(self):
        """
        Release any resources being used by the model (i.e. close the session)
        """
        self.session.close()

    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)


class Configuration(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)
