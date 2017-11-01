import numpy as np
from keras import backend as K


class BaseAttributionMethod(object):
    def __init__(self, batch_size=1, input_batch_size=None):
        self.model = None
        self.batch_size = batch_size
        self.input_batch_size = batch_size
        self.name = 'BaseClass'
        self.eval_x = None
        self.eval_y = None
        self.target_input_idx = None
        self.target_output_idx = None

    """
    Bind model. If target input and output are not first and last layers
    of the network, target layer indeces should also be provided.
    """
    def bind_model(self, model):
        self.model = model
        print ('Target output: %s' % self.model.layers[-1].output)

    def gradient_override_map(self):
        return {}

    def sanity_check(self, eval_x, eval_y, maps):
        pass

    def target_input(self):
        return self.model.layers[0].input

    def target_output(self):
        return self.model.layers[-1].output

    def get_numeric_sensitivity(self, eval_x, eval_y, shape=None, **kwargs):
        """
        Return sensitivity map as numpy array
        :param eval_x: numpy input to the model
        :param eval_y: numpy labels
        :param shape: if provided, the sensitivity will be reshaped to this shape
        :return: numpy array with shape of eval_x
        """
        self.eval_x = eval_x
        self.eval_y = eval_y
        sensitivity = None
        shape = K.int_shape(self.target_input())[1:]

        #y_ = Input(batch_shape=(self.batch_size,) + tuple(eval_y.shape[1:]))  # placeholder for labels
        print ('eval_y shape: ', eval_y.shape)
        y_ = K.placeholder(shape=(None, ) + tuple(eval_y.shape[1:]), name='y_label')  # placeholder for labels
        symbolic_sensitivity = self.get_symbolic_sensitivity(y_)
        evaluate = K.function([self.model.inputs[0], y_], [symbolic_sensitivity])

        for i in range(int(len(eval_x) / self.batch_size) + 1):
            x = eval_x[self.batch_size*i:self.batch_size*(i+1)]
            y = eval_y[self.batch_size*i:self.batch_size*(i+1)]
            self.input_batch_size = len(x)
            if self.input_batch_size > 0:
                tmp = evaluate([x, y])[0]
                if sensitivity is None:
                    sensitivity = tmp
                else:
                    sensitivity = np.append(sensitivity, tmp, axis=0)

        if shape is not None:
            sensitivity = sensitivity.reshape((self.eval_x.shape[0],) + shape)

        # Run sanity check if available
        self.sanity_check(eval_x, eval_y, sensitivity)
        return sensitivity

    def get_symbolic_sensitivity(self, eval_y):
        """
        Return sensitivity map as numpy array
        :param eval_y: placeholder (Tensor) for labels
        :return: Tensor with shape of eval_x
        """
        raise RuntimeError('Calling get_symbolic_sensitivity on abstract class')



