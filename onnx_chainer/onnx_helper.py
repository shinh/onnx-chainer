import collections
import onnx


__func_name = None
__func_to_id = collections.defaultdict(int)


def set_func_name(func_name):
    """Set the name of Chainer function being converted.

    Args:
      func_name (str): The name of Chainer function.
    """
    global __func_name
    __func_name = func_name


def gensym():
    """Returns a unique symbol.

    Returns:
      A unique string symbol.
    """
    assert __func_name is not None
    __func_to_id[__func_name] += 1
    return 'tmp{}_{}'.format(__func_name, __func_to_id[__func_name])


class GraphBuilder(object):
    """A helper class to build consecutive ONNX nodes."""

    def __init__(self):
        self._nodes = []

    def op(self, op_name, input_names, num_outputs=1, **kwargs):
        """Creates a new ONNX node and returns its outputs.

        Args:
          op_name (str): The name of an ONNX op.
          input_names (list of str): The names of input values.
          num_outputs (int): The number of output values.
          **kwargs (dict): ONNX attributes of the node.

        Returns:
          A str of the output name when `num_outputs` is 1.
          A tuple of str of the output names otherwise.
        """
        output_names = [gensym() for i in range(num_outputs)]
        return self.op_output_named(
            op_name, input_names, output_names, **kwargs)

    def op_output_named(
            self, op_name, input_names, output_names, **kwargs):
        """Creates a new ONNX node with output names, and returns its outputs.

        Args:
          op_name (str): The name of an ONNX op.
          input_names (list of str): The names of input values.
          output_names (int of str): The names of output values.
          **kwargs (dict): ONNX attributes of the node.

        Returns:
          A str of the output name when number of output is 1.
          A tuple of str of the output names otherwise.
        """
        # Prevent a common mistake. `input_names="input"` creates a
        # node with 5 inputs.
        assert not isinstance(input_names, str)
        node = onnx.helper.make_node(
            op_name, input_names, output_names, **kwargs)
        self._nodes.append(node)
        if len(output_names) == 1:
            return node.output[0]
        else:
            return tuple(node.output)

    def const(self, array):
        """Creates a Constant node of ONNX.

        Args:
          array (numpy.ndarray): A numpy array.

        Returns:
          A str of the name of the constant value.
        """
        tensor = onnx.numpy_helper.from_array(array)
        return self.op('Constant', [], 1, value=tensor)

    def nodes(self, output_names=None):
        """Returns all nodes created so far.

        Args:
          output_names (list of str): The names of output values to be set at
            the last node.

        Returns:
          A list of `onnx.NodeProto` objects, suitable as the return
          value of converter functions.
        """
        if output_names is not None:
            assert len(self._nodes[-1].output) == len(output_names)
            self._nodes[-1].output[:] = output_names
        return tuple(self._nodes)


def write_tensor_pb(filename, name, value):
    with open(filename, 'wb') as f:
        t = onnx.numpy_helper.from_array(value, name)
        f.write(t.SerializeToString())


def cleanse_param_name(name):
    """Converts Chainer parameter names to ONNX names.

    Note ONNX identifiers must be a valid C identifier.

    Args:
      name (str): A Chainer parameter name (e.g., /l/W).

    Returns
      A valid ONNX name (e.g., param_l_W).
    """
    return 'param' + name.replace('/', '_')
