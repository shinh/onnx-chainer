from __future__ import print_function

import chainer

import onnx

from onnx_chainer.onnx_chainer2 import tracker as tracker_lib
from onnx_chainer.onnx_generator import graph_builder
from onnx_chainer.onnx_generator import mapping


def _extract_value_info(arr, name):
    if isinstance(arr, list):
        assert arr
        assert not isinstance(arr[0], list)
        value_info_proto = onnx.ValueInfoProto()
        value_info_proto.name = name
        sequence_type_proto = value_info_proto.type.sequence_type
        nested = _extract_value_info(arr[0], name)
        tensor_type = sequence_type_proto.elem_type.tensor_type
        tensor_type.CopyFrom(nested.type.tensor_type)
        return value_info_proto
    else:
        return onnx.helper.make_tensor_value_info(
            name=name,
            elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype],
            shape=arr.shape)


class NameGenerator(object):
    def __init__(self):
        self._name_count = {}

    def generate(self, base):
        if base in self._name_count:
            name = '%s_%d' % (base, self._name_count[base])
            self._name_count[base] += 1
            return name
        else:
            self._name_count[base] = 1
            return base


def _real_arrays(a):
    if isinstance(a, (list, tuple)):
        return type(a)(_real_arrays(x) for x in a)
    elif isinstance(a, dict):
        return {k: _real_arrays(x) for k, x in a.items()}
    elif hasattr(a, '_real_array'):
        return _real_arrays(a._real_array())
    elif isinstance(a, chainer.Variable):
        return _real_arrays(a.array)
    return a


class Node(object):
    def __init__(self, name, receiver, args, kwargs, result):
        self.name = name
        self.receiver = _real_arrays(receiver)
        self.args = _real_arrays(args)
        self.kwargs = _real_arrays(kwargs)
        self.result = _real_arrays(result)
        self.func = getattr(receiver, name)
        # When the `receiver` is a bound method.
        if hasattr(self.func, '__func__'):
            self.args.insert(0, _real_arrays(self.func.__self__))
            self.func = self.func.__func__

    def inputs(self):
        r = self.args + list(self.kwargs.values())
        return r

    def outputs(self):
        if isinstance(self.result, (list, tuple)):
            return list(self.result)
        else:
            return [self.result]


def export(model, args, graph_name, opset_version):
    if not isinstance(args, (list, tuple)):
        args = [args]

    tracker = tracker_lib.Tracker()
    with tracker:
        tracked_input = args
        tracker.wrap_model(model)
        tracked_output = model(*[tracker_lib._wrap_array(a) for a in args])
    tracked_output = _real_arrays(tracked_output)
    if not isinstance(tracked_output, (list, tuple)):
        tracked_output = [tracked_output]

    all_nodes = []
    for receiver, name, args, kwargs, result in tracker.get_records():
        all_nodes.append(Node(name, receiver, args, kwargs, result))

    values = {}
    users_map = {}
    producers_map = {}
    for node in all_nodes:
        for v in node.inputs():
            values[id(v)] = v
            if id(v) not in users_map:
                users_map[id(v)] = set()
            users_map[id(v)].add(node)

        for v in node.outputs():
            values[id(v)] = v
            # TODO(hamaji): Should be only for whitelisted values.
            if id(v) in producers_map:
                continue
            producers_map[id(v)] = node

    # TODO(hamaji): Think again about how we track variables.
    tracked_input = _real_arrays(tracked_input)
    input_values = [values[id(i)] for i in tracked_input]
    input_value_ids = {id(i) for i in input_values}
    output_values = [values[id(o)] for o in tracked_output]
    q = list(output_values)
    nodes = set()
    extra_inputs = []
    while q:
        v = q.pop()
        if id(v) not in producers_map:
            if id(v) not in input_value_ids:
                extra_inputs.append(v)
            continue

        n = producers_map[id(v)]
        nodes.add(n)
        for nv in n.inputs():
            q.append(nv)

    sorted_nodes = [n for n in all_nodes if n in nodes]

    if True:
        for node in sorted_nodes:
            inputs = []
            for v in node.inputs():
                inputs.append('%s(%x)' % (type(v), id(v)))
            outputs = []
            for v in node.outputs():
                outputs.append('%s(%x)' % (type(v), id(v)))
            print('%s(%s) => (%s)' %
                  (node.name, ', '.join(inputs), ', '.join(outputs)))

    name_gen = NameGenerator()
    gb = graph_builder.GraphBuilder(graph_name, opset_version)

    value_names = {}
    for i, input_value in enumerate(input_values):
        name = name_gen.generate('Input')
        value_names[id(input_value)] = gb.input(name, input_value)

    for i, output_value in enumerate(output_values):
        name = name_gen.generate('Output')
        value_names[id(output_value)] = gb.output(name, output_value)

    for name, param in model.namedparams():
        name = name.lower().replace('/', '_')
        value_names[id(param.array)] = gb.param(name, param.array)

    def get_name(value, node):
        name = value_names.get(id(value))
        if name is None:
            name = name_gen.generate(node.name)
            value_names[id(value)] = name
        return name

    convert = mapping.get_converter()

    for node in sorted_nodes:
        xnode = convert(gb, node.func, node.receiver, node.args, node.kwargs)

        # Adjust the name of outputs.
        assert len(xnode.output) == len(node.outputs())
        for i, ov in enumerate(node.outputs()):
            if id(ov) in value_names:
                xnode.output[i] = value_names[id(ov)]
            else:
                gb.add_value(xnode.output[i], ov)

    xgraph = gb.make_graph()

    xmodel = onnx.helper.make_model(
        xgraph,
        producer_name='Chainer',
        producer_version=chainer.__version__,
        opset_imports=[onnx.helper.make_opsetid('', opset_version)]
    )
    print(xmodel)

    return xmodel
