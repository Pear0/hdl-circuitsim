import math

from collections import Counter, namedtuple
from util import _signal_name_re

SimVar = namedtuple('SimVar', ('name', 'size'))


class Tiler(object):
    def __init__(self, size=1, left_shift=0):
        self.scale = size
        self.left_shift = left_shift

        self.x = 0
        self.y = 0

    def next_tile(self, height=None):
        try:
            return int((self.left_shift + self.x) * self.scale), int(self.y * self.scale)
        finally:
            if self.y >= 4:
                self.x += 1
                self.y = 0
            else:
                self.y += (height or self.scale) / self.scale


class CircuitLayoutSpec(object):
    def num_inputs(self):
        raise NotImplementedError

    def num_outputs(self):
        raise NotImplementedError

    def get_input_pos(self, pos, index):
        raise NotImplementedError

    def get_output_pos(self, pos, index):
        raise NotImplementedError

    def get_height(self):
        raise NotImplementedError

    def connect_input_direct(self, index):
        return False

    def connect_output_direct(self, index):
        return False


class DipCircuitLayoutSpec(CircuitLayoutSpec):
    def __init__(self, num_inputs, num_outputs):
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

    def num_inputs(self):
        return self._num_inputs

    def num_outputs(self):
        return self._num_outputs

    def get_input_pos(self, pos, index):
        return pos[0], pos[1] + 1 + index

    def get_output_pos(self, pos, index):
        # return pos[0], pos[1] + 1 + index + self._num_inputs
        return pos[0] + 3, pos[1] + 1 + index

    def get_height(self):
        return max(5, self.num_inputs(), self.num_outputs()) + 4


class GateCircuitLayoutSpec(CircuitLayoutSpec):
    def __init__(self, num_inputs, num_outputs):
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

        # 4x4

    def input_offset(self, num, i):
        if num == 1:
            return 0
        if num == 2:
            return [-1, 1][i]

        half = num // 2

        if num % 2 == 1:
            return i - half

        res = i - half
        if res >= 0:
            res += 1
        return res

    def num_inputs(self):
        return self._num_inputs

    def num_outputs(self):
        return self._num_outputs

    def get_input_pos(self, pos, index):
        return pos[0], pos[1] + 2 + self.input_offset(self.num_inputs(), index)

    def get_output_pos(self, pos, index):
        return pos[0] + 4, pos[1] + 2 + self.input_offset(self.num_outputs(), index)

    def get_height(self):
        return max(5, self.num_inputs()) + 1


class GateNotCircuitLayoutSpec(CircuitLayoutSpec):
    def __init__(self, num_inputs, num_outputs):
        assert num_inputs == 1
        assert num_outputs == 1

    def num_inputs(self):
        return 1

    def num_outputs(self):
        return 1

    def get_input_pos(self, pos, index):
        return pos[0], pos[1] + 1

    def get_output_pos(self, pos, index):
        return pos[0] + 3, pos[1] + 1

    def get_height(self):
        return 5


class BufferCircuitLayoutSpec(CircuitLayoutSpec):
    def __init__(self, inputs, outputs):
        assert inputs == 2
        assert outputs == 1

    def num_inputs(self):
        return 2

    def num_outputs(self):
        return 1

    def get_input_pos(self, pos, index):
        if index == 0:
            return pos[0], pos[1] + 1

        return pos[0] + 1, pos[1] + 2

    def get_output_pos(self, pos, index):
        return pos[0] + 2, pos[1] + 1

    def get_height(self):
        return 5


class DFlipFlopCircuitLayoutSpec(CircuitLayoutSpec):
    def __init__(self, inputs, outputs):
        assert inputs == 5
        assert outputs == 1

    def num_inputs(self):
        return 5

    def num_outputs(self):
        return 1

    def get_input_pos(self, pos, index):
        if index == 0:
            return pos[0], pos[1] + 1
        if index == 1:
            return pos[0], pos[1] + 3
        if index == 2:
            return pos[0] + 1, pos[1] + 4
        if index == 3:
            return pos[0] + 2, pos[1] + 4
        if index == 4:
            return pos[0] + 3, pos[1] + 4

        raise ValueError

    def connect_input_direct(self, index):
        return index in [2, 3, 4]

    def get_output_pos(self, pos, index):
        return pos[0] + 4, pos[1] + 1

    def get_height(self):
        return 7


class RegisterCircuitLayoutSpec(CircuitLayoutSpec):
    def __init__(self, inputs, outputs):
        assert inputs == 4
        assert outputs == 1

    def num_inputs(self):
        return 4

    def num_outputs(self):
        return 1

    def get_input_pos(self, pos, index):
        if index == 0:
            return pos[0], pos[1] + 2
        if index == 1:
            return pos[0], pos[1] + 3
        if index == 2:
            return pos[0] + 1, pos[1] + 4
        if index == 3:
            return pos[0] + 2, pos[1] + 4

        raise ValueError

    def connect_input_direct(self, index):
        return index in [2, 3]

    def get_output_pos(self, pos, index):
        return pos[0] + 4, pos[1] + 2

    def get_height(self):
        return 7


class SplitterCircuitLayoutSpec(CircuitLayoutSpec):
    def __init__(self, num_inputs, outputs):
        assert num_inputs == 1
        self.outputs = outputs

    def num_inputs(self):
        return 1

    def num_outputs(self):
        return self.outputs

    def get_input_pos(self, pos, index):
        return pos

    def get_output_pos(self, pos, index):
        return pos[0] + 2, pos[1] + 1 + self.outputs - index

    def get_height(self):
        return 3 + self.outputs


class MultiplexerCircuitLayoutSpec(CircuitLayoutSpec):
    def __init__(self, inputs, outputs):
        self.selector_bits = math.log2(inputs - 1)
        assert (self.selector_bits - int(self.selector_bits) + 1e-5) < 1e-3
        self.selector_bits = int(self.selector_bits)

        assert outputs == 1

    def num_inputs(self):
        return (2 ** self.selector_bits) + 1

    def num_outputs(self):
        return 1

    def get_input_pos(self, pos, index):
        if index == 2 ** self.selector_bits:
            return pos[0] + 1, pos[1] + (2 ** self.selector_bits) + 2

        return pos[0], pos[1] + 1 + index

    def get_output_pos(self, pos, index):
        return pos[0] + 3, pos[1] + (2 ** self.selector_bits + 2) // 2

    def get_height(self):
        return 3 + (2 ** self.selector_bits + 2)


class DecoderCircuitLayoutSpec(CircuitLayoutSpec):
    def __init__(self, inputs, outputs):
        assert inputs == 1
        self.selector_bits = math.log2(outputs)
        assert (self.selector_bits - int(self.selector_bits) + 1e-5) < 1e-3
        self.selector_bits = int(self.selector_bits)

    def num_inputs(self):
        return 1

    def num_outputs(self):
        return 2 ** self.selector_bits

    def get_input_pos(self, pos, index):
        return pos[0] + 2, pos[1] + (2 ** self.selector_bits) + 2

    def get_output_pos(self, pos, index):
        return pos[0] + 3, pos[1] + 1 + index

    def get_height(self):
        return 3 + (2 ** self.selector_bits + 2)


class CircuitSpec:
    def __init__(self, name, layout_cls, inputs, outputs):
        self.name = name
        self.layout = layout_cls(len(inputs), len(outputs))
        self.inputs = inputs
        self.outputs = outputs

    def get_type(self):
        return 'com.ra4king.circuitsim.gui.peers.SubcircuitPeer'

    def properties(self):
        return {
            "Label location": "NORTH",
            "Subcircuit": self.name,
        }


class GateCircuitSpec(CircuitSpec):
    def __init__(self, name, num_inputs, size=1, inputs=None, outputs=None):
        super().__init__(name, gate_special_layouts.get(name, GateCircuitLayoutSpec),
                         inputs or [('A' + str(i), size) for i in range(num_inputs)], outputs or [('X', size)])
        self._type = gate_types[name]

    def get_type(self):
        return self._type

    def properties(self):
        props = super().properties()
        props.pop('Subcircuit')

        props.update({
            "Number of Inputs": str(len(self.inputs)),
            "Direction": "EAST",
            # we assume bitsize if the size of the single output,
            # since most of the "gates" have only one output, this assumption holds.
            "Bitsize": str(self.outputs[0][1]),
        })

        for i in range(len(self.inputs)):
            props["Negate " + str(i)] = "No"

        return props


class BypassCircuitLayoutSpec(CircuitLayoutSpec):
    def __init__(self, num_inputs, outputs):
        assert num_inputs == 1
        assert outputs == 1

    def num_inputs(self):
        return 1

    def num_outputs(self):
        return 1

    def get_input_pos(self, pos, index):
        return pos

    def get_output_pos(self, pos, index):
        return pos

    def get_height(self):
        return 3


class BypassCircuitSpec(CircuitSpec):
    def __init__(self, size=1):
        super().__init__('bypass', BypassCircuitLayoutSpec, [('A', size)], [('B', size)])

    def get_type(self):
        return 'bypass'

    def properties(self):
        props = super().properties()
        return props


class MultiplexerCircuitSpec(CircuitSpec):
    def __init__(self, size=1, selector=1):
        self.selector = selector
        inputs = [(str(i), size) for i in range(2 ** selector)] + [('Selector', selector)]
        super().__init__('multiplexer', MultiplexerCircuitLayoutSpec, inputs, [('Out', size)])

    def get_type(self):
        return 'com.ra4king.circuitsim.gui.peers.plexers.MultiplexerPeer'

    def properties(self):
        props = super().properties()
        props['Selector bits'] = str(self.selector)
        props['Direction'] = 'EAST'
        props['Bitsize'] = str(self.outputs[0][1])
        return props


class DecoderCircuitSpec(CircuitSpec):
    def __init__(self, selector=1):
        self.selector = selector
        outputs = [(str(i), 1) for i in range(2 ** selector)]
        super().__init__('decoder', DecoderCircuitLayoutSpec, [('Selector', selector)], outputs)

    def get_type(self):
        return 'com.ra4king.circuitsim.gui.peers.plexers.DecoderPeer'

    def properties(self):
        props = super().properties()
        props['Selector bits'] = str(self.selector)
        props['Direction'] = 'EAST'
        return props


class ROMCircuitLayoutSpec(CircuitLayoutSpec):
    def __init__(self, num_inputs, outputs):
        assert num_inputs == 2
        assert outputs == 1

    def num_inputs(self):
        return 2

    def num_outputs(self):
        return 1

    def get_input_pos(self, pos, index):
        if index == 0:
            return pos[0], pos[1] + 2

        return pos[0] + 4, pos[1] + 5

    def get_output_pos(self, pos, index):
        return pos[0] + 9, pos[1] + 2

    def get_height(self):
        return 7


class ROMCircuitSpec(CircuitSpec):
    def __init__(self, data):
        self.address_bits = data['address_bits']
        self.data_bits = data['data_bits']
        self.contents = data.get('contents', '')
        super().__init__('rom', ROMCircuitLayoutSpec,
                         [('Address', self.address_bits), ('Enable', 1)], [('Data', self.data_bits)])

    def get_type(self):
        return 'com.ra4king.circuitsim.gui.peers.memory.ROMPeer'

    def properties(self):
        props = super().properties()
        props['Address bits'] = str(self.address_bits)
        props['Bitsize'] = str(self.data_bits)
        props['Contents'] = self.contents
        return props


gate_types = {
    'and': 'com.ra4king.circuitsim.gui.peers.gates.AndGatePeer',
    'not': 'com.ra4king.circuitsim.gui.peers.gates.NotGatePeer',
    'xor': 'com.ra4king.circuitsim.gui.peers.gates.XorGatePeer',
    'nor': 'com.ra4king.circuitsim.gui.peers.gates.NorGatePeer',
    'nand': 'com.ra4king.circuitsim.gui.peers.gates.NandGatePeer',
    'or': 'com.ra4king.circuitsim.gui.peers.gates.OrGatePeer',
    'xnor': 'com.ra4king.circuitsim.gui.peers.gates.XnorGatePeer',

    # special
    'buffer': 'com.ra4king.circuitsim.gui.peers.gates.ControlledBufferPeer',

    # don't have a custom layout yet and need special treatment because needs vertical wires
    'ram': 'com.ra4king.circuitsim.gui.peers.memory.RAMPeer',
    'dflipflop': 'com.ra4king.circuitsim.gui.peers.memory.DFlipFlopPeer',
    'register': 'com.ra4king.circuitsim.gui.peers.memory.RegisterPeer',
}

gate_special_layouts = {
    'buffer': BufferCircuitLayoutSpec,
    'dflipflop': DFlipFlopCircuitLayoutSpec,
    'register': RegisterCircuitLayoutSpec,
    'not': GateNotCircuitLayoutSpec,
}

BarSpec = CircuitSpec('Bar', DipCircuitLayoutSpec, [('A', 1), ('B', 1)], [('C', 1)])


class SplitterCircuitSpec(CircuitSpec):
    def __init__(self, bit_size, output_splits):
        self.bit_size = bit_size
        self.output_splits = output_splits

        self.bit_mapping = [-1 for _ in range(bit_size)]
        for split_i, split in enumerate(output_splits):
            for i in range(split[1], split[0] + 1):
                if self.bit_mapping[i] != -1:
                    raise ValueError
                self.bit_mapping[i] = split_i

        unmapped_pins = 0
        for i, v in enumerate(self.bit_mapping):
            if v == -1:
                self.bit_mapping[i] = len(output_splits)
                unmapped_pins += 1

        outputs = [
            ('A({})'.format(split[0]) if split[0] == split[1] else 'A({}-{})'.format(*split), split[1] - split[0] + 1)
            for split in output_splits]

        if unmapped_pins:
            outputs.append(('Extra', unmapped_pins))

        super().__init__('splitter', SplitterCircuitLayoutSpec, [('A', bit_size)], outputs)

    def get_type(self):
        return 'com.ra4king.circuitsim.gui.peers.wiring.SplitterPeer'

    def properties(self):
        props = super().properties()
        props['Input location'] = 'Left/Top'
        props['Bitsize'] = str(self.bit_size)
        props['Fanouts'] = str(len(self.outputs))

        for i in range(self.bit_size):
            props["Bit " + str(i)] = str(self.bit_mapping[i])

        return props


class CircuitEmbedding:
    def __init__(self, spec, input_map, output_map):
        self.spec = spec
        self.input_map = input_map
        self.output_map = output_map

    def get_all_signal_splits(self):
        all_signals = set(self.input_map.values()).union(set(self.output_map.values()))
        splits = []

        for signal in all_signals:
            sig_match = _signal_name_re.match(signal)
            if not sig_match:
                raise ValueError('Could not parse: ' + signal)

            if sig_match.group(2):
                end = int(sig_match.group(2))
                start = end
                if sig_match.group(3):
                    start = int(sig_match.group(3))

                splits.append((sig_match.group(1), end, start))

        return splits


class SimCircuit(object):
    def __init__(self, spec):
        self.spec = spec
        self.input_signals = [('F', 5), ('FA', 1)]
        self.output_signals = [('X', 1)]
        self.internal_signals = [('FB', 1), ('FC', 1), ('A', 5)]

        self.tiler = Tiler(55, left_shift=1)
        self.component_counter = Counter()

        self.components = [
            CircuitEmbedding(BarSpec, {'A': 'F(1)', 'B': 'F(0)'}, {'C': 'F(3-1)'}),
            CircuitEmbedding(BarSpec, {'A': 'FA', 'B': 'FB'}, {'C': 'FC'}),
            CircuitEmbedding(GateCircuitSpec('and', 4),
                             {'A0': 'A(0)', 'A1': 'A(1)', 'A2': 'A(2)', 'A3': 'A(3)', 'A4': 'A(4)'}, {'X': 'X'}),
            CircuitEmbedding(GateCircuitSpec('buffer', 2),
                             {'A0': 'A(0)', 'A1': 'A(1)'}, {'X': 'X'}),

            CircuitEmbedding(GateCircuitSpec('dflipflop', 5),
                             {'A0': 'A(0)', 'A1': 'A(1)', 'A2': 'A(2)', 'A3': 'A(3)', 'A4': 'A(4)'}, {'X': 'X'})
        ]

    @property
    def _all_signals(self):
        return self.internal_signals + self.input_signals + self.output_signals

    @staticmethod
    def materialize_splitters(all_signals):
        new_comps = []

        for signal, data in all_signals.items():
            if data['splits']:
                single_splits = []
                other_splits = []
                for split in data['splits']:
                    if split[0] == split[1]:
                        single_splits.append(split)
                    else:
                        other_splits.append(split)

                if single_splits:
                    spec = SplitterCircuitSpec(data['size'], single_splits)

                    output_map = {}
                    for split in single_splits:
                        if split[0] == -1:
                            continue
                        f = '({})'.format(split[0]) if split[0] == split[1] else '({}-{})'.format(*split)
                        output_map['A' + f] = signal + f

                    embedding = CircuitEmbedding(spec, {'A': signal}, output_map)
                    new_comps.append(embedding)

                for other_split in other_splits:
                    spec = SplitterCircuitSpec(data['size'], [other_split])

                    output_map = {}
                    f = '({})'.format(other_split[0]) if other_split[0] == other_split[1] else '({}-{})'.format(
                        *other_split)
                    output_map['A' + f] = signal + f

                    embedding = CircuitEmbedding(spec, {'A': signal}, output_map)
                    new_comps.append(embedding)

        print(all_signals)
        return new_comps

    @staticmethod
    def get_io_offset(size):
        if size == 1:
            return 2, 1
        if size <= 8:
            return size, 1

        if size > 24:
            return 8, 3
        if size > 16:
            return 8, 2

        return 8, 1

    def render_inputs_outputs(self):
        sim_components = []
        wires = []

        all_io = [(name, size, True) for name, size in self.input_signals] + \
                 [(name, size, False) for name, size in self.output_signals]

        x, y = 5, 5
        for name, size, is_input in all_io:
            if is_input and name.startswith('__const'):
                sim_components.append({
                    "name": "com.ra4king.circuitsim.gui.peers.wiring.ConstantPeer",
                    "x": x,
                    "y": y,
                    "properties": {
                        "Label location": "NORTH",
                        "Label": "",
                        "Value": '1' * size if name.endswith('on') else '0' * size,
                        "Direction": "WEST",
                        "Bitsize": str(size)
                    }
                })

                offset = self.get_io_offset(size)
                wires.append({
                    "x": x,
                    "y": y + offset[1],
                    "length": offset[0],
                    "isHorizontal": True
                })
            else:
                sim_components.append({
                    "name": "com.ra4king.circuitsim.gui.peers.wiring.PinPeer",
                    "x": x,
                    "y": y,
                    "properties": {
                        "Label location": "WEST",
                        "Label": name,
                        "Is input?": "Yes" if is_input else "No",
                        "Direction": "EAST" if is_input else "WEST",
                        "Bitsize": str(size)
                    }
                })

                if not is_input:
                    offset = self.get_io_offset(size)
                    wires.append({
                        "x": x,
                        "y": y + offset[1],
                        "length": offset[0],
                        "isHorizontal": True
                    })

            offset = self.get_io_offset(size)
            sim_components.append({
                "name": "com.ra4king.circuitsim.gui.peers.wiring.Tunnel",
                "x": x + offset[0],
                "y": y + offset[1] - 1,
                "properties": {
                    "Label": name,
                    "Direction": "WEST",
                    "Bitsize": str(size)
                }
            })
            y += offset[1] * 2 + 2

        return sim_components, wires

    def _determine_bit_size(self, name):
        sig_match = _signal_name_re.match(name)
        if not sig_match:
            raise ValueError('Could not parse: ' + name)

        if sig_match.group(1).startswith('__const1'):
            return 1

        if sig_match.group(2):
            end = int(sig_match.group(2))
            start = end
            if sig_match.group(3):
                start = int(sig_match.group(3))

            assert sig_match.group(1) in map(lambda x: x[0], self._all_signals)
            return end - start + 1

        for signal in self._all_signals:
            if signal[0] == sig_match.group(1):
                return signal[1]

        raise AssertionError('could not find: ' + name)

    def build(self):
        sim_components = [
            # {
            #     "name": "com.ra4king.circuitsim.gui.peers.gates.NotGatePeer",
            #     "x": 47,
            #     "y": 18,
            #     "properties": {
            #         "Label location": "NORTH",
            #         "Negate 0": "No",
            #         "Label": "",
            #         "Direction": "EAST",
            #         "Bitsize": "1"
            #     }
            # },
            # {
            #     "name": "com.ra4king.circuitsim.gui.peers.wiring.Tunnel",
            #     "x": 47,
            #     "y": 24,
            #     "properties": {
            #         "Label": "GEEEEEE",
            #         "Direction": "WEST",
            #         "Bitsize": "1"
            #     }
            # },
            # {
            #     "name": "com.ra4king.circuitsim.gui.peers.wiring.Tunnel",
            #     "x": 70,
            #     "y": 5,
            #     "properties": {
            #         "Label": "GEEEEEE",
            #         "Direction": "WEST",
            #         "Bitsize": "1"
            #     }
            # },
        ]
        wires = []

        io_comps, io_wires = self.render_inputs_outputs()
        sim_components.extend(io_comps)
        wires.extend(io_wires)

        materialized_components = list(self.components)
        all_signals = {name: {'size': s, 'splits': set()} for name, s in self._all_signals}
        for comp in self.components:
            splits = comp.get_all_signal_splits()
            for split in splits:
                all_signals[split[0]]['splits'].add((split[1], split[2]))

        materialized_components.extend(self.materialize_splitters(all_signals))

        for comp in materialized_components:
            layout = comp.spec.layout

            pos = self.tiler.next_tile(layout.get_height())
            pos = (pos[0] + self.tiler.scale // 2, pos[1] + self.tiler.scale // 2)

            self.component_counter[comp.spec.name] += 1

            props = comp.spec.properties()
            props['Label'] = comp.spec.name + str(self.component_counter[comp.spec.name])

            if comp.spec.get_type() != 'bypass':
                sim_components.append({
                    "name": comp.spec.get_type(),
                    "x": pos[0],
                    "y": pos[1],
                    "properties": props,
                })

            left_offset = int(self.tiler.scale * 0.4)

            for i in range(layout.num_inputs()):
                p = layout.get_input_pos(pos, i)

                input_name = comp.spec.inputs[i][0]
                if input_name not in comp.input_map:
                    continue

                mapped_name = comp.input_map[input_name]

                bitsize = self._determine_bit_size(mapped_name)

                name = 'com.ra4king.circuitsim.gui.peers.wiring.Tunnel'
                props = {
                    "Label": mapped_name,
                    "Direction": "WEST",
                    "Bitsize": str(bitsize)
                }

                if mapped_name.startswith('__const1'):
                    name = 'com.ra4king.circuitsim.gui.peers.wiring.ConstantPeer'
                    props['Value'] = '1' * bitsize if mapped_name.endswith('on') else '0' * bitsize
                    props['Label'] = ''

                if layout.connect_input_direct(i):
                    sim_components.append({
                        "name": name,
                        "x": p[0],
                        "y": p[1] - 1,
                        "properties": props,
                    })
                else:
                    sim_components.append({
                        "name": name,
                        "x": p[0] - left_offset,
                        "y": p[1] - 1,
                        "properties": props,
                    })
                    wires.append({
                        "x": p[0] - left_offset,
                        "y": p[1],
                        "length": left_offset,
                        "isHorizontal": True
                    })

            right_offset = int(self.tiler.scale * 0.1)

            for i in range(layout.num_outputs()):
                p = layout.get_output_pos(pos, i)

                output_name = comp.spec.outputs[i][0]
                if output_name not in comp.output_map:
                    continue

                mapped_name = comp.output_map[output_name]

                if layout.connect_output_direct(i):
                    sim_components.append({
                        "name": "com.ra4king.circuitsim.gui.peers.wiring.Tunnel",
                        "x": p[0],
                        "y": p[1] - 1,
                        "properties": {
                            "Label": mapped_name,
                            "Direction": "WEST",
                            "Bitsize": str(self._determine_bit_size(mapped_name))
                        }
                    })
                else:
                    sim_components.append({
                        "name": "com.ra4king.circuitsim.gui.peers.wiring.Tunnel",
                        "x": p[0] + right_offset,
                        "y": p[1] - 1,
                        "properties": {
                            "Label": mapped_name,
                            "Direction": "WEST",
                            "Bitsize": str(self._determine_bit_size(mapped_name))
                        }
                    })
                    wires.append({
                        "x": p[0],
                        "y": p[1],
                        "length": right_offset,
                        "isHorizontal": True
                    })

        return {
            'name': self.spec.name,
            'components': sim_components,
            'wires': wires,
        }


def temp_name_generator():
    i = 1
    while True:
        yield 't' + str(i)
        i += 1
