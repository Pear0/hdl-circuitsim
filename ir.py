from typing import List, Dict

import util


def match_all(x):
    return True


class ASTNode:
    def get_children(self):
        return []

    def get_descendants(self, matcher=match_all):
        combined = []
        for child in self.get_children():
            if matcher(child):
                combined.append(child)
            combined += list(child.get_descendants(matcher=matcher))
        return combined

    def get_all_ports(self) -> List['ASTPort']:
        return self.get_descendants(matcher=lambda x: isinstance(x, ASTPort))


class ASTExpr(ASTNode):
    def get_bitsize(self):
        raise NotImplementedError


class ASTPort(ASTExpr):
    def __init__(self, name, bitsize):
        self.name = str(name)
        self.bitsize = bitsize

    def __getitem__(self, item):
        if type(item) == tuple:
            return ASTSubPort(self.name, item[0], item[1])

        assert type(item) == int
        return ASTSubPort(self.name, item, item)

    def get_render_name(self):
        return self.name

    def get_bitsize(self):
        return self.bitsize


class ASTSubPort(ASTPort):
    def __init__(self, name, start, end):
        super().__init__(name, end - start + 1)
        self.start = start
        self.end = end

    def __getitem__(self, item):
        if type(item) == tuple:
            return ASTSubPort(self.name, self.start + item[0], self.start + item[1])

        assert type(item) == int
        return ASTSubPort(self.name, self.start + item, self.start + item)

    def get_render_name(self):
        if self.start == self.end:
            return '{}({})'.format(self.name, self.start)

        return '{}({}-{})'.format(self.name, self.end, self.start)


class ASTAssign(ASTNode):
    def __init__(self, dst, src, rising_edge=None, enabled=None, write_enable=None):
        if write_enable and not rising_edge:
            rising_edge = ASTPort('clk', 1)
        if rising_edge and not write_enable:
            write_enable = ASTPort('__const1_on', 1)

        self.dst = dst
        self.src = src
        self.rising_edge = rising_edge  # save dst, only update on rising edge
        self.write_enable = write_enable
        self.enabled = enabled  # tri-state whenever this is false

    def get_children(self):
        children = [self.dst, self.src]

        if self.rising_edge:
            children.append(self.rising_edge)
        if self.enabled:
            children.append(self.enabled)

        return children


class ASTBlock(ASTNode):
    def __init__(self, children=None):
        self.children = children or []

    def get_children(self) -> List[ASTNode]:
        return list(self.children)


class ASTLogicGate(ASTExpr):
    def __init__(self, type, children=None):
        self.type = type
        self.children = children or []

    def get_children(self):
        return self.children

    def get_bitsize(self):
        bitsize = None

        for child in self.get_children():
            child_size = child.get_bitsize()
            if bitsize is not None and bitsize != child_size:
                raise ValueError('mismatched bitsize on logic gate')
            bitsize = child_size

        return bitsize


class ASTMultiplexer(ASTExpr):
    def __init__(self, selector: ASTExpr, children: Dict[int, ASTExpr] = None):
        self.selector = selector
        self.children = children or {}

    def get_children(self):
        return [self.selector] + list(self.children.values())

    def get_bitsize(self):
        bitsize = None

        for child in self.children.values():
            child_size = child.get_bitsize()
            if bitsize is not None and bitsize != child_size:
                raise ValueError('mismatched bitsize on logic gate')
            bitsize = child_size

        return bitsize


class ASTDecoder(ASTNode):
    def __init__(self, input: ASTExpr = None, outputs=None):
        self.input = input
        self.outputs = outputs

    def get_children(self):
        return [self.input] + list(self.outputs.values())


class ASTSubCircuit(ASTNode):
    def __init__(self, type, inputs=None, outputs=None, native=False, data=None):
        self.type = type
        self.native = native
        self.data = data
        self.inputs = inputs or {}
        self.outputs = outputs or {}

    def get_children(self):
        return list(self.inputs.values()) + list(self.outputs.values())


class ASTCircuit(ASTBlock):
    def __init__(self, inputs: List[ASTPort], outputs: List[ASTPort], children=None):
        super().__init__(children=children)
        self.input_signals = inputs
        self.output_signals = outputs
        self.internal_signals = {
            '__const1_on': 1,
            '__const1_off': 1,
            '__const1_x': 1,
            # '__const32_on': 32,
            # '__const32_off': 32,
        }  # name -> bitsize

    def get_port(self, name):
        name = str(name)
        for port in self.input_signals + self.output_signals:
            if port.name == name:
                return port

        for port_name, size in self.internal_signals.items():
            if port_name == name:
                return ASTPort(name, size)

    def get_or_create(self, name, size):
        name = str(name)
        p = self.get_port(name)
        if p:
            if p.bitsize != size:
                raise ValueError
            return p

        self.internal_signals[name] = size
        return self.get_port(name)

    def generate_internal_signal(self, bitsize):
        import random

        while True:
            name = ''
            for _ in range(4):
                name += chr(ord('a') + random.randint(0, 25))

            if name not in self.internal_signals:
                self.internal_signals[name] = bitsize
                return name

    def validate(self):
        port_map = dict(self.internal_signals)

        for signal in self.input_signals + self.output_signals:
            if signal.name in port_map:
                raise ValueError('name collision: ' + signal.name)
            port_map[signal.name] = signal.bitsize

        for ref in self.get_all_ports():
            if ref.name not in port_map:
                raise ValueError('cannot find port: ' + ref.name)

            if isinstance(ref, ASTSubPort):
                if ref.end >= port_map[ref.name]:
                    raise ValueError('invalid, sub port {}: end == {} but signal bitsize == {}'
                                     .format(ref.name, ref.bitsize, port_map[ref.name]))
            elif ref.bitsize != port_map[ref.name]:
                raise ValueError('invalid, port {}: bitsize == {} but signal bitsize == {}'
                                 .format(ref.name, ref.bitsize, port_map[ref.name]))








