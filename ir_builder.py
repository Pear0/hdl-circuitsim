import ir
import ir_lowering


class _RootBuilder:
    def __init__(self):
        self.circuit_builder = None
        self.circuits = {}


_root_builder = _RootBuilder()


class CircuitBuilder:
    def __init__(self, name, inputs=None, outputs=None, internal=None):
        self.name = name
        self.inputs = [ir.ASTPort(name, size) for name, size in (inputs or [])]
        self.outputs = [ir.ASTPort(name, size) for name, size in (outputs or [])]

        self.circuit = ir.ASTCircuit(self.inputs, self.outputs)

        for name, size in (internal or []):
            self.circuit.internal_signals[name] = size

    def __enter__(self):
        assert _root_builder.circuit_builder is None
        _root_builder.circuit_builder = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _root_builder.circuit_builder = None

        _root_builder.circuits[self.name] = self.circuit


def _unwrap(x):
    if x is None:
        return x
    if isinstance(x, PortBuilder):
        return x.port
    return x


class PortBuilder:
    def __init__(self, port):
        self.port = port

    def assign(self, src, enabled=None, rising_edge=None):
        _root_builder.circuit_builder.circuit.children.append(
            ir.ASTAssign(self.port, _unwrap(src), enabled=_unwrap(enabled), rising_edge=_unwrap(rising_edge)))

    def __le__(self, other):
        self.assign(other)

    def __getitem__(self, item):
        sub = self.port.__getitem__(item)
        if sub:
            return PortBuilder(sub)
        else:
            raise ValueError('sub port is invalid: ' + str(item))


def port(name, start=None, end=None):
    port = _root_builder.circuit_builder.circuit.get_port(name)
    if not port:
        raise ValueError('Cannot find: ' + str(name))

    if start is not None:
        if end is None:
            end = start
        port = port[start, end]

    return PortBuilder(port)


def gate(name, *args):
    if len(args) <= 2:
        return ir.ASTLogicGate(name, children=[_unwrap(x) for x in args])

    raise ValueError


def g_and(*args):
    return gate('and', *args)


def g_or(*args):
    return gate('or', *args)


def g_not(*args):
    return gate('not', *args)


def g_xor(*args):
    return gate('xor', *args)


def sub_circuit(name, **kwargs):
    _root_builder.circuit_builder.circuit.children.append(
        ir.ASTSubCircuit(name,
                         {k: _unwrap(v) for k, v in kwargs.items()},
                         {k: _unwrap(v) for k, v in kwargs.items()}))


# noinspection PyStatementEffect
def main():
    with CircuitBuilder('Bar', [('a', 2)], [], [('b', 2)]):
        port('a') <= port('a')

        my_logic = g_and(port('a')[0], g_or(port('a', 0), port('a', 1)))

        port('a', 1).assign(my_logic, rising_edge=port('b', 0), enabled=port('b', 0))

        pass

    with CircuitBuilder('Baz', [('a', 2)], []):
        sub_circuit('Bar', a=port('a'))

        port('a') <= port('a')

        pass

    input = ir.ASTPort('in', 2)
    clock = ir.ASTPort('clock', 1)
    output = ir.ASTPort('out', 2)

    root = ir.ASTCircuit([input, clock], [output])

    meh = ir.ASTPort(root.generate_internal_signal(1), 1)
    root.children.append(ir.ASTAssign(meh, clock))

    root.children.append(ir.ASTSubCircuit('Bar', {'a': input[0]}, {'out': clock}))

    root.children.append(ir.ASTAssign(output, input, enabled=ir.ASTLogicGate('and', children=[clock, meh])))

    # other_root = ir.ASTCircuit([ir.ASTPort('in', 1), ir.ASTPort('in2', 1)], [ir.ASTPort('out', 1)])

    _root_builder.circuits['foo'] = root

    ir_lowering.compile_circuits(_root_builder.circuits, filename='circuitsim/gen2.sim')


if __name__ == '__main__':
    main()
