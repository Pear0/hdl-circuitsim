import json

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

    def assign(self, src, enabled=None, write_enable=None):
        _root_builder.circuit_builder.circuit.children.append(
            ir.ASTAssign(self.port, _unwrap(src), enabled=_unwrap(enabled), write_enable=_unwrap(write_enable)))

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


def port_define(name, size):
    try:
        p = port(name)
    except ValueError:
        p = None
    if p:
        raise ValueError('cannot redefine: ' + name)

    _root_builder.circuit_builder.circuit.internal_signals[name] = size
    return port(name)


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
    data = kwargs.pop('config', None)

    _root_builder.circuit_builder.circuit.children.append(
        ir.ASTSubCircuit(name,
                         {k: _unwrap(v) for k, v in kwargs.items()},
                         {k: _unwrap(v) for k, v in kwargs.items()}, data=data))


def decoder(input, outputs):
    _root_builder.circuit_builder.circuit.children.append(
        ir.ASTDecoder(_unwrap(input), {k: _unwrap(v) for k, v in outputs.items()}))


def mux(selector, inputs):
    return ir.ASTMultiplexer(_unwrap(selector), {k: _unwrap(v) for k, v in inputs.items()})


# noinspection PyStatementEffect
def main():
    internal_signals = [
        ('a', 2),
        ('ALU_func', 2), ('OpTest', 1)
    ]

    # order should match signals starting at index 6 of control ROM
    control_signals = ['DrREG', 'DrMEM', 'DrALU', 'DrPC', 'DrOFF', 'LdPC', 'LdIR',
                       'LdMAR', 'LdA', 'LdB', 'LdCmp', 'WrREG', 'WrMEM']

    for sig in control_signals:
        internal_signals.append((sig, 1))

    i_regs = [('iReg' + str(i), 32) for i in range(16)]

    with CircuitBuilder('DataPath', [('clk', 1), ('b', 2)],
                        [('bus', 32), ('PC', 32), ('IR', 32), ('micro_state', 6)] + i_regs + [('IR_imm', 32)],
                        internal_signals):

        # Ld/Dr PC
        port('PC').assign(port('bus'), write_enable=port('LdPC'))
        port('bus').assign(port('PC'), enabled=port('DrPC'))

        # ALU
        port_define('ALU_A', 32).assign(port('bus'), write_enable=port('LdA'))
        port_define('ALU_B', 32).assign(port('bus'), write_enable=port('LdB'))

        sub_circuit('ALU', A=port('ALU_A'), B=port('ALU_B'), Op=port('ALU_func'), Out=port_define('ALU_Out', 32))

        port('bus').assign(port('ALU_Out'), enabled=port('DrALU'))

        # Registers

        sub_circuit('RegFile', clk=port('clk'), Write=port('WrREG'), Index=port_define('RegNo', 4),
                    In=port('bus'), Out=port_define('Reg_Out', 32), **{k: port(k) for k, _ in i_regs})

        port('bus').assign(port('Reg_Out'), enabled=port('DrREG'))

        # Memory
        port_define('MAR', 32).assign(port('bus'), write_enable=port('LdMAR'))

        sub_circuit('Memory', In=port('bus'), Out=port_define('mem_out', 32),
                    Address=port('MAR', 0, 15), Clock=port('clk'),
                    Write=port('WrMEM'))

        port('bus').assign(port('mem_out'), enabled=port('DrMEM'))

        # IR
        port('IR').assign(port('bus'), write_enable=port('LdIR'))

        # poor man's sign extend lmao
        imm = port('IR_imm')
        imm[0, 19] <= port('IR', 0, 19)
        for i in range(20, 32):
            imm[i] <= port('IR', 19)

        port('bus').assign(imm, enabled=port('DrOFF'))

        # Comparison Logic

        sub_circuit('ComparisonLogic', Data=port('bus'), Mode=port('IR', 24, 27), Out=port_define('CmpResult', 1))

        # Misc

        port('RegNo') <= mux(port_define('RegSel', 2), {
            0b00: port('IR', 24, 27),
            0b01: port('IR', 20, 23),
            0b10: port('IR', 0, 3),
        })

        sub_circuit('ControlUnit', clk=port('clk'), RegSel=port('RegSel'), ALU_func=port('ALU_func'),
                    Cmp=port('CmpResult'), OpCode=port('IR', 28, 31),
                    OpTest=port('OpTest'), state=port('micro_state'), **{k: port(k) for k in control_signals})

        # my_logic = g_and(port('a')[0], g_or(port('a', 0), port('a', 1)))

        # port('a', 1).assign(my_logic, rising_edge=port('b', 0), enabled=port('b', 0))

        pass

    with CircuitBuilder('ControlUnit', [('clk', 1), ('OpCode', 4), ('Cmp', 1)],
                        [(x, 1) for x in control_signals] +
                        [('RegSel', 2), ('ALU_func', 2),
                         ('OpTest', 1), ('ChkCmp', 1), ('state', 6)]):

        with open('circuitsim/rom_main.dat') as f:
            control_rom = f.read()

        with open('circuitsim/rom_seq.dat') as f:
            control_seq_rom = f.read()

        with open('circuitsim/rom_cond.dat') as f:
            control_cond_rom = f.read()

        sub_circuit('rom', Address=port('state'), Data=port_define('control_vector', 25), config={
            'address_bits': 6,
            'data_bits': 25,
            'contents': control_rom,
        })

        sub_circuit('rom', Address=port('OpCode'), Data=port_define('next_seq_state', 6), config={
            'address_bits': 4,
            'data_bits': 6,
            'contents': control_seq_rom,
        })

        sub_circuit('rom', Address=port('Cmp'), Data=port_define('next_cmp_state', 6), config={
            'address_bits': 1,
            'data_bits': 6,
            'contents': control_cond_rom,
        })

        port_define('next_state', 6) <= mux(port_define('state_mux', 2), {
            0b00: port('control_vector', 0, 5),
            0b01: port('next_seq_state'),
            0b10: port('next_cmp_state'),
        })

        # every clock we transition state
        port('state').assign(port('next_state'), write_enable=port('__const1_on'))

        for i, v in enumerate(control_signals):
            port(v) <= port('control_vector', 6 + i, 6 + i)

        port('RegSel') <= port('control_vector', 19, 20)
        port('ALU_func') <= port('control_vector', 21, 22)
        port('OpTest') <= port('control_vector', 23)
        port('ChkCmp') <= port('control_vector', 24)

        port('state_mux', 0) <= port('OpTest')
        port('state_mux', 1) <= port('ChkCmp')

        pass

    with CircuitBuilder('RegFile', [('clk', 1), ('Write', 1), ('Index', 4), ('In', 32), ('__const32_off', 32)],
                        [('Out', 32)] + i_regs):
        write_enables = {}
        outputs = {}

        do_write = g_and(port('clk'), port('Write'))

        for i in range(1, 16):
            write_enables[i] = port_define('WrReg{}'.format(i), 1)
            outputs[i] = port_define('OutReg{}'.format(i), 32)

            outputs[i].assign(port('In'), write_enable=g_and(write_enables[i], port('Write')))

        outputs[0] = port('__const32_off')

        for i in range(16):
            port('iReg' + str(i)) <= outputs[i]

        decoder(port('Index'), write_enables)

        port('Out') <= mux(port('Index'), outputs)

    with CircuitBuilder('ALU', [('A', 32), ('B', 32), ('Op', 2), ('__const32_off', 32)], [('Out', 32)]):

        sub_circuit('ADD32', X=port_define('add_result', 32), A=port('A'), B=port('B'), Cin=port('__const1_off'))

        sub_circuit('ADD32', X=port_define('sub_result', 32), A=port('A'), B=g_not(port('B')), Cin=port('__const1_on'))

        port_define('nand_result', 32) <= g_not(g_and(port('A'), port('B')))

        sub_circuit('ADD32', X=port_define('add1_result', 32), A=port('A'),
                    B=port('__const32_off'), Cin=port('__const1_on'))

        port('Out') <= mux(port('Op'), {
            0b00: port('add_result'),
            0b01: port('sub_result'),
            0b10: port('nand_result'),
            0b11: port('add1_result'),
        })

    with CircuitBuilder('ComparisonLogic', [('Data', 32), ('Mode', 4)], [('Out', 1)]):

        or_sigs = [port('Data', i, i) for i in range(32)]
        while len(or_sigs) != 1:
            copy = list(or_sigs)
            or_sigs = []
            for i in range(0, len(copy), 2):
                or_sigs.append(g_or(copy[i], copy[i + 1]))

        port('Out') <= mux(port('Mode'), {
            0b00: g_not(or_sigs[0]),
            0b01: port('Data', 31)
        })

    with CircuitBuilder('ADD32', [('A', 32), ('B', 32), ('Cin', 1)], [('X', 32), ('Cout', 1)]):
        in_cin = 'Cin'

        for i in range(32):
            carry = 'c' + str(i)
            sub_circuit('ADD1', A=port('A', i), B=port('B', i), X=port('X', i),
                        Cin=port(in_cin), Cout=port_define(carry, 1))
            in_cin = carry

        port('Cout') <= port(in_cin)

    with CircuitBuilder('ADD1', [('A', 1), ('B', 1), ('Cin', 1)], [('X', 1), ('Cout', 1)]):

        port_define('AxorB', 1) <= g_xor(port('A'), port('B'))

        port('X') <= g_xor(port('AxorB'), port('Cin'))
        port('Cout') <= g_or(g_and(port('A'), port('B')), g_and(port('AxorB'), port('Cin')))

    input = ir.ASTPort('in', 2)
    clock = ir.ASTPort('clock', 1)
    output = ir.ASTPort('out', 2)

    root = ir.ASTCircuit([input, clock], [output])

    meh = ir.ASTPort(root.generate_internal_signal(1), 1)
    root.children.append(ir.ASTAssign(meh, clock))

    # root.children.append(ir.ASTSubCircuit('Bar', {'a': input[0]}, {'out': clock}))

    root.children.append(ir.ASTAssign(output, input, enabled=ir.ASTLogicGate('and', children=[clock, meh])))

    # other_root = ir.ASTCircuit([ir.ASTPort('in', 1), ir.ASTPort('in2', 1)], [ir.ASTPort('out', 1)])

    _root_builder.circuits['foo'] = root

    with open('circuitsim/intrinsics.sim', 'r') as f:
        native_circuits = json.load(f)['circuits']

    native_defines = [
        ('Memory', [('Address', 16), ('In', 32), ('Clock', 1), ('Write', 1)], [('Out', 32)])
    ]

    ir_lowering.compile_circuits(_root_builder.circuits, filename='circuitsim/gen2.sim',
                                 native_circuits=native_circuits)


if __name__ == '__main__':
    main()
