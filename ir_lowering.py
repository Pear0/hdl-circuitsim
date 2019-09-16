import json

import circuits
import ir


# signal : (name, bit_size)


def flatten_to_signal(body: ir.ASTCircuit, expr: ir.ASTExpr) -> ir.ASTPort:
    if isinstance(expr, ir.ASTPort):
        return expr

    if isinstance(expr, ir.ASTLogicGate):
        gate_bitsize = expr.get_bitsize()

        lowered_children = [flatten_to_signal(body, child) for child in expr.get_children()]

        dst_port = ir.ASTPort(body.generate_internal_signal(gate_bitsize), gate_bitsize)

        body.children.append(ir.ASTAssign(dst_port, ir.ASTLogicGate(expr.type, lowered_children)))

        return dst_port

    raise ValueError('flattening not supported for type: {}'.format(expr))


# flattened:
#
#    ; this will get compiled into a gate
#    port <-[rising_edge: None] gate(port, port, ...)
# OR
#    ; this will get compiled into a d flip flop
#    port <-[rising_edge: port] port
# OR
#    ; this will get compiled into a buffer
#    port <-[enabled: port] port
#
# <-[rising_edge: port, enabled: port] is too complex. it must be
# compiled into two assigns <-[enabled: port] <-[rising_edge: port]
#


def flatten_body(body: ir.ASTCircuit):
    for child_i, child in enumerate(list(body.get_children())):
        if isinstance(child, ir.ASTAssign):
            if child.rising_edge or child.enabled:
                if child.rising_edge and child.enabled:
                    src_signal = flatten_to_signal(body, child.src)
                    edge_signal = flatten_to_signal(body, child.rising_edge)
                    enabled_signal = flatten_to_signal(body, child.enabled)

                    # <-[enabled: port] intermediate_signal <-[rising_edge: port]

                    intermediate_port = ir.ASTPort(body.generate_internal_signal(src_signal.bitsize),
                                                   src_signal.bitsize)

                    # <-[rising_edge: port]
                    body.children.append(ir.ASTAssign(intermediate_port, src_signal, rising_edge=edge_signal))

                    # <-[enabled: port]
                    child.src = intermediate_port
                    child.enabled = edge_signal
                    child.rising_edge = None
                elif child.rising_edge:
                    src_signal = flatten_to_signal(body, child.src)
                    edge_signal = flatten_to_signal(body, child.rising_edge)

                    child.src = src_signal
                    child.rising_edge = edge_signal
                elif child.enabled:
                    src_signal = flatten_to_signal(body, child.src)
                    enabled_signal = flatten_to_signal(body, child.enabled)

                    child.src = src_signal
                    child.enabled = enabled_signal

            elif isinstance(child.src, ir.ASTLogicGate):
                lowered_children = [flatten_to_signal(body, child) for child in child.src.get_children()]

                child.src = ir.ASTLogicGate(child.src.type, lowered_children)
            elif isinstance(child.src, ir.ASTPort):
                pass
            else:
                raise ValueError('cannot flatten root assignment with node: {}'.format(child.src))

        else:
            print('Skipping {}, don\'t know how to flatten'.format(child))


def lower_complex_assigns(body: ir.ASTCircuit):
    for child_i, child in enumerate(list(body.get_children())):
        if isinstance(child, ir.ASTAssign):
            if child.rising_edge:
                assert isinstance(child.rising_edge, ir.ASTPort)
                assert isinstance(child.src, ir.ASTPort)
                assert child.enabled is None

                body.children[child_i] = ir.ASTSubCircuit('register', {
                    'Clock': child.rising_edge,
                    'Enable': ir.ASTPort('__const1_on', 1),
                    'In': child.src,
                }, {'Out': child.dst})

            if child.enabled:
                assert isinstance(child.enabled, ir.ASTPort)
                assert isinstance(child.src, ir.ASTPort)
                assert child.rising_edge is None

                body.children[child_i] = ir.ASTSubCircuit('buffer', {
                    'Enable': child.enabled,
                    'In': child.src,
                }, {'Out': child.dst})


def get_spec(name, other_specs, size=1):
    if name == 'register':
        return circuits.GateCircuitSpec('register', 1, size,
                                        inputs=[('In', size), ('Enable', 1), ('Clock', 1), ('Clear', 1)],
                                        outputs=[('Out', size)])
    if name == 'buffer':
        return circuits.GateCircuitSpec('buffer', 1, size,
                                        inputs=[('In', size), ('Enable', 1)],
                                        outputs=[('Out', size)])
    if name in ('and', 'or', 'not', 'xor'):
        return circuits.GateCircuitSpec(name, 2, size)

    if name == 'bypass':
        return circuits.BypassCircuitSpec()

    return other_specs[name]


def perform_lowering(circuit_name, root: ir.ASTCircuit, other_specs):
    root.validate()

    flatten_body(root)
    lower_complex_assigns(root)

    input_signals = [(x.name, x.bitsize) for x in root.input_signals]
    output_signals = [(x.name, x.bitsize) for x in root.output_signals]
    spec = circuits.CircuitSpec(circuit_name, circuits.DipCircuitLayoutSpec, input_signals, output_signals)
    circuit = circuits.SimCircuit(spec)
    circuit.input_signals = input_signals
    circuit.output_signals = output_signals
    circuit.components = []

    circuit.internal_signals = [(name, bitsize or 1) for name, bitsize in root.internal_signals.items()]

    for child in root.children:
        if isinstance(child, ir.ASTSubCircuit):
            bitsize = 1
            if child.outputs:
                bitsize = next(iter(child.outputs.values())).bitsize

            spec = get_spec(child.type, other_specs, bitsize)  # type: circuits.CircuitSpec

            inputs = {}
            for c_name, c_bitsize in spec.inputs:
                if c_name in child.inputs:
                    inputs[c_name] = child.inputs[c_name].get_render_name()

            outputs = {}
            for c_name, c_bitsize in spec.outputs:
                if c_name in child.outputs:
                    outputs[c_name] = child.outputs[c_name].get_render_name()

            circuit.components.append(circuits.CircuitEmbedding(spec, inputs, outputs))

            pass
        elif isinstance(child, ir.ASTAssign):
            assert child.rising_edge is None
            assert child.enabled is None

            if isinstance(child.src, ir.ASTLogicGate):
                gate = child.src
                spec = get_spec(gate.type, other_specs, gate.get_bitsize())  # type: circuits.CircuitSpec

                inputs = {}
                for i, g_child in enumerate(gate.children):
                    inputs['A' + str(i)] = g_child.get_render_name()

                outputs = {'X': child.dst.get_render_name()}

                circuit.components.append(circuits.CircuitEmbedding(spec, inputs, outputs))
            elif isinstance(child.src, ir.ASTPort):
                spec = get_spec('bypass', other_specs)  # type: circuits.CircuitSpec

                inputs = {'A': child.src.get_render_name()}
                outputs = {'B': child.dst.get_render_name()}
                circuit.components.append(circuits.CircuitEmbedding(spec, inputs, outputs))
            else:
                raise ValueError('die')
        else:
            raise ValueError('die')

    return circuit


def compile_circuits(circuit_map, filename=None, native_circuits=None):
    spec_map = {}

    for name, circuit in circuit_map.items():
        inputs = [(x.name, x.bitsize) for x in circuit.input_signals]
        outputs = [(x.name, x.bitsize) for x in circuit.output_signals]

        spec_map[name] = circuits.CircuitSpec(name, circuits.DipCircuitLayoutSpec, inputs, outputs)

    compiled_circuits = list(native_circuits) if native_circuits else []

    for name, circuit in circuit_map.items():
        compiled = perform_lowering(name, circuit, spec_map)
        compiled_circuits.append(compiled.build())

    if filename:
        data = {
            'version': '1.8.2',
            'globalBitSize': 1,
            'clockSpeed': 1,
            'circuits': compiled_circuits,
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    return compiled_circuits


