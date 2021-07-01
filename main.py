import json
import math
import sys

import pyverilog.dataflow.dataflow as df
from pyverilog.dataflow import visit
from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer
from pyverilog.dataflow.optimizer import VerilogDataflowOptimizer
from pyverilog.vparser import parser
from pyverilog.vparser.ast import Instance, InstanceList, Initial, Decl, RegArray, Parameter, Input, Output

import ir
import ir_lowering


def filter_terms(terms, f):
    return [(str(name[1]), term.msb.value - term.lsb.value + 1) for name, term in terms.items() if f(term.termtype)]


class VLowering:
    def __init__(self, module, terms, binddict, instances, other_modules):
        self.module = module
        self.terms = terms
        self.binddict = binddict
        self.instances = instances
        self.other_modules = other_modules

        terms = {t: terms[t] for t in terms if len(t) == 2}

        inputs = filter_terms(terms, lambda x: 'Input' in x)
        outputs = filter_terms(terms, lambda x: 'Output' in x)
        internal = filter_terms(terms, lambda x: not ('Input' in x or 'Output' in x))

        self.circuit = ir.ASTCircuit([ir.ASTPort(n, s) for n, s in inputs],
                                     [ir.ASTPort(n, s) for n, s in outputs])
        for name, size in internal:
            self.circuit.get_or_create(name, size)

        self.circuit.validate()

        self.resolved_consts = {}

        self.required_intrinsics = set([])

        self.mem_definitions = {}
        self.mem_initial = {}

        for item in self.other_modules[self.module].items:
            if isinstance(item, Decl):
                for arr in item.list:
                    if isinstance(arr, RegArray):
                        self.mem_definitions[arr.name] = arr
            if isinstance(item, Initial):
                item = item.statement.statement
                if item.syscall == 'readmem_sim':
                    with open(item.args[0].value) as f:
                        self.mem_initial[item.args[1].name] = f.read()
                pass

    def _resolve_sc(self, scope_chain):
        return '.'.join(map(str, scope_chain))

    def _do_resolve_const(self, value, bits=None):
        if isinstance(value, int):
            value = bin(value)[2:]
        if "'" in value:
            lit_width, _, value = value.partition("'")
            assert int(lit_width) == bits
            assert value[0].lower() == 'b'
            value = value[1:]

        if any(x not in '10x' for x in value):
            value = bin(int(value))[2:]

        if value == '1' and bits in [1, None]:
            return ir.ASTPort('__const1_on', 1)
        if value == '0' and bits in [1, None]:
            return ir.ASTPort('__const1_off', 1)
        if value == "x" and bits in [1, None]:
            return ir.ASTPort('__const1_x', 1)

        my_sig = self.circuit.generate_internal_signal(bits)
        port = self.circuit.get_port(my_sig)

        value = '0' * (bits - len(value)) + value

        for i in range(bits):
            self.circuit.children.append(ir.ASTAssign(port[i], self.resolve_const(value[len(value) - 1 - i], bits=1)))

        return port

    def resolve_const(self, value, bits=None):
        if (value, bits) in self.resolved_consts:
            return self.resolved_consts[(value, bits)]

        self.resolved_consts[(value, bits)] = self._do_resolve_const(value, bits=bits)
        return self.resolved_consts[(value, bits)]

    def _emit_expr_intrinsic(self, v_name, children, bits=None):
        if v_name == 'Plus':
            assert type(bits) == int
            assert len(children) == 2
            assert all(c.get_bitsize() == bits for c in children)

            result_sig = self.circuit.generate_internal_signal(bits)
            result_port = self.circuit.get_port(result_sig)

            self.required_intrinsics.add(('add', bits))

            self.circuit.children.append(ir.ASTSubCircuit('_add' + str(bits), {
                'A': children[0],
                'B': children[1],
                'Cin': ir.ASTPort('__const1_off', 1),
            }, {
                                                              'X': result_port,
                                                          }))

            return result_port

        raise ValueError

    def _speculative_mux_folding(self, expr, ident=None, bits=None):

        selector = None
        matched_values = {}

        while isinstance(expr, df.DFBranch):
            cond = expr.condnode
            if not (isinstance(cond, df.DFOperator) and cond.operator == 'Eq'):
                break

            cond_term = None
            cond_const = None
            for node in cond.nextnodes:
                if isinstance(node, df.DFIntConst):
                    cond_const = node.eval()
                if isinstance(node, df.DFTerminal):
                    cond_term = node

            if not cond_term or not cond_const:
                break

            if selector is not None and selector != cond_term:
                break

            if selector is None:
                selector = cond_term

            if cond_const in matched_values:
                break

            matched_values[cond_const] = expr.truenode
            expr = expr.falsenode

        if selector is None or len(matched_values) == 0:
            return None

        selector_node = self._lower_expr(selector, cond_term)
        selector_bits = selector_node.get_bitsize()

        lowered_options = {k: (self._lower_expr(v, ident=ident, bits=bits) if v else ident)
                           for k, v in matched_values.items()}

        false_node = self._lower_expr(expr, ident=ident, bits=bits) if expr else ident
        if false_node:
            for i in range(2 ** selector_bits):
                if i not in lowered_options:
                    lowered_options[i] = false_node
        else:
            pass

        assert all(0 <= x < 2 ** selector_bits for x in lowered_options.keys())

        return ir.ASTMultiplexer(selector_node, lowered_options)

    def _lower_expr(self, expr, ident=None, bits=None):
        # if isinstance(expr, )
        if isinstance(expr, df.DFPartselect):
            return self.circuit.get_port(expr.var.name[1])[int(expr.lsb.value), int(expr.msb.value)]
        elif isinstance(expr, df.DFBranch):

            spec_mux = self._speculative_mux_folding(expr, ident=ident, bits=bits)
            if spec_mux:
                return spec_mux

            false_node = self._lower_expr(expr.falsenode, ident=ident, bits=bits) if expr.falsenode else ident
            true_node = self._lower_expr(expr.truenode, ident=ident, bits=bits) if expr.truenode else ident

            if isinstance(expr.condnode, df.DFIntConst):
                if expr.condnode.eval():
                    return true_node
                else:
                    return false_node

            return ir.ASTMultiplexer(self._lower_expr(expr.condnode, bits=1), {
                0: false_node,
                1: true_node,
            })
        elif isinstance(expr, df.DFTerminal):
            return self.circuit.get_port(str(expr.name[1]))
        elif isinstance(expr, df.DFOperator):

            assert len(expr.nextnodes) == 2 or (len(expr.nextnodes) == 1 and expr.operator in ['Unot'])

            if expr.operator == 'Eq' and isinstance(expr.nextnodes[1], df.DFIntConst):
                lhs = self._lower_expr(expr.nextnodes[0])
                bitsize = lhs.get_bitsize()

                lhs_name = self.circuit.generate_internal_signal(bitsize)
                lhs_port = self.circuit.get_port(lhs_name)
                self.circuit.children.append(ir.ASTAssign(lhs_port, lhs))

                const_str = expr.nextnodes[1].value
                if "'" in const_str:
                    _, _, const_str = const_str.partition("'b")

                const_value = int(const_str)

                previous = None

                for i in range(bitsize):
                    my_bit = lhs_port[i]
                    if ((const_value >> i) % 2) == 0:
                        my_bit = ir.ASTLogicGate('not', children=[my_bit])
                    if previous:
                        previous = ir.ASTLogicGate('and', children=[previous, my_bit])
                    else:
                        previous = my_bit

                return previous

            lookup = {'Lor': 'or', 'Or': 'or', 'And': 'and', 'Xor': 'xor', 'Unot': 'not'}

            if expr.operator in ['Plus']:
                return self._emit_expr_intrinsic(expr.operator, [self._lower_expr(x, bits=bits) for x in expr.nextnodes],
                                                 bits=bits)

            return ir.ASTLogicGate(lookup[expr.operator], children=[self._lower_expr(x) for x in expr.nextnodes])
        elif isinstance(expr, df.DFIntConst):
            return self.resolve_const(expr.eval(), bits=bits)
        elif isinstance(expr, df.DFPointer):
            mem_name = str(expr.var.name[1])
            mem_init = ''
            if mem_name in self.mem_initial:
                mem_init = self.mem_initial[mem_name]

            reg_array = self.mem_definitions[mem_name]

            def p(x):
                return df.DFIntConst(x.value).eval()

            assert p(reg_array.width.lsb) == 0
            assert p(reg_array.length.msb) == 0

            address_bits = math.ceil(math.log2(p(reg_array.length.lsb) - p(reg_array.length.msb) + 1))
            data_bits = p(reg_array.width.msb) - p(reg_array.width.lsb) + 1

            address = self._lower_expr(expr.ptr, bits=address_bits)

            data_name = self.circuit.generate_internal_signal(data_bits)
            data_port = ir.ASTPort(data_name, data_bits)

            self.circuit.children.append(ir.ASTSubCircuit('rom', {'Address': address}, {'Data': data_port}, data={
                'address_bits': address_bits,
                'data_bits': data_bits,
                'contents': mem_init,
            }))

            return data_port
        elif isinstance(expr, df.DFConcat):
            pieces = [self._lower_expr(node) for node in expr.nextnodes]
            total_bits = sum(x.get_bitsize() for x in pieces)

            out_name = self.circuit.generate_internal_signal(total_bits)
            out_port = ir.ASTPort(out_name, total_bits)

            high_bit = total_bits - 1
            for piece in pieces:
                low = high_bit - piece.get_bitsize() + 1
                self.circuit.children.append(ir.ASTAssign(ir.ASTSubPort(out_name, low, high_bit), piece))
                high_bit = low - 1

            return out_port
        else:
            raise ValueError

    def _lower_edge(self, info: visit.AlwaysInfo):
        if info.clock_edge is None and info.reset_edge is None:
            return None
        raw_edges = [(str(info.clock_name[1]), info.clock_edge, info.clock_bit)]
        if info.reset_edge:
            raw_edges.append((str(info.reset_name[1]), info.reset_edge, info.reset_bit))

        edges = []
        for raw in raw_edges:
            name = raw[0]
            port = self.circuit.get_port(name)
            if port.bitsize != 1:
                name += '(' + str(raw[2]) + ')'

            my_port = ir.ASTPort(name, 1)
            if raw[1] != 'posedge':
                my_port = ir.ASTLogicGate('not', children=[my_port])

            edges.append(my_port)

        if len(edges) == 2:
            return ir.ASTLogicGate('or', children=edges)

        return edges[0]

    def _lower(self, bv):
        if len(bv.dest) > 2 or (isinstance(bv.tree, df.DFTerminal) and len(bv.tree.name) > 2):
            # this corresponds to sub module param binding which gets lowered later
            return

        if bv.parameterinfo == 'assign':
            dst_port = self.circuit.get_port(str(bv.dest[1]))
            if bv.lsb:
                if bv.lsb == bv.msb:
                    dst_port = dst_port[bv.lsb]
                else:
                    dst_port = dst_port[bv.lsb, bv.msb]

            src = self._lower_expr(bv.tree, ident=dst_port, bits=dst_port.get_bitsize())

            self.circuit.children.append(ir.ASTAssign(dst_port, src))

        elif bv.parameterinfo == 'nonblocking':
            edge = self._lower_edge(bv.alwaysinfo)

            dst_port = self.circuit.get_port(str(bv.dest[1]))
            if bv.lsb:
                if bv.lsb == bv.msb:
                    dst_port = dst_port[bv.lsb]
                else:
                    dst_port = dst_port[bv.lsb, bv.msb]

            src = self._lower_expr(bv.tree, ident=dst_port, bits=dst_port.get_bitsize())

            self.circuit.children.append(ir.ASTAssign(dst_port, src, rising_edge=edge, ))

            pass
        elif bv.parameterinfo == 'parameter':
            pass
        else:
            raise ValueError('unknown parameter info: ' + str(bv.parameterinfo))

    def transform(self):
        for bk, bv in self.binddict.items():
            print('bk:', bk)
            for bvi in bv:
                print('bvi:')
                # print(bvi.tocode())
                print(self._lower(bvi))

        for instance in self.instances:
            module_def = self.other_modules[instance.module]
            module_ports = module_def.portlist.ports
            bound_entries = instance.portlist

            mapping = {}
            for port, arg in zip(module_ports, bound_entries):
                mapping[port.first.name] = self.circuit.get_port(arg.argname.name)

            self.circuit.children.append(ir.ASTSubCircuit(module_def.name, mapping, mapping))

            pass


def generate_intrinsic(info):
    name = '[unknown]'
    circuit = None
    other_intrinsics = set([])

    if info == ('add', 1):
        name = '_add1'
        circuit = ir.ASTCircuit([ir.ASTPort('A', 1), ir.ASTPort('B', 1), ir.ASTPort('Cin', 1)],
                                [ir.ASTPort('X', 1), ir.ASTPort('Cout', 1)])

        circuit.internal_signals['AxorB'] = 1

        circuit.children.append(
            ir.ASTAssign(circuit.get_port('AxorB'),
                         ir.ASTLogicGate('xor', [circuit.get_port('A'),
                                                 circuit.get_port('B')])))

        circuit.children.append(
            ir.ASTAssign(circuit.get_port('X'),
                         ir.ASTLogicGate('xor', [circuit.get_port('AxorB'),
                                                 circuit.get_port('Cin')])))

        circuit.children.append(
            ir.ASTAssign(circuit.get_port('Cout'),
                         ir.ASTLogicGate('or', [ir.ASTLogicGate('and', [circuit.get_port('A'),
                                                                        circuit.get_port('B')]),
                                                ir.ASTLogicGate('and', [circuit.get_port('AxorB'),
                                                                        circuit.get_port('Cin')])
                                                ])))

    elif info[0] == 'add':
        size = info[1]
        other_intrinsics.add(('add', 1))

        name = '_add' + str(size)
        circuit = ir.ASTCircuit([ir.ASTPort('A', size), ir.ASTPort('B', size), ir.ASTPort('Cin', 1)],
                                [ir.ASTPort('X', size), ir.ASTPort('Cout', 1)])

        in_cin = 'Cin'

        for i in range(size):
            carry = 'c' + str(i)
            circuit.internal_signals[carry] = 1

            circuit.children.append(ir.ASTSubCircuit('_add1', {
                'A': circuit.get_port('A')[i],
                'B': circuit.get_port('B')[i],
                'Cin': circuit.get_port(in_cin),
            }, {
                'X': circuit.get_port('X')[i],
                'Cout': circuit.get_port(carry),
            }))
            in_cin = carry

        circuit.children.append(ir.ASTAssign(circuit.get_port('Cout'), circuit.get_port(in_cin)))


    # (circuit, new intrinsics that are needed)
    return name, circuit, other_intrinsics


def compile_module(module_def, other_modules):
    analyzer = VerilogDataflowAnalyzer(sys.argv[1:], module_def.name)
    analyzer.generate()

    directives = analyzer.get_directives()
    terms = analyzer.getTerms()
    binddict = analyzer.getBinddict()

    optimizer = VerilogDataflowOptimizer(terms, binddict)
    optimizer.resolveConstant()

    resolved_terms = optimizer.getResolvedTerms()
    resolved_binddict = optimizer.getResolvedBinddict()
    constlist = optimizer.getConstlist()

    print('Directive:')
    for dr in directives:
        print(dr)

    print('Term:')
    for tk, tv in sorted(resolved_terms.items(), key=lambda x: len(x[0])):
        print(tv.tostr())

    print('Bind:')
    for bk, bv in sorted(resolved_binddict.items(), key=lambda x: len(x[0])):
        print('bk:', bk)
        for bvi in bv:
            print('bvi:')
            print(bvi.tocode())

    print('Const:')
    for ck, cv in sorted(constlist.items(), key=lambda x: len(x[0])):
        print(ck, cv)

    instances = []
    for item in module_def.items:
        if isinstance(item, InstanceList):
            for instance in item.instances:
                instances.append(instance)

    print(instances)

    res = VLowering(module_def.name, resolved_terms, resolved_binddict, instances, other_modules)
    res.transform()

    return res.module, res.circuit, res.required_intrinsics


def main():
    ast, directives = parser.parse(sys.argv[1:])
    ast.show()
    for lineno, directive in directives:
        print('Line %d : %s' % (lineno, directive))

    circuits = {}
    native_circuits = {}
    native_defines = []
    all_intrinsics = set()

    other_modules = {m.name: m for m in ast.description.definitions}

    for module_def in ast.description.definitions:
        native_sim_file = None
        for item in module_def.items:
            if isinstance(item, Decl):
                for param in item.list:
                    if isinstance(param, Parameter):
                        if param.name == 'native_sim':
                            native_sim_file = param.value.var.value

        if native_sim_file:
            sim_file, _, module = native_sim_file.partition(':')

            with open(sim_file, 'r') as f:
                natives = json.load(f)['circuits']

            for native_mod in natives:
                if native_mod['name'] == module:
                    native_circuits[module_def.name] = native_mod
                    break
            else:
                raise ValueError('cannot find native module')

            def filter_ports(cls):
                res = []
                for port in module_def.portlist.ports:
                    if isinstance(port.first, cls):
                        if port.first.width:
                            w = int(port.first.width.msb.value) - int(port.first.width.lsb.value) + 1
                        else:
                            w = 1
                        res.append((port.first.name, w))
                return res

            inputs = filter_ports(Input)
            outputs = filter_ports(Output)

            native_defines.append((module_def.name, inputs, outputs))

        else:
            name, circuit, req_intrinsics = compile_module(module_def, other_modules)
            circuits[name] = circuit
            all_intrinsics.update(req_intrinsics)

    processed_intrinsics = set()
    while all_intrinsics:
        intrinsic = all_intrinsics.pop()
        processed_intrinsics.add(intrinsic)

        name, circuit, others = generate_intrinsic(intrinsic)
        circuits[name] = circuit

        all_intrinsics.update(others.difference(processed_intrinsics))

    print(all_intrinsics, processed_intrinsics)

    # with open('circuitsim/intrinsics.sim', 'r') as f:
    #     native_circuits = json.load(f)['circuits']
    #
    # native_defines = [
    #     ('Memory', [('Address', 16), ('In', 32), ('Clock', 1), ('Write', 1)], [('Out', 32)])
    # ]

    # for entry, _, _ in native_defines:
    #     if entry in circuits:
    #         circuits.pop(entry)

    ir_lowering.compile_circuits(circuits, filename='circuitsim/verilog.sim',
                                 native_circuits=list(native_circuits.values()), defines=native_defines)


if __name__ == '__main__':
    main()
