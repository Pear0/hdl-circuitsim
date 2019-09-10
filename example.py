import json
import re
import sys
import traceback
from collections import namedtuple

from hdlConvertor import HdlConvertor
from hdlConvertor.hdlAst import HdlModuleDef, HdlStmAssign, HdlCall
from hdlConvertor.language import Language
from hdlConvertor.toVhdl import ToVhdl

from circuits import DipCircuitLayoutSpec, BufferCircuitLayoutSpec, DFlipFlopCircuitLayoutSpec, CircuitSpec, SimCircuit, \
    temp_name_generator

c = HdlConvertor()
filenames = ["vhdl/mux.vhd", ]
include_dirs = []
d = c.parse(filenames, Language.VHDL, include_dirs, hierarchyOnly=False, debug=True)


def flatten_visitor(node, name_gen, root_children):
    assert isinstance(node, HdlStmAssign)

    if isinstance(node.src, HdlCall):

        for i, op in enumerate(node.src.ops):
            if isinstance(op, HdlCall):
                t = next(name_gen)

                gen_assign = HdlStmAssign(op, t)
                flatten_visitor(gen_assign, name_gen, root_children)

                node.src.ops[i] = t

    root_children.append(node)


def flatten_module(module):
    name_gen = temp_name_generator()
    children = []

    for root in module.objs:
        flatten_visitor(root, name_gen, children)

    module.objs = children


if __name__ == '__main__':
    for i, root in enumerate(d.objs):
        if isinstance(root, HdlModuleDef) and i == 4:
            flatten_module(root)
        print(root)

    try:
        tv = ToVhdl(sys.stdout)
        tv.print_context(d)
    except ValueError:
        traceback.print_exc()

    with open('circuitsim/gen.sim', 'w') as f:
        json.dump({
            'version': '1.8.2',
            'globalBitSize': 1,
            'clockSpeed': 1,
            'circuits': [SimCircuit(CircuitSpec('foo', DipCircuitLayoutSpec, [], [])).build(), {
                "name": "Bar",
                "components": [
                    {
                        "name": "com.ra4king.circuitsim.gui.peers.gates.AndGatePeer",
                        "x": 23,
                        "y": 16,
                        "properties": {
                            "Negate 1": "No",
                            "Label location": "NORTH",
                            "Negate 0": "No",
                            "Number of Inputs": "2",
                            "Label": "",
                            "Direction": "EAST",
                            "Bitsize": "1"
                        }
                    },
                    {
                        "name": "com.ra4king.circuitsim.gui.peers.wiring.PinPeer",
                        "x": 15,
                        "y": 13,
                        "properties": {
                            "Label location": "WEST",
                            "Label": "A",
                            "Is input?": "Yes",
                            "Direction": "EAST",
                            "Bitsize": "1"
                        }
                    },
                    {
                        "name": "com.ra4king.circuitsim.gui.peers.wiring.PinPeer",
                        "x": 15,
                        "y": 21,
                        "properties": {
                            "Label location": "WEST",
                            "Label": "B",
                            "Is input?": "Yes",
                            "Direction": "EAST",
                            "Bitsize": "1"
                        }
                    },
                    {
                        "name": "com.ra4king.circuitsim.gui.peers.wiring.PinPeer",
                        "x": 30,
                        "y": 17,
                        "properties": {
                            "Label location": "EAST",
                            "Label": "C",
                            "Is input?": "No",
                            "Direction": "WEST",
                            "Bitsize": "1"
                        }
                    }
                ],
                "wires": [
                    {
                        "x": 17,
                        "y": 14,
                        "length": 6,
                        "isHorizontal": True
                    },
                    {
                        "x": 17,
                        "y": 22,
                        "length": 6,
                        "isHorizontal": True
                    },
                    {
                        "x": 23,
                        "y": 14,
                        "length": 3,
                        "isHorizontal": False
                    },
                    {
                        "x": 23,
                        "y": 19,
                        "length": 3,
                        "isHorizontal": False
                    },
                    {
                        "x": 27,
                        "y": 18,
                        "length": 3,
                        "isHorizontal": True
                    }
                ]
            }]
        }, f)
