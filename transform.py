import ir


class TransformBase(object):
    def transform(self, node: ir.ASTNode):
        # must return new node, may be recursive.
        raise NotImplementedError
