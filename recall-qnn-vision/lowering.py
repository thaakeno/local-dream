"""Explicit ONNX numerical lowering, with a conservative sign proof.

HTP qualification must still compare the complete lowered component and model.
This pass changes neither trained parameters nor model output values by policy;
it expresses exp(x) as sigmoid(x)/sigmoid(-x) only when x is proven <= 0.
The denominator is then >= 0.5, avoiding the failing large-negative Exp kernel.
No runtime patching, clipping, epsilon changes or NaN replacement is used.
"""


def lower_nonpositive_exp(model):
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    producers = {out: node for node in graph.node for out in node.output}
    initializers = {x.name: x for x in graph.initializer}
    unknown = frozenset((-1, 0, 1))
    cache = {}

    def array_signs(value):
        if not np.isfinite(value).all():
            return unknown
        return frozenset(np.sign(value).astype(np.int8).reshape(-1).tolist())

    def signs(tensor):
        if tensor in cache:
            return cache[tensor]
        cache[tensor] = unknown  # Also guards malformed cyclic graphs.
        if tensor in initializers:
            result = array_signs(numpy_helper.to_array(initializers[tensor]))
        elif tensor not in producers:
            result = unknown
        else:
            node = producers[tensor]
            op = node.op_type
            if op == "Constant":
                attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                result = array_signs(numpy_helper.to_array(attrs["value"])) if "value" in attrs else unknown
            elif op in {"Softplus", "Exp", "Sigmoid", "Abs"}:
                result = frozenset((0, 1))
            elif op == "Neg":
                result = frozenset(-s for s in signs(node.input[0]))
            elif op in {"Identity", "Transpose", "Reshape", "Squeeze", "Unsqueeze", "Gather", "Slice"}:
                result = signs(node.input[0])
            elif op == "Cast":
                target = next(a.i for a in node.attribute if a.name == "to")
                result = signs(node.input[0]) if target in (TensorProto.FLOAT, TensorProto.FLOAT16,
                                                          TensorProto.DOUBLE, TensorProto.BFLOAT16) else unknown
            elif op in {"Mul", "MatMul"}:
                products = frozenset(a * b for a in signs(node.input[0]) for b in signs(node.input[1]))
                result = products if op == "Mul" or not {-1, 1}.issubset(products) else unknown
            elif op == "Concat":
                result = frozenset(s for source in node.input for s in signs(source))
            else:
                result = unknown
        cache[tensor] = result
        return result

    rewritten, nodes = [], []
    occupied = {out for node in graph.node for out in node.output} | set(initializers)
    for node in graph.node:
        # Constant Exp is already folded by the compiler; do not alter it.
        source = node.input[0] if node.op_type == "Exp" else None
        producer = producers.get(source)
        eligible = source and source not in initializers and producer is not None and producer.op_type != "Constant"
        proof = signs(source) if eligible else unknown
        if node.op_type == "Exp" and eligible and proof.issubset({-1, 0}):
            prefix = node.output[0] + "__stable_decay"
            extra = [prefix + suffix for suffix in ("_sigmoid", "_neg", "_neg_sigmoid")]
            if any(name in occupied for name in extra):
                raise RuntimeError("Stable-decay tensor name collision")
            occupied.update(extra)
            pos, neg, neg_sigmoid = extra
            nodes.extend([
                helper.make_node("Sigmoid", [source], [pos], name=prefix + "_positive"),
                helper.make_node("Neg", [source], [neg], name=prefix + "_negative"),
                helper.make_node("Sigmoid", [neg], [neg_sigmoid], name=prefix + "_denominator"),
                helper.make_node("Div", [pos, neg_sigmoid], list(node.output), name=node.name + "_algebraic"),
            ])
            rewritten.append({"node": node.name, "input": source, "proven_signs": sorted(proof),
                              "identity": "exp(x)=sigmoid(x)/sigmoid(-x), x<=0"})
        else:
            nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    return rewritten
