from pdb import set_trace
import numpy as np

aa = [
  [
    [ 1, 2 ],
    3
  ],
  [
    4,
    5
  ]
]

def traverse(node, leafs=[]):
    if type(node) == list:
        traverse(node[0], leafs)
        traverse(node[1], leafs)
    else:
        leafs.append(node)



def traverse(root, leafs=[]):
    nodes = [root]

    while len(nodes) != 0:
        node = nodes[-1]
        nodes = nodes[:-1]

        if type(node) == list:
            nodes.append(node[0])
            if len(node) > 1:
                nodes.append(node[1])
        else:
            leafs.append(node)

    leafs.reverse()

leafs = []

set_trace()

traverse(aa, leafs)
