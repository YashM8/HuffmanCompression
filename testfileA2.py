from compress import *
from utils import bytes_to_nodes


def test_build_huffman() -> None:
    t = build_huffman_tree({2: 6, 3: 4, 7: 5})
    result = HuffmanTree(None, HuffmanTree(2, None, None),
                HuffmanTree(None, HuffmanTree(3, None, None),
                            HuffmanTree(7, None, None)))
    assert t == result

    freq = {2: 6}
    t = build_huffman_tree(freq)
    result2 = HuffmanTree(None, HuffmanTree(2), HuffmanTree((2 + 1) % 255))
    assert t.left == result2.left

    freq4 = {1: 2, 2: 4, 3: 2, 4: 4, 5: 4}
    t4 = build_huffman_tree(freq4)
    result4 = HuffmanTree(None, HuffmanTree(None, HuffmanTree(2),
                                                HuffmanTree(4)),
                              HuffmanTree(None, HuffmanTree(5),
                                          HuffmanTree(None, HuffmanTree(1),
                                                      HuffmanTree(3))))
    assert t4 == result4


def test_get_codes() -> None:
    tree2 = HuffmanTree(None, HuffmanTree(2, None, None),
                            HuffmanTree(None, HuffmanTree(3, None, None),
                                         HuffmanTree(7, None, None)))
    x = get_codes(tree2)
    assert x == {2: '0', 3: '10', 7: '11'}

    tree3 = HuffmanTree(None, HuffmanTree(None, HuffmanTree(2),
                                                HuffmanTree(4)),
                              HuffmanTree(None, HuffmanTree(5),
                                          HuffmanTree(None, HuffmanTree(1),
                                                      HuffmanTree(3))))
    x2 = get_codes(tree3)
    assert x2 == {1: '110', 2: '00', 3: '111', 4: '01', 5: '10'}


def test_number_nodes() -> None:
    tree1 = HuffmanTree(None, HuffmanTree(1), HuffmanTree(None, HuffmanTree(2), HuffmanTree(9)))
    tree2 = HuffmanTree(None, HuffmanTree(3), HuffmanTree(4))
    tree3 = HuffmanTree(None, HuffmanTree(5), HuffmanTree(6))
    tree4 = HuffmanTree(None, HuffmanTree(7), HuffmanTree(8))
    left = HuffmanTree(None, tree1, tree2)
    right = HuffmanTree(None, tree3, tree4)
    tree = HuffmanTree(None, left, right)
    number_nodes(tree)
    assert tree.number == 7
    assert tree.left.number == 3
    assert tree.right.number == 6
    assert tree.left.left.number == 1
    assert tree.left.left.right.number == 0
    assert tree.left.right.number == 2
    assert tree.right.left.number == 4
    assert tree.right.right.number == 5


def test_avg_length() -> None:
    freq = {1: 2, 2: 4, 3: 2, 4: 4, 5: 4}
    tree = HuffmanTree(None, HuffmanTree(None, HuffmanTree(2),
                                                HuffmanTree(4)),
                              HuffmanTree(None, HuffmanTree(5),
                                          HuffmanTree(None, HuffmanTree(1),
                                                      HuffmanTree(3))))

    assert avg_length(tree, freq) == (2*4 + 2*4 + 2*4 + 3*2 + 3*2) / 16


def test_compress_bytes() -> None:
    d = {0: "0", 1: "10", 2: "11"}
    text2 = bytes([1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0])
    result2 = compress_bytes(text2, d)
    x = [byte_to_bits(byte) for byte in result2]
    assert x == ['10111001', '01110010', '11100000']


def test_generate_tree_general() -> None:
    lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
                ReadNode(1, 4, 1, 1), ReadNode(0, 6, 0, 8),
                ReadNode(1, 3, 1, 0), ReadNode(1, 1, 1, 2)]
    x = generate_tree_general(lst, 2)
    tree = HuffmanTree(None, HuffmanTree(None, \
                                  HuffmanTree(None, HuffmanTree(6, None, None), \
                                              HuffmanTree(8, None, None)), \
                                  HuffmanTree(None, HuffmanTree(5, None, None), \
                                              HuffmanTree(7, None, None))), \
                HuffmanTree(None, HuffmanTree(10, None, None), \
                            HuffmanTree(12, None, None)))
    assert x == tree

    lst2 = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
                ReadNode(1, 4, 1, 5), ReadNode(0, 6, 0, 8),
                ReadNode(1, 3, 1, 0), \
                ReadNode(0, 9, 1, 1), ReadNode(1, 1, 1, 2)]

    y = generate_tree_general(lst2, 2)
    tree2 = HuffmanTree(None, HuffmanTree(None, \
                                  HuffmanTree(None, HuffmanTree(6, None, None), \
                                              HuffmanTree(8, None, None)), \
                                  HuffmanTree(None, HuffmanTree(5, None, None), \
                                              HuffmanTree(7, None, None))), \
                HuffmanTree(None, HuffmanTree(9, None, None), \
                            HuffmanTree(None, HuffmanTree(10, None, None),
                                        HuffmanTree(12, None, None))))
    assert y == tree2


def test_generate_tree_postorder() -> None:
    leaf1 = HuffmanTree(None, HuffmanTree(3), HuffmanTree(4))
    leaf2 = HuffmanTree(None, HuffmanTree(6), HuffmanTree(7))
    leaf3 = HuffmanTree(None, HuffmanTree(10), HuffmanTree(12))
    leaf4 = HuffmanTree(None, HuffmanTree(9), \
                             HuffmanTree(None, HuffmanTree(11),
                                         HuffmanTree(13)))
    branch1 = HuffmanTree(None, leaf1, leaf2)
    branch2 = HuffmanTree(None, leaf3, leaf4)
    final_tree = HuffmanTree(None, branch1, branch2)

    lst = [ReadNode(0, 3, 0, 4), ReadNode(0, 6, 0, 7), \
                ReadNode(1, 0, 1, 0), ReadNode(0, 10, 0, 12),
                ReadNode(0, 11, 0, 13), \
                ReadNode(0, 9, 1, 0), ReadNode(1, 0, 1, 0),
                ReadNode(1, 0, 1, 0)]

    assert generate_tree_postorder(lst, 7) == final_tree

    lst2 = [ReadNode(0, 2, 0, 3), ReadNode(1, 0, 0, 4), ReadNode(1, 0, 0, 5)]
    y = generate_tree_postorder(lst2, 2)
    assert y == HuffmanTree(None,
                HuffmanTree(None, HuffmanTree(None, HuffmanTree(2, None, None), \
                                              HuffmanTree(3, None, None)),
                            HuffmanTree(4, None, None)),
                HuffmanTree(5, None, None))

    lst3 = [ReadNode(0, 1, 0, 2), ReadNode(1, 0, 0, 3),
                ReadNode(1, 1, 0, 4), ReadNode(0, 5, 0, 6),
                ReadNode(0, 7, 0, 8), ReadNode(1, 3, 1, 4),
                ReadNode(1, 2, 1, 5), ReadNode(0, 1, 0, 1)]
    y2 = generate_tree_postorder(lst3, 6)
    assert y2 == HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(None,
                                                                      HuffmanTree(
                                                                          1,
                                                                          None,
                                                                          None),
                                                                      HuffmanTree(
                                                                          2,
                                                                          None,
                                                                          None)),
                                                    HuffmanTree(3, None, None)),
                                  HuffmanTree(4, None, None)), HuffmanTree(None,
                                                                           HuffmanTree(
                                                                               None,
                                                                               HuffmanTree(
                                                                                   5,
                                                                                   None,
                                                                                   None),
                                                                               HuffmanTree(
                                                                                   6,
                                                                                   None,
                                                                                   None)),
                                                                           HuffmanTree(
                                                                               None,
                                                                               HuffmanTree(
                                                                                   7,
                                                                                   None,
                                                                                   None),
                                                                               HuffmanTree(
                                                                                   8,
                                                                                   None,
                                                                                   None))))

    lst4 = [ReadNode(0, 2, 0, 3), ReadNode(0, 1, 1, 0),
                ReadNode(0, 4, 0, 5), ReadNode(1, 0, 1, 0),
                ReadNode(0, 6, 0, 10), ReadNode(0, 8, 0, 9),
                ReadNode(1, 0, 1, 0), ReadNode(1, 0, 1, 0),
                ReadNode(1, 4, 1, 5)]
    y4 = generate_tree_postorder(lst4, 7)
    assert y4 == HuffmanTree(None, HuffmanTree(None,
                                  HuffmanTree(None, HuffmanTree(1, None, None),
                                              HuffmanTree(None,
                                                          HuffmanTree(2, None,
                                                                      None),
                                                          HuffmanTree(3, None,
                                                                      None))),
                                  HuffmanTree(None, HuffmanTree(4, None, None),
                                              HuffmanTree(5, None, None))),
                HuffmanTree(None, HuffmanTree(None, HuffmanTree(6, None, None),
                                              HuffmanTree(10, None, None)),
                            HuffmanTree(None, HuffmanTree(8, None, None),
                                        HuffmanTree(9, None, None))))


def test_decompress_bytes() -> None:
    tree = build_huffman_tree(build_frequency_dict(b'hellomynameissam'))
    number_nodes(tree)
    x = decompress_bytes(tree, \
                          compress_bytes(b'hellomynameissam', get_codes(tree)),
                          len(b'hellomynameissam'))
    assert x == b'hellomynameissam'

    tree2 = build_huffman_tree(build_frequency_dict(b'123456789'))
    number_nodes(tree2)
    x = decompress_bytes(tree2, \
                         compress_bytes(b'123456789', get_codes(tree2)),
                         len(b'123456789'))
    assert x == b'123456789'


def test_improve_tree() -> None:
    left1 = HuffmanTree(None, HuffmanTree(None, \
                                               HuffmanTree(97, None, None),
                                               HuffmanTree(98, None, None)), \
                             HuffmanTree(99, None, None))
    right1 = HuffmanTree(None, HuffmanTree(101, None, None), \
                              HuffmanTree(100, None, None))
    treetwo1 = HuffmanTree(None, left1, right1)
    freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    left2 = HuffmanTree(None, HuffmanTree(None, \
                                               HuffmanTree(101, None, None),
                                               HuffmanTree(100, None, None)), \
                             HuffmanTree(99, None, None))
    right2 = HuffmanTree(None, HuffmanTree(98, None, None), \
                              HuffmanTree(97, None, None))
    treetwo2 = HuffmanTree(None, left2, right2)
    assert avg_length(treetwo1, freq) == 2.49

    improve_tree(treetwo1, freq)
    assert avg_length(treetwo1, freq) == 2.31

    freq2 = {1: 2, 2: 4, 3: 2, 4: 4, 5: 4}
    tree = HuffmanTree(None, HuffmanTree(None, HuffmanTree(1),
                                         HuffmanTree(3)),
                       HuffmanTree(None, HuffmanTree(5),
                                   HuffmanTree(None, HuffmanTree(2),
                                               HuffmanTree(4))))
    assert avg_length(tree, freq2) != (
            2 * 4 + 2 * 4 + 2 * 4 + 3 * 2 + 3 * 2) / 16
    improve_tree(tree, freq2)
    assert avg_length(tree, freq2) == (
                2 * 4 + 2 * 4 + 2 * 4 + 3 * 2 + 3 * 2) / 16


def test_tree_to_bytes() -> None:
    tree = HuffmanTree(None,
                       HuffmanTree(None,
                                   HuffmanTree(1),
                                   HuffmanTree(3)),
                       HuffmanTree(None,
                                   HuffmanTree(5),
                                   HuffmanTree(None,
                                               HuffmanTree(2),
                                               HuffmanTree(4))))
    number_nodes(tree)
    bytes = tree_to_bytes(tree)

    nodes = bytes_to_nodes(bytes)
    newtree = generate_tree_general(nodes, len(nodes) - 1)
    number_nodes(newtree)
    assert tree.number == newtree.number
    assert tree.left.number == newtree.left.number
    assert tree.right.number == newtree.right.number
    assert tree.right.right.number == newtree.right.right.number




