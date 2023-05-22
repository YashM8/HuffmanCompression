
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """
    Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    >>> d1 = build_frequency_dict(bytes([65, 66, 67, 66, 67, 67, 67]))
    >>> d1 == {65: 1, 66: 2, 67: 4}
    True

    ===============================CITATIONS===============================

    https://www.youtube.com/watch?v=JsTptu56GM8
    Video explaining Huffman Trees that helped me visualize the Tree.

    https://pythontutor.com/visualize.html#mode=edit
    I used the Python visualizer to visualize and debug aspects of my code.

    https://cmps-people.ok.ubc.ca/ylucet/DS/Huffman.html
    I used to visualize Huffman Tree with my input typed in as text.

    https://www.programiz.com/python-programming/methods/built-in/bytes
    I used this website to learn about bytes()

    """
    f_dict = {}

    for k in text:
        if k in f_dict:
            f_dict[k] += 1
        else:
            f_dict[k] = 1

    return f_dict


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """
    Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    This function takes a frequency dictionary and returns a HuffmanTree
    corresponding to the best prefix code. There’s a tricky case to watch
    for here: you have to think of what to do when the frequency
    dictionary has only one symbol!.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2),\
    HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    freq_list = []
    order = 0

    for item in freq_dict.items():
        freq_list.append((HuffmanTree(item[0]), item[1], order))
        order += 1

    freq_list = sorted(freq_list, key=lambda x: (x[1], x[2]), reverse=True)

    if len(freq_list) == 1:
        symbol = freq_list[0][0].symbol
        b_random = (symbol + 1) % 256
        dummy_tree = HuffmanTree(b_random)
        return HuffmanTree(None, HuffmanTree(symbol), dummy_tree)

    while len(freq_list) > 1:
        h1 = freq_list[-1]
        h2 = freq_list[-2]
        freq_list = freq_list[:-2]

        tree = HuffmanTree(None, h1[0], h2[0])
        freq_list.append((tree, h1[1] + h2[1], order))
        order += 1

        freq_list = sorted(freq_list, key=lambda x: (x[1], x[2]), reverse=True)

    return freq_list[0][0]


def _recurse_tree(tree: HuffmanTree, dict1: dict[int, str], route: str) -> None:
    """
    Helper for get_codes. Routes for each leaf, concatenating 0's and 1's.
    """
    if tree.is_leaf():
        dict1[tree.symbol] = route
    else:
        _recurse_tree(tree.left, dict1, route + "0")
        _recurse_tree(tree.right, dict1, route + "1")


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """
    Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """

    dict1 = {}
    _recurse_tree(tree, dict1, "")

    return dict1


def _number_assign(tree: HuffmanTree, num: int) -> int:
    """
    Helper for number_nodes. Counting using postorder traversal and
    assigning number.
    """
    if tree is None or tree.is_leaf():
        return num
    else:
        num = _number_assign(tree.left, num)
        num = _number_assign(tree.right, num)
        tree.number = num
        num += 1
        return num


def number_nodes(tree: HuffmanTree) -> None:
    """
    Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    >>> far_left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> left = HuffmanTree(None, far_left, HuffmanTree(4))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.left.number
    0
    >>> tree.left.number
    1
    >>> tree.right.number
    2
    >>> tree.number
    3
    """
    _number_assign(tree, 0)


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """
    Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    codes = get_codes(tree)
    count = 0
    sum_of_f = 0

    for k in codes:
        count += len(codes[k]) * freq_dict[k]
        sum_of_f += freq_dict[k]

    return count / sum_of_f


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """
    Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    lst = []
    str_list = []

    for byte in text:
        str_list.append(codes[byte])

    str1 = ''.join(str_list)

    for i in range(0, len(str1), 8):
        lst.append(bits_to_byte(str1[i: i + 8]))

    return bytes(lst)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """
    Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    tree_list = _postorder_helper(tree)
    lst_bytes = []

    for elem in tree_list:
        if elem.left.is_leaf():
            lst_bytes += [0, elem.left.symbol]

        else:
            lst_bytes += [1, elem.left.number]

        if elem.right.is_leaf():
            lst_bytes += [0, elem.right.symbol]

        else:
            lst_bytes += [1, elem.right.number]

    return bytes(lst_bytes)


def _postorder_helper(tree: HuffmanTree) -> list[HuffmanTree]:
    """
    Helper for tree_to_bytes. Returns a list of HuffmanTree in postorder
    form.
    """
    if tree.is_leaf():
        return []

    return _postorder_helper(tree.left) + _postorder_helper(tree.right) + [tree]


def compress_file(in_file: str, out_file: str) -> None:
    """
    Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """
    Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    if len(node_lst) == 1:
        return HuffmanTree(None, HuffmanTree(node_lst[root_index].l_data),
                           HuffmanTree(node_lst[root_index].r_data))

    else:
        if node_lst[root_index].r_type == 0:
            right = HuffmanTree(node_lst[root_index].r_data)

        else:
            right = generate_tree_general(node_lst, node_lst[root_index].r_data)

        if node_lst[root_index].l_type == 0:
            left = HuffmanTree(node_lst[root_index].l_data)

        else:
            left = generate_tree_general(node_lst, node_lst[root_index].l_data)

        return HuffmanTree(None, left, right)


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """
    Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.
    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    copy = root_index

    if node_lst[root_index].r_type == 1:
        right = HuffmanTree(None, HuffmanTree(
            node_lst[root_index - 1].l_data), HuffmanTree(
            node_lst[root_index - 1].r_data))
        root_index -= 1
    else:
        right = generate_tree_postorder(node_lst, root_index - 1)

    if node_lst[copy].l_type == 1:
        left = HuffmanTree(None, HuffmanTree(
            node_lst[root_index - 1].l_data), HuffmanTree(
            node_lst[root_index - 1].r_data))

    else:
        left = generate_tree_postorder(node_lst, root_index - 1)

    return HuffmanTree(None, left, right)


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """
    Use Huffman tree <tree> to decompress <size> bytes from <text>.
    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    node = tree
    bit_str = ''

    for byte in text:
        bit_str += byte_to_bits(byte)

    byte_list = []

    for bit in bit_str:
        if len(byte_list) == size:
            break

        if bit == '1':
            node = node.right
        else:
            node = node.left

        if node.is_leaf():
            byte_list += [node.symbol]
            node = tree

    return bytes(byte_list)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    queue1 = [tree]

    sort_list = [k[0] for k in sorted(freq_dict.items(), key=lambda x: x[1])]

    while len(queue1) > 0:
        curr = queue1[-1]
        queue1 = queue1[:-1]

        if curr.is_leaf() and len(sort_list) > 0:
            curr.symbol = sort_list[-1]
            sort_list = sort_list[:-1]

        if curr.left is not None:
            queue1.insert(0, curr.left)

        if curr.right is not None:
            queue1.insert(0, curr.right)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
