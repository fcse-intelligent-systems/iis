def save_embeddings(file_path, embs, nodes):
    """Save node embeddings

    :param file_path: path to the output file
    :type file_path: str
    :param embs: matrix containing the embedding vectors
    :type embs: numpy.array
    :param nodes: list of node names
    :type nodes: list(int)
    :return: None
    """
    with open(file_path, 'w') as f:
        f.write(f'{embs.shape[0]} {embs.shape[1]}\n')
        for node, emb in zip(nodes, embs):
            f.write(f'{node} {" ".join(map(str, emb.tolist()))}\n')


def read_embeddings(file_path):
    """ Load node embeddings
    :param file_path: path to the embedding file
    :type file_path: str
    :return: dictionary containing the node names as keys
    and the embeddings vectors as values
    :rtype: dict(int, numpy.array)
    """
    with open(file_path, 'r') as f:
        f.readline()
        embs = {}
        line = f.readline().strip()
        while line != '':
            parts = line.split()
            embs[int(parts[0])] = np.array(list(map(float, parts[1:])))
            line = f.readline().strip()
    return embs
