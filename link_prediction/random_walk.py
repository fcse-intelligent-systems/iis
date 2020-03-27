import numpy as np


def get_stochastic_transition_matrix(A):
    """ Calculates the stochastic transition matrix

    :param A: adjacency matrix with edge strengths
    :type A: numpy.array
    :return: stochastic transition matrix
    :rtype: numpy.array
    """
    Q_prim = 1 / np.sum(A, axis=1).reshape(-1, 1) * A
    return Q_prim


def get_transition_matrix(Q_prim, start_node, alpha):
    """ Calculate the transition matrix from given stochastic transition matrix,
    start node and restart probability

    :param Q_prim: stochastic transition matrix
    :type Q_prim: numpy.array
    :param start_node: index of the start node
    :type start_node: int
    :param alpha: restart probability
    :type alpha: float
    :return: transition matrix
    :rtype: numpy.array
    """
    one = np.zeros(Q_prim.shape)
    one[:, start_node] = 1
    return (1 - alpha) * Q_prim + alpha * one


def iterative_page_rank(trans, epsilon, max_iter):
    """ Iterative power-iterator like computation of PageRank vector p

    :param trans: transition matrix
    :type trans: numpy.array
    :param epsilon: tolerance parameter
    :type epsilon: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :return: stationary distribution
    :rtype: numpy.array
    """
    p = np.ones((1, trans.shape[0])) / trans.shape[0]
    p_new = np.dot(p, trans)
    for t in range(max_iter):
        if np.allclose(p, p_new, rtol=0, atol=epsilon):
            break
        p = p_new
        p_new = np.dot(p, trans)
    return p_new[0]


def random_walk(A, source, alpha=0.3, max_iter=100):
    """ Random walk with given parameters and directed graph

    :param A: adjacency matrix
    :type A: numpy.array
    :param source: index of source node
    :type source: int
    :param alpha: restart probability
    :type alpha: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :return: p vector for every source node
    :rtype: numpy.array
    """
    epsilon = 1e-12

    Q_prim = get_stochastic_transition_matrix(A)
    Q = get_transition_matrix(Q_prim, source, alpha)
    p = iterative_page_rank(Q, epsilon, max_iter)

    return p
