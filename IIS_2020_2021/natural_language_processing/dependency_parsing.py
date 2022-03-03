import os
import nltk
from nltk.parse.stanford import StanfordParser, StanfordDependencyParser
from nltk.tree import Tree

os.environ['JAVAHOME'] = 'C:/Program Files/Java/jdk-12.0.1/bin'
nltk.internals.config_java('C:/Program Files/Java/jdk-12.0.1/bin/java')
path_to_jar = 'D:/StanfordParser/stanford-parser.jar'
path_to_models_jar = 'D:/StanfordParser/stanford-english-corenlp-2018-10-05-models.jar'


def parse_tree(sentence):
    parser = StanfordParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
    tree = parser.raw_parse(sentence).__next__()
    return tree


def tree_to_dict(tree):
    tdict = {}
    for t in tree:
        if isinstance(t, Tree) and isinstance(t[0], Tree):
            tdict[t.label()] = tree_to_dict(t)
        elif isinstance(t, Tree):
            tdict[t.label()] = {t[0]: None}
    return tdict


def parse(node, tree, depth=1):
    result = []
    if tree[node] is None:
        return [[node] * depth]
    else:
        res = []
        for next_node in tree[node]:
            res.extend(parse(next_node, tree[node], depth + 1))
        for r in res:
            r[depth - 1] = node
            result.append(r)
        return result


def parse_tree_features(tree):
    tree_dict = tree_to_dict(tree)
    paths = parse(list(tree_dict.keys())[0], tree_dict)
    return ['-'.join(path[-3:]).lower() for path in paths]


def dependency_parse(sentence):
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
    dependencies = dependency_parser.raw_parse(sentence).__next__()
    rel = list()
    for dependency in list(dependencies.triples()):
        rel.append([dependency[0][0].lower(), dependency[2][0].lower()])
    return rel


if __name__ == '__main__':
    parse_tree_features(parse_tree('The quick brown fox jumps over the lazy dog.'))
    dependency_parse('The quick brown fox jumps over the lazy dog.')
