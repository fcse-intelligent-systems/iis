def relation_with_positive_intensifier(word, relations):
    POSITIVE_INTENSIFIERS = ['absolutely', 'completely', 'extremely', 'highly', 'really', 'so', 'too', 'totally',
                             'utterly', 'very', 'much', 'lots', 'pretty', 'high', 'huge', 'most', 'more', 'deeply',
                             'clearly', 'strongly']
    for rel in relations:
        if rel[0] == word.lower() and rel[1] in POSITIVE_INTENSIFIERS:
            return True
        if rel[1] == word.lower() and rel[0] in POSITIVE_INTENSIFIERS:
            return True
    return False


def relation_with_negative_intensifier(word, relations):
    NEGATIVE_INTENSIFIERS = ['scarcely', 'little', 'few', 'some', 'small', 'hardly', 'barely']
    for rel in relations:
        if rel[0] == word.lower() and rel[1] in NEGATIVE_INTENSIFIERS:
            return True
        if rel[1] == word.lower() and rel[0] in NEGATIVE_INTENSIFIERS:
            return True
    return False


def relation_with_negation(word, relations):
    NEGATIONS = ['not', 'never', 'none', 'nobody', 'nowhere', 'nothing', 'neither', 'no', 'noone', 'n\'t']
    for rel in relations:
        if rel[0] == word.lower() and rel[1] in NEGATIONS:
            return True
        if rel[1] == word.lower() and rel[0] in NEGATIONS:
            return True
    return False


def relation_with_connector(word, relations):
    CONNECTORS = ['although', 'however', 'but', 'notwithstanding', 'nevertheless', 'nonetheless', 'yet', 'instead',
                  'moreover', 'still', 'unfortunately', 'originally', 'surprisingly', 'ideally', 'apparently', 'though',
                  'despite', 'conversely', 'while', 'whereas', 'unlike']
    for rel in relations:
        if rel[0] == word.lower() and rel[1] in CONNECTORS:
            return True
        if rel[1] == word.lower() and rel[0] in CONNECTORS:
            return True
    return False
