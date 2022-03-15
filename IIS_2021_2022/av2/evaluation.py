from sklearn.metrics import accuracy_score, f1_score


def calculate_metrics(test_targets, predictions):
    """Calculation of accuracy score, F1 micro and F1 macro"""
    print(f'\tAccuracy score: {accuracy_score(test_targets, predictions)}')
    print(f'\tF1-micro: {f1_score(test_targets, predictions, average="micro")}')
    print(f'\tF1-macro: {f1_score(test_targets, predictions, average="macro")}')
