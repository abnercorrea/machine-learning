from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def make_classification_train_test(n_samples, n_classes, n_features, n_informative, n_redundant, test_size, random_state):
    # define dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state
    )

    # split train and test data
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return xtr, xte, ytr, yte
