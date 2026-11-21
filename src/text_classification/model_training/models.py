from sklearn.linear_model import LogisticRegression


def define_logreg(c: float, max_iter: int, random_state: int) -> LogisticRegression:
    """Define a Logistic Regression classifier.
    Args:
        c: Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization. C=np.inf results in unpenalized logistic regression.
        max_iter: Maximum number of iterations taken for the solvers to converge.

    Returns:
        Logistic regression model.
    """
    model = LogisticRegression(C=c, max_iter=max_iter, random_state=random_state)

    return model
