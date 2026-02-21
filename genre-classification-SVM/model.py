from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def train_svm(X_train, y_train, kernel='rbf', C=10):
    """
    Train a Support Vector Machine (SVM) classifier on the extracted audio features.

    Parameters
    ----------
    X_train : array-like
        Feature matrix (MFCC-based vectors) for training samples.

    y_train : array-like
        Corresponding genre labels for each training sample.

    kernel : str, optional (default='rbf')
        The kernel type to be used in the SVM.
        'rbf' works well for non-linear decision boundaries, which are
        common in real-world audio feature spaces.

    C : float, optional (default=10)
        Regularization parameter.
        Controls the trade-off between maximizing margin and minimizing
        classification error.
        Higher C → less tolerance for misclassification.

    Returns
    -------
    model : sklearn Pipeline
        A trained pipeline consisting of:
        - StandardScaler (feature normalization)
        - SVC classifier
    """

    # SVMs are sensitive to feature scaling.
    # Since MFCC values can vary in magnitude, we standardize them first.
    # The pipeline ensures scaling is applied consistently during training
    # and future predictions.
    model = make_pipeline(
        StandardScaler(),      # Normalize features (mean=0, std=1)
        SVC(kernel=kernel, C=C)  # Core SVM classifier
    )

    # Train (fit) the model on the training data
    # This is where the SVM learns the decision boundaries
    # that separate the different music genres.
    model.fit(X_train, y_train)

    return model