from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test, genre_labels=None):
    """
    evaluate a trained model on test data.

    prints classification metrics and saves a confusion matrix plot.
    """

    # generate predictions from the trained model
    # this is where we test how well the model generalizes
    y_pred = model.predict(X_test)

    # print detailed precision / recall / f1 metrics per genre
    # this gives more insight than accuracy alone
    print("===== classification report =====")
    print(classification_report(y_test, y_pred))

    # compute confusion matrix
    # rows = true labels, columns = predicted labels
    cm = confusion_matrix(y_test, y_pred)

    # create a heatmap for better visual interpretation
    # darker diagonal values indicate strong performance
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=genre_labels,
        yticklabels=genre_labels
    )

    plt.title("confusion matrix")
    plt.ylabel("true label")
    plt.xlabel("predicted label")
    plt.tight_layout()

    # save the figure so it can be added to the README
    # useful for documentation and reproducibility
    plt.savefig("outputs/confusion_matrix.png", dpi=200)

    # display the plot
    plt.show()