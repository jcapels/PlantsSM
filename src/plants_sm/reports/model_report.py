import json
import os

import numpy as np
from matplotlib import pyplot, pyplot as plt, cm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve, \
    roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, average_precision_score

import seaborn as sns

from plants_sm.data_structures.dataset import Dataset
from plants_sm.models.constants import BINARY
from plants_sm.models.model import Model


class ModelReport:

    def __init__(self, model: Model, task_type: str, dataset: Dataset, reports_directory="./"):
        """
        Constructor for the ModelReport class.

        Parameters
        ----------
        model: Model
            model to generate report for
        task_type: str
            Type of task, e.g. BINARY, MULTICLASS, REGRESSION
        dataset:
            Dataset to evaluate the model on
        reports_directory:
            Directory to save the reports to
        """
        self.model = model
        self.task_type = task_type
        self.dataset = dataset
        os.makedirs(reports_directory, exist_ok=True)
        self.reports_directory = reports_directory

    def generate_metrics_report(self):
        """
        Generates a report for the model.

        """
        if self.task_type == BINARY:
            self._generate_binary_classification_report()

    def _generate_confusion_matrix(self, confusion_matrix_data):
        """
        Generates a confusion matrix for the model.

        Parameters
        ----------
        confusion_matrix_data: np.ndarray
            Confusion matrix data

        """
        ax = plt.subplot()
        sns.heatmap(confusion_matrix_data, annot=True, fmt='g',
                    ax=ax)  # annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['Negative', 'Positive'])
        ax.yaxis.set_ticklabels(['Negative', 'Positive'])
        pyplot.savefig(os.path.join(self.reports_directory, "confusion_matrix.png"))
        plt.show()

    def _generate_roc_auc_curve(self):
        """
        Generates a ROC AUC curve for the model.

        """
        ns_probs = [0 for _ in range(len(self.dataset.y))]
        probs = self.model.predict_proba(self.dataset)
        # keep probabilities for the positive outcome only
        # calculate scores
        # summarize scores
        ns_fpr, ns_tpr, _ = roc_curve(self.dataset.y, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(self.dataset.y, probs)
        # plot the roc curve for the model
        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Model')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.savefig(os.path.join(self.reports_directory, "roc_curve.png"))
        pyplot.show()

    def _generate_history(self):
        """
        Generates a history plot for the model.
        """
        ax2 = plt.subplot()
        sns.lineplot(self.model.history["loss"], ax=ax2)
        pyplot.savefig(os.path.join(self.reports_directory, "loss_history.png"))
        plt.show()

        ax2 = plt.subplot()
        sns.lineplot(self.model.history["metric_results"], ax=ax2)
        pyplot.savefig(os.path.join(self.reports_directory, "metric_results_history.png"))
        plt.show()

    def _generate_precision_recall_curve(self, predictions_proba):
        """
        Generates a precision recall curve for the model.

        """
        probability_thresholds = np.linspace(0, 1, num=100)
        precision_scores = []
        recall_scores = []

        # Find true positive / false positive rate for each threshold
        for p in probability_thresholds:

            y_test_preds = []

            for prob in predictions_proba:
                if prob > p:
                    y_test_preds.append(1)
                else:
                    y_test_preds.append(0)

            precision = precision_score(self.dataset.y, y_test_preds)
            recall = recall_score(self.dataset.y, y_test_preds)

            precision_scores.append(precision)
            recall_scores.append(recall)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall_scores, precision_scores, label='Model')
        baseline = len(self.dataset.y[self.dataset.y == 1]) / len(self.dataset.y)
        ax.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='center left')
        ax.set_title('Precision-Recall Curve')
        pyplot.savefig(os.path.join(self.reports_directory, "precision_recall_curve.png"))
        plt.show()

    def _generate_binary_classification_report(self):
        """
        Generates a report for a binary classification model.
        """
        predictions = self.model.predict(self.dataset)
        accuracy = accuracy_score(self.dataset.y, predictions)
        precision = precision_score(self.dataset.y, predictions)
        recall = recall_score(self.dataset.y, predictions)
        f1 = f1_score(self.dataset.y, predictions)
        balanced_accuracy = balanced_accuracy_score(self.dataset.y, predictions)
        mcc = matthews_corrcoef(self.dataset.y, predictions)
        confusion_matrix_data = confusion_matrix(self.dataset.y, predictions)
        predictions_proba = self.model.predict_proba(self.dataset)
        roc_auc = roc_auc_score(self.dataset.y, predictions_proba)
        precision_recall_auc = average_precision_score(self.dataset.y, predictions_proba)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced accuracy": balanced_accuracy,
            "roc_auc": roc_auc,
            "mcc": mcc,
            "precision_recall_auc": precision_recall_auc
        }

        self.create_visual(metrics)
        pyplot.show()

        self._generate_confusion_matrix(confusion_matrix_data)
        json.dump(metrics, open(os.path.join(self.reports_directory, "metrics.json"), "w"))

        self._generate_roc_auc_curve()
        self._generate_history()

        self._generate_precision_recall_curve(predictions_proba)

    def create_visual(self, metrics):
        colors = []
        colormap = plt.get_cmap('Blues')

        for values in metrics.values():
            # z_score = (y_val - mean) / stds[n]
            # p_value = st.norm.cdf(z_score)
            colors.append(colormap(values))

        cbar = plt.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(),
                                              cmap=colormap),
                            orientation='horizontal',
                            shrink=0.5,
                            pad=0.0625,
                            ax=plt.gca())
        for l in cbar.ax.xaxis.get_ticklabels():
            l.set_fontsize(8)
        cbar.ax.tick_params(length=0)
        cbar.outline.set_linewidth(0.25)

        xAxis = [key for key, value in metrics.items()]
        yAxis = [value for key, value in metrics.items()]

        plt.bar(xAxis,
                yAxis,
                edgecolor='k',
                lw=.25,
                color=colors,
                width=.5)
        plt.grid(False, zorder=0)

        plt.title('Model metrics',
                  fontsize=10,
                  alpha=0.8)

        i = list(np.arange(len(xAxis)))
        plt.xticks(i,
                   tuple(list(metrics.keys())))

        plt.ylim([0, 1])
        # plt.xlim([-0.5, 3.5])
        plt.gca().tick_params(length=0)

        plt.xticks(fontsize=8,
                   alpha=0.8, rotation=45)
        plt.yticks(fontsize=8,
                   alpha=0.8)

        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        plt.gca().tick_params(length=0)
        pyplot.savefig(os.path.join(self.reports_directory, "metrics.png"))
