import json
import os

import numpy as np
from matplotlib import pyplot, pyplot as plt, cm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, roc_curve, \
    roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

from plants_sm.models.constants import BINARY


class ModelReport:

    def __init__(self, model, task_type, dataset, reports_directory="./"):
        self.model = model
        self.task_type = task_type
        self.dataset = dataset
        os.makedirs(reports_directory, exist_ok=True)
        self.reports_directory = reports_directory

    def generate_metrics_report(self):
        if self.task_type == BINARY:
            self._generate_binary_classification_report()

    def _generate_binary_classification_report(self):
        predictions = self.model.predict(self.dataset)
        accuracy = accuracy_score(self.dataset.y, predictions)
        precision = precision_score(self.dataset.y, predictions)
        recall = recall_score(self.dataset.y, predictions)
        f1 = f1_score(self.dataset.y, predictions)
        balanced_accuracy = balanced_accuracy_score(self.dataset.y, predictions)
        mcc = matthews_corrcoef(self.dataset.y, predictions)
        predictions = self.model.predict_proba(self.dataset)
        roc_auc = roc_auc_score(self.dataset.y, predictions)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
            "roc_auc": roc_auc,
            "mcc": mcc
        }

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
        pyplot.show()
        pyplot.savefig(os.path.join(self.reports_directory, "roc_curve.png"))

        xAxis = [key for key, value in metrics.items()]
        yAxis = [value for key, value in metrics.items()]

        # fig = pyplot.figure()
        # pyplot.bar(xAxis, yAxis, color='green')
        # pyplot.xlabel('variable')
        # pyplot.ylabel('value')
        # pyplot.show()

        self.create_visual(metrics, y_val=1)
        pyplot.show()

        json.dump(metrics, open(os.path.join(self.reports_directory, "metrics.json"), "w"))

    def create_visual(self, metrics, y_val=30000):
        colors = []
        colormap = plt.get_cmap('seismic_r')

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

        barlist = plt.bar(xAxis,
                          yAxis,
                          # yerr=[1.96 * std for std in stds],
                          # error_kw=dict(lw=1.5,
                          #               capsize=7.5,
                          #               capthick=1.5,
                          #               ecolor='green'),
                          edgecolor='k',
                          lw=.25,
                          color=colors,
                          width=.5)

        plt.title('Dynamic Coloration Demonstration',
                  fontsize=10,
                  alpha=0.8)

        i = list(np.arange(len(xAxis)))
        plt.xticks(i,
                   tuple(list(metrics.keys())))
        # plt.xlim([-0.5, 3.5])
        plt.gca().tick_params(length=0)

        plt.xticks(fontsize=8,
                   alpha=0.8)
        plt.yticks(fontsize=8,
                   alpha=0.8)

        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        plt.gca().tick_params(length=0)

        plt.axhline(color='green',
                    lw=0.5,
                    y=y_val)