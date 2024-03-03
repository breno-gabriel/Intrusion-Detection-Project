
"""Run experiments and create figs"""
import itertools
import seaborn as sns
import os
import pickle
import matplotlib
import pandas as pd 
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


import dga_classifier.bigram as bigram
import dga_classifier.lstm as lstm

from scipy import interpolate
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

RESULT_FILE = 'results.pkl'

def run_experiments(isbigram=True, islstm=True, nfolds=15):
    """Runs all experiments"""
    bigram_results = None
    lstm_results = None

    if isbigram:
        bigram_results = bigram.run(nfolds=nfolds)

    if islstm:
        lstm_results = lstm.run(nfolds=nfolds)

    return bigram_results, lstm_results
        
    # return lstm_results

def create_figs(isbigram=True, islstm=True, nfolds=15, force=False):
    """Create figures"""
    # Generate results if needed
    # if force or (not os.path.isfile(RESULT_FILE)):
    bigram_results, lstm_results = run_experiments(isbigram, islstm, nfolds)

    # lstm_results = run_experiments(isbigram, islstm, nfolds)

    results = {'bigram': bigram_results, 'lstm': lstm_results}

    # results = {'bigram': None, 'lstm': lstm_results}

    # return results

    #     pickle.dump(results, open(RESULT_FILE, 'wb'))
    # else:
    #     results = pickle.load(open(RESULT_FILE, 'rb'))

    # Extract and calculate bigram ROC
    if results['bigram']:
        bigram_results = results['bigram']
        fpr = []
        tpr = []
        for bigram_result in bigram_results:
            t_fpr, t_tpr, _ = roc_curve(bigram_result['y'], bigram_result['probs'])
            fpr.append(t_fpr)
            tpr.append(t_tpr)
        bigram_binary_fpr, bigram_binary_tpr, bigram_binary_auc = calc_macro_roc(fpr, tpr)
        

    # xtract and calculate LSTM ROC
    if results['lstm']:
        lstm_results = results['lstm']
        fpr = []
        tpr = []
        for lstm_result in lstm_results:
            t_fpr, t_tpr, _ = roc_curve(lstm_result['y'], lstm_result['probs'])
            fpr.append(t_fpr)
            tpr.append(t_tpr)
        lstm_binary_fpr, lstm_binary_tpr, lstm_binary_auc = calc_macro_roc(fpr, tpr)
        
    # Save figure
    with plt.style.context('bmh'):
        plt.plot(lstm_binary_fpr, lstm_binary_tpr,
                 label='LSTM (AUC = %.4f)' % (lstm_binary_auc, ), rasterized=True)
        plt.plot(bigram_binary_fpr, bigram_binary_tpr,
                 label='Bigrams (AUC = %.4f)' % (bigram_binary_auc, ), rasterized=True)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=22)
        plt.ylabel('True Positive Rate', fontsize=22)
        plt.title('ROC - Binary Classification', fontsize=26)
        plt.legend(loc="lower right", fontsize=22)

        plt.tick_params(axis='both', labelsize=22)
        plt.savefig('results.png')

    lstm_confusion_matrix = confusion_matrix(lstm_results[0]['y'], lstm_results[0]['probs'] > 0.5)
    bigram_confusion_matrix = confusion_matrix(bigram_results[0]['y'], bigram_results[0]['probs'] > 0.5)

    plot_confusion_matrix(bigram_confusion_matrix, classes=['benign', 'malicious'], model_name='Bigrams')
    plot_confusion_matrix(lstm_confusion_matrix, classes=['benign', 'malicious'], model_name='LSTM')

    """Plot and save metrics table"""
    lstm_metrics = calculate_metrics(lstm_results)
    bigram_metrics = calculate_metrics(bigram_results)

    table_data = {
        'Model': ['LSTM', 'Bigrams'],
        'Precision': [lstm_metrics['precision'], bigram_metrics['precision']],
        'Recall': [lstm_metrics['recall'], bigram_metrics['recall']],
        'F1-Score': [lstm_metrics['f1_score'], bigram_metrics['f1_score']]
    }

    table_df = pd.DataFrame(table_data)
    plt.figure(figsize=(12, 6))
    sns.set(font_scale=1.2)
    
    # Plot metrics table
    plt.subplot(1, 2, 1)
    sns.heatmap(table_df.set_index('Model').T, annot=True, cmap='Blues', fmt=".3f", linewidths=.5)
    plt.title('Metrics Table', fontsize=20)

    # Plot ROC curve for LSTM
    plt.subplot(1, 2, 2)
    with plt.style.context('bmh'):
        plt.plot(lstm_binary_fpr, lstm_binary_tpr,
                 label='LSTM (AUC = %.4f)' % (lstm_binary_auc, ), rasterized=True)
        plt.plot(bigram_binary_fpr, bigram_binary_tpr,
                 label='Bigrams (AUC = %.4f)' % (bigram_binary_auc, ), rasterized=True)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('ROC - Binary Classification', fontsize=20)
        plt.legend(loc="lower right", fontsize=16)

    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig('metrics_and_roc.png')

def calculate_metrics(results):
    """Calculate precision, recall, and f1-score"""
    y_true = results[0]['y']
    y_pred = (results[0]['probs'] > 0.5).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {'precision': precision, 'recall': recall, 'f1_score': f1}

def plot_confusion_matrix(conf_matrix, classes, model_name):
    """Plot and save confusion matrix"""
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
    with plt.style.context('bmh'):
        plt.figure(figsize=(8, 6))
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix - {model_name}', fontsize=26)
        plt.xlabel('Predicted Label', fontsize=22)
        plt.ylabel('True Label', fontsize=22)
        plt.tick_params(axis='both', labelsize=18)
        plt.savefig(f'confusion_matrix_{model_name.lower()}.png')

def calc_macro_roc(fpr, tpr):
    """Calcs macro ROC on log scale"""
    # Create log scale domain
    all_fpr = sorted(itertools.chain(*fpr))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(tpr)):
        if np.any(np.diff(fpr[i]) == 0):
            # Handle points with the same x coordinate
            indices = np.where(np.diff(fpr[i]) == 0)[0]
            fpr[i][indices + 1] += 1e-8  # Add a small epsilon to avoid division by zero
        interp_func = interpolate.interp1d(fpr[i], tpr[i], kind='linear', fill_value='extrapolate')
        mean_tpr += interp_func(all_fpr)

    return all_fpr, mean_tpr / len(tpr), auc(all_fpr, mean_tpr) / len(tpr)


if __name__ == "__main__":
    create_figs(nfolds=1) # Run with 1 to make it fast
