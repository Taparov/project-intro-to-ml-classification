from sklearn.metrics import accuracy_score,confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def evaluate_model(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def plot_roc_curve(y_test, y_pred_proba, title="Model"):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {title}')
    plt.legend(loc="lower right")
    plt.show()

def prepare_data():
    df = pd.read_csv('./dataset/UCI_Credit_Card.csv')
    df.dropna(inplace=True)
    df['SEX'] = np.where(df['SEX'] == 1, 0, 1)
    # drop roes where PAY_0,2,3,4,5,6 is -2
    df = df[~df['PAY_0'].isin([-2])]
    df = df[~df['PAY_2'].isin([-2])]
    df = df[~df['PAY_3'].isin([-2])]
    df = df[~df['PAY_4'].isin([-2])]
    df = df[~df['PAY_5'].isin([-2])]
    df = df[~df['PAY_6'].isin([-2])]
    # change valuse in pay from -1 to 0
    df['PAY_0'] = np.where(df['PAY_0'] == -1, 0, df['PAY_0'])
    df['PAY_2'] = np.where(df['PAY_2'] == -1, 0, df['PAY_2'])
    df['PAY_3'] = np.where(df['PAY_3'] == -1, 0, df['PAY_3'])
    df['PAY_4'] = np.where(df['PAY_4'] == -1, 0, df['PAY_4'])
    df['PAY_5'] = np.where(df['PAY_5'] == -1, 0, df['PAY_5'])
    df['PAY_6'] = np.where(df['PAY_6'] == -1, 0, df['PAY_6'])
    df.drop(['ID'], axis=1, inplace=True)
    
    return df