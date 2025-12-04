import json
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import ingestion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit


class Model:
    def __init__(self,data,model, name):
        self.data  = data
        self.features = feature_config()
        self.X  = self.data[self.features['features']]
        self.Y =  self.data[self.features['label']]
        self.model = model
        self.name = name
        self.metrics_results = {'accuracy': [], 'precision': [], 'recall': [], 'cm': []}

    def evaluate(self):
        tscv = TimeSeriesSplit(n_splits=5)
        for fold,(train_index, test_index) in enumerate(tscv.split(self.X)):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.Y.iloc[train_index], self.Y.iloc[test_index]

            print(f"\n[Fold {fold + 1}] Train Size: {len(X_train)}, Test Size: {len(X_test)}")
            #Train model
            self.model.fit(X_train,y_train)
            #Predict
            y_pred = self.model.predict(X_test)

            #Evaluate
            self.metrics_results['accuracy'].append(accuracy_score(y_test, y_pred))
            self.metrics_results['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            self.metrics_results['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            self.metrics_results['cm'].append(confusion_matrix(y_test, y_pred))

        avg_accuracy = np.mean(self.metrics_results['accuracy'])
        cm_array = np.sum(self.metrics_results['cm'], axis=0)
        self.confusion_plot(cm_array)
        
        return  {
            'Mean_Accuracy': avg_accuracy,
            'Mean_Precision': np.mean(self.metrics_results['precision']),
            'Mean_Recall': np.mean(self.metrics_results['recall']),
            'Total_CM': np.sum(self.metrics_results['cm'], axis=0).tolist()
        }
    def confusion_plot(self, cm_array):
        TN, FP, FN, TP = cm_array.ravel()
        Total = TN + FP + FN + TP

        # Calculate percentages for annotation
        TN_pct = f'{TN / Total:.1%}'
        FP_pct = f'{FP / Total:.1%}'
        FN_pct = f'{FN / Total:.1%}'
        TP_pct = f'{TP / Total:.1%}'

        # Create combined labels for the heatmap cells (Count + Percentage)
        labels = (np.asarray([['TN\n' + str(TN), 'FP\n' + str(FP)],
                            ['FN\n' + str(FN), 'TP\n' + str(TP)]]))
        annotations = (np.asarray([['TN: ' + str(TN) + f' ({TN_pct})', 'FP: ' + str(FP) + f' ({FP_pct})'],
                           ['FN: ' + str(FN) + f' ({FN_pct})', 'TP: ' + str(TP) + f' ({TP_pct})']]))
        
        plt.figure(figsize=(8, 6))

        # Create the heatmap using the raw counts
        sns.heatmap(cm_array, 
                    annot=annotations,       # Use the custom labels
                    fmt='s',                 # Format as string to display our custom labels
                    cmap='Blues',            # Color scheme
                    cbar=True,               # Display color bar
                    xticklabels=['Predicted 0 (Down/Flat)', 'Predicted 1 (Up)'],
                    yticklabels=['Actual 0 (Down/Flat)', 'Actual 1 (Up)'])

        plt.title(f'Confusion Matrix (Aggregated CV Folds) for {self.name}', fontsize=14)
        plt.ylabel('Actual Class (True Label)', fontsize=12)
        plt.xlabel('Predicted Class', fontsize=12)
        plt.show()


def feature_config(filename='data/features_config.json'):
    path = Path(filename)
    data = json.loads(path.read_text())
    return data

def model_params(filename='data/model_params.json'):
    path = Path(filename)
    data = json.loads(path.read_text())
    return data

if __name__ == '__main__':
    data  = ingestion()
    params = model_params()
    model1 = LogisticRegression(C=params['LogisticRegression']['C'],max_iter=params['LogisticRegression']['max_iter'] )
    model2 = RandomForestClassifier(n_estimators=params['RandomForestClassifier']['n_estimators'],max_depth=params['RandomForestClassifier']['max_depth'] )
    
    lr = Model(data,model1,'LogisticRegression')
    rf = Model(data,model2,'RandomForestClassifier')
    print(lr.evaluate())
    print(rf.evaluate())