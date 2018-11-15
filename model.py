from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import IsolationForest


def creat_model(model_name, params):
    if model_name == 'OneClassSVM':
        return OneClassSVM(params)
    if model_name == 'isolation':
        return isolation(params)


def OneClassSVM(params):
    return svm.OneClassSVM(nu=params['nu'], kernel=params['kernel'], gamma=params['gamma'])


def isolation(params):
    return IsolationForest(n_estimators=params['n_estimators'], max_samples=params['max_samples'],
                           contamination=params['contamination'],
                           max_features=params['max_features'], bootstrap=params['bootstrap'], n_jobs=params['n_jobs'],
                           behaviour='new',random_state=params['random_state'], verbose=params['verbose']
                           )


def evaluate(y_true, y_pred, scores):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    tp, fn, fp, tn = confusion_matrix.ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y_true, scores)
    AUC = metrics.auc(fpr, tpr)
    f1_score = metrics.f1_score(y_true, y_pred)
    precision = tp / (tp + fp)
    Specificity = tn / (tn + fp)
    Sensitivity = tp / (tp + fn)
    eval_r = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'AUC': AUC, 'f1_score': f1_score, 'precision': precision,
            'Specificity': Specificity,'Sensitivity':Sensitivity}
    return eval_r



