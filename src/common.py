"""Пути к данным, метрики и построение графиков."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
DATA = ROOT / 'data'
ART = ROOT / 'results'
ART.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'figure.dpi': 120})

def write_output(name, lines):
    """Печать результатов и сохранение того же текста в results."""
    text = "\n".join(lines)
    print(text, flush=True)
    (ART / (name + '.txt')).write_text(text + "\n", encoding='utf-8')

def regression_lines(values):
    names = {'MAE': 'Mean Absolute Error', 'MSE': 'Mean Squared Error',
             'MedianAE': 'Median Absolute Error', 'EVS': 'Explained Variance', 'R2': 'R2'}
    return [f"{names[key]} = {value:.4f}" for key, value in values.items()]

def classification(y, p):
    return dict(accuracy=metrics.accuracy_score(y,p),
        precision_weighted=metrics.precision_score(y,p,average='weighted',zero_division=0),
        recall_weighted=metrics.recall_score(y,p,average='weighted',zero_division=0),
        f1_weighted=metrics.f1_score(y,p,average='weighted',zero_division=0))

def regression(y,p):
    return dict(MAE=metrics.mean_absolute_error(y,p), MSE=metrics.mean_squared_error(y,p),
        MedianAE=metrics.median_absolute_error(y,p), EVS=metrics.explained_variance_score(y,p),
        R2=metrics.r2_score(y,p))

def boundaries(model, X, y, name, title):
    # Границы классов на сетке с шагом 0.01.
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-1,X[:,0].max()+1,.01),
                         np.arange(X[:,1].min()-1,X[:,1].max()+1,.01))
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.pcolormesh(xx,yy,pred,cmap='Pastel1',shading='auto',vmin=0,vmax=8)
    sc = ax.scatter(X[:,0],X[:,1],c=y,cmap='tab10',vmin=0,vmax=9,edgecolors='black',s=45)
    ax.legend(*sc.legend_elements(),title='Класс',loc='best')
    ax.set(xlabel='Признак 1',ylabel='Признак 2',title=title)
    fig.tight_layout(); fig.savefig(ART/(name+'.png')); plt.close(fig)
    return pred
