from common import *
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def main():
    y=[2,0,0,2,4,4,1,0,3,3,3]; p=[2,1,0,2,4,3,1,0,1,3,3]
    cm=confusion_matrix(y,p)
    fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay(cm).plot(ax=ax,cmap='Greys',colorbar=False)
    ax.set(xlabel='Предсказанный класс',ylabel='Истинный класс',title='Матрица ошибок из п. 5')
    fig.tight_layout(); fig.savefig(ART/'p05_confusion.png'); plt.close(fig)
    report=classification_report(y,p,target_names=[f'Class-{i}' for i in range(5)],digits=4,zero_division=0)
    result = dict(matrix=cm,report=metrics.classification_report(y,p,output_dict=True,zero_division=0),
        class1_one_vs_rest=dict(TP=1,FP=2,FN=0,TN=8))
    write_output('p05', ['Classification report:', report.rstrip(),
                         'Class 1: TP=1, FP=2, FN=0, TN=8'])
if __name__=='__main__': main()
