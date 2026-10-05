from common import *
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_validate

def main():
    data=np.loadtxt(DATA/'data_multivar_nb.txt',delimiter=','); X,y=data[:,:-1],data[:,-1].astype(int)
    full=GaussianNB().fit(X,y)
    boundaries(full,X,y,'p04_all','Naive Bayes: обучение на всей выборке')
    tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=3)
    model=GaussianNB().fit(X[tr],y[tr]); pred=model.predict(X[te])
    boundaries(model,X[te],y[te],'p04_test','Naive Bayes: тестовые точки, обучение на 80%')
    cv=cross_validate(GaussianNB(),X,y,cv=3,scoring=['accuracy','precision_weighted','recall_weighted','f1_weighted'])
    result = dict(shape=X.shape,train_n=len(tr),test_n=len(te),
        all_data_accuracy=full.score(X,y),test=classification(y[te],pred),
        cv={k[5:]:dict(folds=v,mean=v.mean()) for k,v in cv.items() if k.startswith('test_')},
        test_indices=te,y_test=y[te],predictions=pred)
    lines = [f"Naive Bayes: train={len(tr)}, test={len(te)}",
             f"Accuracy on all data: {result['all_data_accuracy']:.2%}",
             'Metric                 Test      CV (3 folds)']
    for key, value in result['test'].items():
        lines.append(f"{key:22} {value:8.2%} {result['cv'][key]['mean']:10.2%}")
    write_output('p04', lines)
if __name__=='__main__': main()
