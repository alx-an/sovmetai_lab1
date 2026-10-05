from common import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split,cross_val_score
from collections import Counter

class MixedEncoder(TransformerMixin,BaseEstimator):
    """Числа сохраняются; категории кодируются как в методичке, только по train."""
    numeric=(0,2,4,10,11,12)
    def fit(self,X,y=None):
        X=np.asarray(X); self.encoders_={i:LabelEncoder().fit(X[:,i]) for i in range(X.shape[1]) if i not in self.numeric}
        return self
    def transform(self,X):
        X=np.asarray(X); out=np.empty(X.shape,dtype=float)
        for i in range(X.shape[1]):
            if i in self.numeric: out[:,i]=X[:,i].astype(float)
            else:
                mapping={s:j for j,s in enumerate(self.encoders_[i].classes_)}
                out[:,i]=[mapping.get(s,-1) for s in X[:,i]]
        return out

def main():
    rows=[]; counts=Counter(); skipped=0
    for line in (DATA/'income_data.txt').read_text().splitlines():
        if not line.strip(): continue
        if '?' in line: skipped+=1; continue
        row=[v.strip() for v in line.split(',')]
        if counts[row[-1]]<25000: rows.append(row); counts[row[-1]]+=1
    data=np.array(rows); X,y=data[:,:-1],data[:,-1]
    tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=5)
    probes=np.array([
        ['37','Private','215646','HS-grad','9','Never-married','Handlers-cleaners','Not-in-family','White','Male','0','0','40','United-States'],
        ['52','Self-emp-inc','120000','Masters','14','Married-civ-spouse','Exec-managerial','Husband','White','Male','15000','0','60','United-States'],
        ['22','Private','180000','Some-college','10','Never-married','Other-service','Own-child','White','Female','0','0','20','United-States']])
    fields=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
    configs=[('LinearSVC C=0.1',OneVsOneClassifier(LinearSVC(C=.1,dual=False,max_iter=20000,random_state=0))),
        ('LinearSVC C=1',OneVsOneClassifier(LinearSVC(C=1,dual=False,max_iter=20000,random_state=0))),
        ('RBF SVC C=1',SVC(C=1,kernel='rbf',gamma='scale',cache_size=512))]
    results=[]
    for name,est in configs:
        print('Fitting:',name,flush=True)
        # Масштабирование необходимо из-за fnlwgt/capital-gain; fit внутри CV исключает утечку.
        model=make_pipeline(MixedEncoder(),StandardScaler(),est).fit(X[tr],y[tr])
        p=model.predict(X[te]); cv=cross_val_score(model,X,y,cv=3,scoring='f1_weighted')
        results.append(dict(name=name,test=classification(y[te],p),cv_f1_folds=cv,cv_f1_mean=cv.mean(),
            predictions=model.predict(probes)))
    result = dict(n=len(y),class_counts=dict(counts),skipped_missing=skipped,train_n=len(tr),test_n=len(te),
        fields=fields,probes=probes,experiments=results,baseline_majority_accuracy=max(Counter(y[te]).values())/len(te))
    lines = [f"Income data: {len(y) + skipped} rows; {skipped} missing; {len(y)} retained",
             f"Train={len(tr)}, test={len(te)}", '']
    for item in results:
        m = item['test']
        lines += [item['name'],
                  f"Accuracy={m['accuracy']:.2%}; Precision={m['precision_weighted']:.2%}; Recall={m['recall_weighted']:.2%}",
                  f"F1={m['f1_weighted']:.2%}; F1 CV (3 folds)={item['cv_f1_mean']:.2%}",
                  'Predictions: ' + '; '.join(f"point {i+1} -> {value}" for i, value in enumerate(item['predictions'])), '']
    write_output('p06', lines)
if __name__=='__main__': main()
