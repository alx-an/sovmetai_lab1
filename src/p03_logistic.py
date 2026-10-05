from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

def main():
    X=np.array([[3.1,7.2],[4,6.7],[2.9,8],[5.1,4.5],[6,5],[5.6,5],[3.3,.4],[3.9,.9],[2.8,1],[.5,3.4],[1,4],[.6,4.9]])
    y=np.repeat(np.arange(4),3)
    results=[]; grids=[]
    for c in [1,100]:
        model=OneVsRestClassifier(LogisticRegression(solver='liblinear',C=c,random_state=0,max_iter=2000)).fit(X,y)
        grids.append(boundaries(model,X,y,f'p03_C{c}',f'Логистический классификатор, C={c}'))
        coef=np.vstack([e.coef_ for e in model.estimators_])
        results.append(dict(C=c,train_accuracy=model.score(X,y),coef=coef,
            intercept=[e.intercept_[0] for e in model.estimators_],coef_norm=np.linalg.norm(coef),
            mean_max_probability=model.predict_proba(X).max(axis=1).mean()))
    result = dict(experiments=results,grid_disagreement=np.mean(grids[0]!=grids[1]),
        probe=[3,3], note='Доля различий относится к равномерной сетке рисунка, не к тестовой точности.')
    lines = ['Logistic Regression:']
    for item in results:
        lines += [f"C={item['C']}: accuracy={item['train_accuracy']:.2%}",
                  f"  Coefficient norm={item['coef_norm']:.4f}; mean max probability={item['mean_max_probability']:.4f}"]
    lines.append(f"Different predictions on the plot grid: {result['grid_disagreement']:.2%}")
    write_output('p03', lines)
if __name__=='__main__': main()
