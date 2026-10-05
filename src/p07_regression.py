from common import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle

def main():
    d=np.loadtxt(DATA/'data_singlevar_regr.txt',delimiter=','); X,y=d[:,:-1],d[:,-1]; n=int(.8*len(y))
    model=LinearRegression().fit(X[:n],y[:n]); p=model.predict(X[n:])
    with (ART/'model.pkl').open('wb') as f: pickle.dump(model,f)
    with (ART/'model.pkl').open('rb') as f: loaded=pickle.load(f)
    restored=loaded.predict(X[n:]); assert np.array_equal(p,restored)
    fig,ax=plt.subplots(figsize=(7,4.6)); ax.scatter(X[:n,0],y[:n],label='Обучение',alpha=.55)
    ax.scatter(X[n:,0],y[n:],label='Тест',color='green'); order=np.argsort(X[:,0])
    ax.plot(X[order,0],model.predict(X[order]),color='black',label='Линейная модель')
    ax.set(xlabel='x',ylabel='y',title='Одномерная регрессия'); ax.legend(); fig.tight_layout()
    fig.savefig(ART/'p07_single.png'); plt.close(fig)
    result = dict(n=len(y),train_n=n,test_n=len(y)-n,metrics=regression(y[n:],p),
        coefficient=model.coef_,intercept=model.intercept_,pickle_equal=np.array_equal(p,restored),
        restored_MAE=metrics.mean_absolute_error(y[n:],restored))
    lines = ['Single variable regression:', f'Train={n}, test={len(y)-n}']
    lines += regression_lines(result['metrics'])
    lines += [f"y = {model.coef_[0]:.4f} * x + {model.intercept_:.4f}",
              f"Predictions after loading are equal: {result['pickle_equal']}",
              f"MAE after loading = {result['restored_MAE']:.4f}"]
    write_output('p07_single', lines)
    d=np.loadtxt(DATA/'data_multivar_regr.txt',delimiter=','); X,y=d[:,:-1],d[:,-1]; n=int(.8*len(y))
    linear=LinearRegression().fit(X[:n],y[:n]); poly=PolynomialFeatures(degree=10)
    xp=poly.fit_transform(X[:n]); pm=LinearRegression().fit(xp,y[:n]); probe=np.array([[7.75,6.35,5.56]])
    pl=linear.predict(X[n:]); pp=pm.predict(poly.transform(X[n:]))
    result = dict(n=len(y),train_n=n,test_n=len(y)-n,degree=10,expanded_features=xp.shape[1],
        probe=probe,linear_prediction=linear.predict(probe),polynomial_prediction=pm.predict(poly.transform(probe)),
        linear_test=regression(y[n:],pl),polynomial_test=regression(y[n:],pp),
        linear_train=regression(y[:n],linear.predict(X[:n])),polynomial_train=regression(y[:n],pm.predict(xp)),
        nearest_row=d[np.argmin(np.linalg.norm(X-probe,axis=1))])
    lines = ['', 'Multivariable regression:', f'Train={n}, test={len(y)-n}',
             'Point: [7.75, 6.35, 5.56]',
             f"Linear prediction = {result['linear_prediction'][0]:.4f}",
             f"Polynomial prediction (degree=10) = {result['polynomial_prediction'][0]:.4f}",
             'Linear test metrics:']
    lines += regression_lines(result['linear_test'])
    lines += ['Polynomial test metrics:'] + regression_lines(result['polynomial_test'])
    write_output('p07_multi', lines)
if __name__=='__main__': main()
