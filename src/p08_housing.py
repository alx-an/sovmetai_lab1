from common import *
from sklearn.svm import SVR
from sklearn.utils import shuffle

def main():
    path=ROOT/'data'/'boston_house_prices.csv'
    d=np.loadtxt(path,delimiter=',',skiprows=2); assert d.shape==(506,14)
    X,y,indices=shuffle(d[:,:-1],d[:,-1],np.arange(len(d)),random_state=7); n=int(.8*len(y))
    probes=np.vstack([[3.7,0,18.4,1,.87,5.95,91,2.5052,26,666,20.2,351.34,15.27],X[n],X[n+1]])
    results=[]
    # Изменяем C при постоянном epsilon, затем epsilon при постоянном C.
    for c,e in [(1,.1),(.1,.1),(10,.1),(1,.01),(1,1)]:
        print(f'Fitting SVR C={c}, epsilon={e}',flush=True)
        model=SVR(kernel='linear',C=c,epsilon=e).fit(X[:n],y[:n]); pred=model.predict(X[n:])
        results.append(dict(C=c,epsilon=e,metrics=regression(y[n:],pred),predicted_prices=model.predict(probes)))
    result = dict(source='https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.1.3/sklearn/datasets/data/boston_house_prices.csv',
        shape=d.shape,train_n=n,test_n=len(y)-n,
        probes=probes,probe_source_rows=[None,int(indices[n])+1,int(indices[n+1])+1],probe_actual_prices=[None,y[n],y[n+1]],experiments=results)
    lines = [f'Boston Housing: {len(y)} objects, {X.shape[1]} features; train={n}, test={len(y)-n}',
             'SVR with linear kernel', '']
    for item in results:
        a, b, c = item['predicted_prices']
        lines += [f"C={item['C']}, epsilon={item['epsilon']}",
                  f"MSE={item['metrics']['MSE']:.4f}; Explained Variance={item['metrics']['EVS']:.4f}",
                  f'Predicted prices: A={a:.4f}; B={b:.4f}; C={c:.4f}', '']
    write_output('p08', lines)
    fig,axes=plt.subplots(1,2,figsize=(10,4)); labels=[f'C={r["C"]}\nε={r["epsilon"]}' for r in results]
    axes[0].bar(labels,[r['metrics']['MSE'] for r in results]); axes[0].set(title='Ошибка на тестовой выборке',ylabel='MSE, (тыс. долларов)²')
    axes[1].bar(labels,[r['metrics']['EVS'] for r in results]); axes[1].set(title='Объяснённая дисперсия',ylabel='EVS')
    fig.tight_layout(); fig.savefig(ART/'p08_comparison.png'); plt.close(fig)
if __name__=='__main__': main()
