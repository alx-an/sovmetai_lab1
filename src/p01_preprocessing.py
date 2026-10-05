from common import *
from sklearn import preprocessing

def main():
    x=np.array([[5.1,-2.9,3.3],[-1.2,7.8,-6.1],[3.9,.4,2.1],[7.3,-9.9,-4.5]])
    scaled=preprocessing.scale(x)
    result=dict(input=x,binarized=preprocessing.Binarizer(threshold=2.1).transform(x),
        mean_before=x.mean(axis=0),std_before=x.std(axis=0),standardized=scaled,
        mean_after=scaled.mean(axis=0),std_after=scaled.std(axis=0),
        minmax=preprocessing.MinMaxScaler().fit_transform(x),
        l1=preprocessing.normalize(x,norm='l1'),l2=preprocessing.normalize(x,norm='l2'))
    assert np.allclose(result['mean_after'],0) and np.allclose(result['std_after'],1)
    assert np.allclose(np.abs(result['l1']).sum(axis=1),1)
    assert np.allclose((result['l2']**2).sum(axis=1),1)
    result = result
    lines = []
    for key, label in [('input', 'Input matrix'), ('binarized', 'Binarization'),
                       ('standardized', 'Standardization'), ('minmax', 'MinMax scaling'),
                       ('l1', 'L1 normalization'), ('l2', 'L2 normalization')]:
        lines += [label + ':', np.array2string(result[key], precision=4, suppress_small=True), '']
    for key in ['mean_before', 'std_before', 'mean_after', 'std_after']:
        lines.append(key + ': ' + np.array2string(result[key], precision=4, suppress_small=True))
    write_output('p01', lines)
if __name__=='__main__': main()
