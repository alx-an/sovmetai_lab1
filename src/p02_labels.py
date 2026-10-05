from common import *
from sklearn.preprocessing import LabelEncoder

def main():
    labels=['red','black','red','green','black','yellow','white']
    encoder=LabelEncoder().fit(labels)
    test=['green','red','black']
    encoded=encoder.transform(test)
    # Обратное преобразование тем же обученным кодировщиком.
    decoded=encoder.inverse_transform([3,0,4,1])
    assert list(encoder.inverse_transform(encoded))==test
    result = dict(mapping={s:i for i,s in enumerate(encoder.classes_)},
        test=test,encoded=encoded,decode_input=[3,0,4,1],decoded=decoded)
    lines = ['Label mapping:']
    lines += [f'{label} -> {code}' for label, code in result['mapping'].items()]
    lines += [f"Encoded: {test} -> {encoded.tolist()}",
              f"Decoded: [3, 0, 4, 1] -> {decoded.tolist()}"]
    write_output('p02', lines)
if __name__=='__main__': main()
