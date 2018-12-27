import numpy as np
from LCSTS_char.data_utils import get_datasets_train
if __name__ == '__main__':
    filename_text = '../DATA/LCSTS/PART_I.txt'
    _, text = get_datasets_train(filename_text)
    l = []
    for line in text:
        l.append(len(line))
    mean = sum(l) / len(l)
    var = np.std(l)
    result = mean + var
    result = int(round(result))
    print('mean:', mean)
    print('result:', result)