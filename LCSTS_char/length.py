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
    print('max:', max(l))      # test:140  # summary:30
    print('min', min(l))       # test:80   # summary:8
    print('mean:', mean)       # test:103  # summary:17
    print('mean+var:', result) # test:114  # summary:22