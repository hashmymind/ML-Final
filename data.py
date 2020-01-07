import pytreebank
import pickle
import numpy as np

dataset = pytreebank.load_sst()


def get_sst(cate='5'):

    train_X = []
    train_y = []

    for e in dataset['train']:
      label, sentence = e.to_labeled_lines()[0]
      if cate == '2' and label == 2:
        continue
      if cate == '2':
        label = 1 if label >2 else 0
      train_X.append(sentence)
      train_y.append(label)

    test_X = []
    test_y = []

    for e in dataset['test']:
      label, sentence = e.to_labeled_lines()[0]
      if cate == '2' and label == 2:
        continue
      if cate == '2':
        label = 1 if label >2 else 0
      test_X.append(sentence)
      test_y.append(label)
      
    return (train_X,train_y), (test_X, test_y)
    

        
if __name__ == '__main__':
    pass