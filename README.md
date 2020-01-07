# About this repo

This is a homework of my ML class, and I only took one night to do it, so there could be a lot of space to improve.

## Dataset
* SST-fine-grained (5 classes)

## Workflow

### Traditional Path
1. clean some punctuation
2. extract TF-IDF features
3. train a naive bayes classifier
5. show the metrics

### Deep Learning Path
1. clean some punctuation
2. tokenize
3. word embedding(GloVe pretrain weights)
4. try LSTM w/o bidirection and w/o self attention layer
5. show the metrics


## Final result

| Model                                        | Accuracy |
|----------------------------------------------|----------|
| Naive Bayes Classifier                       | 27.37%   |
| LSTM                                         | 44.84%   |
| Bidirectional LSTM                           | 39.46%   |
| LSTM with Self Attention Layer               | 45.24%   |
| Bidirectional LSTM with Self Attention Layer | 45.52    |
