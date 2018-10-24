

Here are the steps followed in implementing Active Learning with Rationales (ref: http://www.aclweb.org/anthology/N15-1047):

1) Load test/train data as per the arguments; Each line in the train file is a document/instance.  
2) Pass the data through a series of pipes for preprocessing (tokenize, lowercase, stopword removal, convert to feature (strings to ints), convert labels to ints, vectorize data) and get instances.
 - These instances have term frequencies as values; we need tf-idf values and hence the next step.  
3) Calculating tf-idf values using mallet's FeatureCounter for idf and update instances.  
4) Update instances using "r" factor. (r = 1; o = 0.1)  
5) Select 10 random instances for initial training.  
6) Now, using incrementalTrain, keep adding the top 5 most uncertain instances of type-3 (based on the current model) and check accuracy.  
7) Repeat the previous step until we run out of labeling Budget. (budget used = 100)  