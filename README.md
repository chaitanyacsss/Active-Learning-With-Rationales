### Introduction  ###
Abstract of the paper "Active Learning with Rationales for text Classification" (ref: http://www.aclweb.org/anthology/N15-1047):
```
We present a simple and yet effective approach
that can incorporate rationales elicited
from annotators into the training of any offthe-
shelf classifier. We show that our simple
approach is effective for multinomial na¨ıve
Bayes, logistic regression, and support vector
machines. We additionally present an active
learning method tailored specifically for the
learning with rationales framework.
```

* A "Rationale" is simply a word (from the data sample) that the user provides as the reason for labelling that sample with that particular class (Positive/Negative in our case).
Ex: A single data sample looks like this:

```
This is definitely a stupid, bad-taste movie. Eddie Murphy stars in what is written like a sitcom. He is surrounded with his perfect family, full of good family values. If you're looking for politically correct entertainment, this movie is for you. But if you hate the idea of being the only one not to laugh at obscene gags in a movie-theater full of pop-corn addicts, just flee.
```
And as a part of active learning, the user labels this as "Negative" and the rationale given would be any of the following words:
```
stupid, bad-taste, obscene
```

### Implementation ###
We implement Active Learning with Rationales in the following steps:

1) Load test/train data as per the arguments; Each line in the train file is a document/instance.  
	- The dataset is provided in resources folder (mallet-train.msv/mallet-test.msv)
2) Pass the data through a series of pipes for preprocessing (tokenize, lowercase, stopword removal, convert to feature (strings to ints), convert labels to ints, vectorize data) and get instances.
	- These instances have term frequencies as values; we need tf-idf values and hence the next step.  
3) Calculating tf-idf values using mallet's FeatureCounter for idf and update instances.  
4) Update instances using "r" factor. (r = 1; o = 0.1)  
5) Select 10 random instances for initial training.  
6) Now, using incrementalTrain, keep adding the top 5 most uncertain instances of type-3 (based on the current model) and check accuracy.  
7) Repeat the previous step until we run out of labeling Budget. (budget used = 100)  


### Commands ###
Use the following command to run the training (dependencies are packed in the jar):
```
java -jar run_model.jar {train_path} {test_path}
```


### Results ###
A test accuracy of ~88% is obtained on 25000 (test) samples with just 110 training samples! 