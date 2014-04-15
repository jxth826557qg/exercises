""" Classification

The objective of this task is to build a classifier that can tell us whether a new, unseen deal 
requires a coupon code or not. 

We would like to see a couple of steps:

1. You should use bad_deals.txt and good_deals.txt as training data
2. You should use test_deals.txt to test your code
3. Each time you tune your code, please commit that change so we can see your tuning over time

Also, provide comments on:
    - How general is your classifier?
    - How did you test your classifier?

"""

import numpy as np
import itertools

from gensim import corpora, matutils

from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier




class Task3():
    """
    This class reads training and test data.
    """
    def __init__(self):
        """
        initialization
        """
        self.dict = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def readTrain(self, good_deals_file, bad_deals_file):
        """
        read training data
        @param good_deals_file
        @param bad_deals_file
        @return
        """
        self.dict = corpora.Dictionary(line.split() for line in itertools.chain(open(good_deals_file, 'r'), open(good_deals_file, 'r')))
        #remove terms which only appear once to get more stable model
        onceIds = [tokenid for tokenid, docfreq in self.dict.dfs.iteritems() if docfreq == 1]
        self.dict.filter_tokens(onceIds)
        self.dict.compactify()
        corpus_good = [self.dict.doc2bow(line.split()) for line in open(good_deals_file, 'r')]
        corpus_bad = [self.dict.doc2bow(line.split()) for line in open(bad_deals_file, 'r')]
        corpus = corpus_good + corpus_bad
        self.X_train = matutils.corpus2csc(corpus, num_terms=len(self.dict)).T
        self.y_train = np.array([1]*len(corpus_good) + [0]*len(corpus_bad))
        #self.dict.save('task3_train.dict')
        #joblib.dump(self.X_train, 'task3_X_train.pkl')
        #joblib.dump(self.y_train, 'task3_y_train.pkl')
        print 'readTrain Done!'

    def readTest(self, test_deals_file):
        """
        read test data
        @param test_deals
        @return
        """
        corpus = [self.dict.doc2bow(line.split()) for line in open(test_deals_file, 'r')]
        self.X_test = matutils.corpus2csc(corpus, num_terms=len(self.dict)).T
        #joblib.dump(self.X_test, 'task3_X_test.pkl')
        print 'readTest Done!'


def cv(estimator, X_train, y_train, k_fold=5, scorelist=['accuracy', 'precision', 'recall', 'roc_auc']):
    """
    use cross_val_score in sklearn module to estimate prediction preformance
    @param estimator: sklearn estimator type
    @param X_train: training data X
    @param y_train: training data y
    @param k_folds: num of folds
    @param scorelist: list specifies scores to use
    @return scores: dictionary stroes a list of scores for each score type
    """
    scores = {}
    for score in scorelist:
            scoreArray = cross_val_score(estimator, X_train, y_train, score, k_fold)
            print("%s = %0.3f (%0.3f)" % (score, scoreArray.mean(), scoreArray.std()))
            scores[score] = scoreArray
    return scores


def gridSearch(X_train, y_train, estimator, tuning_params, k_fold=5, scorelist=['accuracy', 'precision', 'recall', 'roc_auc']):
    """
    use GridSearchCV in sklearn module to find the best tuning parameters
    @param X_train: training data X
    @param y_train: training data y
    @param estimator: sklearn estimator type
    @param tuning_params: list of dictionaries specifies tuning parameters
    @param k_folds: num of folds
    @param scorelist: list specifies scores to use
    @return models: dictionary stores GridSearchCV type models for each score type
    """
    models = {}
    for score in scorelist:
            print('tuning parameters for %s' % score)
            model = GridSearchCV(estimator, tuning_params, cv = k_fold, scoring = score)
            model.fit(X_train, y_train)
            for params, mean_score, scores in model.grid_scores_:
                print("%0.3f (%0.3f) for %r" % (mean_score, scores.std(), params))
            models[score] = model
    return models


if __name__ == '__main__':
    doTask3 = True
    if(doTask3):
        print 'Task3 Start!'
        t3 = Task3()
        t3.readTrain('good_deals_cleaned.txt', 'bad_deals_cleaned.txt')
        t3.readTest('test_deals_cleaned.txt')
        print('The shape of training data X: (%s, %s)' % t3.X_train.shape)
        print('The shape of test data X: (%s, %s)' % t3.X_test.shape)
        print 'Task3 Done!'

    """
    Naive Bayes classifier
    """
    doNB = True
    if(doNB):
        print 'NB Start!'
        np.random.seed(997)
        """
        The performance of NB classifier, based on cross-validation, is not good. I think
        that's because it cannot handle cases where the number of dimensions is greater than
        the number of samples, which is our case here.

        accuracy = 0.333(0.075)
        precision = 0.000
        recall = 0.000
        roc_auc = 0.000
        """
        nb = MultinomialNB()
        #cross-validation
        nb_scores = cv(nb, t3.X_train, t3.y_train)
        nb.fit(t3.X_train, t3.y_train)
        nb_pred_train = nb.predict(t3.X_train)
        nb_pred_test = nb.predict(t3.X_test)
        print 'NB Done!'


    """
    SVM classifier
    """
    doSVM = True
    if(doSVM):
        print 'SVM Start!'
        np.random.seed(997)
        
        doSVM_Search = True
        if(doSVM_Search):
            print "SVM_Search start!"
            
            """
            This was my first try for tuning parameters. It turned out that liear kernel is not good,
            but rbf kernel really does a good job.
        
            accuracy = 0.983(0.003): C makes no difference, gamma = 1
            precision = 0.971(0.057): C makes no difference, gamma = 1
            recall = 1.000(0.000): C, gamma make no difference
            roc_auc = 1.000(0.000): C, gamma make no difference
            """
            svm_tuning_params = [{'kernel':['rbf'],'C':range(100, 0, -10), 'gamma':range(10,0,-1)}, {'kernel':['linear'], 'C':range(100, 0, -10)}]
            svm_estimator = svm.SVC(probability=True)
            svcs = gridSearch(t3.X_train, t3.y_train, svm_estimator, svm_tuning_params)
        
        
            """
            Then I narrowed down the range of tuning parameters. It turned out svm did a really good
            job on this data. As far as I know, that's because svm still does well in high dimensional
            spaces, even where number of dimensions is greater than the number of samples.

            accuracy = 1.000(0.000): C makes no difference, gamma = 0.1-0.8
            precision = 1.000(0.000): C makes no difference, gamma = 0.1-0.8
            recall = 1.000(0.000): C, gamma make no difference
            roc_auc = 1.000(0.000): C, gamma make no difference
            """
            svm_tuning_params = {'kernel':['rbf'], 'C':range(1000, 0, -100), 'gamma':np.arange(1,0,-0.1)}
            svm_estimator = svm.SVC(probability=True)
            svcs = gridSearch(t3.X_train, t3.y_train, svm_estimator, svm_tuning_params)

            print "SVM_Search done!"
        

        """
        Finally, I choose C = 1, gamma = 0.1 for my svm classifier, because the smaller these values are,
        the less overfitting there is. These scores are based on cross-validation.

        accuracy = 1.000
        precision = 1.000
        recall = 1.000
        roc_auc = 1.000
        """
        svc = svm.SVC(gamma=0.1, probability=True)
        svc_scores = cv(svc, t3.X_train, t3.y_train)
        svc.fit(t3.X_train, t3.y_train)
        svm_pred_train = svc.predict(t3.X_train)
        svm_pred_test = svc.predict(t3.X_test)

        print 'SVM Done!'


    """
    AdaBoost classifier
    """
    doAdaBoost = True
    if(doAdaBoost):
        print 'AdaBoost Start!'
        np.random.seed(997)
        
        doAda_Search = True
        if(doAda_Search):
            print "Ada_Search start!"
            
            """
            The performance of AdaBoost classifier is not very good even with many weak learners.
            I think it's also because of high dimension issue in our data.
        
            accuracy = 0.667(0.053): n_estimator = 350
            precision = 1.000(0.000): n_estimator = 150-200, 300, 450-500
            recall = 0.333((0.105): n_estimator = 400
            roc_auc = 0.631(0.089): n_estimator = 100-150
            """
            boost_tuning_params = {'n_estimators' : range(500, 0, -50)}
            boost_estimator = AdaBoostClassifier()
            boosts = gridSearch(t3.X_train.toarray(), t3.y_train, boost_estimator, boost_tuning_params)

            print "Ada_Search done!"

        print 'AdaBoost Done!'


    """
    Random Forest classifier
    """
    doRF = True
    if(doRF):
        print 'RF Start!'
        np.random.seed(997)
        
        doRF_Search = True
        if(doRF_Search):
            print "RF_Search start!"
            
            """
            The performance of Random Forest classifier is also not good, similar to AdaBoost classifier.
            I think the reason is also similar.
        
            accuracy = 0.533(0.041): n_estimator = 150, 400, 500
            precision = 0.400(0.490): n_estimator = 300-400
            recall = 0.067(0.082): n_estimator = 50, 200-250, 350
            roc_auc = 0.728(0.090): n_estimator = 50
            """
            rf_tuning_params = {'n_estimators' : range(500, 0, -50)}
            rf_estimator = RandomForestClassifier()
            rfs = gridSearch(t3.X_train.toarray(), t3.y_train, rf_estimator, rf_tuning_params)

            print "RF_Search done!"

        print 'RF Done!'
        

""" Report
I completed this task by following steps:
1. Read data using Task3 class.
2. Build different types of classifiers, including Naive Bayes classifier, SVM classifier,
   AdaBoost classifier, Random Forest classifier. And it turned out only SVM works well.

Comments:
- How general is your classifier?
  All the classifiers I built are kind of general. They can handle multiple classes problem,
  either in nature, like Naive Bayes, AdaBoost, Random Forest, or using one-verus-all or
  one-versu-one method, like SVM. They can also handle different types of features, either
  continuous or categorical.
- How did you test your classifier?
  Because I don't have labels with the test data, I just used cross-validation technique to
  estimate prediction performance of my classifier. If I have labels with the test data, I should
  have a final test score of the prediction performance of my classifier.
"""
