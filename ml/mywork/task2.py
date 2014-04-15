""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""


import numpy as np
import pylab as pl

from gensim import corpora, models, matutils

from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.grid_search import GridSearchCV
from sklearn import metrics


class Task2:
    """
    This class uses gensim module to create corpus from file.
    """
    def __init__(self):
        """
        initialization
        """
        #gensim dictionary type
        self.dictionary = None
        #corpus
        self.corpus = None

    def readData(self, trainFile, dictionary = None):
        """
        read data
        @param trainFile
        @param dictionary: gensim dictionary type; if None, create from inputFile
        @return
        """
        if(dictionary != None):
            self.dictionary = corpora.Dictionary.load(dictionary)
        else:
            with open(trainFile, 'r') as f:
                self.dictionary = corpora.Dictionary(line.split() for line in f)
            #remove terms which only appear once to get more stable model
            onceIds = [tokenid for tokenid, docfreq in self.dictionary.dfs.iteritems() if docfreq == 1]
            self.dictionary.filter_tokens(onceIds)
            self.dictionary.compactify()
            #self.dictionary.save('task2_train.dict')
        print 'makeDict Done!'

        with open(trainFile, 'r') as f:
            self.corpus = [self.dictionary.doc2bow(line.split()) for line in f]
        #corpora.MmCorpus.serialize('task2_train.mm', self.corpus)
        print 'makeCorpus Done!'



if __name__ == '__main__':
    """
    Read data first.
    """
    doTask2 = True
    if(doTask2):
        print 'Task2 Start!'
        t2 = Task2()
        t2.readData('deals_cleaned.txt')
        print 'Task2 Done!'

    """
    Fit LDA model, using gensim.models.ldamodel module, to discover topics.
    """
    doLDA = True
    if(doLDA):
        print 'LDA start!'
        np.random.seed(997)
        
        doLDA_Search = True
        if(doLDA_Search):
            print "LDA_Search Start!"

            """
            This was my first try for tuning paramter num_topics. I use 5-fold cross-validation to estimate
            the performance of each LDA model, in terms of per word perplexity.
            """
            ldas = {}
            perplexs_train = {}
            perplexs_test = {}
            for num_topics in xrange(200, 0, -10):
                data = list(t2.corpus)
                np.random.shuffle(data)
                l = int(len(data) * 0.8)
                train = data[0:l]
                test = data[l:]
                lda = models.ldamodel.LdaModel(corpus=train, id2word=t2.dictionary, num_topics=num_topics, alpha='auto')
                ldas[num_topics] = lda
                perplex_train = np.exp2(-lda.log_perplexity(train))
                perplexs_train[num_topics] = perplex_train
                perplex_test = np.exp2(-lda.log_perplexity(test))
                perplexs_test[num_topics] = perplex_test
                print('num_topics = %s, perplexity_train = %s, perplexity_test = %s' % (num_topics, perplex_train, perplex_test))
            

            """
            For this dataset, as we can see from the plot, per word perplexity of test data
            decreases a lot as number of topics decreases. Actually also does per word perplexity of
            training data. But because of the scale, it not very clear on the plot. I choose number of
            topics to be 10 for my final LDA model. But, I think the best way to determine it is that
            we have some prior knowledge or domain knowledge about this.
            """
            pl.figure()
            pl.xlabel('Number of topics')
            pl.ylabel('Per word perplexity')
            pl.plot(xrange(200, 0, -10), [perplexs_train[key] for key in xrange(200, 0, -10)], 'go-', label = 'Train')
            pl.plot(xrange(200, 0, -10), [perplexs_test[key] for key in xrange(200, 0, -10)], 'ro-', label = 'Test')
            pl.legend(loc = 'upper left')
            pl.savefig('lda_cv.png')

            #joblib.dump(ldas, 'task2_ldas.pkl')
            #joblib.dump(perplexs_train, 'task2_perp_train.pkl')
            #joblib.dump(perplexs_test, 'taks2_perp_test.pkl')

            """
            I also did this to confirm myself the per word perplexity is always decreasing.
            And it actually does.
            """
            for num_topics in xrange(9, 0, -1):
                data = list(t2.corpus)
                np.random.shuffle(data)
                l = int(len(data) * 0.8)
                train = data[0:l]
                test = data[l:]
                lda = models.ldamodel.LdaModel(corpus=train, id2word=t2.dictionary, num_topics=num_topics, alpha='auto')
                perplex_train = np.exp2(-lda.log_perplexity(train))
                perplex_test = np.exp2(-lda.log_perplexity(test))
                print('num_topics = %s, perplexity_train = %s, perplexity_test = %s' % (num_topics, perplex_train, perplex_test))

            print "LDA_Search Done!"
            

        """
        Fit final LDA model using whole dataset and summarize the 10 topics. Actually for some topic,
        it's not very clear what it is. I just give my best guess.
        Topic 0: free shipping
        Topic 1: promotion
        Topic 2: ticket
        Topic 3: link, page
        Topic 4: gift card
        Topic 5: hotel
        Topic 6: price
        Topic 7: online
        Topic 8: date, time
        Topic 9: save
        """
        lda_final = models.ldamodel.LdaModel(corpus=t2.corpus, id2word=t2.dictionary, num_topics=10, alpha='auto')
        topics = lda_final.show_topics()
        for topic in xrange(10):
            print('The distribution of first 10 terms of topic %s is :' % topic)
            print topics[topic]
            print
        #transform original data to topics data to find groups
        topics_data = matutils.corpus2dense(lda_final[t2.corpus], 10).T

        print 'LDA Done!'



    """
    Fit Kmeans model, using sklearn.cluster.KMeans module, to discover groups using derived data of topics.
    """
    doKmeans = True
    if(doKmeans):
        print 'Kmeans Start!'
        np.random.seed(997)
        
        doKmeans_Search = True
        if(doKmeans_Search):
            print "Kmeans_Search start!"
            
            """
            This was my first try for tuning paramter n_clusters. I use 5-fold cross-validation to estimate
            the performance of each Kmeans model, in terms of silhouette score.
            """
            kmeans_models = {}
            kmeans_scores = {}
            for n_clusters in xrange(100,0,-10):
                data = np.array(list(topics_data))
                np.random.shuffle(data)
                l = int(len(data) * 0.8)
                train = data[0:l]
                test = data[l:]
                kmeans = KMeans(n_clusters)
                kmeans.fit(train)
                kmeans_models[n_clusters] = kmeans
                labels_test = kmeans.predict(test)
                score = metrics.silhouette_score(test, labels_test, metric='euclidean')
                kmeans_scores[n_clusters] = score
                print('n_clusters = %s, silhouette_score = %s' % (n_clusters, score))
                print


            """
            As we can see from the plot, silhouette score of test data is highest when number of clusters
            is 10, which is also the number of topics. I think this makes sense because we use the derived
            data, where each row is a mixture of 10 topics with different weights. Ideally each cluster
            should correspond to each topic.
            """
            pl.figure()
            pl.xlabel('Number of clusters')
            pl.ylabel('Test silhouette score')
            pl.plot(xrange(100, 0, -10), [kmeans_scores[key] for key in xrange(100, 0, -10)], 'bo-')
            pl.savefig('kmeans_cv.png')

            print "Kmeans_Search done!"

        """
        Fit final Kmeans model with 10 clusters using the whole dataset. As I find out, each cluster
        actually corresponds to each topic.
        """
        kmeans_final = KMeans(10)
        kmeans_final.fit(topics_data)
        #find topics with highest weight in each cluster center
        cluster_topics = [cluster_center.argmax() for cluster_center in kmeans_final.cluster_centers_]
        print('Topics with highest weight in each cluster center: ')
        print cluster_topics
        print('All cluster labels: ')
        print kmeans_final.labels_

        print 'Kmeans Done!'
        


""" Report
I completed this task by following steps:
1. Read data using Task2 class.
2. Fit LDA model to discover topics.
3. Fit Kmeans model to discover groups, using derived topics data.

Issues:
1. I tried to fit a clustering model using the original data. But I always got a ValueError, which I
   have not solved yet.
2. Maybe I need a better way to determine the number of topics.
3. I meant to use multinomial mixture model to do the clustering which I think should work better here,
   but I could not find it in sklearn module, although the algorithm is not very hard to implement.
4. Kmeans model needs number of clusters to be determined before model fitting. I tried to use hierarchical
   clustering model to avoid this. But it ran out of memory on my laptop.
"""
