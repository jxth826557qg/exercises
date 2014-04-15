""" Features

The objective of this task is to explore the corpus, deals.txt. 

The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:

1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?

"""

import numpy as np
import string
from pattern.en import singularize
from pprint import pprint


class dataClean:
    """
    This class does data cleaning: transform terms to lowercase, remove terms in stoplist,
    remove some punctuations if it is at the end of the term, and singularize terms
    """
    def __init__(self, stoplistFile = None, punctuations = None):
        """
        initialization
        @param stoplistFile: file stores stop words; If None, use default provided
        @param punctuations: tuple specifies puncuations to remove; If None, use default provided
        @return
        """
        if(stoplistFile == None):
            self.stoplist = set('for a of the and to in'.split())
        else:
            self.stoplist = set()
            with open(stoplistFile, 'r') as stoplistData:
                for line in stoplistData: self.stoplist.add(line.strip())

        if(punctuations == None):
            self.punctuations = ('!', ',', '.', '?', ';', '-')
        else:
            self.punctuations = punctuations

    def clean(self, inputFile, outputFile):
        """
        data cleaning
        @param inputData: input data file
        @param outputData: output data file
        @return
        """
        with open(inputFile, 'r') as inputData, open(outputFile, 'w') as outputData:
            for line in inputData:
                out = []
                splits = line.strip().lower().split(' ')
                for term in splits:
                    #remove punctuation
                    if(term.endswith(self.punctuations)): term = term[:-1]
                    #remove term in stoplist, singularize it
                    if(term != '' and term not in self.stoplist): out.append(singularize(term))
                outputData.write(' '.join(out) + '\n')
        print 'Clean Done!'
        

class Task1:
    """
    This class reads and processes the file in a MapReduce stype, but not exactly because
    it also outputs a dictionary storing all types of guitars.
    """
    def __init__(self):
        """
        initialization
        """
        #intermediate result from mapper. key: term, value: list of 1
        self.intermediate = {}
        #final result from reducer. key: term, value: frequency
        self.result = {}
        #dictionary stores all types of guitars. key: guitar, value: list of corresponding records
        self.guitars = {}

    def mapper(self, record):
        """
        mapper
        @param record: one line in deals.txt file
        @return
        """
        splits = record.strip().split(' ')
        for index, term in enumerate(splits):
            #emit intermediate result
            self.emit_intermediate(term, 1)
            #keep record containing guitar
            if(term == 'guitar'):
                guitar = term if index == 0 else splits[index-1] + ' ' + term
                self.guitars.setdefault(guitar, [])
                self.guitars[guitar].append(record.strip())
                
    def emit_intermediate(self, key, value):
        """
        emit intermediate result
        @param key
        @param value
        @return
        """
        self.intermediate.setdefault(key, [])
        self.intermediate[key].append(value)

    def reducer(self, key, list_of_values):
        """
        reducer
        @param key
        @param list_of_values
        @return
        """
        freq = np.array(list_of_values).sum()
        self.emit(key, freq)

    def emit(self, key, value):
        """
        emit final result
        @param key
        @param value
        @return
        """
        self.result[key] = value

    def execute(self, inputFile):
        """
        call other functions to do all the data processing
        @param data: imput data file
        @return
        """
        with open(inputFile, 'r') as data:
            for line in data:
                self.mapper(line)
            for key, value in self.intermediate.items():
                self.reducer(key, value)
        print 'MapReduce Done!'

    def mostPop(self):
        """
        return and print most popular term
        @return mostPopTerm
        """
        mostPopTerm = max(self.result, key = self.result.get)
        print('The most popular term is \'%s\'.' % mostPopTerm)
        return mostPopTerm

    def leastPop(self):
        """
        return and print least popular term
        @return leastPopTerm
        """
        leastPopTerm = min(self.result, key = self.result.get)
        print('The least popular term is \'%s\'.' % leastPopTerm)
        return leastPopTerm

    def allGuitars(self):
        """
        print all types of guitars in ascending order
        @return
        """
        print('There are %s types of guitars mentioned in the file. They are:' % len(self.guitars))
        pprint(sorted(self.guitars.keys()))
    
        

if __name__ == '__main__':
    """
    Clean all datasets first.
    """
    doClean = True
    if(doClean):
        print 'Clean Start!'
        clean = dataClean(stoplistFile = 'stoplist.txt')
        clean.clean('good_deals.txt', 'good_deals_cleaned.txt')
        clean.clean('bad_deals.txt', 'bad_deals_cleaned.txt')
        clean.clean('test_deals.txt', 'test_deals_cleaned.txt')
        clean.clean('deals.txt', 'deals_cleaned.txt')
        print 'Clean Done!'

    """
    Use Task1 class to answer all questions.
    """
    doTask1 = True
    if(doTask1):
        print 'Task1 Start!'
        t1 = Task1()
        t1.execute('deals_cleaned.txt')
        mostPop = t1.mostPop()
        leastPop = t1.leastPop()
        t1.allGuitars()
        print 'Task1 Done!'


""" Report
I completed this task by following steps:
1. Data cleaning using dataClean class and all following work are based on cleaned data.
2. Data processing using execute function in Task1 class.
3. Use specific functions in Task1 class to answer the 3 questions.

But there are still some issues I did not handle very well.
1. Although I did data cleaning first, it's not very sophisticated.
   More work should be done here.
2. For question 1 and 2, there are multiple max and min terms. I just return one.
3. For question 3, I just keep all distinct terms of guitar with its previous term(if there
   is one). This is method is crude. Some results are even not valid guitar type. More
   sophisticated methods should be used.
"""
