import os
import traceback
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, SingleClassifierEnhancer, MultipleClassifiersCombiner, FilteredClassifier, \
    PredictionOutput, Kernel, KernelClassifier
from weka.classifiers import Evaluation
from weka.filters import Filter
from weka.core.classes import Random, from_commandline
import weka.plot.classifiers as plot_cls
import weka.plot.graph as plot_graph
#import weka.core.types as types

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from weka.core.converters import Loader
jvm.start()
def vote_classifier_train(train_data):
    meta = MultipleClassifiersCombiner(classname="weka.classifiers.meta.Vote"
        ,options=['-S', '1', '-B', 'weka.classifiers.trees.J48 -C 0.25 -M 2', '-B', 'weka.classifiers.trees.RandomTree -K 6 -M 1.0 -V 0.001 -S 1',
        '-B', 'weka.classifiers.meta.Bagging -P 100 -S 1 -num-slots 1 -I 10 -W weka.classifiers.trees.REPTree -- -M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0',
        '-B', 'weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump', '-B',
        'weka.classifiers.meta.Bagging -P 100 -S 1 -num-slots 1 -I 10 -W weka.classifiers.trees.REPTree -- '
        '-M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0', '-B', 'weka.classifiers.bayes.NaiveBayes ', '-R', 'AVG'])
    # classifiers = [
    #         Classifier(classname="weka.classifiers.trees.J48"),
    #         Classifier(classname="weka.classifiers.trees.RandomTree"),
    #         Classifier(classname="weka.classifiers.meta.Bagging"),
    #         Classifier(classname="weka.classifiers.meta.AdaBoostM1"),
    #         Classifier(classname="weka.classifiers.meta.Bagging"),
    #         Classifier(classname="weka.classifiers.bayes.NaiveBayes")
    #
    # ]
    # meta.classifiers = classifiers
    #meta.options=["-R","AVG"]
    #train_data=str(train_data)
    meta.build_classifier(train_data)
    return meta

def predict(cls,data):
    for index, inst in enumerate(data):
        pred = cls.classify_instance(inst)
        dist = cls.distribution_for_instance(inst)
        print(str(index + 1) + ": label index=" + str(pred) + ", class distribution=" + str(dist))
    return  pred,dist

loader=Loader(classname="weka.core.converters.CSVLoader")
data=loader.load_file('./data/final/3.csv')
data.class_is_last()
# meta = MultipleClassifiersCombiner(classname="weka.classifiers.meta.Vote")
# classifiers = [
#         Classifier(classname="weka.classifiers.functions.SMO"),
#         Classifier(classname="weka.classifiers.trees.J48")
#     ]
# meta.classifiers = classifiers

# print(meta.options)
#cls=vote_classifier_train(data)
from weka.classifiers import Evaluation

meta = MultipleClassifiersCombiner(classname="weka.classifiers.meta.Vote"
                                   , options=['-S', '1', '-B', 'weka.classifiers.trees.J48 -C 0.25 -M 2', '-B',
                                              'weka.classifiers.trees.RandomTree -K 6 -M 1.0 -V 0.001 -S 1',
                                              '-B',
                                              'weka.classifiers.meta.Bagging -P 100 -S 1 -num-slots 1 -I 10 -W weka.classifiers.trees.REPTree -- -M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0',
                                              '-B',
                                              'weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump',
                                              '-B',
                                              'weka.classifiers.meta.Bagging -P 100 -S 1 -num-slots 1 -I 10 -W weka.classifiers.trees.REPTree -- '
                                              '-M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0', '-B',
                                              'weka.classifiers.bayes.NaiveBayes ', '-R', 'AVG'])


from weka.core.classes import Random
eval=Evaluation(data)
cls=Classifier(classname='weka.classifiers.trees.J48',options=['-C','.025'])
print(cls.options)
pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
eval.crossvalidate_model(cls,data,3,Random(1),pout)
print(eval.percent_correct)
print(eval.summary())
print(eval.class_details())
print('true positive:'+str(eval.true_positive_rate(1)))
print('false positive:'+str(eval.false_positive_rate(1)))

jvm.stop()