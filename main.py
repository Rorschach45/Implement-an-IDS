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
def vote_classifier():
    meta = MultipleClassifiersCombiner(classname="weka.classifiers.meta.Vote")
    classifiers = [
            Classifier(classname="weka.classifiers.trees.J48",options=["−C","0.25","−M", "2"]),
            Classifier(classname="weka.classifiers.trees.RandomTree",options=["−K", "6", "−M", "1.0", "−V", "0.001"
            , "−S", "1"]),
            Classifier(classname="weka.classifiers.meta.Bagging", options=["−P", "100", "−S", "1", "−num-slots", "1" ,"−I"
                                                                           ,"10" "−W" "weka.classifiers.trees.REPTree", "−M", "2"
                                                                        ,"−V" ,"0.001","−N", "3", "−S", "1" "−L" "−1", "−I" ,"0.0"]),
            Classifier(classname="weka.classifiers.meta.AdaBoostM1", options=["−P", "100" ,"−S" ,"1", "−I", "10" "−W",
                                                                              "weka.classifiers.trees.DecisionStump"]),
            Classifier(classname="weka.classifiers.meta.Bagging",options=["−P", "100","−S", "1", "−num-slots", "1", "−I",
                                                                          "10", "−W", "weka.classifiers.trees.REPTree − ","−M",
                                                                          "2" ,"−V" ,"0.001", "−N", "3", "−S", "1" ,"−L","−1", "−I", "0.0"]),
            classifiers()

    ]
    meta.classifiers = classifiers

loader=Loader(classname="weka.core.converters.CSVLoader")
data=loader.load_file('./data/final/1.csv')
data.class_is_last()
print()
# print("generic Vote instantiation")
# meta = MultipleClassifiersCombiner(classname="weka.classifiers.meta.Vote")
# classifiers = [
#         Classifier(classname="weka.classifiers.functions.SMO"),
#         Classifier(classname="weka.classifiers.trees.J48")
#     ]
# meta.classifiers = classifiers
# print(meta.to_commandline())