import weka.core.jvm as jvm
from weka.classifiers import Classifier, MultipleClassifiersCombiner, PredictionOutput
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.core.converters import Loader
import gc


def vote_classifier_train(dicrectory, nameOfDataSet, flag):
    loader = Loader(classname="weka.core.converters.CSVLoader")
    data = loader.load_file(dicrectory)
    data.class_is_last()
    meta = MultipleClassifiersCombiner(classname="weka.classifiers.meta.Vote",
                                       options=['-S', '1', '-B', 'weka.classifiers.trees.J48 -C 0.25 -M 2',
                                                '-B', 'weka.classifiers.trees.RandomTree -K 6 -M 1.0 -V 0.001 -S 1',
                                                '-B',
                                                'weka.classifiers.meta.Bagging -P 100 -S 1 -num-slots 1 -I 10 -W weka.classifiers.trees.REPTree -- '
                                                '-M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0', '-B',
                                                'weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump',
                                                '-B',
                                                'weka.classifiers.meta.Bagging -P 100 -S 1 -num-slots 1 -I 10 -W weka.classifiers.trees.REPTree -- '
                                                '-M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0', '-B',
                                                'weka.classifiers.bayes.NaiveBayes ', '-R', 'AVG'])
    eval = Evaluation(data)
    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
    if flag:
        eval.crossvalidate_model(meta, data, 10, Random(1), pout)
    else:
        eval.evaluate_train_test_split(meta, data, 80.0, Random(1), pout)
    gc.collect()
    print_and_save('Proposed model', flag, nameOfDataSet, eval)


def j48(dicrectory, nameOfDataSet, flag):
    loader = Loader(classname="weka.core.converters.CSVLoader")
    data = loader.load_file(dicrectory)
    data.class_is_last()
    cls = Classifier(classname='weka.classifiers.trees.J48', options=['-C', '.025'])
    eval = Evaluation(data)
    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
    if flag:
        eval.crossvalidate_model(cls, data, 10, Random(1), pout)
    else:
        eval.evaluate_train_test_split(cls, data, 80.0, Random(1), pout)
    print_and_save('J48 model', flag, nameOfDataSet, eval)
    gc.collect()


def naive_bayse(dicrectory, nameOfDataSet, flag):
    loader = Loader(classname="weka.core.converters.CSVLoader")
    data = loader.load_file(dicrectory)
    data.class_is_last()
    cls = Classifier(classname='weka.classifiers.bayes.NaiveBayes')
    eval = Evaluation(data)
    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
    if flag:
        eval.crossvalidate_model(cls, data, 10, Random(1), pout)
    else:
        eval.evaluate_train_test_split(cls, data, 80.0, Random(1), pout)
    print_and_save('Naive Bayes model', flag, nameOfDataSet, eval)
    gc.collect()


def random_tree(dicrectory, nameOfDataSet, flag):
    loader = Loader(classname="weka.core.converters.CSVLoader")
    data = loader.load_file(dicrectory)
    data.class_is_last()
    cls = Classifier(classname='weka.classifiers.trees.RandomTree', options=['-K', '6'])
    eval = Evaluation(data)
    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
    if flag:
        eval.crossvalidate_model(cls, data, 10, Random(1), pout)
    else:
        eval.evaluate_train_test_split(cls, data, 80.0, Random(1), pout)
    print_and_save('random tree model', flag, nameOfDataSet, eval)
    gc.collect()


def print_and_save(modelname, flag, nameOfDataSet, eval):
    if flag:
        file = open('./result/result_kfold.txt', 'a')
    else:
        file = open('./result/result_split_1run.txt', 'a')
    print('*************************************************')
    file.write('*************************************************')
    print('\n')
    file.write('\n')
    print('the result of ' + modelname + ' on:' + nameOfDataSet)
    file.write('the result of ' + modelname + ' on:' + nameOfDataSet)
    file.write('\n')
    print('*************************************************/')
    file.write('*************************************************/')
    file.write('\n')
    print(eval.percent_correct)
    file.write(str(eval.percent_correct))
    print(eval.summary())
    file.write(str(eval.summary()))
    print(eval.class_details())
    file.write(str(eval.class_details()))
    print('true positive:' + str(eval.true_positive_rate(1)))
    file.write('\n')
    file.write('true positive:' + str(eval.true_positive_rate(1)))
    print('false positive:' + str(eval.false_positive_rate(1)))
    file.write('\n')
    file.write('false positive:' + str(eval.false_positive_rate(1)))
    print('\n')
    file.write('\n')
    print('\n')
    file.write('\n')
    file.write('\n')
    file.write('\n')
    file.write('\n')
    gc.collect()


def main():
    jvm.start()
    vote_classifier_train('./data/final/bolean_for_weka.csv', 'boolean_target', True)
    vote_classifier_train('./data/final/bolean_for_weka.csv', 'boolean_target', False)
    j48('./data/final/bolean_for_weka.csv', 'boolean_target', True)
    j48('./data/final/bolean_for_weka.csv', 'boolean_target', False)
    naive_bayse('./data/final/bolean_for_weka.csv', 'boolean_target', True)
    naive_bayse('./data/final/bolean_for_weka.csv', 'boolean_target', False)
    random_tree('./data/final/bolean_for_weka.csv', 'boolean_target', True)
    random_tree('./data/final/bolean_for_weka.csv', 'boolean_target', False)

    vote_classifier_train(
        './data/final/20 Percent Training Set reducedAttacks_data feature selected with normalized data.csv',
        'reduced attacks to 4', True)
    vote_classifier_train(
        './data/final/20 Percent Training Set reducedAttacks_data feature selected with normalized data.csv',
        'reduced attacks to 4', False)
    j48('./data/final/20 Percent Training Set reducedAttacks_data feature selected with normalized data.csv',
        'reduced attacks to 4', True)
    j48('./data/final/20 Percent Training Set reducedAttacks_data feature selected with normalized data.csv',
        'reduced attacks to 4', False)
    naive_bayse('./data/final/20 Percent Training Set reducedAttacks_data feature selected with normalized data.csv',
                'reduced attacks to 4', True)
    naive_bayse('./data/final/20 Percent Training Set reducedAttacks_data feature selected with normalized data.csv',
                'reduced attacks to 4', False)
    random_tree('./data/final/20 Percent Training Set reducedAttacks_data feature selected with normalized data.csv',
                'reduced attacks to 4', True)
    random_tree('./data/final/20 Percent Training Set reducedAttacks_data feature selected with normalized data.csv',
                'reduced attacks to 4', False)
    jvm.stop()


if __name__ == '__main__':
    main()