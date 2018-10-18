import ID3, parse, random
from copy import copy
import numpy as np
import matplotlib.pyplot as plt

def testID3AndEvaluate():
  data = [dict(a=1, b=0, Class=1), dict(a=1, b=1, Class=1)]
  tree = ID3.ID3(data, 0)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=1, b=0))
    if ans != 1:
      print("ID3 test failed.")
    else:
      print("ID3 test succeeded.")
  else:
    print("ID3 test failed -- no tree returned")

def testPruning():
  # data = [dict(a=1, b=1, c=1, Class=0), dict(a=1, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1), dict(a=0, b=0, c=0, Class=1), dict(a=0, b=0, c=1, Class=0)]
  # validationData = [dict(a=0, b=0, c=1, Class=1)]
  data = [dict(a=0, b=1, c=1, d=0, Class=1), dict(a=0, b=0, c=1, d=0, Class=0), dict(a=0, b=1, c=0, d=0, Class=1), dict(a=1, b=0, c=1, d=0, Class=0), dict(a=1, b=1, c=0, d=0, Class=0), dict(a=1, b=1, c=0, d=1, Class=0), dict(a=1, b=1, c=1, d=0, Class=0)]
  validationData = [dict(a=0, b=0, c=1, d=0, Class=1), dict(a=1, b=1, c=1, d=1, Class = 0)]
  tree = ID3.ID3(data, 0)
  ID3.prune(tree, validationData)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=0, b=0, c=1, d=0))
    if ans != 1:
      print("pruning test failed.")
    else:
      print("pruning test succeeded.")
  else:
    print("pruning test failed -- no tree returned.")


def testID3AndTest():
  trainData = [dict(a=1, b=0, c=0, Class=1), dict(a=1, b=1, c=0, Class=1), 
  dict(a=0, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1)]
  testData = [dict(a=1, b=0, c=1, Class=1), dict(a=1, b=1, c=1, Class=1), 
  dict(a=0, b=0, c=1, Class=0), dict(a=0, b=1, c=1, Class=0)]
  tree = ID3.ID3(trainData, 0)
  fails = 0
  if tree != None:
    acc = ID3.test(tree, trainData)
    if acc == 1.0:
      print("testing on train data succeeded.")
    else:
      print("testing on train data failed.")
      fails = fails + 1
    acc = ID3.test(tree, testData)
    if acc == 0.75:
      print("testing on test data succeeded.")
    else:
      print("testing on test data failed.")
      fails = fails + 1
    if fails > 0:
      print("Failures: ", fails)
    else:
      print("testID3AndTest succeeded.")
  else:
    print("testID3andTest failed -- no tree returned.")	

# inFile - string location of the house data file
def testPruningOnHouseData(inFile):
  data = parse.parse(inFile)
  sizes = np.arange(10,300,10)
  pruned_accuracies = np.zeros(len(sizes))
  without_prune_accuracies = np.zeros(len(sizes))
  for k in range(len(sizes)):
      withPruning = []
      withoutPruning = []
      for i in range(100):
            random.shuffle(data)
            data_copy = copy(data[0:sizes[k]])
            train = data_copy[:len(data_copy)//2]
            valid = data_copy[len(data)//2:3*len(data_copy)//4]
            test = data_copy[3*len(data_copy)//4:]
          
            tree = ID3.ID3(train, ID3.getmodeByFeature(data_copy,'Class'))
    #        acc = ID3.test(tree, train)
    #        print("training accuracy: ",acc)
    #        acc = ID3.test(tree, valid)
    #        print("validation accuracy: ",acc)
    #        acc = ID3.test(tree, test)
    #        print("test accuracy: ",acc)
    #      
    #        ID3.prune(tree, valid)
    #        acc = ID3.test(tree, train)
    #        print("pruned tree train accuracy: ",acc)
    #        acc = ID3.test(tree, valid)
    #        print("pruned tree validation accuracy: ",acc)
            acc = ID3.test(tree, test)
    #        print("pruned tree test accuracy: ",acc)
            withPruning.append(acc)
            tree = ID3.ID3(train+valid, ID3.getmodeByFeature(data_copy,'Class'))
            acc = ID3.test(tree, test)
    #        print("no pruning test accuracy: ",acc)
            withoutPruning.append(acc)
      pruned_accuracies[k] = np.average(np.array(withPruning))
      without_prune_accuracies[k] = np.average(np.array(withoutPruning))
  fig, axs = plt.subplots(2, 1, constrained_layout = True)
  axs[0].plot(sizes, pruned_accuracies)
  axs[0].set_title('With Pruning')
  axs[0].set_xlabel('training set size')
  axs[0].set_ylabel('accuracy')
  axs[1].plot(sizes, without_prune_accuracies)
  axs[1].set_title('No Pruning')
  axs[1].set_xlabel('training set size')
  axs[1].set_ylabel('accuracy')
  plt.show()
#  print(withPruning)
#  print(withoutPruning)
#  print("average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning))