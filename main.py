# Test mask set
# segment: metto flag a 0

from conversion import keras_array_of_cases
from building_and_train import train
from output_check import check_output

trainOriginals,trainLabels, testOriginals, preds_train_t = train()


# Test mask set
# segment: metto flag a 0
testMasks = keras_array_of_cases(0, 75, 100)
print('testMasks_ done')


#I repeat the image plotting 10 times

check_output(10,testOriginals,testMasks,preds_train_t)