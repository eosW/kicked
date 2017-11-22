from preproc import process
from train_test import train,test

process("data/training.csv", "data/processed_train.csv")
process("data/test.csv", "data/processed_test.csv", train=False)

train("data/processed_train.csv","kicked.model",0.01)
test("data/processed_test.csv","data/predicted.csv","kicked.model")