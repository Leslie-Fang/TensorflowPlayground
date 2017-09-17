from trainModel2 import trainModel
from verifyModel import verifyData

if __name__ == "__main__":
    fileName = "testResult.txt"
    with open(fileName,'w+') as f:
        f.write("epoch\tlearning_rate\tresult\n")
        for epoch in range(1000,3500,100):
            for num in range(1,11,1):
                learning_rate = num * 0.05
                print(epoch)
                print(learning_rate)
                # f.write("{}\t{}\tresult\n")
                trainModel(epoch,learning_rate)
                result = verifyData()
                f.write("{}\t{}\t{}\n".format(epoch,learning_rate,result))

