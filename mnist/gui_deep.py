from Tkinter import *
import tkMessageBox
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import preprocessimage as ppi
import inference_deep2 as infer
import serial

class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.cap = cv2.VideoCapture(0)
        self.resimage = cv2.imread('test.png')
        self.predictionResult = StringVar()
        self.predictionResult.set("The image we predicted is: unKnown")
        self.sess,self.y_conv,self.keep_prob,self.x = infer.restoreModel()
        self.createWidgets()
        self.res = 0

    def __del__(self):
        self.cap.release()

    def createWidgets(self):
        self.helloLabel = Label(self, text='Welcome to the HandWriting Digit App!')
        self.helloLabel.grid(columnspan=4,sticky=W)

        self.inputLabel = Label(self, text='Image path: ')
        self.inputLabel.grid(row=1, column=0,sticky=W)

        self.nameInput = Entry(self)
        self.nameInput.insert(END, './pic3/')
        self.nameInput.grid(row=1, column=1)

        self.inputLabel2 = Label(self, text='Image name: ')
        self.inputLabel2.grid(row=1, column=2,sticky=W)

        self.nameInput2 = Entry(self)
        self.nameInput2.insert(END, 'figure_0.png')
        self.nameInput2.grid(row=1, column=3)

        self.getImage = Button(self, text='Shoot', command=self.shoot)
        self.getImage.grid(row=2, column=0)

        self.saveButton = Button(self, text='Save', command=self.saveImage)
        self.saveButton.grid(row=2, column=1)

        self.quitButton = Button(self, text='Process', command=self.imagePreprocess)
        self.quitButton.grid(row=2, column=2)

        self.quitButton = Button(self, text='Predict', command=self.imagePredict)
        self.quitButton.grid(row=2, column=3)

        self.inputLabel3 = Label(self, textvariable=self.predictionResult)
        self.inputLabel3.grid(row=3, columnspan=4,sticky=W)

        self.sendButton = Button(self, text='Send', command=self.send)
        self.sendButton.grid(row=4, column=1,columnspan=1)

        self.quitButton = Button(self, text='Quit', command=self.quit)
        self.quitButton.grid(row=4, column=2,columnspan=1)

    def shoot(self):
        #vidcap = cv2.VideoCapture()
        ret, image = self.cap.read()
        self.resimage = image
        plt.imshow(image)
        plt.show()

    def saveImage(self):
        inputPath = self.nameInput.get()
        inputName = self.nameInput2.get()
        print inputPath
        if os.path.exists(inputPath):
            pass
        else:
            print "mkdir"
            os.system("mkdir "+inputPath)
        cv2.imwrite(inputPath+inputName, self.resimage)

    def imagePreprocess(self):
        inputPath = self.nameInput.get()
        inputName = self.nameInput2.get()
        ppi.preprocess(inputPath, inputName)

    def imagePredict(self):
        inputPath = self.nameInput.get()
        inputName = self.nameInput2.get()
        self.res = infer.main(inputPath, inputName,self.sess,self.y_conv,self.keep_prob,self.x)
        self.predictionResult.set("The image we predicted is: "+str(self.res))

    def send(self):
        ser = serial.Serial('/dev/tty.usbserial-A6YT1ITY', 57600, timeout=1)
        sendData = str(self.res) + "\n"
        ser.write(sendData)
        #time.sleep(1)
        ser.close()

if __name__ == "__main__":
    app = Application()
    app.master.title('HandWriting Digit Recognize')
    app.mainloop()