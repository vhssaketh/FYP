from faceRecognitionMain import FaceRecognition
from tkinter import *

trainPath=""
testPath=""
printAcc=True
##Get the parameters

def click():
    trainPath=train.get()
    testPath=test.get()
    print(trainPath, testPath)
    faceRec = FaceRecognition(24, 8)
    model = faceRec.trainRecognizer(trainPath=trainPath, printAcc=printAcc, testPath=testPath)
    close_window()


def close_window():
    window.destroy()
def store_selection():
    printAcc=var1.get()


window = Tk()
window.title("Face Recognition")
window.configure(background="black")
Label (window, text="Welcome to Real Time Face Detection ", bg="black", fg="white", font="none 12 bold"). grid (row=1, column=1, sticky=W)

Label (window, text="Enter train path : ", bg="black", fg="white", font="none 12 bold"). grid (row=2, column=0, sticky=W)
train= Entry(window, width=20, bg="white")
train.grid(row=2, column=1, sticky=W)
Label (window, text="Enter test path : ", bg="black", fg="white", font="none 12 bold"). grid (row=4, column=0, sticky=W)
test= Entry(window, width=20, bg="white")
test.grid(row=4, column=1, sticky=W)
var1=IntVar()
c1=Checkbutton(window, text="Print Accuracy?",variable=var1, onvalue=True, offvalue=False, command=store_selection)
c1.grid(row=5)
Button(window, text="SUBMIT", width=6, command=click) .grid(row=6)
window.mainloop()


