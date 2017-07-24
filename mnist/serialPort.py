import time
import serial

#cd /dev
#ls -l tty*
#find the name of the usb device
if __name__ == "__main__":
    print("Hello world!")
    i = 0
    ser = serial.Serial('/dev/tty.usbserial-A6YT1ITY', 57600, timeout=1)
    while 1:
        #readline: would read all the data unit "\n"
        '''
        line = ser.readline()
        print line
        time.sleep(5)
        i = i+1
        if i == 1:
            ser.write("1\n")
        elif i == 2:
            ser.write("2\n")
        elif i == 3:
            ser.write("3\n")
        elif i == 4:
            ser.write("4\n")
        if i == 4:
            i = 0
        '''

        # send the data as input
        line = ser.readline()
        print line
        number = raw_input("Input the recognized number: ")
        sendData = number + "\n"
        ser.write(sendData)
        if number == 11:
            break

    ser.close()
