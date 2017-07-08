from ctypes import *
import time

if __name__ == '__main__':
    time_begin = time.clock()

    dll = CDLL("./main.so")
    print(dll.add(2, 6))
    #dll.print_sum(10000)

    t = time.clock() - time_begin
    print("\nUse time: %s" % t)