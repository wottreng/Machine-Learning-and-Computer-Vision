"""
code credit goes to Mariya:
https://github.com/MariyaSha/
https://www.youtube.com/c/PythonSimplified
"""
import time

# EXAMPLE 2 - FIBONACCI

def fibonacci(idx):
    # iteration
    seq = [0,1]
    for i in range(idx):
        seq.append(seq[-1]+seq[-2])
    return seq[-2]


def Fibonacci(idx):
    # recursion
    if idx <= 1:
        return idx
    else:
        return Fibonacci(idx-1)+Fibonacci(idx-2)

#speed test for both fibonacci functions
print("***** recursion *****")
rec = time.time()
print(Fibonacci(8))
print("speed : " + str(time.time()-rec))
print("***** iteration *****")
it = time.time()
print(fibonacci(8))
print("speed : " + str(time.time()-it))

# EXAMPLE 1 - EVEN NUMS
def EvenNums(num):
    """
    A Recurssive Function that returns the
    positive even numbers smaller or equal to given number
    """
    print(num)
    if num % 2 != 0:
        return print("please select an even number")
    elif num == 2:
        #base case
        return num
    else:
        #recursive function call
        return EvenNums(num-2)

#EvenNums(20)
