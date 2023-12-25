import sys
import math
import string


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    
    X = dict()
    
    with open (filename,encoding='utf-8') as f:
        
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        line = f.read()
        upper_line = line.upper()
        
        for i in alphabet:
            X[i] = 0
            
        for i in upper_line:
            if i in X:    
               X[i] = X[i] + 1
        return X


print("Q1")

sfile = shred("letter.txt")
for i in sfile:
    print(i , sfile[i])
print("\n")

print("Q2")

x1 = sfile['A']
english1 = x1 * math.log(get_parameter_vectors()[0][0])
espanol1 = x1 * math.log(get_parameter_vectors()[1][0])

print("%.4f" %english1)
print("%.4f" %espanol1)
print("\n")
 
print("Q3")

f_english = 0.0
f_espanol = 0.0

for i in range(26):
    x = list(sfile.values())[i]
    f_english += x * math.log(get_parameter_vectors()[0][i])
    f_espanol += x * math.log(get_parameter_vectors()[1][i])

englishResult = math.log(0.6) + f_english
espanolResult = math.log(0.4) + f_espanol

print("%.4f" %englishResult)
print("%.4f" %espanolResult)
print("\n")  

print("Q4")

result = 0

if espanolResult - englishResult >= 100: 
    result = 0
elif espanolResult - englishResult <= -100: 
    result = 1
else:
    result = 1/(1 + math.exp(espanolResult - englishResult))

print("%.4f" %result)


           
# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
