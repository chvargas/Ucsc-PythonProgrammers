
# coding: utf-8

# In[ ]:

""" ####################################   Question 1 #########################################################################

 A queue follows FIFO (first-in, first-out). FIFO is the case where the first 
element added is the first element that can be retrieved. 
Consider a list with values [4, 2, 9]. Create functions queueadd and 
queueretrieve to add and pop elements from the list in FIFO order 
respectively. After each function call, print the content of the list.
Add 7 to the queue and then follow the FIFO rules to pop elements 
until the list is empty

"""


# In[1]:

# Defining Class
class Sclass:
    #List instantiation 
    def __init__(self):
        self.myListLIFO = [4,2,9]
    
    #funtion to append list (adding values)
    def Sadd(self, firstNumber):
        self.myListLIFO.append(firstNumber)
        print(self.myListLIFO)
    
    #funtion to retieve information list (getting values)
    def Sretrieve(self):
        if len(self.myListLIFO) == 0 :
            print( " The list is empty. " )
            return
        self.myListLIFO.pop()
        print(self.myListLIFO)


testListLIFO = Sclass()

testListLIFO.Sadd(7)
testListLIFO.Sretrieve()
testListLIFO.Sretrieve()
testListLIFO.Sretrieve()
testListLIFO.Sretrieve()
testListLIFO.Sretrieve()


# In[ ]:

"""  ###################################     QUESTION 2      ###################################################################

 Implement an encoding scheme.
The string 
WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWBWWWWWWWWWWWWWW 
has 67 characters. Write a function called getcompressed to convert this string to 
12W1B12W3B24W1B14W. The new string is created
by calculating the number of times a characters appears consecutively and
placing the character next to it. The new string only needs 18 character,
thus compressing the original string by 73%.
 """


# In[20]:

string = "WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWBWWWWWWWWWWWWWW"
count = 1
length = ""

for i in range ( 1 , len(string) ):
    
    if string[i-1] == string[i]:
        count = count + 1
    else :
        length = length + string[i-1] + str(count)
        count = 1
        
length = length + string[i] + str(count)

print (length)


# In[ ]:




# In[ ]:



