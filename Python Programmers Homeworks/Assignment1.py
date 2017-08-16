# Question 1 (4 points) :  places = {'San Antonio':100, 'Austin':50, 'Houston':200, 'Dallas':75}. 
# The key in this dictionary is the name of a place and the value is the distance of that place from some origin.  
# To the standard output print the place and the distance that is closest to the origin.

places = {"San Antonio": 100, "Austin": 50, "Houston": 200, "Dallas": 75}
 
# initializing the closest city to infinite and the closest city to an empty string
closestDistance = float("inf")
closestCity = ""
 
# interrating over the map
for city, Distance in places.items():
    if Distance < closestDistance:
        # this distance is now the closest distance
        closestDistance = Distance
        closestCity = city
 
print "%s is the closest city with a distance of %d" % (closestCity, closestDistance)
 
# Question 2 (3 points) 
# mylist = ['tape', 'copy', 'tape', 'pencil', 'pen', 'tape', 'copy', 'clip', 'copy', 'pencil', 'pen', 'copy']. 
# Create a dictionary with the word as the key and the frequency of its occurrence as the value.

mylist = ["tape", "copy", "tape", "pencil", "pen", "tape", "copy", "clip", "copy", "pencil", "pen", "copy"]
dictionary1 = {x:mylist.count(x) for x in mylist}
print dictionary1

# Question 3 (3 points) Keep getting names from the user until the user decides to quit by entering 'Q' or 'q'. 
# If the user doesn't enter a name, then inform them that they can't leave it blank. 
# Check the number of characters in the name. Print it to the standard output.


while True:  
    username = raw_input ("Please enter a name: ")
    if username == "q" or username == "Q":
        print "You decided to stop entering name."
        break
    elif username == "":
        print "Please enter a name. You can't leave it blank!"
    else:
        print "The name %s has %d characters"  %(username, len(username))
