def demo_func(num1,num2,num3=4):
    """Demo function for testing purpose

    >>> demo_func(1,1,1)
    """
    return abs(num1-num2-num3)


#Function as input
def multiply(arg1,arg2):
    return arg1*arg2

def func_as_func(fn,arg1,arg2):
    return fn(arg1,arg2)

print(func_as_func(multiply,3,4))

def squared(x):
    return x%5

print(max([30,32,45,76,86,45,6,53,45],key=squared))

lambda_func = lambda x: x % 5


print('Greater than 5' if 6 < 5 else 'Not greater')

print(max([30,32,45,76,86,45,6,53,45],key=lambda_func))
print(lambda_func(7))

#Slicing 

fruits = ['apple','pearl','grape','oragne','strawberry','passionfruit','pinapple','banana']

print(fruits[1:3])
print(fruits[-1])
print(fruits[:6])
print(fruits[2:])
print(fruits[-3:])

fruits[0:3]=['app','pear','grap','grapas','adasd']
print(fruits)



planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

for i in range(len(planets)):
    planets[i] = planets[i] * 2 

for planet in planets:
    print(planet,end=' ')

for i, planet in enumerate(planets):
    print(planet,planets[i])

list(enumerate(planets))

#Unpacking the tuple
x = 0.125
numerator, denominator = x.as_integer_ratio()

#List of tuples
nums = [
    ('one', 1, 'I'),
    ('two', 2, 'II'),
    ('three', 3, 'III'),
    ('four', 4, 'IV'),
]


#List comprehensation
#SQL like syntax
#Create a reduced list from the original
reduced_planet = [planet.upper() for planet in planets if len(planet)<6]
print(reduced_planet)

#One line if

#Lamba
#Like anonymous methods
f = lambda x: x*2
print(max([2,3,4,5,6,7,84,3,34,2,2.1],key = f))

number_list = [2, 6, 6, 7,1,-2,3,-9,-10,-23,-43]

print(len([num for num in number_list if num<0]))

splitting_text = "This text will be splitted"

list_words= splitting_text.split()

print('%'.join(word.upper() for word in list_words))

print('{} is a very great {} language'.format('Pyhton','Programming'))

#External libraries
import math

print("Its a math: {}".format(type(math)))