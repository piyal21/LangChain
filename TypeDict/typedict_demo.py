from typing import TypedDict



class Person(TypedDict):
    name:str 
    age : int 
    
    
    
person1 : Person={'name': 'piyal', 'age':25}


print(person1['age'])