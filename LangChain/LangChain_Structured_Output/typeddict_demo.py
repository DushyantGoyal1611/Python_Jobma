from typing import TypedDict

class Person(TypedDict):
    name: str
    age:int

new_person: Person = {'name':'Dushyant', 'age':22}
print(new_person)
print(type(new_person['age']))