from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name : str = 'Dushyant'
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(ge=0, le=10, default=5, description='This field represents the cgpa of a student')

new_student = {'age':22, 'email':'goyaldushyant1611@gmail.com', 'cgpa':10}
student = Student(**new_student)
print(student)