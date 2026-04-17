import dataclasses

from langchain.tools import tool

@dataclasses
class UserProfile:
    name: str
    age: int
    location: str
    interests: list[str]


@tool
def user_profile_retriever() -> UserProfile:
    user_profile = UserProfile()


    return  user_profile