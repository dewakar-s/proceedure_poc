from pydantic import BaseModel

class StartRequest(BaseModel):
    steps: list
    thread_id: str

class ResumeRequest(BaseModel):
    thread_id: str
    user_input: str
