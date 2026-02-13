from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field

class Resource(BaseModel):
    type: str
    name: str
    link: HttpUrl  
    priority: Optional[str] = "medium"

class SkillGap(BaseModel):
    skill: str
    description: str
    learning_time_estimate: str
    learning_resources: List[Resource]

class CareerRoadmap(BaseModel):
    target: str
    time_allocated: str
    gaps: List[SkillGap]
    roadmap_summary: str