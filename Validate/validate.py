from typing import List, Optional, Tuple
import re
from pydantic import BaseModel, HttpUrl

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

def verify_time_budget(roadmap: CareerRoadmap, limit_str: str) -> Tuple[bool, str]:
    """
    Validation Link: Checks if the AI's math matches the user's budget.
    """
    limit_match = re.search(r'(\d+)', limit_str)
    if not limit_match: return True, ""
    
    allowed_val = int(limit_match.group(1))
    unit = "week" if "week" in limit_str.lower() else "month"
    
    actual_val = 0
    for gap in roadmap.gaps:
        val_match = re.search(r'(\d+)', gap.learning_time_estimate)
        if val_match:
            actual_val += int(val_match.group(1))

    if actual_val > allowed_val:
        return False, f"Time budget exceeded! Plan is {actual_val} {unit}s, limit is {allowed_val} {unit}s."
    return True, "Budget verified."