import os
import sys

# --- PATH INJECTION ---
# This ensures the 'Validate' folder is visible to this script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from Validate.validate import CareerRoadmap, SkillGap, Resource

load_dotenv()

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=os.getenv("HF_TOKEN")
)

SYSTEM_INSTRUCTION = (
    "You are a Career Path AI. Always reply with ONLY a valid JSON object inside a fenced ```json ... ``` block. "
    "The JSON keys must exactly match the schema provided."
    "Limit each skill to a maximum of 2 high-quality resources. Be concise in descriptions."
)

def build_prompt(current_role: str, target_role: str, time_period: str) -> str:
    # Use Pydantic to get the schema automatically
    schema = CareerRoadmap.model_json_schema()
    
    return f"""SYSTEM: {SYSTEM_INSTRUCTION}
    SCHEMA: {json.dumps(schema)}

    User: I'm a {current_role} transitioning to {target_role}. I have {time_period} for this.
    
    Return a JSON following the schema. Ensure 'learning_resources' includes 'type', 'name', 'link', and 'priority'.
    """

# --- Execution ---
prompt = build_prompt(".Net developer", "ML Engineer", "1 month")

response = client.chat_completion(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=2000,
    temperature=0.1
)

raw_output = response.choices[0].message.content

# 1. Robust cleaning: strip everything before the first '{' and after the last '}'
def clean_json_string(text: str) -> str:
    # Find the first '{' and the last '}'
    start_index = text.find('{')
    end_index = text.rfind('}')
    
    if start_index != -1 and end_index != -1:
        return text[start_index:end_index + 1]
    return text.strip()

json_string = clean_json_string(raw_output)

try:
    # 1. Validate and Parse
    roadmap = CareerRoadmap.model_validate_json(json_string)
    
    # Header Section
    print("\n" + "="*60)
    print(f"CAREER TRANSITION ROADMAP: {roadmap.target.upper()}")
    print(f"ALLOCATED TIMEFRAME: {roadmap.time_allocated}")
    print("="*60)
    
    # Skill Gaps Analysis
    print("TECHNICAL SKILL GAP ANALYSIS")
    print("-" * 30)
    
    for i, gap in enumerate(roadmap.gaps, 1):
        print(f"\n{i}. {gap.skill.upper()}")
        print(f"   TIME ESTIMATE : {gap.learning_time_estimate}")
        print(f"   DESCRIPTION   : {gap.description}")
        
        print(f"   LEARNING RESOURCES:")
        for res in gap.learning_resources:
            # Determine priority label
            p_label = f"[{res.priority.upper()}]"
            
            # Print Resource Details
            print(f"      - {p_label:<10} {res.name}")
            print(f"        TYPE       : {res.type.upper()}")
            print(f"        SOURCE     : {res.link}")

    print("\n" + "="*60)
    print("END OF GENERATED ROADMAP")

except Exception as e:
    print(f"CRITICAL ERROR: Data validation failed.")
    print(f"DETAILS: {str(e)}")

def fix_incomplete_json(json_string: str) -> str:
    """Basic helper to close dangling JSON structures."""
    json_string = json_string.strip()
    
    # Count opening vs closing
    braces_needed = json_string.count('{') - json_string.count('}')
    brackets_needed = json_string.count('[') - json_string.count(']')
    
    # Add missing closing characters
    json_string += ']' * brackets_needed
    json_string += '}' * braces_needed
    
    return json_string

# Use it before validation:
json_string = clean_json_string(raw_output)
json_string = fix_incomplete_json(json_string) # <-- Add this line

