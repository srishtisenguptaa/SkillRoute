import os
import sys
import json
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# --- PATH INJECTION ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Validate.validate import CareerRoadmap

load_dotenv()

# Using a highly available model
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=os.getenv("HF_TOKEN")
)

SYSTEM_INSTRUCTION = (
    "You are a Career Path AI. You must output ONLY a valid JSON object. "
    "Do not include any introductory or concluding text. "
    "The JSON must strictly follow the provided schema."
)

def build_prompt(current_role: str, target_role: str, time_period: str) -> str:
    schema = CareerRoadmap.model_json_schema()
    return (
        f"I am a {current_role} transitioning to {target_role}. I have {time_period} for this.\n"
        f"Return the roadmap following this JSON schema: {json.dumps(schema)}"
    )

def extract_clean_json(text: str) -> str:
    """Uses regex to find the first JSON-like block to avoid trailing character errors."""
    # Look for content inside markdown code blocks first
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: find the first '{' and the last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        return text[start:end+1].strip()
    
    return text.strip()

# --- Execution ---
user_prompt = build_prompt(".Net developer", "DevOps Engineer", "2 months")

try:
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=2000,
        temperature=0.1
    )

    raw_output = response.choices[0].message.content
    json_string = extract_clean_json(raw_output)

    # Validate and Parse using Pydantic
    roadmap = CareerRoadmap.model_validate_json(json_string)
    
    # Output formatting
    print("\n" + "="*60)
    print(f"CAREER TRANSITION ROADMAP: {roadmap.target.upper()}")
    print(f"ALLOCATED TIMEFRAME: {roadmap.time_allocated}")
    print("="*60)
    
    for i, gap in enumerate(roadmap.gaps, 1):
        print(f"\n{i}. {gap.skill.upper()} ({gap.learning_time_estimate})")
        print(f"   {gap.description}")
        print(f"   RESOURCES:")
        for res in gap.learning_resources:
            print(f"     - [{res.priority.upper()}] {res.name} ({res.link})")

    print("\n" + "="*60)
    print("END OF GENERATED ROADMAP")

except Exception as e:
    print(f"\nCRITICAL ERROR: {type(e).__name__}")
    print(f"DETAILS: {str(e)}")
    # If it fails, print the raw output to see what the LLM actually said
    if 'raw_output' in locals():
        print("\n--- RAW OUTPUT FROM LLM ---")
        print(raw_output)