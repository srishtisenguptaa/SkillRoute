import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

# --- PATH INJECTION ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Validate.validate import CareerRoadmap, verify_time_budget

load_dotenv()

class RoadmapGenerator:
    def __init__(self):
        # 1. Initialize the Model (using the LangChain wrapper)
        self.llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            temperature=0.1,
            max_new_tokens=2000
        )
        
        # 2. Set up the Pydantic Parser
        self.parser = PydanticOutputParser(pydantic_object=CareerRoadmap)

    def get_chain(self):
        # 3. Define a strict prompt to prevent hallucinations
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an elite Career Coach. You provide roadmap data in strict JSON format.\n"
                "ANTI-HALLUCINATION RULES:\n"
                "1. Only suggest real, well-known platforms (Coursera, Udemy, EdX, Official Docs).\n"
                "2. The total time of all skills MUST match the user's budget exactly.\n"
                "3. If the timeframe is short, provide fewer, more intense skill gaps.\n"
                "{format_instructions}"
            )),
            ("user", "Transition: {current_role} to {target_role}. Timeframe: {time_period}.")
        ])

        # 4. Create the Chain using LCEL (| operator)
        return prompt | self.llm | self.parser

    def generate_roadmap(self, current_role, target_role, time_period):
        chain = self.get_chain()
        
        # Format instructions tell the LLM exactly how the JSON should look
        format_instructions = self.parser.get_format_instructions()
        
        try:
            # 5. Execute the Chain
            roadmap = chain.invoke({
                "current_role": current_role,
                "target_role": target_role,
                "time_period": time_period,
                "format_instructions": format_instructions
            })

            # 6. Final Logic Check (The "Gatekeeper")
            is_valid, msg = verify_time_budget(roadmap, time_period)
            if not is_valid:
                print(f"⚠️ Budget Warning: {msg}")
            
            return roadmap

        except OutputParserException as e:
            print(f"❌ Failed to parse JSON: {e}")
        except Exception as e:
            print(f"❌ Chain Error: {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    generator = RoadmapGenerator()
    result = generator.generate_roadmap(".Net Developer", "DevOps Engineer", "2 weeks")
    
    if result:
        print(f"\n✅ Generated Roadmap for: {result.target}")
        print(f"Summary: {result.roadmap_summary}")
        for gap in result.gaps:
            print(f"- {gap.skill} ({gap.learning_time_estimate})")