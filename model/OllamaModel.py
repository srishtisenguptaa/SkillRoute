import os
import sys
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Setup Paths ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FastParallelRoadmap:
    def __init__(self):
        # STRATEGY MODEL: Mistral-Nemo (12B) for high-quality logic
        self.smart_llm = ChatOllama(model="mistral-nemo", temperature=0.1)
        
        # SEARCH MODEL: Llama 3.2 (3B) or Phi-3.5 for speed/formatting
        # Note: Ensure you run `ollama pull llama3.2` before executing
        self.fast_llm = ChatOllama(model="qwen2.5:3b-instruct-q4_k_m", temperature=0)

    def _get_course_search_chain(self):
        """Branch 1: Focuses on finding platforms using the FAST model."""
        prompt = ChatPromptTemplate.from_template(
            "List real courses from Coursera, Udemy, or edX for transitioning "
            "from {current_role} to {target_role}. Format: 'Course Name - Platform'."
        )
        # Use the faster, smaller model here
        return prompt | self.fast_llm | StrOutputParser()

    def _get_strategy_chain(self):
        """Branch 2: Focuses on milestones using the SMART model."""
        prompt = ChatPromptTemplate.from_template(
            "Create a 3-step milestone strategy for a {time_period} transition "
            "from {current_role} to {target_role}. Focus on critical technical skills."
        )
        # Use the smarter, larger model here
        return prompt | self.smart_llm | StrOutputParser()

    def generate_roadmap(self, current_role, target_role, time_period):
        # 1. Run both LLM queries at the same time
        map_chain = RunnableParallel(
            courses=self._get_course_search_chain(),
            strategy=self._get_strategy_chain(),
            # Keep original variables for the final printout
            target=RunnablePassthrough(),
            time=RunnablePassthrough()
        )

        # 2. Instant Python Formatting (No LLM needed here)
        def combine_results(data):
            return (
                f"--- FINAL CAREER ROADMAP ---\n"
                f"Target: {target_role} in {time_period}\n\n"
                f"### STRATEGIC MILESTONES\n{data['strategy']}\n\n"
                f"### RECOMMENDED RESOURCES\n{data['courses']}\n"
                f"-----------------------------"
            )

        # 3. Construct the full pipeline
        # Data flows: Input -> Parallel LLMs -> Python Formatter
        final_chain = map_chain | RunnableLambda(combine_results)

        return final_chain.invoke({
            "current_role": current_role,
            "target_role": target_role,
            "time_period": time_period
        })

if __name__ == "__main__":
    # Ensure Ollama is running before starting
    try:
        generator = FastParallelRoadmap()
        print("Generating roadmap... (Running parallel local models)\n")
        
        result = generator.generate_roadmap(".Net Developer", "ML Engineer", "4 weeks")
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Make sure 'ollama serve' is running and you have pulled 'mistral-nemo' and 'llama3.2'.")