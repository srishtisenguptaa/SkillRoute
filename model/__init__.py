llm_base = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    temperature=0.1,
    max_new_tokens=2000
)
self.llm = ChatHuggingFace(llm=llm_base)