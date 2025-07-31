from llama_cpp import Llama

llm = Llama(
    model_path="C:\\Users\\vansh\\llama-models\\mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=-1,  # Use GPU acceleration (for RTX 3060)
    n_ctx=2048,
    verbose=True
)

# Basic chat loop
print("🤖 LLaMA is ready! Type 'exit' to stop.");
while True:
    prompt = input("🧑 You: ")
    if prompt.lower() == "exit":
        break

    response = llm(
        f"[INST] {prompt} [/INST]",
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
        stop=["</s>"]
    )
    print(f"🤖 Bot: {response['choices'][0]['text'].strip()}")
