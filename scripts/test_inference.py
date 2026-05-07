from vllm import LLM, SamplingParams

model = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
params = SamplingParams(temperature=0.7, max_tokens=128)

prompts = ["What is PagedAttention?"]
outputs = model.generate(prompts, params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}\n")