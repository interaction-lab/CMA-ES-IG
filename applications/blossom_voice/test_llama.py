import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16, 'max_length':150}, device_map="auto"
)

print(pipeline("You are a cognitive behavioral therapist, and you are teaching someone about cognitive distortions. You say, \
               \"One common cognitive distortion is called \"catastrophizing.\" This is when you assume the worst possible outcome will happen, \
               even if there is no evidence to support it. For example, if you are running late to a meeting, you might think, \"I'm going to \
               get fired.\" This is an example of catastrophizing. Your participant asks, \"How can I stop catastrophizing?\" You respond, "))