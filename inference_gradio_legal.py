import json
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModelForCausalLM
from utils.prompter import Prompter


tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
prompt_template_name = "alpaca_summarization"
# template = json.load(open(template, "r"))


model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b",
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, "lora-falcon-7b-legal-summary", adapter_name="legal")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompter = Prompter(prompt_template_name)

def generate(input_prompt, temperature, top_p, top_k, num_beams, max_tokens):

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        num_beams=num_beams,
        # max_tokens=max_tokens,
        repetition_penalty=1.15,
    )

    # PROMPT =f'''Below is an instruction that describes a task. Write a response that appropriately completes the request. Write a detailed summary of the meeting in the input.
    # ### Instruction:
    # {input_prompt}
    # ### Response:
    # '''
    # PROMPT = template['prompt_no_input'].format(instruction=input_prompt)

    PROMPT = prompter.generate_prompt(input_prompt)
    prompt_len = len(PROMPT)

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].cuda()

    print("Generating...")
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.batch_decode(generation_output.sequences)[0][prompt_len:]
    print(response)
    return response



import gradio as gr

gr.Interface(
    fn=generate,
    inputs=[
        gr.components.Textbox(
            lines=2,
            label="Instruction",
            placeholder="Tell me about the stock market.",
        ),
        # gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        gr.components.Slider(
            minimum=0, maximum=1, value=0.6, label="Temperature"
        ),
        gr.components.Slider(
            minimum=0, maximum=1, step=0.05, value=0.95, label="Top p"
        ),
        gr.components.Slider(
            minimum=0, maximum=100, step=1, value=50, label="Top k"
        ),
        gr.components.Slider(
            minimum=1, maximum=4, step=1, value=1, label="Beams"
        ),
        gr.components.Slider(
            minimum=1, maximum=2000, step=32, value=128, label="Max tokens"
        ),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=5,
            label="Output",
        )
    ],
    title="🦙🧑‍⚖️⚖️ Legal Alpaca-LoRA",
    description="Legal Alpaca-LoRA is a 7B-parameter LLaMA model. It is trained on the [multi-lexsum dataset](https://huggingface.co/datasets/allenai/multi_lexsum) dataset and makes use of the Huggingface LLaMA implementation. For more information, refer to the [original implementation](https://github.com/tloen/alpaca-lora)",  # noqa: E501
).queue().launch()