from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModelForCausalLM
import gradio as gr


model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

descriptions = {
    "tiiuae/falcon-7b": "Falcon-7B is a 7B parameters causal decoder-only model built by TII and trained on 1,500B tokens of RefinedWeb enhanced with curated corpora. It is made available under the Apache 2.0 license.",

}


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

    PROMPT =f'''Below is an instruction that describes a task. Write a response that appropriately completes the request. Write a detailed summary of the meeting in the input.
    ### Instruction:
    {input_prompt}
    ### Response:
    '''

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
    title="🦅 Falcon 7B",
    description=descriptions[model_id],  # noqa: E501
).queue().launch()