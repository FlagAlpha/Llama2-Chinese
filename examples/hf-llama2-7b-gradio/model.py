from threading import Thread
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = '/path/models--meta-llama--Llama-2-7b-chat-hf'

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map='auto',
        load_in_8bit=True
    )
else:
    model = None
tokenizer = AutoTokenizer.from_pretrained(model_id)


def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    for user_input, response in chat_history:
        texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
    texts.append(f'{message.strip()} [/INST]')
    return ''.join(texts)


def run(message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50) -> Iterator[str]:
    prompt = get_prompt(message, chat_history, system_prompt)
    inputs = tokenizer([prompt], return_tensors='pt').to("cuda")

    streamer = TextIteratorStreamer(tokenizer,
                                    timeout=10.,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield ''.join(outputs)
