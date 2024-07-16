import cohere
import google.generativeai as genai
from groq import Groq

import settings


def generate_with_cohere(
    prompt: str, temperature: float, model_name: str, token_limit: int
):
    co = cohere.Client(settings.CO_API_KEY)
    response = co.chat(
        message=prompt,
        model=model_name,
        temperature=temperature,
        max_tokens=token_limit,
    )
    return response.text


def generate_with_gemini(
    prompt: str, temperature: float, model_name: str, token_limit: int
):
    genai.configure(api_key=settings.GOOGLE_API_KEY)
    generation_config = {"temperature": temperature, "max_output_tokens": token_limit}
    gen_model = genai.GenerativeModel(model_name, generation_config=generation_config)
    response = gen_model.generate_content(prompt)
    return response.text


def generarte_with_groq(
    prompt: str, temperature: float, model_name: str, token_limit: int
):
    client = Groq(api_key=settings.GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        temperature=temperature,
        max_tokens=token_limit,
    )
    response = chat_completion.choices[0].message.content
    return response
