## LLM Playground

Welcome to the LLM Playground, a powerful and intuitive web app built using Mesop by Google. This playground allows you to seamlessly interact with and compare responses from three advanced language models:

- **Command-R-Plus** by Cohere
- **Gemini 1.5 Flash** by Google
- **Llama3-8B-8192** by Meta AI

### Key Features

- **User-Friendly Interface**: Write your prompts and receive responses with ease, thanks to a clean and intuitive UI.
- **Model Selection**: Choose between three powerful language models to generate responses.
- **Adjustable Settings**: Customize your model’s temperature and token limit directly from the interface.
- **Live Code Generation**: Get the running code for each model with your selected settings, ready for integration into your projects.
- **Automatic Text Streaming**: View model responses in real-time as they are generated.
- **Hot Reload**: Experience rapid development with automatic browser reloads that preserve the state.

### Benefits of Mesop

- **Ease of Use**: Quickly build and iterate on your web app without the need for frontend expertise.
- **Rich Component Library**: Leverage ready-to-use components, including chat functionalities.
- **Strong Type Safety**: Enjoy robust IDE support and type safety for a smooth development experience.
- **Flexibility**: Build custom UIs without writing JavaScript, CSS, or HTML, and compose your UI into reusable components.

# Get Started

## Requirements

- Python 3.8+
- Install dependencies from `requirements.txt`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Kirouane-Ayoub/Mesop-LLM-Playground.git
    cd Mesop-LLM-Playground
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory and add your API keys:
    ```ini
    CO_API_KEY=your_cohere_api_key
    GEMINI_API_KEY=your_gemini_api_key
    GROQ_API_KEY=....
    ```

## Usage 

```sh
mesop app.py
```
![Screenshot from 2024-06-23 11-44-56](https://github.com/Kirouane-Ayoub/Mesop-LLM-Playground/assets/99510125/c83e2cd9-d698-4fa7-9eb1-82ce11b82799)
![Screenshot from 2024-06-23 11-45-43](https://github.com/Kirouane-Ayoub/Mesop-LLM-Playground/assets/99510125/235d18c5-e23e-40ed-86cf-7ee9d2a4bbec)
![Screenshot from 2024-06-23 11-45-56](https://github.com/Kirouane-Ayoub/Mesop-LLM-Playground/assets/99510125/b429478e-7fd4-4224-8a8c-8261a9df6b42)
