# Streamlit chat assistant

A Streamlit chat interface powered by OpenRouter's official Python SDK and Nemotron 3.5 Lightning free model, with streamed responses, reasoning enabled, and conversation history preserved between turns.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Add your OpenRouter API key to a `.env` file in the project root. `.env` is ignored by Git.

```toml
OPENROUTER_API_KEY= your-openrouter-api-key
```

Run the app:

```bash
streamlit run app.py
```

The app uses the official [`openrouter` Python SDK](https://pypi.org/project/openrouter/) to call `nvidia/nemotron-3.5-lightning:free` with reasoning enabled. Assistant messages are retained with their `reasoning_details` so they can be passed back on later turns.