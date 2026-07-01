# lite squad
Lightweight cli tool for pooling results from an ensemble of LLMs. 

Sometimes two (or three) heads are better than one.

## Usage
Install, and then use, at your command line:

    pip install litesquad
    litesquad "plan such and such project"

To run the default with workers from different vendors, you will need API keys for [Gemini](https://aistudio.google.com/app/apikey), [OpenAI](https://openai.com/index/openai-api/), and [Anthropic](https://platform.claude.com/docs/en/api/admin/api_keys/retrieve). Store your api keys in .env (or whatever). 

One LLM (opus) acts as a manager who assigns your question to some worker LLMs (gemini, gpt4, sonnet). Another (gpt5) acts as a critic that gives feedback to the workers. They revise their response. The manager generates an integrated reply for the user. 

### Basic tests
To see if api keys are working (from activated env):  `python -m litesquad.check_keys`

Check on specific models: `litesquad --check` 

### To do
Allow "pm only" option so you don't have to invoke entire 11-call machinery for every query. 

### Caveats
This is a glorified LLM (an ensemble of LLMs orchestrated by a really good LLM) meant to provide better answers than any single LLM. It is not agentic: it doesn't do anything besides text processing. 

To work in agentic mode, see [squad](https://github.com/bradygaster/squad). 
