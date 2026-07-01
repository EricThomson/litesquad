# lite squad
Lightweight cli tool for working with a team of LLMs. 

Sometimes three heads are better than one.

## Usage
Install, and then use, at your command line:

    pip install litesquad
    litesquad "plan such and such project"

To run the default, you will need API keys for [Gemini](https://aistudio.google.com/app/apikey), [OpenAI](https://openai.com/index/openai-api/), and [Anthropic](https://platform.claude.com/docs/en/api/admin/api_keys/retrieve). Store your api keys in .env (or whatever). 

Query is distibuted to three worker LLMs (gemini, gpt4.1-mini, sonnet). Another (gpt5) acts as a critic that gives feedback to the workers. They revise their response. The judge (opus) uses these modified responses to synthesize a final reply for the user. 

### Basic tests
To see if api keys are working (from activated env):  `python -m litesquad.check_keys`

Check on specific models: `litesquad --check` 

Offline test: `litesquad --smoke --mock`


### Caveats
This is an ensemble of LLMs meant to provide better answers than a single LLM. This is not agentic: it currently doesn't do anything besides text processing. 

With apologies to [squad](https://github.com/bradygaster/squad). 
