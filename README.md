# lite squad
Tool for working with a team of LLMs. Sometimes two, or six, heads are better than one.

## Usage
Install, and then use, at your command line:

    pip install litesquad
    litesquad --web # to run interactive browser version (Dash)

CLI run:

    litesquad "plan such and such project"
    litesquad "such and such" --quick # get a quick answer, bypassing the squad

litesquad can take a few minutes to run when not in deep think mode. There's a lot of LLM calls going on under the hood (see below). 

Query is distributed to worker LLMs (gemini, gpt4.1-mini, sonnet, plus deepseek, mistral, and llama via OpenRouter). Another (gpt5) acts as a critic that gives feedback to the workers. They revise their response. An intermediate representation of these responses to extract their main content is extracted, and clustered into categories of suggestions. A judge (opus) converts these suggestions into a final coherent answer for the user. The independent worker chains run in parallel (the run.max_parallel config setting controls how many at once; set it to 1 for fully serial runs).

To run the default, you will need API keys for [Gemini](https://aistudio.google.com/app/apikey), [OpenAI](https://openai.com/index/openai-api/), [Anthropic](https://platform.claude.com/docs/en/api/admin/api_keys/retrieve), and [OpenRouter](https://openrouter.ai/keys). One OpenRouter key reaches the whole openrouter.ai catalog (deepseek, mistral, llama, grok, qwen, ...), which is how the worker roster grows wide without a key per provider. Store your api keys in .env (or whatever). 

### Basic tests
To see if api keys are working (from activated env):  `python -m litesquad.check_keys`

Check on specific models: `litesquad --check` 

Offline test: `litesquad --smoke --mock`


### Caveats
This provides an interface to an ensemble of LLMs to try to generate a better answer than with a single LLM. No agentic tool-usage, web calls from the LLMs. Just vanilla text processing.

With apologies to [squad](https://github.com/bradygaster/squad). 
