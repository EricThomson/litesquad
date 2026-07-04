# lite squad
Tool for working with a team of LLMs. Because sometimes two, or five, heads are better than one.

## Usage
Install with pip:

    pip install litesquad

Use in browser: 

    litesquad --web 

Use at command line:

    litesquad "plan such and such project"
    litesquad "such and such" --quick # get a quick answer, bypassing the squad

litesquad takes a few minutes to run when running in default (deep) mode. There's a lot of LLM calls going on under the hood.

Query is distributed to worker LLMs (gemini and sonnet direct, plus deepseek, mistral, and llama via OpenRouter). Another (grok) acts as a critic that gives feedback to the workers. They revise their response. An intermediate representation of these responses is extracted to pull the content and clustered into categories of suggestions (gpt5). A judge (opus) converts these suggestions into a final coherent answer for the user. 

## What about Fusion?
After I pushed version 0.0.2 I found out about OpenRouter's [Fusion](https://openrouter.ai/blog/announcements/fusion-beats-frontier/).

It has the same goal as litesquad, and a similar architecture, but is built by a professional team that can actually maintain the product. When I did testing of litesquad vs Fusion, Fusion consistently beat litesquad. They both produce good results, but if you want to *use* something (versus play in a workbench) I would recommend Fusion. 

Also, my guess is there will be a squadron of these applications within a couple of years. It's just too obvious a need. 

### So. Many. APIs.
Todo: replace with OpenRouter API key only.

To run properly, litesquad currently needs API keys for [Gemini](https://aistudio.google.com/app/apikey), [OpenAI](https://openai.com/index/openai-api/), [Anthropic](https://platform.claude.com/docs/en/api/admin/api_keys/retrieve), and [OpenRouter](https://openrouter.ai/keys). One OpenRouter key reaches the whole openrouter.ai catalog (deepseek, mistral, llama, grok, qwen, ...), which is how the worker roster grows wide without a key per provider. You can store your API keys in `.env`.

### Basic tests
To see if api keys are working (from activated env):  `python -m litesquad.check_keys`

Check on specific models: `litesquad --check` 

Offline test: `litesquad --smoke --mock`

### Caveats
This provides an interface to a swarm of LLMs to try to generate a better answer than when using a single LLM. Unlike Fusion, there is no tool-usage such as web calls from the LLMs. Just reasoning. 

With apologies to [squad](https://github.com/bradygaster/squad). 

