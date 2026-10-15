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

Query is distributed to worker LLMs (e.g., GPT, Gemini). Another LLM acts as a critic that gives feedback to the workers. They revise their response. An intermediate representation of these responses is extracted, and this is then converted into categories/clusters of suggestions. A final frontier judge model converts these suggestions into a final coherent answer for the user. 

### API key management
To run properly, litesquad currently needs an [OpenRouter](https://openrouter.ai/keys) api key. One OpenRouter key reaches the whole openrouter.ai catalog (deepseek, mistral, llama, grok, qwen, ...), which is how the worker roster grows wide without a key per provider. Store your API key in `.env`.

### Basic tests
To see if openrouter api key is set up and working against default models:

    litesquad --check 

Offline test: `litesquad --smoke --mock`

### Caveats
This provides an interface to a swarm of LLMs to try to generate a better answer than when using a single LLM. There is no tool-usage such as web calls from the LLMs. Just reasoning. 

With apologies to [squad](https://github.com/bradygaster/squad). 

### To do
- Consider adding web search capabilities to litesquad. This would be a major change, so will need to think about how to handle it. Not how/if I want to do this, so sleep on it for a while (the worry is we would lose heterogeneity among workers, so I'd need to do it right). 

### What about Fusion?
After developing a beta version of litesquad, I found out about OpenRouter's [Fusion](https://openrouter.ai/blog/announcements/fusion-beats-frontier/). Fusion is very similar to litesquad, and is maintained by a professional team. I'm not sure if it is still being maintained, but my guess is there will be lots of tools like this coming online in the near future -- it's just too obvious a need!
