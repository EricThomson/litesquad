# lite squad
Tool for working with a team of LLMs. Because sometimes two, or five, heads are better than one.

With apologies to [squad](https://github.com/bradygaster/squad). 

## Usage
Install with pip:

    pip install litesquad

Open application in browser (recommended): 

    litesquad --web 

To use at command line:

    litesquad "plan such and such project"
    litesquad "plan such and such" --quick # get a quick answer, bypassing the squad

litesquad takes a few minutes to run when running in default (deep) mode. There's a lot of LLM calls going on under the hood.

### How it works
Query is distributed to worker LLMs (e.g., GPT, Gemini). Another LLM acts as a critic that gives feedback to the workers. They revise their responses. An intermediate representation of these responses is extracted, and this is converted into suggestion clusters. 

A final frontier judge model converts these suggestions into a final coherent answer for the user. In the web interface, this can be downloaded as a standalone file for later perusal. 

### API key management
To run properly, litesquad currently needs an [OpenRouter](https://openrouter.ai/keys) api key. One OpenRouter key reaches the whole openrouter.ai catalog (deepseek, mistral, llama, etc). Store your API key in `.env`.

### Basic tests
To see if openrouter api key is set up and working against default models:

    litesquad --check 

Offline test: `litesquad --smoke --mock`

### Caveats
This provides an interface to a swarm of LLMs to try to generate a better answer than when using a single LLM. There is no tool-usage such as web calls from the LLMs. Just reasoning. 

> I considered adding web search to some of the workers, but the goal is to maximize heterogeneity among workers: web search would dampen this, and add complexity to litesquad. It would be easy to add sometime in the future.

### What about Fusion?
After developing a beta version of litesquad, I found out about OpenRouter's [Fusion](https://openrouter.ai/blog/announcements/fusion-beats-frontier/). Fusion is very similar to litesquad, and is maintained by a professional team. Fusion may not currently be maintained, but my guess is there will be lots of tools like this coming online in the near future!
