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

Query is distributed to worker LLMs (e.g., GPT, Gemini). Another LLM acts as a critic that gives feedback to the workers. They revise their response. An intermediate representation of these responses is extracted to pull the content, and this is then converted into categories/clusters of suggestions. A final frontier judge model converts these suggestions into a final coherent answer for the user. 

### So. Many. APIs.
To run properly, litesquad currently needs an [OpenRouter](https://openrouter.ai/keys) api key. One OpenRouter key reaches the whole openrouter.ai catalog (deepseek, mistral, llama, grok, qwen, ...), which is how the worker roster grows wide without a key per provider. You can store your API key in `.env`.


### Basic tests
To see if openrouter api key is set up: `litesquad --check` 

Offline test: `litesquad --smoke --mock`

### Caveats
This provides an interface to a swarm of LLMs to try to generate a better answer than when using a single LLM. There is no tool-usage such as web calls from the LLMs. Just reasoning. 

With apologies to [squad](https://github.com/bradygaster/squad). 

### What about Fusion?
After developing a beta version of litesquad, I found out about OpenRouter's [Fusion](https://openrouter.ai/blog/announcements/fusion-beats-frontier/).

Fusion has the same goal as litesquad, and a fairly similar architecture (though it is more sophisticated, and built by a professional team). Unfortunately, Fusion lately has not been working for me: I'm not sure if openrouter is still maintaining it. 

When I did testing of litesquad vs Fusion, they both produced good results. Often Fusion is better. On some questions, I prefer the output of litesquad (for instance for more open-ended personal topics when there is clearly no single best answer). 

My guess is there will be a squadron of these applications within a couple of years. It's just too obvious a need, and an intrinsically interesting problem. 

### To do
- Just do openrouter instead of multiple api tokens that are a pain to maintain. 
- Make it easier to select models. 
- Add web search agent capabilities to litesquad:  make this a toggle. 

