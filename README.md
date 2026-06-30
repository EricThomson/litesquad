# lite squad
Lightweight cli tool for multiple LLM collaboration.

One acts as a project manager who assigns a task to other LLMs. Another acts as a critic. The PM then summarizes, integrates the information and feeds it back to you. 

## Usage
Install, and then use, at your command line:

    pip install litesquad
    litesquad "plan such and such project"

To run the default with workers from different vendors, you will need API keys for [Gemini](https://aistudio.google.com/app/apikey), [OpenAI](https://openai.com/index/openai-api/), and [Anthropic](https://platform.claude.com/docs/en/api/admin/api_keys/retrieve). Store your api keys in .env (or whatever). 


Default: 
Opus is the PM, feeds info to two workers that work independently (defaults to sonnet, gemini), and then critic model analyzes both proposals (gpt 5). Finally, opus synthesizes all of the above into a final answer. 


## Testing 
To see if api keys are working (from activated env):

    python -m litesquad.check_keys

    litesquad --check 