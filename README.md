# lite squad
Lightweight cli tool for collaborating with a little squad of LLMs. 

One LLM acts as a project manager who assigns a task to some worker LLMs. Another acts as a critic that gives feedback to the workers. They revise their resspond. The PM then generates an integrated reply for the user. 

## Usage
Install, and then use, at your command line:

    pip install litesquad
    litesquad "plan such and such project"

To run the default with workers from different vendors, you will need API keys for [Gemini](https://aistudio.google.com/app/apikey), [OpenAI](https://openai.com/index/openai-api/), and [Anthropic](https://platform.claude.com/docs/en/api/admin/api_keys/retrieve). Store your api keys in .env (or whatever). 


### Default settings 
Opus is the PM, feeds info to three workers (from gemini, claude, and openai) that work independently, and then critic model (gpt5) analyzes their answers, gives them a chance to update.

Their updated answers are given to the PM to synthesize the results into a single coherent answer.


## Testing 
To see if api keys are working (from activated env):

    python -m litesquad.check_keys

Check on specific models:
    litesquad --check 


### To do
Allow "pm only" option so you don't have to invoke entire 11-call machinery for every query. 