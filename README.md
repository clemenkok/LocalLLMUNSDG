# Instructions for Replication

## Dependencies

- You will need Python installed on your local machine.  
- Activate the `venv` by running the following command in your terminal:

```
python -m venv .venv
\.venv\Scripts\activate.bat
```

- You will also need to download the weights for the model we are using - GPT4ALL 13B Snoozy. Download it [here](https://gpt4all.io/index.html). Keep it within the same directory as our python script.  

## Having a Go

Once done, install the dependencies:  

```
\.venv\Scripts\python -m pip install langchain gpt4all pandas
```

Subsequently, run `python script.py`.  

## Current issues

- Each inference with the current prompt and question (i.e. module content from each row) takes about 5 minutes on my computer. It might be worth considering moving to a more beefy (but secure) platform, or consider a less computationally expensive approach such as bag of words.  

