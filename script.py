from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pandas as pd

df = pd.read_csv("ICA curriculum module report.csv")

template = """
You are an expert at identifying UN sustainable development goals from textual data. Your job is to classify information I tell you into one of seventeen UN Sustainable Development Goals: [No poverty (SDG 1), Zero hunger (SDG 2), Good health and well-being (SDG 3), Quality education (SDG 4), Gender equality (SDG 5), Clean water and sanitation (SDG 6), Affordable and clean energy (SDG 7), Decent work and economic growth (SDG 8), Industry, innovation and infrastructure (SDG 9), Reduced inequalities (SDG 10), Sustainable cities and communities (SDG 11), Responsible consumption and production (SDG 12), Climate action (SDG 13), Life below water (SDG 14), Life on land (SDG 15), Peace, justice, and strong institutions (SDG 16), Partnerships for the goals (SDG 17).]

Some examples with the output answers include:

Example:  Sustainable Electrical Systems 1 Context, drivers and policy 2 Sustainable energy technologies - Wind power - Solar photovoltaic power - Hydro power - Heat networks - Offshore transmission 3 Network integration issues - Intermittency study: modelling of capacity credit, reserve and balancing - Technical Issues for Distributed Generation: modelling of voltage rise and power loss issues - Network Planning for Distributed Generation - Smart Grids
Output:
Affordable and Clean Energy,  Industry, Innovation and Infrastructure, Climate action

Example: Power System Economics Fundamentals of electricity markets: Demand curve and elasticity of the demand Supply curve and marginal cost Market equilibrium and social welfare Consumers� surplus and suppliers� economic profit Participants to electricity markets: Generation company (GENCO) Network Companies Retailer / supplier Large / small Consumers Economics models: Transmission economics: marginal transmission prices, congestion surplus, network revenue, network investment Economics and reliability, concept of customer worth of supply. Co-optimisation of energy and reserve in a centralised electricity market Strategic planning for generation companies
Output:
Affordable and Clean Energy,  Industry, Innovation and Infrastructure, Climate action

Now I want you to label the following example: {question}
Output?"""

prompt = PromptTemplate(template=template, input_variables=["question"])

local_path = (
    "./gpt4all-13b-snoozy-q4_0.gguf"  # replace with your desired local file path
)

answers = []

# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]

# Verbose is required to pass to the callback manager
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

# If you want to use a custom model add the backend parameter
# Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
llm = GPT4All(model=local_path, backend="gptj", callbacks=callbacks, verbose=True)

llm_chain = LLMChain(prompt=prompt, llm=llm)

for index, row in df.iterrows():
    print(row["Module Subject Code"])
    question = row["Module Content"]  
    print(llm_chain.run(question))

# Add the 'Answers' column to the DataFrame
# df['Answers'] = answers

# Now, your DataFrame 'df' will have a new column 'Answers' containing the answers.
# You can save this DataFrame back to a CSV file if needed.
# df.to_csv('output.csv', index=False)  # Save the DataFrame to a new CSV file.
