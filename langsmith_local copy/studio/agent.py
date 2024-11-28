from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool, Tool
from langchain_openai import ChatOpenAI
from openai import OpenAI
import requests
import os, getpass
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from datetime import datetime
# Web Search
import boto3
import os 
from langgraph.checkpoint.memory import MemorySaver
# from helper.helper_functions import get_product_info
from classes.productClass import prospectsparams,fb_metrices,DataStore,insta_metrics,get_business_metrics,get_column_values
import pandas as pd


from typing import Any
# Math
# Create global instance

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
LANGCHAIN_TRACING_API_KEY=os.getenv('LANGCHAIN_TRACING_API_KEY')
# _set_env("LANGCHAIN_TRACING_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "HAWK"

def load_data():
    # Assuming headers are in row 2 (index 1)
    return pd.read_excel('Sample Data for the Model.xlsx', header=1)

def find_prospects(category, location):
    """
    Finds prospects based on the given category and location.
    Helps sales reps discover high-potential leads tailored to their preferences

    Args:
        category (str): The primary category to filter prospects.
        location (str): The city to filter prospects.

    Returns:
        pd.DataFrame: A filtered DataFrame containing prospects matching the criteria.

    Raises:
        ValueError: If no prospects are found for the given category and location.
    """
    data = load_data()
    # print(data.columns)
    
    # Clean column names
    data.columns = data.columns.str.strip().str.replace('\n', '', regex=True).str.replace(' ', '_')
    
    # Update column names in filters
    filtered_prospects = data[(data['Category_-_Primary'] == category) & 
                              (data['City'] == location)]
    if filtered_prospects.empty:
        raise ValueError(f"No prospects found for category '{category}' in location '{location}'.")
    return filtered_prospects
@tool
def prospects_gen_tool(params:prospectsparams):
    """
    asks all the params in the patameters of prospectsparams for asking the questions
    Args:
        city: the specific city which the user wants to target for
        category: what specific category the user wants to target for
    Returns:
        returns the params
    """

    try:
        global final_data
        print(f"----{params.city}-----{params.category}")
        final_data=find_prospects(params.category,params.city)
        print(f"-----The data received--{len(final_data)}---{final_data}--")
        if len(final_data.columns)>2:
            print(f"----{final_data.columns}----")
        else:
            raise ValueError(f"No prospects found for category '{params.city}' in location '{params.category}'.")
        final_data=final_data.to_dict('records')
        DataStore.set_data(final_data)
        return {"city":params.city,"category":params.category}

    except Exception as e:
        print(f"The issue is ---{e}---")
        raise ValueError(f"No prospects found for category '{params.city}' in location '{params.category}'.")

@tool
def prospectinsight_gen_tool(final_data: Any):
    """
    Delivers deep analysis of selected prospects to aid decision-making. your main job is to do analysis
    Args:
        meta points:  never ask for this fields input from user meta metrices to take decision on 
        insta points: never ask for this fields input from user insta points to take decision on
        business_overall_performance_metrics: original performance metrics of the business never ask the fields input from user
        competitor_1_data: competitor1 data analyse it to make the user more strong never ask the fields input from user
        competitor_2_data: competitor2 data analyse it to make the user more strong never ask the fields input from user

    Returns:
        returns the params
    """
    try:
        print(f'ENTERING THE TOOL')
        stored_data = DataStore.get_data()
        if stored_data is None:
            raise ValueError("No prospect data available")
        metaavg_viewsCount,metaavg_likes,metaavg_shares=fb_metrices(stored_data)
        avg_videoViewCount,avg_videoPlayCount,avg_likesCount,avg_commentsCount=insta_metrics(stored_data)
        # All_Signals.2   All_Signals.1   All_Signals   All_Signals/SMB_Data_Points
        business_overall_performance_metrics=get_business_metrics('All_Signals/SMB_Data_Points',stored_data)
        competitor_1_data=get_business_metrics('All_Signals',stored_data)
        competitor_2_data=get_business_metrics('All_Signals.1',stored_data)
        print(f"------THE MAIN DATA----{metaavg_viewsCount}--{metaavg_likes}----{metaavg_shares}-")
        return_data={"meta":{
                "metaavg_viewsCount":metaavg_viewsCount,
                "metaavg_likes":metaavg_likes,
                "metaavg_shares":metaavg_shares
                },
                "instagram":{"avg_videoViewCount":avg_videoViewCount,"avg_videoPlayCount":avg_videoPlayCount,"avg_likesCount":avg_likesCount,"avg_commentsCount":avg_commentsCount},
                "business_overall_performance_metrics":business_overall_performance_metrics,
                "competitor_1_data":competitor_1_data,
                "competitor_2_data":competitor_2_data
                }
        return return_data
        # return {
        #         "metaavg_viewsCount":metaavg_viewsCount,
        #         "metaavg_likes":metaavg_likes,
        #         "metaavg_shares":metaavg_shares
        #         }
    except Exception as e:
        print(f"The issue is ---{e}---")

@tool
def communication_agent(final_data: Any):
    """
    Extracts prospect data and drafts a personalized email based on their needs.
    This agent retrieves stored prospect data, extracts specified columns, 
    and uses the retrieved information to draft a persuasive and personalized 
    email. The email focuses on addressing the prospect's needs and showcasing 
    the product or service as a tailored solution.
    remember the users name and personal info should be present in the email copy drafted

    Args:
        final_data (Any): Placeholder for any additional input data required by the agent 
                          (not currently used in this implementation).
                          never ask for any fields input all are provided to you

    Returns:
        dict: A dictionary containing:
            - "email_draft": A string containing the personalized email content.
    """
    try:
        print(f'ENTERING THE TOOL')
        stored_data = DataStore.get_data()
        if stored_data is None:
            raise ValueError("No prospect data available")
        column_names = ['Customer', 'Products', 'User_Name', 'UID', 'Business_Name', 'Address', 'State', 'City', 'Zip', 'Phone', 'Email']
        result = get_column_values(stored_data, column_names)
        email_draft = '''Dear {customer_name},\n\n"
            We at [Your Company Name] have been analyzing businesses in {city} and have identified "
            opportunities where {business_name} could.....'''
        return result,email_draft
    except Exception as e:
        print(f"-------{e}----")


tools=[prospects_gen_tool,prospectinsight_gen_tool,communication_agent]
# prompt = client.pull_prompt("mainagentprompt",include_model=True)
# input_variables_main_agent,template_textmain_agent,model_name_main_agent,temperature_main_agent=fetch_prompt(prompt)
# Instantiate LLM
llm = ChatOpenAI(model='gpt-4o', temperature=0.7, streaming=True)
llm_with_tools = llm.bind_tools(tools)
# template_textmain_agent=template_textmain_agent.format(time_now=datetime.now())
# print("complete prompt of agent: ", template_textmain_agent)
messages =[( "system", """
                        -Begin by asking for the required fields to generate prospects using the prospects_gen_tool
                        - Ensure all necessary details are collected to produce accurate and meaningful results.
                        - After completing the prospects_gen_tool, proceed to the prospectinsight_gen_tool
                        - Leverage your expertise in metrics, including Meta and Instagram performance data, to analyze the provided data
                        - Identify key issues and suggest actionable improvements based on insights
                        - Use the provided business_overall_performance_metrics and competitor data to perform a detailed SWOT analysis
                        - Highlight strengths, weaknesses, opportunities, and threats clearly, and point out areas for improvement where the user is falling behind
                        - Utilize the communication_agent to craft a compelling and engaging email tailored to the users needs
                        - The email should
                        - Incorporate personalized details about the user and their business
                        - Persuasively outline how your solutions address their specific pain points and improve performance
                    **General Guidelines
                     - Do ask for user confirmation then only proceed in each step 
                     - Always interact with precision and professionalism
                     - Ensure the user's name and personal information are seamlessly integrated into the email copy to establish trust and a personal connection
                     - Always invoke tools when generating content or performing analyses to ensure accuracy and alignment with user needs"""),
    ("human", f"The today's date is {datetime.now}")
]

memory = MemorySaver()



def human_feedback(state: MessagesState):
    pass

# Assistant node
def assistant(state: MessagesState):
   print("state in assistant: ", state)
   return {"messages": [llm_with_tools.invoke(messages+ state["messages"])]}
   


# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human_feedback", human_feedback)

# Define edges: these determine the control flow
builder.add_edge(START, "human_feedback")
builder.add_edge("human_feedback", "assistant")
builder.add_conditional_edges(
    "assistant",

    tools_condition,
)
builder.add_edge("tools", "assistant")
graph = builder.compile(checkpointer=memory)



