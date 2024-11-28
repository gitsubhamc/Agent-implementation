from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool, Tool
from langchain_openai import ChatOpenAI
from openai import OpenAI
from enum import Enum, IntEnum
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import json
import pydantic
from langgraph.checkpoint.memory import MemorySaver
# from helper.helper_functions import search_location, fb_pages, parse_ad_schedule


class prospectsparams(BaseModel):
    category: Optional[str] = Field(
        "",
        description="This field resembles the category of product the user wants to target user can pass this in argument"
    )
    city: Optional[str] = Field(
        "",
        description="This field resembles the city in which user wants to search for"
    )

# class prospectinsightparams(BaseModel):
import json
import ast
import pandas as pd
def fb_metrices(prospects):
    print(f"ENTERING THE LOGIC-------------")
    if isinstance(prospects, dict):
        prospects = pd.DataFrame([prospects])
    elif isinstance(prospects, list):
        prospects = pd.DataFrame(prospects)
    else:
        prospects = prospects
    print(f"THE COLUMNS IN DATA IS ----{prospects.columns}")
    data_dict = prospects['FB_latest_posts']
    data_json_string = data_dict.to_json()
    test_data = json.loads(data_json_string)
    avg_viewsCount = 0
    avg_likes = 0
    avg_shares = 0
    print(f"THE DATA IS--{test_data}---")
    data_list = ast.literal_eval(test_data['0'])
    # data_list=test_data
    for idx, post in enumerate(data_list[:10]):
        # print(f"----{post}----")
        try:
            if 'viewsCount' in post:
                avg_viewsCount += int(post['viewsCount'])
            if 'likes' in post:
                avg_likes += int(post['likes'])
            if 'shares' in post:
                avg_shares += int(post['shares'])
        except:
            pass
    num_posts = min(len(data_list), 10)
    avg_viewsCount = avg_viewsCount / num_posts if num_posts > 0 else 0
    avg_likes = avg_likes / num_posts if num_posts > 0 else 0
    avg_shares = avg_shares / num_posts if num_posts > 0 else 0
    print(f"---{avg_viewsCount}---{avg_likes}---{avg_shares}")
    return avg_viewsCount,avg_likes,avg_shares

def insta_metrics(prospects):
    print(f"ENtering the insta metrics-------")
    if isinstance(prospects, dict):
        prospects = pd.DataFrame([prospects])
    elif isinstance(prospects, list):
        prospects = pd.DataFrame(prospects)
    else:
        prospects = prospects
    insta_dict = prospects['Insta_Latest_Posts']  # Retrieve data
    insta_dict_string = insta_dict.to_json()     # Convert to JSON string
    insta_data = json.loads(insta_dict_string)   # Parse JSON into a dictionary

    avg_videoViewCount = 0
    avg_videoPlayCount = 0
    avg_likesCount = 0
    avg_commentsCount = 0

    data_list = ast.literal_eval(insta_data['0'])

    for idx, post in enumerate(data_list[:10]):  # Limit iterations to 10
        print(f"---{post}---")
        try:
            if 'videoViewCount' in post:
                avg_videoViewCount += int(post['videoViewCount'])
            if 'videoPlayCount' in post:
                avg_videoPlayCount += int(post['videoPlayCount'])
            if 'likesCount' in post:
                avg_likesCount += int(post['likesCount'])
            if 'commentsCount' in post:
                avg_commentsCount += int(post['commentsCount'])
        except:
            pass

    num_posts = min(len(data_list), 10)  # Use the smaller value between 10 and the list length
    avg_videoViewCount = avg_videoViewCount / num_posts if num_posts > 0 else 0
    avg_videoPlayCount = avg_videoPlayCount / num_posts if num_posts > 0 else 0
    avg_likesCount = avg_likesCount / num_posts if num_posts > 0 else 0
    avg_commentsCount = avg_commentsCount / num_posts if num_posts > 0 else 0

    print(f"--- Avg Video Views: {avg_videoViewCount}, Avg Video Plays: {avg_videoPlayCount}, "
        f"Avg Likes: {avg_likesCount}, Avg Comments: {avg_commentsCount} ---")
    return avg_videoViewCount,avg_videoPlayCount,avg_likesCount,avg_commentsCount

def get_business_metrics(column_name,prospects):
    # smb_data=prospects['All_Signals/SMB_Data_Points']
    if isinstance(prospects, dict):
        prospects = pd.DataFrame([prospects])
    elif isinstance(prospects, list):
        prospects = pd.DataFrame(prospects)
    else:
        prospects = prospects
    smb_data=prospects[column_name]
    smb_data_json=smb_data.to_json()
    smb_data=json.loads(smb_data_json)
    smb_data = ast.literal_eval(smb_data['0'])
    return_data={}
    # print(smb_data[0])
    Social_Media_Presence=smb_data[0].get('Social Media Presence','')
    if Social_Media_Presence!='':
        social_media_data={'facebook_business_page- presence':Social_Media_Presence['facebook_business_page- presence'],
                        'facebook_business_page- fb_followers':Social_Media_Presence['facebook_business_page- fb_followers'],
                        'twitter_business_profile- tweet_count':Social_Media_Presence['twitter_business_profile- tweet_count'],
                        'youtube_business_channel- yt_videos_counts':Social_Media_Presence['youtube_business_channel- yt_videos_counts'],
                        'youtube_business_channel- yt_views_counts':Social_Media_Presence['youtube_business_channel- yt_views_counts'],
                        'fb_fans- Facebook Likes':Social_Media_Presence['fb_fans- Facebook Likes']
                        }
        return_data['social_media_data']=social_media_data
    Most_listings_google=smb_data[0].get('Local Business Presence','')
    if Most_listings_google!='':
        most_listing_data={
            'google_places-local_directory_googleplaces_reviews':Most_listings_google['google_places-local_directory_googleplaces_reviews'],
        }
        return_data['most_listing_data']=most_listing_data
    website_features=smb_data[0].get('Infrastructure Robustness','')
    if website_features!='':
        website_features_unique={
            'PageSpeed Score (Desktop)':website_features['PageSpeed Score (Desktop)'],
            'Server Response Time':website_features['Server Response Time']
        }
        return_data['website_features_unique']=website_features_unique
    FIRMOGRAPHICS=smb_data[0].get('FIRMOGRAPHICS','')
    if FIRMOGRAPHICS!='':
        return_data['FIRMOGRAPHICS']=FIRMOGRAPHICS
    BuzzScore=smb_data[0]['BUSINESS INFO'].get('BuzzScore','')
    if BuzzScore!='':
        # return_data.append(BuzzScore)
        return_data['BuzzScore']=BuzzScore
    business_name=smb_data[0]['BUSINESS INFO'].get('Business Name','')
    if business_name!='':
        return_data["business_name"]=business_name
    
    return return_data

def get_column_values(prospects, column_names):
    """
    Retrieve the values for specified columns from a dataframe if they exist.
    
    Args:
        prospects (pd.DataFrame): The dataframe containing prospect data.
        column_names (list): List of column names to retrieve values for.
    
    Returns:
        dict: A dictionary with column names as keys and their respective values as lists or a message if not found.
    """
    if isinstance(prospects, dict):
        prospects = pd.DataFrame([prospects])
    elif isinstance(prospects, list):
        prospects = pd.DataFrame(prospects)
    else:
        prospects = prospects
    column_values = {}
    for col in column_names:
        if col in prospects.columns:
            column_values[col] = prospects[col].tolist()  # Convert column to a list of values
        else:
            column_values[col] = f"Column '{col}' not found in the dataframe."
    
    return column_values

class DataStore:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.prospect_data = None
        return cls._instance
    
    @classmethod
    def set_data(cls, data):
        if cls._instance is None:
            cls()
        cls._instance.prospect_data = data
    
    @classmethod
    def get_data(cls):
        if cls._instance is None:
            cls()
        return cls._instance.prospect_data
