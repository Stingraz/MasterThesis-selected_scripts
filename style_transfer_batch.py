# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:23:13 2024

@author: micha

This script has 3 steps:
    1) It first creates a dataframe containing all unstylized articles to be 
       style transferred, their class assignments, and a unique id for each.
    2) It then formats each row of the dataframe into an API request, creates 
       a .jsonl batch file out of all 300 requests, and uploads the batch file 
       to OpenAI.
    3) This final step costs money, so is protected by a sys.exit() command 
       prior to it. The batch request is actually sent here. When the file is 
       processed, the file id is returned for a binary file, which then is 
       converted into .json format and the body of text for each stylized 
       article is extracted, saved to the dataframe, and exported as a csv.
    
"""

import pandas as pd
import numpy as np
import sys
from openai import OpenAI
import os
import json

# Adjust display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Specify seed value for reproducibility
seed_value = 42
np.random.seed(seed_value)

# Specify the API key to access OpenAI account
client = OpenAI(api_key="api_key_here")

# Set temperature for API request (lower is more coherent; higher is more creative) 
temp = 1.0  # Don't set higher than 1.0! It starts outputting symbols and other weird stuff
max_tokens = 4000

# Specify the type of stylization to apply:
# "shakespeare", "political", "Gucci_ad", "cheerful", "stupid_and_rude", "preschooler"
style = "political"

# ------------------- Load in data and prepare dataframes ------------------- #    

# Load in the unstylized articles
file_path_unstylized = "datasets/single-category/reuters_recategorized-single-category-300_subset2.csv"

with open(file_path_unstylized, 'r') as file:
    df_unstylized_articles = pd.read_csv(file, index_col="Unnamed: 0")
    
# Copy unstylized DataFrame, drop the unneeded column, and rename the "body_of_article" column
df_unstylized_and_stylized = df_unstylized_articles.copy()
df_unstylized_and_stylized = df_unstylized_and_stylized.drop(df_unstylized_and_stylized.columns[[0,1,3,4,5,6]], axis=1)
df_unstylized_and_stylized = df_unstylized_and_stylized.rename(columns={"body_of_article": "unstylized"})

# Save the true category label columns 
df_true_labels = df_unstylized_articles.copy()
df_true_labels = df_true_labels.drop(['unique_id', 'topics', 'body_of_article'], axis=1)

# Create an empty column to add the stylized articles to
df_unstylized_and_stylized[style] = np.nan

# Add true label columns back on, so the file format matches the unstylized one
df_unstylized_and_stylized = pd.concat([df_unstylized_and_stylized, df_true_labels], axis=1)


# ------------------------- Create the batch file --------------------------- #

# Define the data to be written to the .jsonl file
data = []

for index in range(0, len(df_unstylized_and_stylized)):
#for index in range(0, 2):
    article_text = df_unstylized_and_stylized.iloc[index]["unstylized"]
    request = {"custom_id": f"request-{index}",
             "method": "POST",
             "url": "/v1/chat/completions",
             "body": {"model": "gpt-3.5-turbo-0125",
                      "messages": [{"role": "system", "content": "You are a writing assistant, skilled in applying style transfer onto provided text inputs."},
                                   {"role": "user", "content": f"Rewrite the following text in the style of a political speech: {article_text}."}],
                                    "max_tokens": max_tokens}}

    data.append(request)

# Specify the output file name
batch_filename = os.path.join("datasets/stylized_articles/subset2", "batch_requests.jsonl")

# Write the data to a .jsonl file
with open(batch_filename, 'w') as file:
    for entry in data:
        json.dump(entry, file)
        file.write('\n')

print(f"Data successfully written to {batch_filename}")


# Upload batch file to OpenAI:
batch_input_file = client.files.create(
  file=open(batch_filename, "rb"),
  purpose="batch"
)

batch_input_file_id = batch_input_file.id  


sys.exit()


####################### Send API Request to OpenAI API ########################
#                   (this costs money every time it's run!)

client.batches.create(input_file_id=batch_input_file_id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                        metadata={"description": "subset 2 style transfer"})

# Check batch status at any time:
client.batches.retrieve("batch-name-here")

# Retrieve the file once it's done processing:  
# replace "file-name-here" with the value they give for output_file_id in batch status check above
content = client.files.content("file-name-here")  

# Convert response content to bytes
results = content.read()
    
# Save the binary object to a json file:
results_filename = os.path.join("datasets/stylized_articles/subset2", "results.jsonl")

with open(results_filename, 'wb') as file:
    file.write(results)    
    
    
# Read and process the .jsonl file
results_list = []
with open(results_filename, 'r') as file:
    for line in file:
        results_list.append(json.loads(line))

""" From OpenAI: Note that the output line order may not match the input line order. 
Instead of relying on order to process your results, use the custom_id field which
will be present in each line of your output file and allow you to map requests in 
your input to results in your output."""

for result in results_list:
    # Get the body of the style transferred article
    stylized_article = result["response"]["body"]["choices"][0]["message"]["content"]
    
    # Get the custom_id and extract just the number (this is the row number from the original df)
    custom_id = result["custom_id"]
    # Extract the value following "request-"
    request_number = int(custom_id.split("request-")[1])
    
    # Save the article text to the dataframe in the correct row
    df_unstylized_and_stylized.loc[request_number, style] = stylized_article

# Save the completed file    
df_unstylized_and_stylized.to_csv(f"datasets/stylized_articles/subset2/{style}_articles.csv")
    
 
    
    
    
    