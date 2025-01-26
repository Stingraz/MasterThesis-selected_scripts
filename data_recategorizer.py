# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:13:56 2024

@author: micha

This script reads in the csv file created by data_preprocessing.py which contains
6596 Reuters articles and original Reuters Topics categories. It visualizes the 
distribution of these original Topics categories for each article.

It then does clustering through PCA and t-SNE and plots visualizations (these
                                            results are not present in paper) 

It recategorizes the articles according to the 4 classes and depending 
on the case_type chosen, either filters out single-category articles, 
multi-category articles, or doesn't filter out either type (for the paper, 
multi-category articles are removed). The resulting class distribution is plotted.

Finally, the dataframe is shuffled and 300 articles are sampled to get a balanced
subset. Resulting dataset is exported as a CSV. Visualizes results also. 

Bottom section (lines 640-686) is to extract out-of-sample subset without needing 
to rerun entire script again.

"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import re
import gensim.downloader as api
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Specify whether or not to save plots after generating (this might overwrite any you alread have!)
save_plots = True

# Specify the type of dataset you want to generate:
    # to filter out single-category articles, select "multi-category", 
    # to filter out multi-category article, select "single-category", 
    # and to just recategorize but not filter out either type, select "mixed-category"
case_type = "single-category"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fixes an issue with how matplotlib handles threading with OpenMP 

# Set the random seed for reproducibility (used in later functions as well)
seed_value = 42
np.random.seed(seed_value)

# Adjust display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Change the current working directory
os.chdir("datasets")

# Specify the filename for reading:
filename = 'reuters_cleaned.csv'

# Specify the number of categories to create:
num_categories = 4   
    
# Read the CSV file into a Pandas DataFrame:
df = pd.read_csv(filename, index_col='Unnamed: 0')  # index_col identifies the existing index column in the csv file


# --------- Check for NaN values in the DataFrame and remove them ----------- #

df.dropna(inplace=True)
nan_values = df.isna().any()

# Print columns with NaN values, if any
if nan_values.any():
    print("Columns with NaN values:")
    print(nan_values[nan_values].index)
else:
    print(f"No NaN values found in the {len(df)} article DataFrame.")
    
    
# ------- Check for or create new folder to store generated datasets ------- #

case_type_folder = case_type
if not os.path.exists(case_type_folder):
    os.makedirs(case_type_folder)


# ---------------------- Determine new category groups -----------------------#

# Split the topics by commas
topics = df['topics'].str.split(', ')

# Flatten the list-like objects within the topics Series
flattened_topics = topics.explode()
set_of_topics = set(flattened_topics)
print(set_of_topics)
num_topics = flattened_topics.nunique()
print(f'\nNumber of topics: {num_topics}')
# Gives 118; too high for visualization or for zero-shot classification


# Manually create groupings for the topic categories (details in Reuters cat_descriptions.txt):
# (These will be the actual categories I give to the zero-shot classifier later on)

agriculture_topics = ['cocoa', 'tea', 'groundnut', 'sorghum', 'oilseed', 'coconut', 'corn-oil',
               'rubber', 'wheat', 'rapeseed', 'sunseed', 'soy-oil', 'lin-oil', 'linseed', 
               'coffee', 'cotton', 'groundnut-oil', 'citruspulp', 'coconut-oil', 'plywood',
               'soybean', 'tapioca', 'palmkernel', 'rice', 'castor-oil', 'cotton-oil',
               'sun-oil', 'sugar', 'palm-oil', 'potato', 'red-bean', 'castorseed', 
               'lumber', 'rye', 'grain', 'sun-meal', 'orange', 'cottonseed', 'corn',
               'veg-oil', 'rape-meal', 'rape-oil', 'barley', 'oat', 'copra-cake', 'fishmeal', 
               'lin-meal', 'meal-feed', 'cornglutenfeed', 'soy-meal', 'livestock', 
               'hog', 'f-cattle', 'carcass', 'wool', 'l-cattle', 'pork-belly']

metals_topics = ['iron-steel', 'copper', 'nickel', 'gold', 'alum', 'strategic-metal', 
          'platinum', 'palladium', 'zinc', 'tin', 'lead', 'silver']
           
energy_topics = ['gas', 'heat', 'nat-gas', 'fuel', 'propane', 'crude', 'pet-chem', 
          'naphtha', 'jet'] 

if num_categories == 4:
    economy_topics = ['jobs', 'income', 'retail', 'inventories', 'housing', 'interest', 
               'money-fx', 'money-supply', 'reserves', 'trade', 'yen', 'nzdlr',   
               'dlr', 'instal-debt', 'austdlr', 'ship', 'bop', 'cpi', 'wpi', 'ipi', 
               'cpu', 'gnp', 'lei', 'hk', 'can', 'stg', 'dmk', 'sfr', 'ffr', 'bfr',
               'dfl', 'lit', 'dkr', 'nkr', 'skr', 'saudriyal', 'rand', 'rupiah', 
               'ringgit', 'peseta', 'acq']
    
    # Find topics I forgot to add to these new topic groups
    missing_topics = set_of_topics.difference(set(agriculture_topics + metals_topics + 
                                                  economy_topics + energy_topics))
    
elif num_categories == 5:
    economy_topics = ['jobs', 'income', 'retail', 'inventories', 'housing', 'interest', 
               'money-fx', 'money-supply', 'reserves', 'trade', 'yen', 'nzdlr',   
               'dlr', 'instal-debt', 'austdlr', 'ship', 'bop', 'cpi', 'wpi', 'ipi', 
               'cpu', 'gnp', 'lei', 'hk', 'can', 'stg', 'dmk', 'sfr', 'ffr', 'bfr',
               'dfl', 'lit', 'dkr', 'nkr', 'skr', 'saudriyal', 'rand', 'rupiah', 
               'ringgit', 'peseta']
    mergers_and_acquisitions_topics = ['acq']
    # Find topics I forgot to add to these new topic groups
    missing_topics = set_of_topics.difference(set(agriculture_topics + metals_topics + 
                                                  economy_topics + energy_topics + mergers_and_acquisitions_topics)) 

#print("Missing topics:", missing_topics)
# It says I didn't miss any.
  

####################### Clustering and Visualizations #########################

# -------------- Replace abbreviated topics with full words ----------------- #
    
# Define a dictionary to store topic abbreviations and their corresponding full words
abbreviations_dict = {'alum': 'aluminum',                   'nzdlr': 'New Zealand dollar',
                      'nzdlrs': 'New Zealand dollars',      'austdlr': 'Australian dollar',
                      'austdlrs': 'Australian dollars',     'dlrs': 'dollars',
                      'dlr': 'dollar',                      'lit': 'Italian Lira',
                      'can': 'Canadian dollar',
                      'hk': 'Hong Kong dollar',             'bpd': 'barrels per day',
                      'hks': 'Hong Kong dollars',           'bop': 'balance of payments',         
                      'wpi': 'wholesale price index',       'ipi': 'industrial production index',
                      'cpu': 'capacity utilisation',        'gnp': 'gross national product',
                      'lei': 'leading economic indicators', 'instal-debt': 'instalment debt',
                      'cpi': 'consumer price index',        'dmks': 'Deutsche-Marks',
                      'stg': 'Sterling',                    'dmk': 'Deutsche-Mark',
                      'sfr': 'Swiss Franc',                 'sfrs': 'Swiss Francs',
                      'ffr': 'French Franc',                'ffrs': 'French Francs',
                      'bfr': 'Belgian Franc',               'bfrs': 'Belgian Francs',
                      'dfl': 'Netherlands Florin',          'dfls': 'Netherlands Florins',
                      'lits': 'Italian Liras',              'acq': 'acquisition',
                      'dkr': 'Danish Krone',                'dkrs': 'Danish Krones',
                      'nkr': 'Norwegian Krone',             'nkrs': 'Norwegian Krones',
                      'skr': 'Swedish Krona',               'skrs': 'Swedish Kronas',
                      'saudriyal': 'Saudi Arabian Riyal',   'saudriyals': 'Saudi Arabian Riyals',
                      'rand': 'South African Rand',         'rands': 'South African Rands',
                      'nat-gas': 'natural-gas',             'pet-chem': 'petro-chemicals',
                      'cornglutenfeed': 'corn gluten feed', 'f-cattle': 'cattle',
                      'veg-oil': 'vegetable oil',           'ship': 'shipping',
                      'citruspulp': 'citrus pulp',          'palmkernel': 'palm kernel',
                      'castorseed': 'castor seed',          'l-cattle': 'cattle',
                      'jet': 'jet fuel',                    'cottonseed': 'cotton seed',
                      'rape-oil': 'rapeseed oil',           'money-fx': 'money foreign exchange',
                      'rape-meal': 'rapeseed meal',         'sun-meal': 'sunflower meal',
                      'lin-oil': 'linseed oil',             'pork-belly': 'pork'}



# Convert set_of_topics to a list to enable item assignment
list_of_topics = list(set_of_topics)


# Function to preprocess topics
def preprocess_topic(topic):
    # Replace hyphens with spaces
    topic = topic.replace('-', ' ')
    # Tokenize the topic into words
    words = word_tokenize(topic)
    # Remove non-alphanumeric characters from each word
    words = [re.sub(r'\W+', '', word) for word in words]
    # Convert words to lowercase
    words = [word.lower() for word in words]
    return words

# Iterate over each topic in list_of_topics, preprocess it, and replace abbreviations with their respective values
for i, topic in enumerate(list_of_topics):
    if topic in abbreviations_dict:
        list_of_topics[i] = abbreviations_dict[topic]
    # Preprocess the topic
    list_of_topics[i] = preprocess_topic(list_of_topics[i])

print(list_of_topics)


################## Find word embeddings for Reuters topics ####################

# Load pre-trained word embeddings 
word_vectors = api.load("word2vec-google-news-300")

# Apply get_topic_embedding to each topic to get topic embeddings
topic_embeddings = []
for topic in list_of_topics:
    # Initialize an empty list to store embeddings for each word in the topic
    word_embeddings = []
    for word in topic:
        if word in word_vectors:
            word_embeddings.append(word_vectors[word])
    # Calculate the average embedding for the topic
    if word_embeddings:
        topic_embeddings.append(sum(word_embeddings) / len(word_embeddings))
    else:
        topic_embeddings.append(None)

# Filter out topics with None embeddings
topic_embeddings = [embedding for embedding in topic_embeddings if embedding is not None]

# Calculate semantic similarity matrix between topic embeddings
semantic_similarity_matrix = cosine_similarity(topic_embeddings)


############### Perform dimensionality reduction using PCA ####################

pca = PCA(n_components=2)
reduced_embeddings_pca = pca.fit_transform(semantic_similarity_matrix)

# --------------------- Do k-means to color the clusters -------------------- #

# Perform clustering on the topic embeddings
kmeans = KMeans(n_clusters=4)  
cluster_labels = kmeans.fit_predict(reduced_embeddings_pca)

# Plot the embeddings in a scatterplot with different colors for each cluster
plt.figure(figsize=(10, 6))
for i in range(len(np.unique(cluster_labels))):
    plt.scatter(reduced_embeddings_pca[cluster_labels == i, 0], reduced_embeddings_pca[cluster_labels == i, 1], label=f'Cluster {i}')

plt.title('Semantic Associations of Reuters Topics: PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
if save_plots == True:
    plt.savefig("clustering-PCA.png")
plt.show()


########## Alternatively, use t-SNE for dimensionality reduction ##############

# Specify the random seed for t-SNE (it won't use the global seed I specified earlier)
random_seed = 42

tsne = TSNE(n_components=2, random_state=random_seed)
reduced_embeddings_tsne = tsne.fit_transform(semantic_similarity_matrix)

# --------------------- Do k-means to color the clusters -------------------- #

# Perform clustering on the topic embeddings
kmeans = KMeans(n_clusters=4, random_state=random_seed)  
cluster_labels = kmeans.fit_predict(reduced_embeddings_tsne)

# Plot the embeddings in a scatterplot with different colors for each cluster
plt.figure(figsize=(10, 6))
for i in range(len(np.unique(cluster_labels))):
    plt.scatter(reduced_embeddings_tsne[cluster_labels == i, 0], reduced_embeddings_tsne[cluster_labels == i, 1], label=f'Cluster {i}')
plt.title('Semantic Associations of Reuters Topics: t-SNE')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
if save_plots == True:
    plt.savefig("clustering-tSNE.png")
plt.show()


#### Identify which of the original topics were assigned to which cluster #####

# Initialize a dictionary to store topics assigned to each cluster
cluster_topics = {i: [] for i in range(len(np.unique(cluster_labels)))}

# Iterate through cluster labels and original topics, and populate the dictionary
for cluster_label, original_topic in zip(cluster_labels, list_of_topics):
    cluster_topics[cluster_label].append(original_topic)

# Print the topics assigned to each cluster
for cluster_label, topics in cluster_topics.items():
    print(f"Cluster {cluster_label}:")
    for topic in topics:
        print(topic)
    print()
    
# Based on these results, seems better to put "acq" in economy group and delete "mergers_and_acquisitions"   

                                     
######## Assign all articles to classes based on Topics categoriess ###########    
                                       
# Add 4 new class columns (the above topic groups)
df['agriculture'] = 0
df['metals'] = 0
df['economy'] = 0
df['energy'] = 0

if num_categories == 5:
    df['mergers_and_acquisitions'] = 0
    
# Loop through each row of the dataframe
for index, row in df.iterrows():
    # Initialize a flag to track whether any new column has been assigned a value of 1
    assigned_value = False
    
    # Access the 'topics' cell for the current row
    topics = row['topics'].split(', ')
    
    # Check if any agriculture topic is present
    if any(topic in agriculture_topics for topic in topics):
        df.at[index, 'agriculture'] = 1
        assigned_value = True
    
    # Check if any metals topic is present
    if any(topic in metals_topics for topic in topics):
        df.at[index, 'metals'] = 1
        assigned_value = True
    
    # Check if any economy topic is present
    if any(topic in economy_topics for topic in topics):
        df.at[index, 'economy'] = 1
        assigned_value = True
            
    # Check if any energy topic is present
    if any(topic in energy_topics for topic in topics):
        df.at[index, 'energy'] = 1
        assigned_value = True
   
    if num_categories == 6:
        # Check if any mergers_and_acquisitions topic is present
        if any(topic in mergers_and_acquisitions_topics for topic in topics):
            df.at[index, 'mergers_and_acquisitions'] = 1
            assigned_value = True
            
    # If no new column has been assigned a value of 1, print out the entire row
    if not assigned_value:
        print("\nRow:", index)
        print("Topics:", topics)
        # Nothing printed, so all rows were assigned to at least 1 topic group
        
#print(df.head(20))


############ Plot the Distribution of Category Labels and Counts ##############

new_columns = ['agriculture', 'metals', 'economy', 'energy']

# Calculate the sums of the selected columns and sort 
column_sums = df[new_columns].sum()
column_sums_sorted = column_sums.sort_values(ascending=False)
total_sum = column_sums.sum()
print(total_sum)
# This shows 6888 article categorizations were made, for the 6596 articles remaining,
# meaning some were assigned multiple categories, just like I wanted.


############# Plot the New Category Distribution for Full Dataset #############

# Plot the distribution of the topic group sums 
ax = column_sums_sorted.plot(kind='bar', figsize=(10, 6), edgecolor='black')
plt.title('Distribution of Topic Categories', fontweight='bold')
plt.xlabel("Topic Category", fontweight='bold')
plt.ylabel("Number of Articles", fontweight='bold')
plt.xticks(rotation=0)  

# Add column sums as text annotations at the top of each bar
for i, v in enumerate(column_sums_sorted):
    ax.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

# Add annotation for the total sum of column sums 
#plt.text(-0.4, max(column_sums)-100, "Total: " + f"{total_sum}", 
#         ha='left', va='bottom',  weight='bold', fontsize=12)
plt.tight_layout()  # Adjust layout to prevent clipping of labels
if save_plots == True:
    plt.savefig("recategorized-full_dataset.png")
plt.show()


############ Plot the Distribution of Articles per Category Count #############

# Define a function to count the number of category labels for each article
def count_category_labels(row):
    #if row.name == 0:  # Print only for the first row
        #print(row.iloc[2:])  # Check the values being summed
    return sum(row.iloc[2:9])  # Summing the binary values will give the count of 1s

# Apply the function to each row of the DataFrame and store the counts in a new column
df['category_label_count'] = df.apply(count_category_labels, axis=1)

"""# Plot a histogram of the category label counts
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(df['category_label_count'], bins=np.arange(0.5, df['category_label_count'].max() + 1), 
                           edgecolor='black')
plt.xlabel('Number of Category Labels', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title('Distribution of Category Assignments per Article - Full Dataset', fontweight='bold')
plt.xticks(np.arange(0, df['category_label_count'].max() + 1))
#plt.grid(axis='y', linestyle='--', alpha=0.7)
# Add annotations above each bin
for count, bin_edge in zip(counts, bins):
    bin_mid = bin_edge + 0.5  # Calculate the midpoint of the bin
    plt.text(bin_mid, count, str(int(count)), ha='center', va='bottom', weight='bold') 
if save_plots == True:
    plt.savefig("category_counts-full_dataset.png")
plt.show()"""


# Calculate counts and bins
counts, bins = np.histogram(df['category_label_count'], bins=np.arange(0.5, df['category_label_count'].max() + 1))

# Set the width of each bin
bin_width = 0.6  # This can be adjusted to control the spacing between bars

plt.figure(figsize=(10, 6))
plt.bar(bins[:-1] + 0.5, counts, width=bin_width, edgecolor='black')

plt.xlabel('Number of Category Labels', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title('Distribution of Category Assignments per Article - Full Dataset', fontweight='bold')

# Set x-ticks to the correct positions
plt.xticks(np.arange(0, df['category_label_count'].max() + 1))

# Add annotations above each bin
for count, bin_edge in zip(counts, bins[:-1]+0.5):
    plt.text(bin_edge, count, str(int(count)), ha='center', va='bottom', weight='bold')
plt.tight_layout()
if save_plots:
    plt.savefig("category_counts-full_dataset.png", dpi=300)

plt.show()




############ Filter out single-category or multi-category articles ############

if case_type == "multi-category":
    # Filter, leaving only the multi-category articles
    df = df[df['category_label_count'] > 1]
    df.reset_index(drop=True, inplace=True)
    print(f"\nFiltering out {case_type} articles... Articles remaining in dataframe:", len(df))
    # There are 306 for the 6-category and 296 for the 5-category

elif case_type == "single-category":
    # Count how many articles have 'category_label_count' of only 1
    num_articles_with_one_category = len(df[df['category_label_count'] == 1])
    print("\nNumber of articles with only 1 of the 5 categories assigned:", num_articles_with_one_category)
    # There are 6291 for the 6-category and 6301 for the 5-category

# Filter, leaving only the single-category articles
    df = df[df['category_label_count'] == 1]
    df.reset_index(drop=True, inplace=True)
    print(f"\nFiltering to keep only the {case_type} articles... Articles remaining in dataframe:", len(df))
    # There are 6314 for the 4-category

elif case_type == "mixed-category":
    # Don't filter out either type.
    print("\nNot filtering out single- or multi-category articles. Articles remaining in dataframe:", len(df))
    
    
if case_type in ["mixed-category", "multi-category"]:
    # Plot category counts per article only if there are multi-category articles remaining 
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(df['category_label_count'], bins=np.arange(1.5, df['category_label_count'].max() + 1), color='skyblue', edgecolor='black')
    plt.xlabel('Number of Category Labels', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title('Category Assignments per Article - Filtered Dataset', fontweight='bold')
    plt.xticks(np.arange(2, df['category_label_count'].max() + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add annotations above each bin
    for count, bin_edge in zip(counts, bins):
        bin_mid = bin_edge + 0.5  # Calculate the midpoint of the bin
        plt.text(bin_mid, count, str(int(count)), ha='center', va='bottom', weight='bold', fontsize=12)  
    if save_plots == True:
        plt.savefig(os.path.join(case_type, "category_counts.png"))
    plt.show()
    
# Remove the 'category_label_count' column from the DataFrame
df.drop(columns=['category_label_count'], inplace=True)


if case_type in ["single-category", "multi-category"]:
    # Plot the Category Distribution, only if some articles have been filtered (otherwise it's the same as we plotted above)
    
    # Calculate the new sums for each category, and sort them:
    column_sums = df[new_columns].sum()
    column_sums_sorted = column_sums.sort_values(ascending=False)
    total_sum = column_sums.sum()
    print("\nTotal number of category assignments: ", total_sum)
    
    # Plot the distribution of the topic group sums 
    ax = column_sums_sorted.plot(kind='bar', figsize=(10, 6), edgecolor='black')
    plt.title('Distribution of Topic Categories', fontweight='bold')
    plt.xlabel("Topic Category", fontweight='bold')
    plt.ylabel("Number of Articles", fontweight='bold')
    plt.xticks(rotation=0)  
    
    # Add column sums as text annotations at the top of each bar
    for i, v in enumerate(column_sums_sorted):
        ax.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Add annotation for the total sum of column sums 
    #plt.text(-0.4, max(column_sums)-100, "Total: " + f"{total_sum}", 
    #         ha='left', va='bottom',  weight='bold', fontsize=12)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    if save_plots == True:
        plt.savefig(os.path.join(case_type, "category_distribution_new.png"), dpi=300)
    plt.show()


############ Plot the distribution of true labels for each category ###########

for index, category in enumerate(df.columns[2:7]):     
    plt.figure(figsize=(10, 6)) 
    df[category].value_counts().sort_index().plot(kind='bar')
    # Add annotations to the bars
    for i, value in enumerate(df[category].value_counts().sort_index()):
        plt.annotate(value, xy=(i, value), ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Actual Labels: {category}', fontweight='bold')
    plt.xlabel('Label', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.xticks(rotation=0)
    # Add an annotation to show the distribution values
    num_of_zeros = df[category].value_counts()[0]
    num_of_ones = df[category].value_counts()[1]
    percentage_zeros = (num_of_zeros / (num_of_ones + num_of_zeros)) * 100
    percentage_ones = (num_of_ones / (num_of_ones + num_of_zeros)) * 100
    plt.text(-0.1, 100, f'{percentage_zeros:.2f}%', color='b', fontweight='bold', fontsize=18)
    plt.text(0.9, 100, f'{percentage_ones:.2f}%', color='b', fontweight='bold', fontsize=18)
    
    if save_plots == True:
        plt.savefig(os.path.join(case_type, f"distribution_of_true_labels-{category}.png"))
    plt.show()


# ------ Shuffle and sample the dataframe to get 300 random articles -------- #

df_shuffled = df.sample(frac=1, random_state=seed_value).reset_index(drop=True)

# Add a unique identifier to each article:
df_shuffled.insert(0, 'unique_id', range(1, len(df_shuffled) + 1))   
    
# To save the entire 6314-article dataset without sampling a subset, run this line:
df_shuffled.to_csv(os.path.join(case_type, f"reuters_recategorized-{case_type}-6314articles_4cat.csv"))

# Do basic sampling for the cases involving multi-label articles
if case_type in ["mixed-category", "multi-category"]:
    df_sampled = df_shuffled.sample(n=300, random_state=seed_value).reset_index(drop=True)

# Do balanced sampling for the single category articles dataset
elif case_type == "single-category":
    
    # Group the DataFrame by the category columns
    grouped = df_shuffled.groupby(new_columns)
    
    # Number of samples per category
    n_samples_per_category = 300 // len(new_columns)
    
    # List to hold sampled DataFrames
    sampled_dfs = []
    
    # Sample from each category
    for category, group in grouped:
        sampled_df = group.sample(n=n_samples_per_category, random_state=seed_value)
        sampled_dfs.append(sampled_df)
    
    # Concatenate sampled DataFrames
    sampled_df = pd.concat(sampled_dfs)
    
    # Reset the index
    sampled_df.reset_index(drop=True, inplace=True)
    
    # Check the distribution of categories
    print(sampled_df[new_columns].sum())
    
    
    # --------- Plot the Category Distribution for 300-article subset ----------- #
    
    # Calculate the sums of each category and sort 
    column_sums = sampled_df[new_columns].sum()
    column_sums_sorted = column_sums.sort_values(ascending=False)
    total_sum = column_sums.sum()
    print(total_sum)
    
    # Plot the distribution of the topic group sums 
    ax = column_sums_sorted.plot(kind='bar', figsize=(10, 6), edgecolor='black')
    plt.title('Distribution of Topic Categories - 300-article subset', fontweight='bold')
    plt.xlabel("Topic Category", fontweight='bold')
    plt.ylabel("Number of Articles", fontweight='bold')
    plt.xticks(rotation=0)  
    
    # Add column sums as text annotations at the top of each bar
    for i, v in enumerate(column_sums_sorted):
        ax.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    if save_plots == True:
        plt.savefig("recategorized-300_sampled_dataset.png")
    plt.show()


################## Export a csv with reassigned categories ####################

sampled_df.to_csv(os.path.join(case_type, f"reuters_recategorized-{case_type}-300.csv"))

# Change directories to get back to the main folder
os.chdir("../")

sys.exit()



####### Extract a new 300-article subset (out-of-sample validation set) #######

""" This section assumes you don't want to have to rerun the entire script above.
Just run lines 27-57, then run this part and it loads all needed files itself."""

# Read the first 300-article subset into a DataFrame:
first_subset_300_filename = os.path.join("datasets/single-category/reuters_recategorized-single-category-300.csv")
first_subset_300_df = pd.read_csv(first_subset_300_filename, index_col='Unnamed: 0')

# Read the full (6314-article) single-article dataset into a DataFrame:
full_dataset_filename = os.path.join("datasets/single-category/reuters_recategorized-single-category-6314articles_4cat.csv")
full_dataset_df = pd.read_csv(full_dataset_filename, index_col='Unnamed: 0')

# Remove articles in first_subset_300_df from full_dataset_df so they aren't selected again
remaining_df = full_dataset_df[~full_dataset_df['unique_id'].isin(first_subset_300_df['unique_id'])]

# Run just up to this line to extract subset3 (the remaining 6014 articles without subset1):
#remaining_df.to_csv(os.path.join("datasets/single-category", "reuters_recategorized-single-category-300_subset3.csv"))

columns = ['agriculture', 'metals', 'economy', 'energy']

# Group the DataFrame by the category columns
grouped = remaining_df.groupby(columns)

# Number of samples per category
n_samples_per_category = 300 // len(columns)

# List to hold sampled DataFrames
sampled_dfs = []

# Sample from each category
for category, group in grouped:
    sampled_df = group.sample(n=n_samples_per_category, random_state=seed_value)
    sampled_dfs.append(sampled_df)

# Concatenate sampled DataFrames
sampled_df = pd.concat(sampled_dfs)

# Reset the index
sampled_df.reset_index(drop=True, inplace=True)

# Check the distribution of categories
print(sampled_df[columns].sum())


# Export csv of new 300-article subset
sampled_df.to_csv(os.path.join("datasets/single-category", "reuters_recategorized-single-category-6014_subset2.csv"))
