# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:01:04 2024

@author: micha

Calculates the average word vector for each article, and compares each unstylized 
version to its stylized complement using word embeddings from the pretrained 
spaCy model "en_core_web_md" (medium-sized English model).


"""
import pandas as pd
import csv
import os
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from adjustText import adjust_text


# Adjust display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

np.random.seed(42)

style_list = ["shakespeare", "preschooler", "stupid_and_rude", "Gucci_ad", "legalese", "political", "cheerful"]

encoding = 'utf-8'
 
metric = "auroc"   # "accuracy", "auroc"

save_plots = True

# Load the pretrained spacy model for getting word embeddings
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading 'en_core_web_md' model...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")      
    
############################# Function Definitions ############################


def calculate_average_word_vector(article_texts): 
    """ article_texts must be in list of strings format"""   
    # Tokenize each article's text:
    text_tokens = [nlp(text) for text in article_texts]
    
    # Calculate the average vector of all article texts:
    average_word_vector = np.mean([token.vector for tokens in text_tokens for token in tokens], 
                                      axis=0)
    return average_word_vector


def calculate_cosine_similarity_with_average(stylized_article_text, average_word_vector):
    """ Calculate cosine similarity with the average word vector """     
    if stylized_article_text is None:
        return 0.0  
    # Calculate word vector for the stylized article
    stylized_article_vector = nlp(stylized_article_text).vector
    
    similarity = cosine_similarity([stylized_article_vector], [average_word_vector])[0][0]
    return similarity
    
    
######################## End of Function Definitions ##########################



############ Load the datasets with the overall classifier scores #############        

if metric == "accuracy":
    unstylized_results_file_path="classifier_comparison-unstylized/2024-05-22/single-category/avg_classifier_scores_300articles.csv"
elif metric == "auroc":
    unstylized_results_file_path="classifier_comparison-unstylized/2024-05-22/single-category/deberta/auroc_scores.csv"
    
with open(unstylized_results_file_path, 'r') as file:
    df_results_unstylized = pd.read_csv(file, index_col=None)

# Get the model score for the dataset 
if metric == "accuracy":    
    # For accuracy, row 0 is deberta
    score_unstylized = df_results_unstylized.iloc[0]["accuracy_score"]
elif metric == "auroc":
    # Rename the unnamed column for easier access
    df_results_unstylized.rename(columns={df_results_unstylized.columns[0]: 'metric'}, inplace=True)
    score_unstylized = df_results_unstylized.loc[df_results_unstylized['metric'] == 'Weighted Average', 'AUROC'].values[0]

# Create a dataframe to contain the results for each dataset
df_summary = pd.DataFrame(columns=["dataset", "cos_sim", metric])

# Populate the DataFrame with data for the unstylized dataset
df_summary.loc[0] = ["unstylized", 1.0, score_unstylized]
    

for i, style in enumerate(style_list):
    print(f"Computing cosine similarities for {style}...")
    # Load in the stylized and unstylized articles
    file_path_stylized = f"datasets/stylized_articles/{style}/{style}_articles.csv"
    
    with open(file_path_stylized, 'r', encoding=encoding) as file:
        df_articles_and_true_labels = pd.read_csv(file, index_col=None)        
    
    # Extract article text columns from DataFrame and convert them into lists of strings
    article_texts_unstylized = df_articles_and_true_labels["unstylized"].astype(str).tolist()
    article_texts_stylized = df_articles_and_true_labels[style].astype(str).tolist()
    
    # Extract the true labels and save them to another dataframe
    df_true_labels = df_articles_and_true_labels.copy()
    categories = ["agriculture", "metals", "economy", "energy"]
    df_true_labels = df_true_labels.loc[:, categories]
    # Rename the columns
    new_column_names = {col: col + "_true" for col in categories}
    df_true_labels.rename(columns=new_column_names, inplace=True)
    
    
    # Calculate cos sim between each stylized article and it's unstylized version
    cos_sim_values = []
    for index, (stylized_article, unstylized_article) in enumerate(zip(article_texts_stylized, article_texts_unstylized)):
        similarity_to_self = calculate_cosine_similarity_with_average(stylized_article, calculate_average_word_vector([unstylized_article]))
        cos_sim_values.append(similarity_to_self)
    
    # Calculate the average cosine similarity for the entire stylized dataset
    average_cos_sim_stylized = np.mean(cos_sim_values)
    
    # Load in the article results file
    if metric == "accuracy": 
        stylized_results_file_path = f"disentanglement/deberta_evaluation/original_models/DeBERTa-v2.0/{style}/avg_classifier_scores_300articles.csv"
    elif metric == "auroc":
        stylized_results_file_path = f"disentanglement/deberta_evaluation/original_models/DeBERTa-v2.0/{style}/auroc_scores.csv"
        
    with open(stylized_results_file_path, 'r') as file:
        df_results_stylized = pd.read_csv(file, index_col=None)
    
    # Get the model score for the dataset 
    if metric == "accuracy":
        # For accuracy, row 0 is deberta
        score_stylized = df_results_stylized.iloc[0]["accuracy_score"]
    elif metric == "auroc":
        # Rename the unnamed column for easier access
        df_results_stylized.rename(columns={df_results_stylized.columns[0]: 'metric'}, inplace=True)
        score_stylized = df_results_stylized.loc[df_results_stylized['metric'] == 'Weighted Average', 'AUROC-0'].values[0]
        
    # Populate the DataFrame with data for the stylized dataset of this intensity value
    df_summary.loc[i+1] = [style, average_cos_sim_stylized, score_stylized]


# -------- Plot average cos sim against model score for each dataset -------- #

plt.figure(figsize=(10, 6))
plt.scatter(df_summary["cos_sim"], df_summary[metric] * 100, color='blue')
plt.xlabel("Average Cosine Similarity", fontweight='bold')
plt.ylabel("AUROC [%]", fontweight='bold')
plt.title("AUROC Scores vs Cosine Similarities", fontweight='bold')
#plt.xticks([0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
#plt.yticks([0.8, 0.81, 0.82, 0.83, 0.84, 0.85,0.86, 0.87, 0.88, 0.89, 0.9])

# Fit a linear regression line
#m, b = np.polyfit(df_summary["cos_sim"], df_summary["accuracy"], 1)
#plt.plot(df_summary["cos_sim"], m*df_summary["cos_sim"] + b, color='red')

"""# Fit a polynomial curve
z = np.polyfit(df_summary["cos_sim"], df_summary["accuracy"], 3)
p = np.poly1d(z)
xp = np.linspace(min(df_summary["cos_sim"]), max(df_summary["cos_sim"]), 100)
plt.plot(xp, p(xp), color='red')"""

for i, txt in enumerate(df_summary["dataset"]):
    plt.annotate(txt, (df_summary["cos_sim"][i], df_summary[metric][i] * 100), textcoords="offset points", xytext=(0,10), ha='center')
plt.grid(True, alpha=0.3)
plt.ylim(88, 100)
plt.xlim(0.845, 1.01)
plt.tight_layout()
if save_plots == True:
    plt.savefig(f"disentanglement/deberta_evaluation/original_models/DeBERTA-v2.0/cos_sims-{metric}.png")
plt.show()


# ------------- Plot the accuracy for each dataset on a 1D plot ------------- #

plt.figure(figsize=(10, 2))
plt.scatter(df_summary["accuracy"], np.zeros(len(df_summary)), color='blue', marker='o', clip_on=False) # clip_on sets the points in front
plt.yticks([])  # Hides the y-axis
plt.xlabel('Accuracy Score', fontweight='bold')
plt.title('Accuracy Scores of All Styled Datasets', fontweight='bold')
plt.xlim(0.7, 0.9) 
plt.ylim(-0.1, 0)  
plt.gca().invert_yaxis()  # Invert y-axis
plt.gca().spines['top'].set_visible(False)  # Hide top spine
plt.gca().spines['right'].set_visible(False)  # Hide right spine
plt.gca().spines['left'].set_visible(False)  # Hide left spine

# Annotate each point with its style name
texts = []
for i, dataset in enumerate(df_summary["dataset"]): 
    texts.append(plt.text(df_summary["accuracy"][i], 0, dataset, ha='center', va='bottom'))

# Adjust text positions to avoid overlap
adjust_text(texts, x=df_summary["accuracy"], y=np.zeros(len(df_summary)), arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

plt.tight_layout()
if save_plots == True:
    plt.savefig("disentanglement/deberta_evaluation/original_models/DeBERTa-v2.0/all_accuracy_scores.png")
plt.show()




