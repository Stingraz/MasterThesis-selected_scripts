# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:19:38 2024

@author: micha

This script uses a specified pretrained Hugging Face model to predict the content 
labels (classes) of a subset of articles from the 4-class, unstylized Reuters 
dataset. 

This scipt should not be run directly, but rather imported to Colab and run by 
the run_all_3_classifiers.ipynb Jupyter notebook. Specific instructions are 
included there.

"""

############################# Function Definitions ############################

def classify_article(classifier, text, categories, df_pred_single_article, multi_label):
    """Make predictions for a single article.
    
    Parameters:
    - classifier: HuggingFace pipeline containing a model
    - df_pred: DataFrame to append scores onto
    - text: String of text to classify
    - categories: List of strings representing label names
    
    Returns:
    - df_pred: DataFrame containing classification predictions"""
    import pandas as pd            
    prediction_dict = classifier(text, categories, multi_label=multi_label)
    
    # Print results for each category and append to DataFrame:
    #for label, score in zip(prediction_dict["labels"], prediction_dict["scores"]):
    #    print(f"{label}: {score}")  
    #    df_pred[label] = [score]
    
    # Create a dictionary to hold the new row
    new_row = {label: score for label, score in zip(prediction_dict["labels"], prediction_dict["scores"])}
    
    # Concatenate the new row to the bottom of the DataFrame
    df_pred_single_article = pd.concat([df_pred_single_article, pd.DataFrame(new_row, index=[0])], ignore_index=True)

    return df_pred_single_article


######################## End of Function Definitions ########################## 

def main(model, dataset):
    """ model: a string specifying which pretrained classifier to use ("deberta", "distilbert", or "distilroberta")
        dataset: a set of articles, in DataFrame format """
    import pandas as pd
    import numpy as np
    from transformers import pipeline
    import time

    np.random.seed(42)
    
    # Start the timer
    start_time = time.time()

    # Define the labels to be used for classification:        
    categories = list(dataset.columns[2:])
    
    # --------------------- Test the classification models ---------------------- #
    
    if model == "distilbert":
        print("\nRunning the distilbert model...\n")
        classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        # multi_label should be False for distilbert (seems to not do well when True) 
        multi_label = False
        
    elif model == "deberta":
        print("\nRunning the deberta model...\n")
        classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0") 
        #classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33")
        multi_label = False
        # Suppress the numpy deprecation warning by setting the deprecated alias to int explicitly
        np.int = int
    
    elif model == "distilroberta":
        print("\nRunning the distilroberta model...\n")
        classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")
        multi_label = False
    
    else:
        print("\nModel not available. Please use deberta, distilbert, or distilroberta.\n")
    
    
    ################### Make predictions on Reuters Dataset ################### 
    
    # Create a DataFrame to store results for all articles
    df_predictions = dataset.copy()
    
    # Create a new set of columns for category value, to be added to the predictions dataframe later
    pred_columns = [f"{column}_pred" for column in df_predictions.columns[2:]]

    # Loop through each article, and compute prediction
    for index, row in df_predictions.iterrows():
        print(f"Predicting on article {index} of {len(df_predictions)}...")
        article_text = row['body_of_article']
        
        # Create an empty DataFrame to store predictions
        df_pred_single_article = pd.DataFrame(columns=categories)
        
        # Get the float prediction values using the zero-shot classifier and record
        df_pred_single_article = classify_article(classifier=classifier, text=article_text, categories=categories, 
                                   df_pred_single_article=df_pred_single_article, multi_label=multi_label)
    
        # Record the scores in the results dataframe 
        df_predictions.loc[index, pred_columns] = df_pred_single_article.iloc[0].values

    
    # Stop the timer and calculate elapsed time for this model
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Create and export a csv file of df_results_single_classifier
    predictions_filename = f"{model}_predictions.csv"
    df_predictions.to_csv(predictions_filename, index=False)
    print(f"\nClassification model predictions exported as: {predictions_filename}") 
    
    return inference_time
   
if __name__ == "__main__":
    main()
