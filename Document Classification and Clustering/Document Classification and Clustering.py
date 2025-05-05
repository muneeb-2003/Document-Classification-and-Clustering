import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import nltk
from nltk.stem import PorterStemmer
import pickle
import warnings
import csv
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
import PREPROCESSING_FILE as pp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext


# reading the TF_IDF file
df = pd.read_csv('data.csv',encoding='latin1')

df.head()

df.tail()

class_labels = {
    "Explainable Artificial Intelligence": [1, 2, 3, 7],
    "Heart Failure": [8, 9, 11],
    "Time Series Forecasting": [12, 13, 14, 15, 16],
    "Transformer Model": [17, 18, 21],
    "Feature Selection": [22, 23, 24, 25, 26]
}

# Create an empty list to store labels
labels_list = []

# Iterate through the DataFrame rows
for index, row in df.iterrows():
    documentID = row['documentID']
    # Iterate through class_labels dictionary
    for label, doc_ids in class_labels.items():
        # Check if the document ID is in the list of IDs for the current label
        if documentID in doc_ids:
            # Append the label to the labels_list
            labels_list.append(label)
            break  # Stop iterating through class_labels once the label is found

# Assign the labels_list to the 'labels' column in the DataFrame
df['labels'] = labels_list

df.head()

df

# showing the unique documentID in the table
df['documentID'].unique()

#dropping documentID and labels
x = df.drop(['documentID','labels'],axis=1)
y = df['labels']
y = pd.get_dummies(y)

y

# showing the empty values in the x and y 
print("Missing values in x:", x.isnull().sum())
print("Missing values in y:", y.isnull().sum())

# this function is the preprocessing function for query in this function we can read the documents and do tokenization
def Preprocessing_for_query(query): # in this we clean the tokens and do stemming also
        ps = PorterStemmer()#init porter stemmer of index(if we cant load it) and query
        stopWords= pp.getFileContent('Stopword-List.txt') #get a preset stopword list
        stopWords=word_tokenize(stopWords) #toeknize the stopwords

        cleanedWords=pp.tokenizeAndClean(query,stopWords)
        words=[]
        for key in cleanedWords.keys():
            word=ps.stem(key)
            for i in range(cleanedWords[key]):
                words.append(word)
        return words

# in this function we are creating the vector space for query
def Query_vector_creation(query, termArray,idf):#Process the query then run it in the vectore space model(index) and return result
        terms= Preprocessing_for_query(query)
        result=[0 for i in range(len(termArray))]

        for term in terms:
            if(term in termArray):
                index=termArray.index(term)
                result[index]+=1

        for term in terms:
            weight=0
            if(term in termArray):
                index=termArray.index(term)
                result[index]*=idf[term]
                weight+=(result[index]**2)

        weight=weight**0.5
        for i in range(len(result)):
            if(result[i]!=0):
                result[i]/=weight

        return result
    
Term_Array=None
IDF=None
with open('term_array.pickle','rb') as f:
    Term_Array=pickle.load(f)
with open('idf_score.pickle','rb') as f:
    IDF=pickle.load(f)

data = pd.read_csv('data.csv', encoding='latin1')
X = data.drop('documentID', axis=1)
print(X.shape)

class_labels = {
    "Explainable Artificial Intelligence": [1, 2, 3, 7],
    "Heart Failure": [8, 9, 11],
    "Time Series Forecasting": [12, 13, 14, 15, 16],
    "Transformer Model": [17, 18, 21],
    "Feature Selection": [22, 23, 24, 25, 26]
}

# Create an empty list to store labels
labels_list = []

# Iterate through the DataFrame rows
for index, row in df.iterrows():
    documentID = row['documentID']
    # Iterate through class_labels dictionary
    for label, doc_ids in class_labels.items():
        # Check if the document ID is in the list of IDs for the current label
        if documentID in doc_ids:
            # Append the label to the labels_list
            labels_list.append(label)
            break  # Stop iterating through class_labels once the label is found

# Assign the labels_list to the 'labels' column in the DataFrame
df['labels'] = labels_list
Y = df['labels']

# Ignore the warning about feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")
#applying knn model here
# knn_model = KNeighborsClassifier()
knn_model = KNeighborsClassifier()
#fitting the model
knn_model.fit(X,Y)
# prediting the model 
y_pred_train = knn_model.predict(X)

class_names = knn_model.classes_
# printing the accuracy
print(f"Accuracy on Training data: {accuracy_score(Y,y_pred_train)}\n")#accuracy
precision_scores = precision_score(Y, y_pred_train, average=None)#precision_score
recall_scores = recall_score(Y, y_pred_train, average=None)#recall_score
f1_scores = f1_score(Y, y_pred_train, average=None)#f1_score
for i in range(len(class_names)):
    print(f'Class: {class_names[i]}')
    print(f'Precision: {precision_scores[i]}')
    print(f'Recall: {recall_scores[i]}')
    print(f'F1 Score: {f1_scores[i]}\n')
# print(classification_report(Y, y_pred_train)) 

#example
# print('===============================')
# paragraph1="""Heart failure, also known as congestive heart failure, is a chronic condition where the heart is unable to pump enough blood to meet the body's needs. This can lead to symptoms such as shortness of breath, fatigue, swelling in the legs, ankles, or abdomen, and rapid or irregular heartbeat. There are several types of heart failure, including left-sided, right-sided, systolic, and diastolic heart failure, each with its own causes and characteristics. Treatment for heart failure typically involves medications to help the heart pump more effectively, lifestyle changes such as diet and exercise, and in some cases, surgical interventions. Managing heart failure requires a comprehensive approach that addresses both the physical and emotional aspects of the condition, and regular monitoring by healthcare professionals is essential to ensure the best possible outcomes for patients."""

# paragraph2 = """
# Explainable Artificial Intelligence (XAI) is a crucial field focused on making AI systems and their decisions understandable and transparent to humans. XAI techniques aim to provide insights into the inner workings of AI models, allowing users to comprehend why a particular decision was made. This transparency is vital, especially in high-stakes applications like healthcare and finance, where AI decisions can have significant impacts. XAI approaches include using interpretable models, determining feature importance, providing local explanations for individual predictions, promoting human-AI collaboration, and using visualizations to represent complex decision-making processes. By incorporating XAI, AI systems can be more accountable, fair, and trustworthy, fostering greater acceptance and adoption in various domains.
# """

# paragraph3= """

# Transparency and interpretability in artificial intelligence (AI) are essential for building trust and understanding in AI systems. Transparent AI refers to the ability to understand the logic, process, and outcomes of AI models, ensuring that users can trust the decisions made by these systems. Interpretable AI focuses on making the internal workings of AI models understandable to humans, enabling users to comprehend how inputs are processed to produce outputs. By promoting transparency and interpretability, AI systems become more accountable, allowing users to verify the fairness, accuracy, and ethical considerations of these systems. This transparency also encourages collaboration between humans and AI, facilitating more effective and trustworthy decision-making processes across various domains.
# """

# paragraph4 = """
# This is a crucial aspect of machine learning and data mining research, aimed at identifying the most relevant variables or features from a dataset to improve model performance and interpretability. This process involves selecting a subset of features that are most informative for the target variable, while excluding irrelevant or redundant ones. Research in feature selection encompasses various techniques, including filter methods that rank features based on statistical measures, wrapper methods that use the model's performance as a criterion, and embedded methods that integrate feature selection into the model's training process. Recent advancements in this field focus on developing efficient algorithms for high-dimensional datasets, handling missing data, and addressing the challenges posed by complex data structures such as time series or text data. Feature selection research plays a pivotal role in enhancing the efficiency, accuracy, and interpretability of machine learning models across diverse application domains.
# """

# paragraph5 = """
# Analyzing sequential data is crucial for understanding trends and making informed decisions. By examining past data points, patterns, and trends, businesses can anticipate changes and adapt strategies. This analysis helps organizations predict demand, identify seasonality, and manage risk. In finance, this approach aids in predicting stock prices, currency exchange rates, and economic indicators. Weather forecasting also benefits from this method, predicting future conditions based on historical data. Overall, analyzing sequential data is essential for data-driven decision-making, operational optimization, and competitive advantage across industries.
# """

# paragraph6 = """
# Transformer are translating text and speech in near real-time, opening meetings and classrooms to diverse and hearing-impaired attendees. They’re helping researchers understand the chains of genes in DNA and amino acids in proteins in ways that can speed drug design.
# """
# Vector_for_query=Query_vector_creation(paragraph6,Term_Array,IDF)
# Vector_for_query = np.array(Vector_for_query).reshape(1, -1)
# Answer=knn_model.predict(Vector_for_query)
# print(Answer)

# Part 2
# K-Mean Clustering

df # showing the table

# Assuming df is your dataframe
if 'documentID' in df.columns:
    X = df.drop(['documentID', 'labels'], axis=1)  # Remove documentID column
else:
    X = df.drop(['labels'], axis=1)  # If documentID is not present, only remove labels column

# Perform Standard Scaling
standard_scaler = StandardScaler()
# fitting it 
X_scaled = standard_scaler.fit_transform(X)

# Perform KMeans clustering
kmeans_model = KMeans(n_clusters=5)  # No of Clusters Choosen is 5
df['cluster'] = kmeans_model.fit_predict(X_scaled)

# making a new table
df[['labels','documentID', 'cluster']]

# Group by cluster and labels, then aggregate documentID into a list
# here we are grouping the clusters based on unique cluster and labels
cluster_class_mapping = df.groupby(['cluster', 'labels'])['documentID'].apply(list).reset_index()

print(cluster_class_mapping)

# Compute Purity
True_Labels = df['labels']

def Score_f_Purity(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

Purity_Score = Score_f_Purity(True_Labels, df['cluster'])
print("Purity Score:", Purity_Score)

# Compute Silhouette Score
Silhouette_Score = metrics.silhouette_score(X, df['cluster'])
print("Silhouette Score:", Silhouette_Score)

# Compute Rand Index
Rand_Index = metrics.adjusted_rand_score(True_Labels, df['cluster'])
print("Rand Index:", Rand_Index)

# Assuming X is your data and df['cluster'] contains the cluster assignments
# Perform PCA to reduce the dimensionality to 2 components
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# making the PCA for clarification
# Plot the clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clustering Results')
plt.colorbar(scatter, label='Cluster')
plt.show()

# just for an example
paragraph7 = """
Transparency and interpretability in artificial intelligence (AI) are essential for building trust and understanding in AI systems. Transparent AI refers to the ability to understand the logic, process, and outcomes of AI models, ensuring that users can trust the decisions made by these systems. Interpretable AI focuses on making the internal workings of AI models understandable to humans, enabling users to comprehend how inputs are processed to produce outputs. By promoting transparency and interpretability, AI systems become more accountable, allowing users to verify the fairness, accuracy, and ethical considerations of these systems. This transparency also encourages collaboration between humans and AI, facilitating more effective and trustworthy decision-making processes across various domains.
"""
Vector_Query=Query_vector_creation(paragraph7,Term_Array,IDF)
# np.reshap()
Vector_query = np.array(Vector_Query).reshape(1, -1)
result=kmeans_model.predict(Vector_query)
cluster_labels = kmeans_model.labels_
print("Cluster Labels:", cluster_labels)
print("Predicted Cluster for New Text:", result)

#gui
# Function to perform classification
def classify():
    # Get the text from the classification input field
    query_text = classification_entry.get()

    # Perform classification
    query_vector = Query_vector_creation(query_text, Term_Array, IDF)
    query_vector = np.array(query_vector).reshape(1, -1)
    classification_result = knn_model.predict(query_vector)

    # Display classification result in the text area
    result_text = f"Classification Result: {classification_result}\n"
    text_area.delete('1.0', tk.END)  # Clear previous text
    text_area.insert(tk.END, result_text)

# Function to perform clustering
def cluster():
    # Get the text from the clustering input field
    query_text = clustering_entry.get()

    # Perform clustering
    query_vector = Query_vector_creation(query_text, Term_Array, IDF)
    query_vector = np.array(query_vector).reshape(1, -1)
    cluster_result = kmeans_model.predict(query_vector)

    # Display clustering result in the text area
    result_text = f"Cluster Prediction: {cluster_result}\n"
    text_area.delete('1.0', tk.END)  # Clear previous text
    text_area.insert(tk.END, result_text)

# Function to clear the text area
def clear_text():
    text_area.delete('1.0', tk.END)

# Create a Tkinter window
window = tk.Tk()
window.title("Document Classification and Clustering")
window.geometry("800x600")  # Larger window size

# Add an entry field for classification input
classification_entry_label = ttk.Label(window, text="Enter text for classification:")
classification_entry_label.grid(column=0, row=0, padx=10, pady=10)
classification_entry = ttk.Entry(window, width=50)
classification_entry.grid(column=1, row=0, padx=10, pady=10)

# Add a button to trigger classification
classify_button = ttk.Button(window, text="Classify", command=classify)
classify_button.grid(column=2, row=0, padx=10, pady=10)

# Add a button to clear classification text area
clear_classification_button = ttk.Button(window, text="Clear Classification", command=clear_text)
clear_classification_button.grid(column=3, row=0, padx=10, pady=10)

# Add an entry field for clustering input
clustering_entry_label = ttk.Label(window, text="Enter text for clustering:")
clustering_entry_label.grid(column=0, row=1, padx=10, pady=10)
clustering_entry = ttk.Entry(window, width=50)
clustering_entry.grid(column=1, row=1, padx=10, pady=10)

# Add a button to trigger clustering
cluster_button = ttk.Button(window, text="Cluster", command=cluster)
cluster_button.grid(column=2, row=1, padx=10, pady=10)

# Add a button to clear clustering text area
clear_clustering_button = ttk.Button(window, text="Clear Clustering", command=clear_text)
clear_clustering_button.grid(column=3, row=1, padx=10, pady=10)

# Add a text area to display results
text_area = scrolledtext.ScrolledText(window, width=90, height=30)  # Increased size
text_area.grid(column=0, row=2, columnspan=4, padx=10, pady=10)

# Run the Tkinter event loop
window.mainloop()