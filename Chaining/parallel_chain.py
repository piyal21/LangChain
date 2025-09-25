from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from huggingface_hub import login
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
import os

load_dotenv()
api_token = os.getenv('HUGGINGFACE_API_TOKEN')
login(api_token)

prompt = PromptTemplate(
    template= 'Write 5 interesting facts about {topic}',
    input_variables=['topic']
)

# --> setting up the model. 
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-Prover-V2-671B",
    task= "text-generation"
)
model = ChatHuggingFace(llm=llm)

# --> setting up parser 
parser = StrOutputParser()



# prompt 1 
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}" ,
    input_variables=['text']
)

# prompt 2 
prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=['text']
)

#prompt 3 
prompt3 = PromptTemplate(
    template='Merge the provided notes nad quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

# --> Runnable Parallel --> can execute as many parallel chains as user like
parallel_chain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz': prompt2 | model | parser
})

# --> merged chain 
merge_chain = prompt3 | model | parser

# input text 
text ="""Skip to main content
scikit-learn homepage
Install
User Guide
API
Examples
Community
GitHub
Section Navigation

Release Highlights
Biclustering
Calibration
Classification
Clustering
Covariance estimation
Cross decomposition
Dataset examples
Decision Trees
Decomposition
Developing Estimators
Ensemble methods
Examples based on real world datasets
Feature Selection
Frozen Estimators
Gaussian Mixture Models
Gaussian Process for Machine Learning
Generalized Linear Models
Inspection
Kernel Approximation
Manifold learning
Miscellaneous
Advanced Plotting With Partial Dependence
Comparing anomaly detection algorithms for outlier detection on toy datasets
Comparison of kernel ridge regression and SVR
Displaying Pipelines
Displaying estimators and complex pipelines
Evaluation of outlier detection estimators
Explicit feature map approximation for RBF kernels
Face completion with a multi-output estimators
Introducing the set_output API
Isotonic Regression
Metadata Routing
Multilabel classification
ROC Curve with Visualization API
The Johnson-Lindenstrauss bound for embedding with random projections
Visualizations with Display Objects
Missing Value Imputation
Model Selection
Multiclass methods
Multioutput methods
Nearest Neighbors
Neural Networks
Pipelines and composite estimators
Preprocessing
Semi Supervised Classification
Support Vector Machines
Working with text documents
Examples
Miscellaneous
ROC Curve with Visualization API
ROC Curve with Visualization API
Scikit-learn defines a simple API for creating visualizations for machine learning. The key features of this API is to allow for quick plotting and visual adjustments without recalculation. In this example, we will demonstrate how to use the visualization API by comparing ROC curves.

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
Load Data and Train a SVC
First, we load the wine dataset and convert it to a binary classification problem. Then, we train a support vector classifier on a training dataset.

import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_wine(return_X_y=True)
y = y == 2

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
svc = SVC(random_state=42)
svc.fit(X_train, y_train)

SVC
?i
SVC(random_state=42)


Plotting the ROC Curve
Next, we plot the ROC curve with a single call to sklearn.metrics.RocCurveDisplay.from_estimator. The returned svc_disp object allows us to continue using the already computed ROC curve for the SVC in future plots.

svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)
plt.show()
plot roc curve visualization api
Training a Random Forest and Plotting the ROC Curve
We train a random forest classifier and create a plot comparing it to the SVC ROC curve. Notice how svc_disp uses plot to plot the SVC ROC curve without recomputing the values of the roc curve itself. Furthermore, we pass alpha=0.8 to the plot functions to adjust the alpha values of the curves.

rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8)
plt.show()
plot roc curve visualization api
Total running time of the script: (0 minutes 0.157 seconds)

Related examples


Receiver Operating Characteristic (ROC) with cross validation

Detection error tradeoff (DET) curve

Release Highlights for scikit-learn 0.22

Multiclass Receiver Operating Characteristic (ROC)
Gallery generated by Sphinx-Gallery

previous

Multilabel classification

next

The Johnson-Lindenstrauss bound for embedding with random projections

 On this page
Load Data and Train a SVC
Plotting the ROC Curve
Training a Random Forest and Plotting the ROC Curve
This Page
Show Source
 Download source code
 Download Jupyter notebook
 Download zipped
Launch JupyterLite
Launch binder
© Copyright 2007 - 2025, scikit-learn developers (BSD License).

 """


final_chain = parallel_chain | merge_chain 
result = final_chain.invoke({'text':text})
print(result)

#--> How the parallel chain works
print(final_chain.get_graph().print_ascii())