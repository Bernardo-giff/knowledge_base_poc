# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 02L - Embeddings, Vector Databases, and Search
# MAGIC
# MAGIC
# MAGIC In this lab, we will apply the text vectorization, search, and question answering workflow that you learned in the demo. The dataset we will use this time will be on talk titles and sessions from [Data + AI Summit 2023](https://www.databricks.com/dataaisummit/). 
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Learn how to use Chroma to store your embedding vectors and conduct similarity search
# MAGIC 1. Use OpenAI GPT-3.5 to generate response to your prompt

# COMMAND ----------

# MAGIC %pip install chromadb==0.3.21 tiktoken==0.3.3

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data Setup
# MAGIC
# MAGIC In this step I will run necessary libraries in a separate notebook (from the Databricks course, which should be adapted to real use cases)

# COMMAND ----------

## Running the set up to use the LLM models from the course
%run ../knowledge_base_poc/setup/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read data
# MAGIC
# MAGIC **Note: I am reduicing the scope here to experiment (i.e: filtering everything for order relationship) --> In order to use all of the scope, a strategy for vectorization would be needed**

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH wholesale_orders AS (
# MAGIC   SELECT * FROM prd.gold.fct_order
# MAGIC   WHERE lower(ds_record_type_name) != 'Retail'
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC   Id
# MAGIC   , concat('Subject: ', Subject, ' \n Body: ', TextBody) AS corpus
# MAGIC FROM prd.bronze.salesforce_emailmessage
# MAGIC -- FOR EXPERIMENTATION PURPOSES, ONLY KEEP E-MAILS RELATED TO ORDERS
# MAGIC WHERE RelatedToId IN (SELECT pk_order FROM wholesale_orders)
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   Id
# MAGIC   , Body AS corpus FROM prd.bronze.salesforce_feeditem
# MAGIC -- FOR EXPERIMENTATION PURPOSES, ONLY CHATTER MESSAGES
# MAGIC WHERE Type = 'TextPost' -- i.e: Chatter message
# MAGIC -- AND AGAIN ONLY KEEPING CHATTER POSTS RELATED TO ORDERS
# MAGIC AND ParentId IN (SELECT pk_order FROM wholesale_orders)
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   Id
# MAGIC   , concat('Subject: ', Subject, ' \n Description: ', Description) AS corpus
# MAGIC FROM prd.bronze.salesforce_task
# MAGIC -- AND AGAIN ONLY TASKS CHATTER POSTS RELATED TO ORDERS
# MAGIC WHERE WhatId IN (SELECT pk_order FROM wholesale_orders)

# COMMAND ----------

# MAGIC %md
# MAGIC As a POC, we are using the untreated data from the Bronze layer. The performance of this model, both in terms of speed and accuracy, can be approved by using dedicated data curated for this (i.e: filtering out automated tasks, test and so forth)

# COMMAND ----------

import pandas as pd

# Turn communications query into Pandas DF
communications = _sqldf.toPandas()

# Display the results
display(communications)

# COMMAND ----------

# Print the number of records
print(f"Number of records: {len(communications)}")

# COMMAND ----------

# MAGIC %md
# MAGIC The next step is to devide the text into vectors. This choice is arbitrary, can be done by words, sentences, paragraphs or sections. For this POC, we will use sentences to preserve context whilst keeping specificity. Experimentation would tell the best strategy.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Library: FAISS
# MAGIC
# MAGIC Vector libraries are often sufficient for small, static data. Since it's not a full-fledged database solution, it doesn't have the CRUD (Create, Read, Update, Delete) support. Once the index has been built, if there are more vectors that need to be added/removed/edited, the index has to be rebuilt from scratch. 
# MAGIC
# MAGIC That said, vector libraries are easy, lightweight, and fast to use. Examples of vector libraries are [FAISS](https://faiss.ai/), [ScaNN](https://github.com/google-research/google-research/tree/master/scann), [ANNOY](https://github.com/spotify/annoy), and [HNSM](https://arxiv.org/abs/1603.09320).
# MAGIC
# MAGIC FAISS has several ways for similarity search: L2 (Euclidean distance), cosine similarity. You can read more about their implementation on their [GitHub](https://github.com/facebookresearch/faiss/wiki/Getting-started#searching) page or [blog post](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/). They also published their own [best practice guide here](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).
# MAGIC
# MAGIC If you'd like to read up more on the comparisons between vector libraries and databases, [here is a good blog post](https://weaviate.io/blog/vector-library-vs-vector-database#feature-comparison---library-versus-database).
# MAGIC
# MAGIC The overall workflow of FAISS is captured in the diagram below. 
# MAGIC
# MAGIC <img src="https://miro.medium.com/v2/resize:fit:1400/0*ouf0eyQskPeGWIGm" width=700>
# MAGIC
# MAGIC Source: [How to use FAISS to build your first similarity search by Asna Shafiq](https://medium.com/loopio-tech/how-to-use-faiss-to-build-your-first-similarity-search-bf0f708aa772).

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 2: Vectorize text into embedding vectors
# MAGIC We will be using `Sentence-Transformers` [library](https://www.sbert.net/) to load a language model to vectorize our text into embeddings. The library hosts some of the most popular transformers on [Hugging Face Model Hub](https://huggingface.co/sentence-transformers).
# MAGIC Here, we are using the `model = SentenceTransformer("all-MiniLM-L6-v2")` to generate embeddings.

# COMMAND ----------

DA.paths.datasets

# COMMAND ----------

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2", 
    cache_folder=DA.paths.datasets
)  # Use a pre-cached model
faiss_corpus_embedding = model.encode(communications['corpus'].values.tolist())
len(faiss_corpus_embedding), len(faiss_corpus_embedding[0])

# COMMAND ----------

# MAGIC %md
# MAGIC This indexing takes at least 30 minutes to run. Therefore, we need to have the vector search ready at night or use a vector database approach (i.e: Storing this information in a dedicated database for this). It also goes without saying that this would be better with all text already translated to English.
# MAGIC
# MAGIC To extend this functionality to Confluence spaces, we would need to somehow access this data. One option is this library: https://atlassian-python-api.readthedocs.io/confluence.html#get-page-info
# MAGIC
# MAGIC Another, long term and better approach would be to ingest this data via AppFlow or a similar App.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 3: Saving embedding vectors to FAISS index
# MAGIC Below, we create the FAISS index object based on our embedding vectors, normalize vectors, and add these vectors to the FAISS index. 

# COMMAND ----------

import numpy as np
import faiss

communications_to_index = communications.set_index(["Id"], drop=False)
id_index = np.array(communications_to_index.id.values).flatten().astype("int")

content_encoded_normalized = faiss_corpus_embedding.copy()
faiss.normalize_L2(content_encoded_normalized)

# Index1DMap translates search results to IDs: https://faiss.ai/cpp_api/file/IndexIDMap_8h.html#_CPPv4I0EN5faiss18IndexIDMapTemplateE
# The IndexFlatIP below builds index
index_content = faiss.IndexIDMap(faiss.IndexFlatIP(len(faiss_corpus_embedding[0])))
index_content.add_with_ids(content_encoded_normalized, id_index)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Search for relevant documents
# MAGIC
# MAGIC We define a search function below to first vectorize our query text, and then search for the vectors with the closest distance. 

# COMMAND ----------

def search_content(query, pdf_to_index, k=3):
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)

    # We set k to limit the number of vectors we want to return
    top_k = index_content.search(query_vector, k)
    ids = top_k[1][0].tolist()
    similarities = top_k[0][0].tolist()
    results = pdf_to_index.loc[ids]
    results["similarities"] = similarities
    return results

# COMMAND ----------

display(search_content("animal", pdf_to_index))
