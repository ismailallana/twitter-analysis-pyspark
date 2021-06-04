#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, Word2Vec, StringIndexer, IndexToString
from pyspark.ml.linalg import *
from pyspark.sql.types import * 
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row


# In[3]:


spark = SparkSession     .builder     .appName("11-iall-ismail-allana")     .getOrCreate()


# In[4]:


df = spark.read.option("multiline","true").json('tweets.json').cache()
spark.conf.set("spark.sql.shuffle.partitions", 20)


# # Workload 1 (Top 5 users with similar interest for user 2305663328)

# Preparing the data for feature extraction

# In[5]:


#Extracting the relevant three columns from the dataframe for the workload and converting to RDD
workload1_df = df.select(['user_id','replyto_id','retweet_id'])
workload1_rdd = workload1_df.rdd.map(list)

#Filtering out the null values and combining the reply and retweet columns into one
replyrdd = workload1_rdd.map(lambda x: (x[0],x[1])).filter(lambda x: x[1] != None)
retweetrdd = workload1_rdd.map(lambda x: (x[0],x[2])).filter(lambda x: x[1] != None)
joinedrdd = retweetrdd.union(replyrdd)

#Combining all tweets for each user into a single document representation
doc_representation = joinedrdd.groupByKey().mapValues(list).map(lambda x: (str(x[0]),x[1]))
final_df = doc_representation.toDF().selectExpr("_1 as UserID", "_2 as DocumentRepresentation").cache()


# TFIDF Vectorisation

# In[6]:


#Converting the tweet ids into vectors and performing tfidf
df_tfidf = final_df.withColumn("DocumentRepresentation",
   concat_ws(" ",col("DocumentRepresentation")))
tokenizer = Tokenizer(inputCol="DocumentRepresentation", outputCol="words")
wordsData = tokenizer.transform(df_tfidf)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Calculating cosine similarity for all users with user 2305663329 (Row 25)
v1_tfidf = rescaledData.select('features').collect()[24][0]
userid = rescaledData.select('UserID').collect()[24][0]
rdd_tfidf = rescaledData.rdd.map(list).map(lambda x: [x[0],x[-1]]).filter(lambda x: x[0] != userid)
cossim_tfidf = rdd_tfidf.mapValues(lambda x: float(x.dot(v1_tfidf) / (x.norm(2) * v1_tfidf.norm(2)))).sortBy(lambda x: -x[1])
cossim_tfidf.toDF().selectExpr("_1 as UserID", "_2 as TFIDF_Cosine_Similarity").show(5)


# Word2Vec Vectorisation

# In[15]:


#Converting tweet ids into word vectors
df_word2vec = final_df.withColumn('DocumentRepresentation', final_df.DocumentRepresentation.cast("array<string>"))
word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="DocumentRepresentation", outputCol="result")
model = word2Vec.fit(df_word2vec)
result = model.transform(df_word2vec)

# Calculating cosine similarity for all users with user 2305663329 (Row 25)
userid = result.select('UserID').collect()[24][0]
rdd_word2vec = result.rdd.map(list).map(lambda x: [x[0],x[-1]]).filter(lambda x: x[0] != userid)
v1_word2vec = result.select('result').collect()[24][0]
cossim_word2vec = rdd_word2vec.mapValues(lambda x: float(x.dot(v1_word2vec) / (x.norm(2) * v1_word2vec.norm(2)))).sortBy(lambda x: -x[1])
cossim_word2vec.toDF().selectExpr("_1 as UserID", "_2 as Word2Vec_Cosine_Similarity").show(5)


# # Workload 2

# Data preparation for collaborative filtering

# In[7]:


#Extract relevant 2 columns for workload 2 and grouping all the users and user mentions to be fed to ALS model
workload2_df = df.select(['user_id','user_mentions'])
workload2_rdd = workload2_df.rdd.filter(lambda x: x[1] != None).map(lambda x: ((x[0],list(x[1][0])[0]),1)).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0][0],x[0][1],x[1]))
user_itemdf = workload2_rdd.toDF()

# Convert values to ALS suitable format

stringIndexer = StringIndexer(inputCol="_1", outputCol="user_index",
    stringOrderType="frequencyDesc")
stringIndexer.setHandleInvalid("error")
model_user = stringIndexer.fit(user_itemdf)
model_user.setHandleInvalid("error")
user_itemdf = model_user.transform(user_itemdf)

stringIndexer = StringIndexer(inputCol="_2", outputCol="item_index",
    stringOrderType="frequencyDesc")
stringIndexer.setHandleInvalid("error")
model_item = stringIndexer.fit(user_itemdf)
model_item.setHandleInvalid("error")
user_itemdf = model_item.transform(user_itemdf)

user_itemdf = user_itemdf.withColumn("_3", user_itemdf["_3"].cast(FloatType()))

#Feed the dataframe to collaborative filtering algorithm

(training, test) = user_itemdf.randomSplit([0.8, 0.2])
als = ALS(maxIter=5, regParam=0.01, userCol="user_index", itemCol="item_index", ratingCol="_3",
          coldStartStrategy="drop")
model = als.fit(training)
userRecs = model.recommendForAllUsers(5)

#Transformations on model output to display top 5 recommendations

userrec_rdd = userRecs.rdd

userrec_finalrdd = userrec_rdd.map(lambda x: (x[0],x[1][0][0],x[1][1][0],x[1][2][0],x[1][3][0],x[1][4][0]))

recommendations = userrec_finalrdd.toDF()

inverter = IndexToString(inputCol="_1", outputCol="User", labels=model_user.labels)
recommendations = inverter.transform(recommendations).drop('_1')
inverter = IndexToString(inputCol="_2", outputCol="Recommendation 1", labels=model_item.labels)
recommendations = inverter.transform(recommendations).drop('_2')
inverter = IndexToString(inputCol="_3", outputCol="Recommendation 2", labels=model_item.labels)
recommendations = inverter.transform(recommendations).drop('_3')
inverter = IndexToString(inputCol="_4", outputCol="Recommendation 3", labels=model_item.labels)
recommendations = inverter.transform(recommendations).drop('_4')
inverter = IndexToString(inputCol="_5", outputCol="Recommendation 4", labels=model_item.labels)
recommendations = inverter.transform(recommendations).drop('_5')
inverter = IndexToString(inputCol="_6", outputCol="Recommendation 5", labels=model_item.labels)
recommendations = inverter.transform(recommendations).drop('_6')


# In[8]:


recommendations.show()

