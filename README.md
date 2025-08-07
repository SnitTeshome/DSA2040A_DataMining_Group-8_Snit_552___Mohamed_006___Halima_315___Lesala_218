# *DSA2040A:Group 8 project End-to-End Data Mining using Olist E-commerce Dataset*


![alt text](output_screenshots/image-1.png)


---

## *Table of Contents*

* [*Project Overview*](#dsa2040agroup-8-project-end-to-end-data-mining-using-olist-e-commerce-dataset)
* [*Team Members & Contributions*](#team-members--contributions)
* [*Project Summary*](#project-summary)
* [*Project Folder Structure*](#project-folder-structure)
* [*Week-by-Week Progress*](#week-by-week-progress)

  * [*Week 1 & 2 – ETL Process*](#week-1--2--kickoff--dataset-selection-mohamed-mohamed1_extract_transformipynb)
  * [*Week 3 – Exploratory Analysis*](#week-3--exploratory--statistical-analysishalima-mohammed2_exploratory_analysisipynb)
  * [*Week 4 – Data Mining & NLP*](#week-4--data-minning-snit_teshome-3_data_miningipynb)
* [*Tools and Technologies*](#tools--technologies)
* [*How to Run*](#how-to-run)
* [*Data Source*](#data-source)
* [*License*](#license)

---


*This dataset, generously provided by Olist—the largest department store on Brazilian marketplaces—offers a comprehensive view of over 100,000 orders placed between 2016 and 2018. It captures multiple dimensions of the e-commerce experience, including order status, pricing, payment behavior, freight logistics, customer reviews, product categories, and geolocation.*

*Each order may include multiple items fulfilled by different sellers. The data has been anonymized, with store and partner names replaced by Game of Thrones great house names to preserve privacy. Additionally, a separate geolocation dataset enables mapping of Brazilian ZIP codes to latitude and longitude coordinates.*

*Olist connects small businesses across Brazil to larger marketplaces, streamlining sales and logistics through a single contract. After purchases are fulfilled, customers receive review requests, providing valuable feedback data.*

*Note: The dataset is divided into multiple linked tables for clarity and modular analysis.*

---



## *Team Members & Contributions*
*Group 8 – Domain: E-commerce*

| *Name*                   | *Student ID* | *GitHub Username*                          | *Role & Contribution*                          |
|--------------------------|--------------|---------------------------------------------|------------------------------------------------|
| *Mohamed Mohamed*        | *670006*     | [@mohayo8](https://github.com/mohayo8)               | *ETL Lead – Responsible for data cleaning & transformation 1_extract_transform.ipynb*      |
| *Halima Mohammed*        | *670315*     | [@halima-04](https://github.com/halima-04)           | *Analyst – Leads EDA and data interpretation 2_exploratory_analysis.ipynb  and executive_summary.pdf*        |
| *Lesala Phillip Monaheng*| *669218*     | [@Lesala](https://github.com/Lesala)                 | *Visualizer – Dashboards, charts, and final insights 4_insights_dashboard.ipynb*      |
| *Snit Teshome*           | *670552*     | [@SnitTeshome](https://github.com/SnitTeshome)       | *Documenter & Data Mining – Report writing, classification/clustering ── 3_data_mining.ipynb & README.md*      |

## *Project Summary*

*This project leverages the Olist E-commerce dataset, which represents a Brazilian marketplace that connects small and medium-sized retailers to customers across Brazil.*


![Output screenshot](output_screenshots/image-2.png)





---
### *Questions addressed include:*
*1.Can we build predictive models to classify review sentiment and identify high-value customer segments?*

*2.What actionable business insights can be visualized through dashboards for stakeholders?*
## *Project Folder Structure*

```
DataMining_GroupProject_Group8/
├── data/
│   ├── raw/                ← Place your original CSV file here
│   ├── transformed/        ← Cleaned dataset goes here
│   └── final/              ← Dataset used for mining/dashboards
├── notebooks/
│   ├── 1_extract_transform.ipynb
│   ├── 2_exploratory_analysis.ipynb
│   ├── 3_data_mining.ipynb
│   └── 4_insights_dashboard.ipynb
├── report/
│   ├── executive_summary.pdf
│   └── presentation.pptx
├── requirements.txt
├── .gitignore
└── README.md
```


# *`Week-by-Week Progress`*
# *Week 1 & 2 – Kickoff & Dataset Selection :Mohamed Mohamed:1_extract_transform.ipynb*
### *I. Import Required Libraries*

*Essential Python libraries such as `pandas`, `numpy`, and `scikit-learn` are imported for data manipulation, cleaning, and feature engineering.*




### *Define Data Paths*

*A raw data path is set using a relative directory reference to organize the ETL structure and simplify file access.*

```python
RAW_PATH = "../data/raw"
```

# *I. Extract*


*All CSV files in the raw data directory are located using `glob`, loaded into individual pandas DataFrames, and stored in a dictionary for structured access.*

```python
csv_files = glob(os.path.join(RAW_PATH, "*.csv"))
datasets = {}
for file in csv_files:
    base = os.path.basename(file).replace("olist_", "").replace(".csv", "")
    datasets[base] = pd.read_csv(file)
```


### *Preview Datasets*

*Each dataset is previewed by printing its name, shape (rows × columns), and the first few rows to understand its contents and schema.*

---

### *Inspect Data Structure*

*The `.info()` method is used on each DataFrame to examine data types, null values, and column structure for initial profiling.*
```python
for name, df in datasets.items():
    print(f"\n--- Structure of Dataset: {name.upper()} ---\n")
    df.info()
    print("\n" + "-" * 80)
```

![alt text](output_screenshots/image-3.png)

# *II.Transformation*
### *Data Cleaning & Transformation (Wrangling)*

*A `wrangle()` function is created to standardize and clean all datasets. This function performs the following operations:*

* *Column names are standardized to lowercase with underscores.*
* *Duplicate rows are removed.*
* *Columns with more than 50% missing values are dropped (exceptions apply).*
* *All date columns are parsed into datetime format.*
* *Whitespace is stripped from string-based fields.*
* *Low-cardinality object columns are converted to categorical types.*
* *Missing values are imputed: numeric columns with mean, and categorical/object columns with a constant placeholder.*
* *Columns relevant to customer and product categories are explicitly converted to categorical type.*

---
```python
def wrangle(df):
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df = df.drop_duplicates()
    exceptions = ['review_comment_title', 'review_comment_message''product_category_name']
    df = df.loc[:, (df.isnull().mean() < 0.5) | df.columns.isin(exceptions)]
    drop_cols = [
        "product_name_lenght", "product_description_lenght", "product_photos_qty",
        "payment_sequential", "review_id"
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    date_cols = [col for col in df.columns if "date" in col or "timestamp" in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda col: col.str.strip())
    cols_to_check = [
        'order_status', 'customer_city', 'customer_state',
        'product_category_name', 'seller_city', 'seller_state',
        'product_category_name_english'
    ]
    object_cols = df.select_dtypes(include="object").columns¬¬
    for col in cols_to_check:
        if col in object_cols:
            unique_vals = df[col].nunique()
            ratio = unique_vals / len(df)
            if ratio <= 0.05:
                df[col] = df[col].astype("category")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy="mean")
                df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    cat_object_cols = df.select_dtypes(include="object").columns
    if len(cat_object_cols) > 0:
        obj_imputer = SimpleImputer(strategy="constant", fill_value="unknown")
        df[cat_object_cols] = obj_imputer.fit_transform(df[cat_object_cols])
    for col in cols_to_check:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df
```


### *Apply Cleaning Function to All Datasets*

*The `wrangle()` function is applied to each DataFrame in the dataset collection to create a uniformly cleaned dataset collection.*

![alt text](output_screenshots/image-4.png)
---

### *Merge Cleaned Datasets*

*The cleaned datasets are sequentially merged into a unified dataset (`olist_full_data`) using appropriate foreign key relationships:*

* *Orders are merged with customers, items, payments, reviews, products, sellers, category translations, and geolocation data.*
* *The final DataFrame contains an enriched view of each transaction across time, geography, customer behavior, and product attributes.*


```python
olist_full_data = cleaned_datasets["orders_dataset"]
olist_full_data = olist_full_data.merge(cleaned_datasets["customers_dataset"], on="customer_id", how="left")
olist_full_data = olist_full_data.merge(cleaned_datasets["order_items_dataset"], on="order_id", how="left")
olist_full_data = olist_full_data.merge(cleaned_datasets["order_payments_dataset"], on="order_id", how="left")
olist_full_data = olist_full_data.merge(cleaned_datasets["order_reviews_dataset"], on="order_id", how="left")
olist_full_data = olist_full_data.merge(cleaned_datasets["products_dataset"], on="product_id", how="left")
olist_full_data = olist_full_data.merge(cleaned_datasets["sellers_dataset"], on="seller_id", how="left")
olist_full_data = olist_full_data.merge(cleaned_datasets["product_category_name_translation"], on="product_category_name", how="left")
olist_full_data = olist_full_data.merge(
    cleaned_datasets["geolocation_dataset"].drop_duplicates("geolocation_zip_code_prefix"),
    left_on="customer_zip_code_prefix",
    right_on="geolocation_zip_code_prefix",
    how="left"
)
```
![alt text](output_screenshots/image-5.png)
---

### *Final Data Cleaning*

*Post-merge cleanup includes logical imputation for key date fields, conversion of ZIP codes and order item IDs to string, and removal of rows with critical missing values.*

---

### *Feature Engineering*

*New features are created to enhance analytical capability:*

* *Financial metrics:* `profit_margin`, `freight_ratio`
* *Logistics metrics:* `estimated_delay`, `order_processing_time`
* *Customer segmentation metrics (RFM):* `recency_days`, `purchase_frequency`, `monetary_value`
* *Product-level metrics:* `product_volume_cm3`, `product_density`, `category_review_score`
* *Categorical flags:* `is_late`, `high_freight_flag`

---

### *Save Transformed Data*

*The final cleaned and enriched dataset is saved in both `.csv` and `.parquet` formats to support downstream modeling and dashboarding.*


---
![alt text](output_screenshots/image-6.png)

# *III.Loading*


 *Step 1: Read the Transformed Dataset*

 *Step 2: Load transformed_dataset  as Parquet*

 ![alt text](output_screenshots/image-7.png)

---
*step-3 Preview the full_data results*

#### *ETL Summary*
*The raw dataset is loaded and examined for structure and quality. Data types are corrected (e.g., date fields), missing values are handled, and new calculated columns are created (such as profit margin). Cleaned and structured data is then saved for further analysis.*

# *Week 3 – Exploratory & Statistical Analysis:Halima Mohammed:2_exploratory_analysis.ipynb*



### *Data Loading & Overview:*

*Basic methods for loading and previewing data:*

```python
df = pd.read_csv('your_data.csv')
df.info()
```
---

###  *Summary Statistics:*

*Generates descriptive statistics for numerical features:*

```python

numerical_df = df.select_dtypes(include=[np.number])
print(numerical_df.describe().round(2))
```

## *Summary Statistics Overview*

### *1. Price & Freight*

*Price and freight_value show extreme outliers. The maximum price is 6735, while the mean is 120. Freight value has a max of 409.68 with a mean of 20. This suggests a heavy-tailed distribution. Consider log transformation or outlier capping before modeling.*

---

### *2. Payment Installments & Payment Value*

*Customers pay in up to 24 installments, suggesting possible credit behavior. Payment value has a large maximum (13,664.08) and a mean of 172.43, indicating a right-skewed distribution. Scaling or transformation is advisable.*

---

### *3. Product Dimensions & Weight*

*Product weight, length, width, and height show high variability. The max volume reaches nearly 300,000 cm³, pointing to potential errors or very large items. Outlier handling is necessary here too.*

---

### *4. Profitability Metrics*

*Profit margin is extremely skewed, with a max of over 13,000 and a mean of 152. Freight ratio could be informative when comparing cost vs price. These variables may benefit from normalization or log scaling.*

---

### *5. Time-related Metrics*

*On average, deliveries are early by 11.91 days, though some are late by up to 188 days. Most orders are processed on the same day. Recency spans widely, with a mean of 242 days, indicating varying customer activity.*

---

### *6. Customer Behavior*

*Most customers are one-time buyers (purchase frequency median is 1), though some have bought up to 16 times. Monetary value is highly skewed (max over 100K). Average review scores per category are high (around 4).*

---

### *7. Flags*

*Only 6% of orders are marked as late. About 25% of transactions had high freight cost. These binary flags can be useful targets for classification models.*

---

### *Action Points*

- *Handle outliers in price, weight, freight, and profit-related features.*
- *Apply feature scaling or transformations on skewed variables.*
- *Consider customer segmentation using frequency, monetary value, and recency.*
- *Target prediction for flags like is_late or high_freight_flag is feasible.*
![alt text](output_screenshots/image-8.png)
---

## *Univariate Analysis:*

*Visualizes distributions using `histplot`, `boxplot`, `countplot` from Seaborn.*

*Highlights skewed distributions and common value ranges.*

```python
sns.histplot(data=df, x='price', bins=50, kde=True)
```
![alt text](output_screenshots/image-9.png)
---
![alt text](output_screenshots/image-10.png)
## *Bivariate Analysis:*

*Correlation analysis & visualizations:*

```python
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

```
![alt text](output_screenshots/Out_put%20screet_EDA%20_Dataminning/image.png)
---

*Focuses on:*

* *Relationships between `payment_value`, `price`, `profitability`*
* *Group differences across product categories and regions*

---
## *Boxplot of Price by Product Category*
![alt text](output_screenshots/Out_put%20screet_EDA%20_Dataminning/image-1.png)



## *1. Payment Value by Payment Type*

![alt text](output_screenshots/Out_put%20screet_EDA%20_Dataminning/image-2.png)


## *2. Estimated Delay by Seller State*
![alt text](output_screenshots/Out_put%20screet_EDA%20_Dataminning/image-3.png)


## *3. Monetary Value by Customer State*
![alt text](output_screenshots/Out_put%20screet_EDA%20_Dataminning/image-4.png)

## *Customer Behavior: RFM Segmentation*

*Analyzes Recency, Frequency, Monetary value:*

```python
sns.scatterplot(data=df, x='recency_days', y='monetary_value', size='purchase_frequency')

```

### *Time-Based Analysis*
### *Number of Purchases per Month*
![alt text](output_screenshots/Out_put%20screet_EDA%20_Dataminning/image-16.png)


---
*Detailed Breakdown of Trends*  

- *Initial Stage (September 2016 - January 2017)*:  
  The business begins with a *very low order volume*, starting near zero and growing slowly over the first few months. This is *typical for a new venture* finding its footing.

- *Rapid Growth Phase (February 2017 - November 2017)*:  
  Starting in early 2017, the business enters a period of *aggressive and consistent growth*. The number of orders climbs *steeply month-over-month*.  
  This rapid ascent culminates in a *sharp peak in November 2017*, with the number of orders *exceeding 6,000*. This peak is likely driven by *seasonal shopping events* like `*Black Friday* and *pre-holiday sales*.`

- *Maturity and High-Volume Plateau (December 2017 - August 2018)*:  
  Following the November peak, there is a *significant drop* in December 2017, though the volume remains *substantially higher* than the previous year.  
  For the remainder of the period shown, the business establishes a *new, higher baseline* for orders. The volume *consistently fluctuates* in a *high range*, mostly between *5,000 and 6,000 orders per month*.  
  This indicates that the growth achieved in 2017 was *not temporary*; the business has *successfully scaled* and is now operating at a *much higher level of activity*.


*Conclusion & Business Implication*  
The data clearly shows a business that has *successfully transitioned from a startup phase to a mature, high-volume operation*. The key challenge illustrated by the chart is *managing the seasonality and fluctuations* that come with a larger scale.  
The primary business implication is that *strategies should now focus on sustaining this high volume*, *optimizing operations for peak demand* (like the one seen in November), and *identifying new avenues for the next phase of growth*.


# *Week 4 – Data Minning: Snit_Teshome 3_data_mining.ipynb*
##  *Machine Learning & NLP: Clustering, Sentiment & Text Classification*
---

### *Data Loading & Preparation*

*Loaded e-commerce review, order, and product data.*
*Merged datasets to create a unified DataFrame for analysis.*

```python
df_reviews = pd.read_csv('.../olist_order_reviews_dataset.csv')
df_items = pd.read_csv('.../olist_order_items_dataset.csv')
df_products = pd.read_csv('.../olist_products_dataset.csv')
```

---

## *1.Clustering (Unsupervised Learning)*
### *I .Data Preprocessing for K-Means*
*Selected relevant numerical features (e.g., monetary value, purchase frequency, review score).*
*Scaled features and applied K-Means clustering.*
*Used the Elbow Method to determine optimal clusters (k=6).*
*Profiled clusters to understand customer segments.*
![alt text](output_screenshots/Out_put%20screet_EDA%20_Dataminning/image-5.png)
```python
features_to_cluster = ['monetary_value', 'purchase_frequency', 'recency_days', 'review_score', 'profit_margin']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features_to_cluster])
kmeans = KMeans(n_clusters=6, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_data)
```
### *Elbow Method Insights*
![alt text](output_screenshots/Out_put%20screet_EDA%20_Dataminning/image-6.png)

---

### *Key Observations:*

---

*🔵Cluster 0: Moderate overall metrics, with a strong review score (4.66), suggesting customers are fairly satisfied.*
*May be retained with regular engagement.*

*🔴Cluster 1: Low review score (1.59) despite fair monetary value and profit margin.*
*Could signal service/product issues that require quality improvement.*

*🟠Cluster 2: Extremely high monetary value (109,312.64) and profit margin (13,636.07) — but alarmingly low review score (1.00). Represents dissatisfied but valuable clients, possibly bulk buyers or B2B clients needing better service follow-up.*

*🟡Cluster 3: Similar in profile to Cluster 0, but extremely high recency (397.64) indicates inactivity.*
*Should be re-engaged through reactivation campaigns.*

*🟢Cluster 4: High monetary value (7,544.95) and strong profit margin (1,486.24) — possibly premium customers.*
*Slightly lower satisfaction (3.51), which may be improved to strengthen loyalty.*

*💙Cluster 5: High purchase frequency (7.11) with a solid review score (4.41) and decent monetary value.*
*Highly loyal and satisfied customers — best candidates for loyalty programs or special offers.*

---
#### *Overall Trend*

*Customer behavior is diverse and segmented—some are highly profitable but dissatisfied, others are loyal and engaged, while some are drifting away or minimally involved. Understanding these patterns supports more effective decision-making around retention, engagement, and resource allocation.*

### *Natural Language Processing (NLP) & Sentiment Analysis*

*Cleaned and preprocessed review text (removed duplicates, stopwords, punctuation).*
*Generated word clouds to visualize frequent terms in reviews.*
*Used VADER sentiment analysis to classify reviews as Positive, Neutral, or Negative.*
*Visualized sentiment distribution.*

```python
def clean_and_tokenize(text): ...
wordcloud = WordCloud(...).generate(' '.join(df['review_comment_message_clean'].dropna()))
analyzer = SentimentIntensityAnalyzer()
def classify_sentiment(text): ...
df['sentiment'] = df['review_comment_message_clean'].map(classify_sentiment)
```

---
![alt text](output_screenshots/image-12.png)
![alt text](output_screenshots/image-13.png)


![alt text](output_screenshots/Out_put%20screet_EDA%20_Dataminning/image-7.png)

![alt text](output_screenshots/Out_put%20screet_EDA%20_Dataminning/image-8.png)

![alt text](output_screenshots/Out_put%20screet_EDA%20_Dataminning/image-9.png)

### *Text Classification (Supervised Learning)*

*Defined features (review text) and target (sentiment label).*
*Split data into training and test sets.*
*Encoded sentiment labels and vectorized text using TF-IDF.*
*Trained a Logistic Regression model to predict sentiment.*
*Evaluated model with accuracy, confusion matrix, and classification report.*

### *Vectorization with Tfidf means*:
*Vectorization with Tfidf (Term Frequency-Inverse Document Frequency) is a technique used to convert text data into numerical vectors that can be used in machine learning model.* 
 
*🔹 TF-IDF (Term Frequency–Inverse Document Frequency):  
Balances how frequent a word is and how unique it is across the dataset.*

*Importance = Appears often in a given review but rarely in other reviews.*

*Helps filter out generic words and highlight more meaningful, context-specific ones.*

*TF-IDF Score = TF × IDF*
*Combining both:*

*Words with high TF but low IDF (like “the”, “good”) get low TF-IDF → filtered out as uninformative.*

*Words with high TF and high IDF (like “refund”, “delay”) get high TF-IDF → highlighted as important.*


![alt text](output_screenshots/Out_put%20screet_EDA%20_Dataminning/image-10.png)
```python
X = df['review_comment_message']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = LogisticRegression(...)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
```
## *Confusion Matrix*

![alt text](output_screenshots/Out_put%20screet_EDA%20_Dataminning/image-11.png)


### *Confusion Matrix Interpretation (3-Class Sentiment)*

- *Neutral (majority class)*: Excellent performance — **99.4% recall**, minimal confusion.
- *Positive*: Substantial improvement — **90% recall**, misclassifications reduced (esp. Neutral↔Positive).
- *Negative*: Major gains — **82% recall** (from ~23%), fewer errors.

## *Classification Report*
![alt text](output_screenshots/image-14.png)
---
*Given that  the daset is from e-commerce reviews, which metric (precision, recall, F1-score) matters more, especially when considering classes like Negative, Neutral, and Positive?*

*It Depends on Your Goal:*

*1. `Precision:`  
How many of the predicted labels are actually correct?*

*When it's important:*  
*If you're using predicted sentiment to trigger actions, e.g.:*  
*- Auto-responding to Negative reviews.*  
*- Giving discounts to Positive reviewers.*  
*- Flagging Negative reviews for manual moderation.*  

*You don’t want false positives — i.e., misclassifying Neutral/Positive reviews as Negative and wasting resources.*

*📌 In e-commerce, precision is very important for Negative sentiment.  
You don’t want to wrongly treat Neutral/Positive feedback as a crisis.*

*2.` Recall:`  
Out of all actual reviews of a type, how many did we catch?*

*When it's important:*  
*If your goal is not to miss any Negative reviews.*  
*For example, you're doing brand monitoring, and you want to capture every Negative sentiment — even at the risk of over-alerting.*

*📌 In e-commerce, recall is important when customer satisfaction monitoring or crisis management is the goal.*

*3. `F1-score:` 
Harmonic mean of Precision and Recall.  
Balances both.*

*When it's important:*  
*When both precision and recall matter, like in general sentiment analysis dashboards.*  
*If the dataset is imbalanced (like yours — mostly Neutral), F1-score helps you measure how well the model is doing beyond just accuracy.*

*📌 For overall model comparison, F1-score is a safe and fair metric.*

**Overall Accuracy**: *98.5%

## *Tools & Technologies*

- *Python* (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLTK, spaCy, Transformers)
- *Jupyter Notebook*
- *Git & GitHub* for version control and collaboration
- *Plotly*, *Seaborn*, and *Power BI* for data visualization

# *Week 5: – Lesala Phillip Monaheng:4_insights_dashboard.ipynb*

*Created an interactive dashboard to visualize key insights from the analysis.*

![alt text](report/image-1.png)

---

![alt text](report/image-2.png)





## *How to Run*04_insights_dashboard

*Clone the repository Full Workflow (Cloning → Making Changes → Committing → Pushing)*

```bash
git clone https://github.com/SnitTeshome/DSA2040A_DataMining_Group-8_Snit_552___Mohamed_006___Halima_315___Lesala_218.git
cd DSA2040A_DataMining_Group-8_Snit_552___Mohamed_006___Halima_315___Lesala_218
```
``` bash
git status
git add .
git commit -m "Your commit message"
git push origin main
```

## *Data Source*

*Dataset used:* [**Olist E-commerce Dataset**](https://www.kaggle.com/code/rasikagurav/brazilian-e-commerce-eda-nlp)  



---

### *License*

*This project is licensed under the* [MIT License](https://github.com/SnitTeshome/DSA2040A_DataMining_Group-8_Snit_552___Mohamed_006___Halima_315___Lesala_218/blob/main/LICENSE).
*You are free to use, copy, modify, and distribute this software, provided the original license is included.*

---


