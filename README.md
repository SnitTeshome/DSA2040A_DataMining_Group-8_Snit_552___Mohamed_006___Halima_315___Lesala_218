# *DSA2040A:Group 8 project End-to-End Data Mining using Olist E-commerce Dataset*


![alt text](image-1.png)

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

![alt text](image-2.png)




---
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
## *Week 1 – Kickoff & Dataset Selection*



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

![alt text](image-3.png)

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

```python
cleaned_datasets = {}
for name, df in datasets.items():
    print(f"{name}  has been cleaned successfully.")
    cleaned_datasets[name] = wrangle(df)
```
![alt text](image-4.png)
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
![alt text](image-5.png)
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
![alt text](image-6.png)

# *III.Loading*


 *Step 1: Read the Transformed Dataset*

 *Step 2: Load transformed_dataset  as Parquet*

 ![alt text](image-7.png)

---
*step-3 Preview the full_data results*

## *ETL Summary*
*The raw dataset is loaded and examined for structure and quality. Data types are corrected (e.g., date fields), missing values are handled, and new calculated columns are created (such as profit margin). Cleaned and structured data is then saved for further analysis.*
## *Tools & Technologies*

- *Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)*
- *Jupyter Notebook*
- *GitHub for collaboration*
- *Plotly/Seaborn/Power BI for visualization*

---

## *How to Run*

*1. Clone the repository:*

```bash

git clone https://github.com/your-username/DSA2040A_DataMining_Group8.git
cd DSA2040A_DataMining_Group8
```
## *Data Source*

*Dataset used:* [**Olist E-commerce Dataset**](https://www.kaggle.com/code/rasikagurav/brazilian-e-commerce-eda-nlp)  
*Description:* This dataset contains nearly 10,000 records of retail orders, including order dates, product details, customer information, sales, profit, discount, and region. It’s widely used for teaching data analysis and visualization.

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


