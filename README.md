# *DSA2040A_DataMining_Group8*
*Group 8 project for DSA2040A: End-to-End Data Mining using Olist E-commerce Dataset*


## *Team Members & Contributions*

## *Team Members & Contributions*
*Group 8 – Domain: E-commerce*

| *Name*                   | *Student ID* | *GitHub Username*                          | *Role & Contribution*                          |
|--------------------------|--------------|---------------------------------------------|------------------------------------------------|
| *Mohamed Mohamed*        | *670006*     | [@mohayo8](https://github.com/mohayo8)               | *ETL Lead – Responsible for data cleaning & transformation (Notebook 1)*      |
| *Halima Mohammed*        | *670315*     | [@halima-04](https://github.com/halima-04)           | *Analyst – Leads EDA and data interpretation (Notebook 2)*        |
| *Lesala Phillip Monaheng*| *669218*     | [@Lesala](https://github.com/Lesala)                 | *Visualizer – Dashboards, charts, and final insights (Notebook 4)*      |
| *Snit Teshome*           | *670552*     | [@SnitTeshome](https://github.com/SnitTeshome)       | *Documenter & Data Mining – Report writing, classification/clustering (Notebook 3 & report)*      |

## *Project Summary*

*This project leverages the Olist E-commerce dataset, which represents a Brazilian marketplace that connects small and medium-sized retailers to customers across Brazil.*

The goal is to understand order behavior, delivery performance, product segmentation, and customer satisfaction by analyzing the complete pipeline: from ETL, EDA, Data Mining, to Dashboard development.*

*Key questions addressed include:*
- *What product categories are most profitable?*
- *How does discount affect sales and profit?*
- *Are there regional or order-priority trends?*
- *Can we segment customers or transactions using clustering?*

---

## *ETL Summary*


*The raw dataset is loaded and examined for structure and quality. Data types are corrected (e.g., date fields), missing values are handled, and new calculated columns are created (such as profit margin). Cleaned and structured data is then saved for further analysis.*

---

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


