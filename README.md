# AWS-Deployment-Retail-Demand-Forecasting
This repository provides a production-ready retail demand forecasting system using a trained deep learning model in TensorFlow (https://github.com/abolfazl6678/Retail-Demand-Forecasting-DeepLearning) and an API developed by FastAPI, fully deployed on AWS using Docker, ECR, and EC2.
It allows users to send product/store features through an API and receive real-time demand forecasts over the internet.

---

## Project Overview !!!!!!!!
Accurate  product demand forcast is a critical challenge in retail supply chains due to its direct effect on lost sales if it is lower and excess cost due to overstocking if it is higher. In this project, I used  Artificial Neural Networks (ANNs), one of the main methods in Deep Learning, to predict daily product demandin retail stores.
**Business Insights:** Forecasting results provide actionable insights to minimize stock-outs, reduce excess inventory, and optimize replenishment planning.

---
## Project Objectives and assumptions !!!!!!!!!!!!!!!

- Build a model to predict daily demand for each product 
- The model is built using Artificial Neural Networks (ANNs) - Deep Learning
- The model is trained by a dataset containing past sales, inventory levels, and product/store details given for each product in a given day
- **Modeling Assumption1:** Each day is treated as independent, allowing the task to be framed as standard supervised learning.
- **Modeling Assumption2:** Daily features (past sales, inventory, product/store attributes) are assumed to capture relevant patterns, removing the need for explicit time-series modeling.
- **Problem Formulation:** The model is formulated as a supervised regression problem, where daily product/store features are used to predict the expert demand forecast.
- **Frameworks:** The model is implemented using two major deep learning Python libraries: **PyTorch** and **TensorFlow**.
- **GPU:** The model is designed to automatically use GPU (if available) for faster training, evaluation, and prediction.
---
## Tools & Libraries used !!!!!!!!!!!!!
- **Programming Language:** Python
- **Python Libraries:** Numpy, Pandas, Matplotlib, PyTorch, TensorFlow, keras_tuner, optuna
- **Tool:** Google Colab with GPU, Git/GitHub

---
## Dataset & Variables !!!!!!!!!!!!
The dataset was provided in Kaggle (see Acknowledgments section) and includes detailed of retail store inventory as a table named **retail_store_inventory** with below varaibles.
- **number of Rows of the table:** ~73,000  
- **Columns (variabes):** 15

| Column Name             | Description                                                                                                 |
|-------------------------|-------------------------------------------------------------------------------------------------------------|
| `Date`                  | The specific day when the sales or demand data was recorded                                                 |
| `Store ID`              | A unique identifier for each store location                                                                 |
| `Product ID`            | A unique identifier for each product in the catalog                                                         |
| `Category`              | The classification of the product (e.g., electronics, toys, groceries)                                      |
| `Inventory Level`       | The number of products in inventory                                                                         |
| `Region`                | The geographical area where the store is located (e.g., North, South, West )                                |
| `Units Sold`            | The actual number of product units sold on the given date                                                   |
| `Units Ordered`         | The number of product units requested by the store or customers (may differ from units sold if out of stock)|
| `Demand Forecas`        | The predicted number of units expected to be sold, based on forecasting models (by experts)                 |
| `Price`                 | The selling price of the product per unit at the time of sale                                               |
| `Discount`              | Any reduction applied to the original product price (percentage or amount)                                  |
| `Weather Condition`     | Environmental factors such as sunny, rainy, which can influence demand                                      |
| `Holiday/Promotion`     | Indicates if the date coincided with a holiday, festival, or promotional campaign (binary )                 |
| `Competitor Pricing`    | Price of a similar product offered by competitors in the same region or market                              |
| `Seasonality`           | Cyclical or seasonal effects influencing sales patterns (e.g., Autumn, winter, spring)                      |

---
## Methodology !!!!!!!!!!!!!
1. **Data Preprocessing**  
   - Inspecting data
   - Handling missing values
   - Remiving duplicated rows
   - Data type conversion
   - Data scaling
     
2. **Model Development by TensorFlow**  
   - Fully connected ANN
   - Hyperparameter tunning by keras_tunner
   - Evaluate the best model
3. **Model Development by PyTorch**
   - Fully connected ANN
   - Hyperparameter tunning by keras_tunner
   - Evaluate the best model
4. **Evaluation Metrics** 
   - MSE (Mean Squared Error)  
   - R² Score
5. **Evaluation Metrics**
   - Save the best and trained model properly
---
## Results 
| Model                 |  MSE    |   R²    |
|-----------------------|---------|---------|
| ANN (TensorFlow)      |  81.13  |  0.99   |
| ANN (PyTorch)         |  75.71  |  0.99   |

---
## PyTorch vs TensorFlow – My Observations !!!!!!!!!!!!
- **Both frameworks** have shown strong results in prediction of demand forcast.
- **PyTorch:** More intuitive training loop, easier debugging.  
- **TensorFlow/Keras:** More concise, production-ready, easier deployment with TensorFlow Serving. 
---
## Project Structure !!!!!!!!!!!!!
```
Retail-Demand-Forecasting-DeepLearning/
├── data/
│ ├── raw/
│ │   └── retail_store_inventory.csv        #Dataset comes from Kaggle (see Acknowledgments for detail)
│ ├── interim/           
│ │   └── cleaned_data.parquet
│ └── processed/
├── jupyter_notebook_Scripts/
│     ├── 01_Data_Cleaning_inspection.ipynb
│     ├── 02_DeepLearning_model_tensorflow.ipynb 
│     └── 03_DeepLearning_model_pytorch.ipynb
├── output/
│     ├── 01_Data_Cleaning_inspection.docx
│     ├── 02_DeepLearning_model_tensorflow.docx
│     └── 03_DeepLearning_model_pytorch.docx
├── plots/
│     ├── Convergence_Curve_tf.png
│     ├── Residual_Plot_pt.png
│     ├── Residual_Plot_tf.png
│     ├── training_validation_loss_tf.png
│     ├── True_Actual_Demand_Forecast_pt.png
│     └── True_Actual_Demand_Forecast_tf.png
├── DL_model_pytorch.pkl
├── DL_model_pytorch.pkl
└── README.md

```


---
## Future Work !!!!!!!!!!!!!1
- Incorporate external features (holidays, promotions, weather)  
- Experiment with sequence models (RNN/LSTM/GRU) for time-series demand  
- Deploy the model with an API for real-time demand forecasting into cloud

---
## Acknowledgments !!!!!!!!!!
Dataset: [Kaggle-Retail Store Inventory Forecasting](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)  

---

## Author !!!!!!!!!!!!!

**Abolfazl Zolfaghari**  
[Email](ab.zolfaghari.abbasghaleh) | [GitHub](https://github.com/abolfazl6678)

---
