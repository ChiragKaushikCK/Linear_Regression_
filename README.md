# Linear_Regression_

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Matplotlib-EE6633?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
</div>

<h1 align="center" style="color: #28A745; font-family: 'Segoe UI', sans-serif; font-size: 3.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
  <br>
  <a href="https://github.com/your-username/simple-linear-regression-salary-prediction">
    <img src="https://placehold.co/600x200/28A745/FFFFFF?text=Simple+Linear+Regression" alt="Simple Linear Regression Banner" style="border-radius: 15px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
  </a>
  <br>
  📈 Simple Linear Regression for Salary Prediction 💰
  <br>
</h1>

<p align="center" style="font-size: 1.2em; color: #555; max-width: 800px; margin: 0 auto; line-height: 1.6;">
  Uncover the relationship between years of experience and salary with this **Simple Linear Regression** Jupyter Notebook! This project demonstrates a fundamental machine learning algorithm used for predicting a continuous outcome based on a single input variable. It covers data loading, preprocessing (including handling missing values), splitting data, model training, and visualizing the regression line. Ideal for beginners in machine learning and data science! 🚀
</p>

<br>

<details style="background-color: #E6F7FF; border-left: 5px solid #28A745; padding: 15px; border-radius: 8px; margin: 20px auto; max-width: 700px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
  <summary style="font-size: 1.3em; font-weight: bold; color: #333; cursor: pointer;">Table of Contents</summary>
  <ol style="list-style-type: decimal; padding-left: 25px; line-height: 1.8;">
    <li><a href="#about-the-project" style="color: #28A745; text-decoration: none;">📚 About The Project</a></li>
    <li><a href="#dataset" style="color: #28A745; text-decoration: none;">📦 Dataset</a></li>
    <li><a href="#modeling-approach" style="color: #28A745; text-decoration: none;">⚙️ Modeling Approach</a></li>
    <li><a href="#features" style="color: #28A745; text-decoration: none;">🎯 Features</a></li>
    <li><a href="#prerequisites" style="color: #28A745; text-decoration: none;">🛠️ Prerequisites</a></li>
    <li><a href="#how-to-run" style="color: #28A745; text-decoration: none;">📋 How to Run</a></li>
    <li><a href="#example-output" style="color: #28A745; text-decoration: none;">📈 Example Output</a></li>
    <li><a href="#code-breakdown" style="color: #28A745; text-decoration: none;">🧠 Code Breakdown</a></li>
    <li><a href="#contribute" style="color: #28A745; text-decoration: none;">🤝 Contribute</a></li>
  </ol>
</details>

---

<h2 id="about-the-project" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  📚 About The Project
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  This project implements **Simple Linear Regression**, a basic yet powerful statistical method used to model the relationship between two continuous variables. [cite: uploaded:1. Simple Linear Regression.ipynb] In this notebook, the goal is to predict an individual's salary based on their years of experience. The notebook covers the essential steps of a typical machine learning workflow:
</p>
<ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
  <li style="margin-bottom: 10px; background-color: #E6F7FF; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Data Loading:</strong> Reading the dataset containing years of experience and salary.
  </li>
  <li style="margin-bottom: 10px; background-color: #E6F7FF; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Feature Separation:</strong> Clearly defining independent (YearsExperience) and dependent (Salary) variables.
  </li>
  <li style="margin-bottom: 10px; background-color: #E6F7FF; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Missing Value Handling:</strong> Demonstrating how to impute missing values using the mean strategy.
  </li>
  <li style="margin-bottom: 10px; background-color: #E6F7FF; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Train-Test Split:</strong> Dividing the dataset into training and testing sets to evaluate model performance.
  </li>
  <li style="margin-bottom: 10px; background-color: #E6F7FF; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Model Training & Prediction:</strong> Building and training a `LinearRegression` model and making predictions.
  </li>
  <li style="margin-bottom: 10px; background-color: #E6F7FF; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Visualization:</strong> Plotting the regression line against actual data points to visually assess the model's fit.
  </li>
</ul>

---

<h2 id="dataset" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  📦 Dataset
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  The project uses a dataset named `Salary_Data-2.csv` (or similar). [cite: uploaded:1. Simple Linear Regression.ipynb] This dataset is expected to contain at least two columns:
</p>
<ul style="list-style-type: disc; padding-left: 20px; font-size: 1.1em; color: #444;">
  <li><strong style="color: #28A745;">`YearsExperience` (Independent Variable):</strong> Represents the number of years an individual has worked.</li>
  <li><strong style="color: #28A745;">`Salary` (Dependent Variable):</strong> Represents the salary earned by the individual.</li>
</ul>
<p style="font-size: 1.1em; color: #444; line-height: 1.6; margin-top: 10px;">
  An example of the initial data structure is shown below:
</p>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #9CDCFE;">YearsExperience</span>  <span style="color: #9CDCFE;">Salary</span>
<span style="color: #B5CEA8;">1.1</span>              <span style="color: #B5CEA8;">39343.0</span>
<span style="color: #B5CEA8;">1.3</span>              <span style="color: #B5CEA8;">46205.0</span>
<span style="color: #B5CEA8;">1.5</span>              <span style="color: #B5CEA8;">37731.0</span>
<span style="color: #B5CEA8;">2.0</span>              <span style="color: #B5CEA8;">43525.0</span>
<span style="color: #B5CEA8;">...</span></code></pre>

---

<h2 id="modeling-approach" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  ⚙️ Modeling Approach
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  This notebook applies **Simple Linear Regression**, which models the relationship between the independent variable ($X$) and the dependent variable ($Y$) as a straight line:
</p>
<p align="center" style="font-size: 1.3em; font-weight: bold; color: #28A745; margin: 20px 0;">
  $$ Y = \beta_0 + \beta_1 X + \epsilon $$
</p>
<ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
  <li style="margin-bottom: 10px; background-color: #D4EDDA; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">$\beta_0$:</strong> The Y-intercept, representing the expected value of Y when X is 0.
  </li>
  <li style="margin-bottom: 10px; background-color: #D4EDDA; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">$\beta_1$:</strong> The slope, representing the change in Y for a one-unit change in X.
  </li>
  <li style="margin-bottom: 10px; background-color: #D4EDDA; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">$\epsilon$:</strong> The error term, accounting for the variability in Y that cannot be explained by X.
  </li>
</ul>
<p style="font-size: 1.1em; color: #444; line-height: 1.6; margin-top: 15px;">
  The model is trained using the Ordinary Least Squares (OLS) method, which minimizes the sum of the squared differences between the observed and predicted values.
</p>

---

<h2 id="features" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  🎯 Features
</h2>
<ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
  <li style="margin-bottom: 15px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">🚀 End-to-End Regression:</strong> From data loading and preprocessing to model training, prediction, and visualization. [cite: uploaded:1. Simple Linear Regression.ipynb]
  </li>
  <li style="margin-bottom: 15px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">🧹 Missing Value Imputation:</strong> Demonstrates a practical approach to handling missing data using `SimpleImputer`. [cite: uploaded:1. Simple Linear Regression.ipynb]
  </li>
  <li style="margin-bottom: 15px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">📊 Clear Visualization:</strong> Provides an intuitive scatter plot of actual vs. predicted values with the regression line. [cite: uploaded:1. Simple Linear Regression.ipynb]
  </li>
  <li style="margin-bottom: 15px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">✍️ Concise & Understandable Code:</strong> Designed for clarity, making it easy to follow the linear regression process.
  </li>
</ul>

---

<h2 id="prerequisites" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  🛠️ Prerequisites
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  To run this project, ensure you have the following installed:
</p>
<ul style="list-style-type: disc; padding-left: 20px; font-size: 1.1em; color: #444;">
  <li><strong style="color: #28A745;">Python 3.x</strong></li>
  <li><strong style="color: #28A745;">Jupyter Notebook</strong> (or JupyterLab, Google Colab)</li>
  <li>Required Libraries:
    <pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code>pip install pandas numpy scikit-learn matplotlib</code></pre>
  </li>
  <li>**Spark (Optional but present in notebook):** The notebook initially uses `spark.read.csv`. If running locally without Spark, you might need to adjust the data loading to `pd.read_csv` directly.</li>
</ul>

---

<h2 id="how-to-run" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  📋 How to Run
</h2>
<ol style="list-style-type: decimal; padding-left: 20px; font-size: 1.1em; color: #444; line-height: 1.8;">
  <li style="margin-bottom: 10px;">
    <strong style="color: #28A745;">Download the Notebook:</strong>
    <p style="margin-top: 5px;">Download <code>1. Simple Linear Regression.ipynb</code> from this repository.</p>
    <p style="margin-top: 5px;">Alternatively, open it directly in <a href="https://colab.research.google.com/" style="color: #28A745; text-decoration: none;">Google Colab</a> for a zero-setup experience (ensure you upload the dataset if running there).</p>
  </li>
  <li style="margin-bottom: 10px;">
    <strong style="color: #28A745;">Prepare Data:</strong>
    <p style="margin-top: 5px;">Ensure you have your dataset named `Salary_Data-2.csv` (or adjust the file path in the notebook's `pd.read_csv()` or `spark.read.csv()` call) in the same directory as the notebook.</p>
  </li>
  <li style="margin-bottom: 10px;">
    <strong style="color: #28A745;">Install Dependencies:</strong>
    <pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code>pip install pandas numpy scikit-learn matplotlib</code></pre>
    (If using Spark, ensure your Spark environment is correctly set up.)
  </li>
  <li style="margin-bottom: 10px;">
    <strong style="color: #28A745;">Run the Notebook:</strong>
    <p style="margin-top: 5px;">Open <code>1. Simple Linear Regression.ipynb</code> in Jupyter or Colab.</p>
    <p style="margin-top: 5px;">Execute each cell sequentially to perform the linear regression and see the results!</p>
  </li>
</ol>

---

<h2 id="example-output" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  📈 Example Output
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  The primary output will be a scatter plot showing the actual salary points and the fitted regression line.
</p>
<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Salary Prediction Plot:</h3>
<div style="text-align: center; background-color: #F8F8F8; padding: 15px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-top: 20px;">
  <img src="https://placehold.co/600x400/D4EDDA/333333?text=Salary+Prediction+Plot" alt="Salary Prediction Plot Placeholder" style="width: 100%; height: auto; border-radius: 8px; border: 1px solid #ddd;">
  <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
    A scatter plot showing `YearsExperience` vs. `Salary`, with the red dots representing actual test data points and the green line representing the predicted salary by the linear regression model.
  </p>
</div>
<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Sample Predictions vs. Actuals:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #6A9955;"># Sample Predicted Salaries (y_pred)</span>
<span style="color: #B5CEA8;">array([ 75164.26, 114405.22, 101629.09,  46874.26, 113492.64,  91590.71])</span>

<span style="color: #6A9955;"># Sample Actual Salaries (y_test)</span>
<span style="color: #B5CEA8;">array([ 83088., 112635., 113812.,  39891., 116969.,  98273.])</span></code></pre>

---

<h2 id="code-breakdown" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  🧠 Code Breakdown
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  Key parts of the notebook's code structure:
</p>

<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Data Loading & Initial Inspection:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #569CD6;">import</span> <span style="color: #9CDCFE;">pandas</span> <span style="color: #C586C0;">as</span> <span style="color: #9CDCFE;">pd</span>
<span style="color: #569CD6;">import</span> <span style="color: #9CDCFE;">numpy</span> <span style="color: #C586C0;">as</span> <span style="color: #9CDCFE;">np</span>

<span style="color: #6A9955;"># If running in a Spark environment (like Databricks)</span>
<span style="color: #9CDCFE;">df</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">spark.read.option</span>(<span style="color: #CE9178;">"header"</span>,<span style="color: #9CDCFE;">True</span>).<span style="color: #9CDCFE;">option</span>(<span style="color: #CE9178;">"inferSchema"</span>,<span style="color: #9CDCFE;">True</span>).<span style="color: #9CDCFE;">csv</span>(<span style="color: #CE9178;">"/FileStore/tables/Salary_Data-2.csv"</span>)
<span style="color: #9CDCFE;">df</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">df.toPandas</span>()

<span style="color: #6A9955;"># If running locally, you might directly use:</span>
<span style="color: #6A9955;"># df = pd.read_csv('Salary_Data-2.csv')</span>
<span style="color: #9CDCFE;">df</span></code></pre>

<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Separating Variables & Handling Missing Values:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #9CDCFE;">X</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">df.iloc</span>[:, <span style="color: #B5CEA8;">0</span>:<span style="color: #B5CEA8;">1</span>].<span style="color: #9CDCFE;">values</span>
<span style="color: #9CDCFE;">y</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">df.iloc</span>[:,<span style="color: #B5CEA8;">1</span>].<span style="color: #9CDCFE;">values</span>

<span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">sklearn.impute</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">SimpleImputer</span>
<span style="color: #9CDCFE;">imputer</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">SimpleImputer</span>(<span style="color: #9CDCFE;">missing_values</span><span style="color: #CE9178;">=</span><span style="color: #569CD6;">np.NaN</span>,<span style="color: #9CDCFE;">strategy</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">"mean"</span>)
<span style="color: #9CDCFE;">imputer</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">imputer.fit</span>(<span style="color: #9CDCFE;">df.iloc</span>[:,<span style="color: #B5CEA8;">0</span>:<span style="color: #B5CEA8;">2</span>])
<span style="color: #9CDCFE;">df.iloc</span>[:,<span style="color: #B5CEA8;">0</span>:<span style="color: #B5CEA8;">2</span>]<span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">imputer.transform</span>(<span style="color: #9CDCFE;">df.iloc</span>[:,<span style="color: #B5CEA8;">0</span>:<span style="color: #B5CEA8;">2</span>])</code></pre>

<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Train-Test Split & Model Training:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">sklearn.model_selection</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">train_test_split</span>
<span style="color: #9CDCFE;">X_train</span>,<span style="color: #9CDCFE;">X_test</span>,<span style="color: #9CDCFE;">y_train</span>,<span style="color: #9CDCFE;">y_test</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">train_test_split</span>(<span style="color: #9CDCFE;">X</span>,<span style="color: #9CDCFE;">y</span>,<span style="color: #9CDCFE;">test_size</span><span style="color: #CE9178;">=</span><span style="color: #B5CEA8;">0.2</span>)

<span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">sklearn.linear_model</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">LinearRegression</span>
<span style="color: #9CDCFE;">model1</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">LinearRegression</span>()
<span style="color: #9CDCFE;">model1.fit</span>(<span style="color: #9CDCFE;">X_train</span>,<span style="color: #9CDCFE;">y_train</span>)

<span style="color: #9CDCFE;">y_pred</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">model1.predict</span>(<span style="color: #9CDCFE;">X_test</span>)</code></pre>

<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Visualization:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #569CD6;">import</span> <span style="color: #9CDCFE;">matplotlib.pyplot</span> <span style="color: #C586C0;">as</span> <span style="color: #9CDCFE;">plt</span>

<span style="color: #9CDCFE;">plt.scatter</span>(<span style="color: #9CDCFE;">X_test.astype</span>(<span style="color: #CE9178;">'float'</span>),<span style="color: #9CDCFE;">y_test.astype</span>(<span style="color: #CE9178;">'float'</span>),<span style="color: #9CDCFE;">color</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'red'</span>)
<span style="color: #9CDCFE;">plt.plot</span>(<span style="color: #9CDCFE;">X_test</span>,<span style="color: #9CDCFE;">y_pred</span>,<span style="color: #9CDCFE;">color</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'green'</span>)
<span style="color: #9CDCFE;">plt.xlabel</span>(<span style="color: #CE9178;">"Experience"</span>)
<span style="color: #9CDCFE;">plt.ylabel</span>(<span style="color: #CE9178;">"Salary"</span>)
<span style="color: #9CDCFE;">plt.title</span>(<span style="color: #CE9178;">"Salary Prediction"</span>)
<span style="color: #9CDCFE;">plt.show</span>()</code></pre>

---

<h2 id="contribute" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  🤝 Contribute
</h2>
<p align="center" style="font-size: 1.2em; color: #555; max-width: 800px; margin: 0 auto; line-height: 1.6;">
  Contributions are welcome! If you have ideas for improving the code, adding more detailed analysis (e.g., R-squared, Mean Squared Error), or exploring more advanced regression techniques, feel free to open an issue or submit a pull request. Let's enhance this foundational machine learning project together! 🌟
</p>
<p align="center" style="font-size: 1.2em; color: #555; max-width: 800px; margin: 15px auto 0; line-height: 1.6;">
  Star this repo if you find it helpful! ⭐
</p>
<p align="center" style="font-size: 1em; color: #777; margin-top: 30px;">
  Created with 💖 by Chirag
</p>
