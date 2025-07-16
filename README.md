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
  <a href="https://github.com/your-username/linear-regression-models">
    <img src="https://placehold.co/600x200/28A745/FFFFFF?text=Linear+Regression+Models" alt="Linear Regression Models Banner" style="border-radius: 15px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
  </a>
  <br>
  📈 Simple & Multiple Linear Regression for Prediction 💰
  <br>
</h1>

<p align="center" style="font-size: 1.2em; color: #555; max-width: 800px; margin: 0 auto; line-height: 1.6;">
  Explore the power of linear regression with this comprehensive Jupyter Notebook! This project demonstrates both **Simple Linear Regression** (predicting salary based on experience) and **Multiple Linear Regression** (predicting startup profit based on R&D, Administration, Marketing Spend, and State). It covers data loading, preprocessing (including handling missing values and categorical features), data splitting, model training, and visualization. Ideal for anyone looking to understand and apply linear regression in real-world scenarios! 🚀
</p>

<br>

<details style="background-color: #E6F7FF; border-left: 5px solid #28A745; padding: 15px; border-radius: 8px; margin: 20px auto; max-width: 700px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
  <summary style="font-size: 1.3em; font-weight: bold; color: #333; cursor: pointer;">Table of Contents</summary>
  <ol style="list-style-type: decimal; padding-left: 25px; line-height: 1.8;">
    <li><a href="#about-the-project" style="color: #28A745; text-decoration: none;">📚 About The Project</a></li>
    <li><a href="#simple-linear-regression" style="color: #28A745; text-decoration: none;">📊 Simple Linear Regression (Salary Prediction)</a></li>
    <li><a href="#multiple-linear-regression" style="color: #28A745; text-decoration: none;">📈 Multiple Linear Regression (Startup Profit Prediction)</a></li>
    <li><a href="#prerequisites" style="color: #28A745; text-decoration: none;">🛠️ Prerequisites</a></li>
    <li><a href="#how-to-run" style="color: #28A745; text-decoration: none;">📋 How to Run</a></li>
    <li><a href="#code-breakdown" style="color: #28A745; text-decoration: none;">🧠 Code Breakdown</a></li>
    <li><a href="#contribute" style="color: #28A745; text-decoration: none;">🤝 Contribute</a></li>
  </ol>
</details>

---

<h2 id="about-the-project" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  📚 About The Project
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  This project serves as an introduction to linear regression, demonstrating its application in two distinct scenarios:
</p>
<ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
  <li style="margin-bottom: 10px; background-color: #E6F7FF; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Simple Linear Regression:</strong> Models the relationship between a single independent variable and a dependent variable. Here, we predict salary based on years of experience.
  </li>
  <li style="margin-bottom: 10px; background-color: #E6F7FF; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Multiple Linear Regression:</strong> Extends the concept to model the relationship between multiple independent variables and a dependent variable. Here, we predict startup profit using R&D Spend, Administration, Marketing Spend, and State.
  </li>
</ul>
<p style="font-size: 1.1em; color: #444; line-height: 1.6; margin-top: 15px;">
  Both models follow a standard machine learning workflow, including data loading, preprocessing, splitting into training and testing sets, model training, prediction, and evaluation.
</p>

---

<h2 id="simple-linear-regression" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  📊 Simple Linear Regression (Salary Prediction)
</h2>
<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Dataset:</h3>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  This section uses a dataset named `Salary_Data-2.csv`. [cite: uploaded:1. Simple Linear Regression.ipynb] It contains:
</p>
<ul style="list-style-type: disc; padding-left: 20px; font-size: 1.1em; color: #444;">
  <li><strong style="color: #28A745;">`YearsExperience` (Independent Variable):</strong> Number of years worked.</li>
  <li><strong style="color: #28A745;">`Salary` (Dependent Variable):</strong> Salary earned.</li>
</ul>
<p style="font-size: 1.1em; color: #444; line-height: 1.6; margin-top: 10px;">
  **Key Steps:**
</p>
<ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
  <li style="margin-bottom: 10px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Data Loading & Inspection:</strong> Reads the CSV and checks for basic information and missing values.
  </li>
  <li style="margin-bottom: 10px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Missing Value Handling:</strong> Uses `SimpleImputer` to fill any missing values in the dataset using the mean strategy. [cite: uploaded:1. Simple Linear Regression.ipynb]
  </li>
  <li style="margin-bottom: 10px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Train-Test Split:</strong> Divides the data into 80% training and 20% testing sets. [cite: uploaded:1. Simple Linear Regression.ipynb]
  </li>
  <li style="margin-bottom: 10px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Model Training:</strong> A `LinearRegression` model from `sklearn.linear_model` is trained on the training data. [cite: uploaded:1. Simple Linear Regression.ipynb]
  </li>
  <li style="margin-bottom: 10px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Prediction & Visualization:</strong> Predictions are made on the test set, and a scatter plot visualizes the actual data points against the fitted regression line. [cite: uploaded:1. Simple Linear Regression.ipynb]
  </li>
</ul>
<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Mathematical Model:</h3>
<p align="center" style="font-size: 1.3em; font-weight: bold; color: #28A745; margin: 20px 0;">
  $$ Y = \beta_0 + \beta_1 X + \epsilon $$
</p>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  Where:
  <ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
    <li style="margin-bottom: 5px;"><strong style="color: #28A745;">$Y$:</strong> Dependent variable (Salary)</li>
    <li style="margin-bottom: 5px;"><strong style="color: #28A745;">$X$:</strong> Independent variable (YearsExperience)</li>
    <li style="margin-bottom: 5px;"><strong style="color: #28A745;">$\beta_0$:</strong> Y-intercept</li>
    <li style="margin-bottom: 5px;"><strong style="color: #28A745;">$\beta_1$:</strong> Coefficient for X (slope)</li>
    <li style="margin-bottom: 5px;"><strong style="color: #28A745;">$\epsilon$:</strong> Error term</li>
  </ul>
</p>
<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Example Output (Simple Linear Regression):</h3>
<div style="text-align: center; background-color: #F8F8F8; padding: 15px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-top: 20px;">
  <img src="https://placehold.co/600x400/D4EDDA/333333?text=Salary+Prediction+Plot" alt="Salary Prediction Plot Placeholder" style="width: 100%; height: auto; border-radius: 8px; border: 1px solid #ddd;">
  <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
    A scatter plot showing `YearsExperience` vs. `Salary`, with actual test data points and the regression line.
  </p>
</div>

---

<h2 id="multiple-linear-regression" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  📈 Multiple Linear Regression (Startup Profit Prediction)
</h2>
<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Dataset:</h3>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  This section utilizes the `50_Startups.csv` dataset. [cite: uploaded:2. Multi Linear Regression.ipynb] It includes:
</p>
<ul style="list-style-type: disc; padding-left: 20px; font-size: 1.1em; color: #444;">
  <li><strong style="color: #28A745;">`R&D Spend` (Independent Variable)</strong></li>
  <li><strong style="color: #28A745;">`Administration` (Independent Variable)</strong></li>
  <li><strong style="color: #28A745;">`Marketing Spend` (Independent Variable)</strong></li>
  <li><strong style="color: #28A745;">`State` (Categorical Independent Variable)</strong></li>
  <li><strong style="color: #28A745;">`Profit` (Dependent Variable)</strong></li>
</ul>
<p style="font-size: 1.1em; color: #444; line-height: 1.6; margin-top: 10px;">
  **Key Steps:**
</p>
<ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
  <li style="margin-bottom: 10px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Data Loading & Inspection:</strong> Reads the CSV and checks data types and null values. [cite: uploaded:2. Multi Linear Regression.ipynb]
  </li>
  <li style="margin-bottom: 10px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Categorical Encoding:</strong> Converts the `State` categorical column into numerical format using one-hot encoding (`pd.get_dummies`). [cite: uploaded:2. Multi Linear Regression.ipynb]
  </li>
  <li style="margin-bottom: 10px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Outlier Handling:</strong> Includes a custom function `Handling_Outliers` to remove outliers based on the Interquartile Range (IQR) method. [cite: uploaded:2. Multi Linear Regression.ipynb]
  </li>
  <li style="margin-bottom: 10px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Feature Selection (Backward Elimination):</strong> Demonstrates a manual backward elimination process using `statsmodels.api.OLS` to identify statistically significant features. [cite: uploaded:2. Multi Linear Regression.ipynb]
  </li>
  <li style="margin-bottom: 10px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Train-Test Split:</strong> Divides the data into training and testing sets. [cite: uploaded:2. Multi Linear Regression.ipynb]
  </li>
  <li style="margin-bottom: 10px; background-color: #F0FFF0; padding: 10px 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);">
    <strong style="color: #28A745;">Model Training & Evaluation:</strong> A `LinearRegression` model is trained, and its performance is evaluated.
  </li>
</ul>
<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Mathematical Model:</h3>
<p align="center" style="font-size: 1.3em; font-weight: bold; color: #28A745; margin: 20px 0;">
  $$ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon $$
</p>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  Where:
  <ul style="list-style-type: none; padding: 0; font-size: 1.1em; color: #444;">
    <li style="margin-bottom: 5px;"><strong style="color: #28A745;">$Y$:</strong> Dependent variable (Profit)</li>
    <li style="margin-bottom: 5px;"><strong style="color: #28A745;">$X_i$:</strong> Independent variables (R&D Spend, Administration, Marketing Spend, State_Florida, State_New York)</li>
    <li style="margin-bottom: 5px;"><strong style="color: #28A745;">$\beta_0$:</strong> Y-intercept</li>
    <li style="margin-bottom: 5px;"><strong style="color: #28A745;">$\beta_i$:</strong> Coefficients for each independent variable</li>
    <li style="margin-bottom: 5px;"><strong style="color: #28A745;">$\epsilon$:</strong> Error term</li>
  </ul>
</p>
<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Example Output (Multiple Linear Regression - Data Head):</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #9CDCFE;">R&D Spend</span>  <span style="color: #9CDCFE;">Administration</span>  <span style="color: #9CDCFE;">Marketing Spend</span>  <span style="color: #9CDCFE;">Profit</span>  <span style="color: #9CDCFE;">State_Florida</span>  <span style="color: #9CDCFE;">State_New York</span>
<span style="color: #B5CEA8;">165349.20</span>  <span style="color: #B5CEA8;">136897.80</span>       <span style="color: #B5CEA8;">471784.10</span>        <span style="color: #B5CEA8;">192261.83</span>  <span style="color: #B5CEA8;">0</span>              <span style="color: #B5CEA8;">1</span>
<span style="color: #B5CEA8;">162597.70</span>  <span style="color: #B5CEA8;">151377.59</span>       <span style="color: #B5CEA8;">443898.53</span>        <span style="color: #B5CEA8;">191792.06</span>  <span style="color: #B5CEA8;">0</span>              <span style="color: #B5CEA8;">0</span>
<span style="color: #B5CEA8;">...</span></code></pre>

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
    <pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code>pip install pandas numpy scikit-learn matplotlib statsmodels</code></pre>
  </li>
  <li>**Spark (Optional but present in notebooks):** The notebooks initially use `spark.read.csv`. If running locally without Spark, you might need to adjust the data loading to `pd.read_csv` directly.</li>
</ul>

---

<h2 id="how-to-run" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  📋 How to Run
</h2>
<ol style="list-style-type: decimal; padding-left: 20px; font-size: 1.1em; color: #444; line-height: 1.8;">
  <li style="margin-bottom: 10px;">
    <strong style="color: #28A745;">Download the Notebooks:</strong>
    <p style="margin-top: 5px;">Download <code>1. Simple Linear Regression.ipynb</code> and <code>2. Multi Linear Regression.ipynb</code> from this repository.</p>
    <p style="margin-top: 5px;">Alternatively, open them directly in <a href="https://colab.research.google.com/" style="color: #28A745; text-decoration: none;">Google Colab</a> for a zero-setup experience (ensure you upload the datasets if running there).</p>
  </li>
  <li style="margin-bottom: 10px;">
    <strong style="color: #28A745;">Prepare Data:</strong>
    <p style="margin-top: 5px;">Ensure you have your datasets (`Salary_Data-2.csv` and `50_Startups.csv` or similar) in the same directory as the notebooks, or adjust the file paths in the notebooks accordingly.</p>
  </li>
  <li style="margin-bottom: 10px;">
    <strong style="color: #28A745;">Install Dependencies:</strong>
    <pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code>pip install pandas numpy scikit-learn matplotlib statsmodels</code></pre>
    (If using Spark, ensure your Spark environment is correctly set up.)
  </li>
  <li style="margin-bottom: 10px;">
    <strong style="color: #28A745;">Run the Notebooks:</strong>
    <p style="margin-top: 5px;">Open each notebook (<code>1. Simple Linear Regression.ipynb</code> and <code>2. Multi Linear Regression.ipynb</code>) in Jupyter or Colab.</p>
    <p style="margin-top: 5px;">Execute each cell sequentially to perform the linear regression analyses and see the results!</p>
  </li>
</ol>

---

<h2 id="code-breakdown" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  🧠 Code Breakdown
</h2>
<p style="font-size: 1.1em; color: #444; line-height: 1.6;">
  Key parts of the notebooks' code structure:
</p>

<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Common Imports:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #569CD6;">import</span> <span style="color: #9CDCFE;">pandas</span> <span style="color: #C586C0;">as</span> <span style="color: #9CDCFE;">pd</span>
<span style="color: #569CD6;">import</span> <span style="color: #9CDCFE;">numpy</span> <span style="color: #C586C0;">as</span> <span style="color: #9CDCFE;">np</span>
<span style="color: #569CD6;">import</span> <span style="color: #9CDCFE;">matplotlib.pyplot</span> <span style="color: #C586C0;">as</span> <span style="color: #9CDCFE;">plt</span>
<span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">sklearn.model_selection</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">train_test_split</span>
<span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">sklearn.linear_model</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">LinearRegression</span>
<span style="color: #569CD6;">from</span> <span style="color: #9CDCFE;">sklearn.impute</span> <span style="color: #C586C0;">import</span> <span style="color: #9CDCFE;">SimpleImputer</span>
<span style="color: #569CD6;">import</span> <span style="color: #9CDCFE;">statsmodels.api</span> <span style="color: #C586C0;">as</span> <span style="color: #9CDCFE;">sm</span> <span style="color: #6A9955;"># For Multiple Linear Regression's statistical summary</span></code></pre>

<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Simple Linear Regression Specifics:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #6A9955;"># Data Loading (adjust path if not using Spark)</span>
<span style="color: #9CDCFE;">df_salary</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">pd.read_csv</span>(<span style="color: #CE9178;">'Salary_Data-2.csv'</span>)

<span style="color: #6A9955;"># Feature and Target Separation</span>
<span style="color: #9CDCFE;">X_salary</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">df_salary.iloc</span>[:, <span style="color: #B5CEA8;">0</span>:<span style="color: #B5CEA8;">1</span>].<span style="color: #9CDCFE;">values</span>
<span style="color: #9CDCFE;">y_salary</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">df_salary.iloc</span>[:, <span style="color: #B5CEA8;">1</span>].<span style="color: #9CDCFE;">values</span>

<span style="color: #6A9955;"># Missing Value Imputation</span>
<span style="color: #9CDCFE;">imputer</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">SimpleImputer</span>(<span style="color: #9CDCFE;">missing_values</span><span style="color: #CE9178;">=</span><span style="color: #569CD6;">np.NaN</span>, <span style="color: #9CDCFE;">strategy</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">"mean"</span>)
<span style="color: #9CDCFE;">X_salary</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">imputer.fit_transform</span>(<span style="color: #9CDCFE;">X_salary</span>)
<span style="color: #9CDCFE;">y_salary</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">imputer.fit_transform</span>(<span style="color: #9CDCFE;">y_salary.reshape</span>(-<span style="color: #B5CEA8;">1</span>, <span style="color: #B5CEA8;">1</span>)).<span style="color: #9CDCFE;">flatten</span>()

<span style="color: #6A9955;"># Train-Test Split</span>
<span style="color: #9CDCFE;">X_train_s</span>, <span style="color: #9CDCFE;">X_test_s</span>, <span style="color: #9CDCFE;">y_train_s</span>, <span style="color: #9CDCFE;">y_test_s</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">train_test_split</span>(<span style="color: #9CDCFE;">X_salary</span>, <span style="color: #9CDCFE;">y_salary</span>, <span style="color: #9CDCFE;">test_size</span><span style="color: #CE9178;">=</span><span style="color: #B5CEA8;">0.2</span>)

<span style="color: #6A9955;"># Model Training</span>
<span style="color: #9CDCFE;">model_s</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">LinearRegression</span>()
<span style="color: #9CDCFE;">model_s.fit</span>(<span style="color: #9CDCFE;">X_train_s</span>, <span style="color: #9CDCFE;">y_train_s</span>)

<span style="color: #6A9955;"># Prediction</span>
<span style="color: #9CDCFE;">y_pred_s</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">model_s.predict</span>(<span style="color: #9CDCFE;">X_test_s</span>)

<span style="color: #6A9955;"># Visualization</span>
<span style="color: #9CDCFE;">plt.scatter</span>(<span style="color: #9CDCFE;">X_test_s</span>.<span style="color: #9CDCFE;">astype</span>(<span style="color: #CE9178;">'float'</span>), <span style="color: #9CDCFE;">y_test_s</span>.<span style="color: #9CDCFE;">astype</span>(<span style="color: #CE9178;">'float'</span>), <span style="color: #9CDCFE;">color</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'red'</span>)
<span style="color: #9CDCFE;">plt.plot</span>(<span style="color: #9CDCFE;">X_test_s</span>, <span style="color: #9CDCFE;">y_pred_s</span>, <span style="color: #9CDCFE;">color</span><span style="color: #CE9178;">=</span><span style="color: #CE9178;">'green'</span>)
<span style="color: #9CDCFE;">plt.xlabel</span>(<span style="color: #CE9178;">"Years of Experience"</span>)
<span style="color: #9CDCFE;">plt.ylabel</span>(<span style="color: #CE9178;">"Salary"</span>)
<span style="color: #9CDCFE;">plt.title</span>(<span style="color: #CE9178;">"Salary Prediction (Simple Linear Regression)"</span>)
<span style="color: #9CDCFE;">plt.show</span>()</code></pre>

<h3 style="color: #28A745; font-size: 1.8em; margin-top: 25px;">Multiple Linear Regression Specifics:</h3>
<pre style="background-color: #2D2D2D; color: #E0E0E0; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.95em; box-shadow: 0 2px 10px rgba(0,0,0,0.15);"><code><span style="color: #6A9955;"># Data Loading (adjust path if not using Spark)</span>
<span style="color: #9CDCFE;">df_startup</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">pd.read_csv</span>(<span style="color: #CE9178;">'50_Startups.csv'</span>)

<span style="color: #6A9955;"># Categorical Encoding</span>
<span style="color: #9CDCFE;">df_startup</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">pd.get_dummies</span>(<span style="color: #9CDCFE;">data</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">df_startup</span>, <span style="color: #9CDCFE;">columns</span><span style="color: #CE9178;">=</span>[<span style="color: #CE9178;">"State"</span>], <span style="color: #9CDCFE;">drop_first</span><span style="color: #CE9178;">=</span><span style="color: #9CDCFE;">True</span>)

<span style="color: #6A9955;"># Outlier Handling Function (defined earlier in the notebook)</span>
<span style="color: #569CD6;">def</span> <span style="color: #795E26;">Handling_Outliers</span>(<span style="color: #9CDCFE;">dataset</span>):
  <span style="color: #569CD6;">for</span> <span style="color: #9CDCFE;">col</span> <span style="color: #569CD6;">in</span> <span style="color: #9CDCFE;">dataset</span>:
    <span style="color: #9CDCFE;">q1</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">dataset</span>[<span style="color: #9CDCFE;">col</span>].<span style="color: #9CDCFE;">astype</span>(<span style="color: #CE9178;">'float'</span>).<span style="color: #9CDCFE;">quantile</span>(<span style="color: #B5CEA8;">0.25</span>)
    <span style="color: #9CDCFE;">q3</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">dataset</span>[<span style="color: #9CDCFE;">col</span>].<span style="color: #9CDCFE;">astype</span>(<span style="color: #CE9178;">'float'</span>).<span style="color: #9CDCFE;">quantile</span>(<span style="color: #B5CEA8;">0.75</span>)
    <span style="color: #9CDCFE;">iqr</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">q3</span> - <span style="color: #9CDCFE;">q1</span>
    <span style="color: #9CDCFE;">lower_percentile</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">q1</span> - <span style="color: #B5CEA8;">1.5</span> * <span style="color: #9CDCFE;">iqr</span>
    <span style="color: #9CDCFE;">upper_percentile</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">q3</span> + <span style="color: #B5CEA8;">1.5</span> * <span style="color: #9CDCFE;">iqr</span>
    <span style="color: #9CDCFE;">dataset</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">dataset</span>[<span style="color: #9CDCFE;">dataset</span>[<span style="color: #9CDCFE;">col</span>].<span style="color: #9CDCFE;">astype</span>(<span style="color: #CE9178;">'float'</span>) <span style="color: #CE9178;">>=</span> <span style="color: #9CDCFE;">lower_percentile</span>]
    <span style="color: #9CDCFE;">dataset</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">dataset</span>[<span style="color: #9CDCFE;">dataset</span>[<span style="color: #9CDCFE;">col</span>].<span style="color: #9CDCFE;">astype</span>(<span style="color: #CE9178;">'float'</span>) <span style="color: #CE9178;"><=</span> <span style="color: #9CDCFE;">upper_percentile</span>]
  <span style="color: #569CD6;">return</span> <span style="color: #9CDCFE;">dataset</span>

<span style="color: #9CDCFE;">df_startup_clean</span> <span style="color: #CE9178;">=</span> <span style="color: #795E26;">Handling_Outliers</span>(<span style="color: #9CDCFE;">df_startup</span>)

<span style="color: #6A9955;"># Feature and Target Separation</span>
<span style="color: #9CDCFE;">y_startup</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">df_startup_clean.pop</span>(<span style="color: #CE9178;">"Profit"</span>)
<span style="color: #9CDCFE;">X_startup</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">df_startup_clean</span>

<span style="color: #6A9955;"># Adding a constant for the intercept term for statsmodels OLS</span>
<span style="color: #9CDCFE;">X_startup</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">sm.add_constant</span>(<span style="color: #9CDCFE;">X_startup</span>)

<span style="color: #6A9955;"># Train-Test Split</span>
<span style="color: #9CDCFE;">X_train_m</span>, <span style="color: #9CDCFE;">X_test_m</span>, <span style="color: #9CDCFE;">y_train_m</span>, <span style="color: #9CDCFE;">y_test_m</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">train_test_split</span>(<span style="color: #9CDCFE;">X_startup</span>, <span style="color: #9CDCFE;">y_startup</span>, <span style="color: #9CDCFE;">test_size</span><span style="color: #CE9178;">=</span><span style="color: #B5CEA8;">0.2</span>, <span style="color: #9CDCFE;">random_state</span><span style="color: #CE9178;">=</span><span style="color: #B5CEA8;">100</span>)

<span style="color: #6A9955;"># Model Training (using statsmodels for detailed summary)</span>
<span style="color: #9CDCFE;">model_m</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">sm.OLS</span>(<span style="color: #9CDCFE;">y_startup</span>, <span style="color: #9CDCFE;">X_startup</span>).<span style="color: #9CDCFE;">fit</span>()
<span style="color: #9CDCFE;">print</span>(<span style="color: #9CDCFE;">model_m.summary2</span>())

<span style="color: #6A9955;"># Example of Backward Elimination (manual step-by-step as in notebook)</span>
<span style="color: #9CDCFE;">X_startup_optimized</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">X_startup.iloc</span>[:, [<span style="color: #B5CEA8;">0</span>, <span style="color: #B5CEA8;">1</span>, <span style="color: #B5CEA8;">2</span>, <span style="color: #B5CEA8;">3</span>]] <span style="color: #6A9955;"># Example: dropping 'State_New York' after analysis</span>
<span style="color: #9CDCFE;">model_m_optimized</span> <span style="color: #CE9178;">=</span> <span style="color: #9CDCFE;">sm.OLS</span>(<span style="color: #9CDCFE;">y_startup</span>, <span style="color: #9CDCFE;">X_startup_optimized</span>).<span style="color: #9CDCFE;">fit</span>()
<span style="color: #9CDCFE;">print</span>(<span style="color: #9CDCFE;">model_m_optimized.summary2</span>())</code></pre>

---

<h2 id="contribute" style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; border-bottom: 3px solid #FFC107; padding-bottom: 10px;">
  🤝 Contribute
</h2>
<p align="center" style="font-size: 1.2em; color: #555; max-width: 800px; margin: 0 auto; line-height: 1.6;">
  Contributions are welcome! If you have ideas for improving the code, adding more detailed analysis (e.g., R-squared, Mean Squared Error, cross-validation), or exploring more advanced regression techniques, feel free to open an issue or submit a pull request. Let's enhance this foundational machine learning project together! 🌟
</p>
<p align="center" style="font-size: 1.2em; color: #555; margin: 15px auto 0; line-height: 1.6;">
  Star this repo if you find it helpful! ⭐
</p>
<p align="center" style="font-size: 1em; color: #777; margin-top: 30px;">
  Created with 💖 by Chirag
</p>
