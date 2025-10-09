<h3>Bank Fraud Detection using Machine Learning</h3>

<h3>Objective:</h3>
<p>
  To build a machine learning model that detects fraudulent transactions from multiple datasets by cleaning, merging, and 
  modeling data efficiently.
</p>

<h3>Challenges Faced:</h3>
<pre>
- Multiple datasets with different shapes and schemas.
- Missing or inconsistent value columns across files.
- Non-binary target encoding issues (0, 2 instead of 0, 1).
- Presence of outliers and inconsistent value scales.
- Difficulty in merging data due to the absence of common columns.
</pre>


<h3>Steps Taken:</h3>
<h3>1. Data Inspection & Cleaning</h3>

<li>Loaded multiple CSVs and verified their shapes.</li>

<li>Inspected head() and info() to understand data structure.</li>

<li>Handled missing values.</li>

<h3>2. Data Integration</h3>

<li>Created a common “DataFrame” key column across train and test datasets to enable merging.</li>

<li>Added missing columns (like value) where required.</li>

<li>Ensured all dataframes had consistent feature structures before merging</li>

<h3>3. Model Building</h3>

<li>Split data into train/test sets (using data column).</li>

<li>Trained a Random Forest Classifier as the main model.</li>

<li>Evaluated with classification report, accuracy, precision, recall, F1-score.</li>

<h3>5. Results</h3>

<li>Achieved strong classification metrics with minimal overfitting.</li>

<li>Compared model performance across different random seeds and parameters.</li>
