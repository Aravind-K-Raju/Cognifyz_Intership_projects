**Machine Learning Internship Projects – Cognifyz Technologies**

**📌 Overview**
This repository showcases the tasks completed as part of my Machine Learning Internship at Cognifyz Technologies. Each task focused on applying core data science and machine learning techniques to real-world problems, including regression modeling, recommendation systems, and predictive analytics.

**🗂️ Tasks Completed
✅ Task 1 – Predicting Restaurant Ratings**
  Built a Decision Tree Regression model to predict restaurant aggregate ratings.
  Preprocessed data by handling missing values and encoding categorical variables.
  Trained the model and evaluated its performance using:
      R-squared Score (R²)
      Mean Squared Error (MSE)
  Analyzed the most influential features affecting restaurant ratings.
  Visualized:
    Feature Importance
    Correlation Matrix

**✅ Task 2 – Restaurant Recommendation System**
  Developed a Content-Based Filtering recommendation engine.
  Preprocessed data and created a combined "content" feature using:
    Cuisine type
    Experience level (rating text)
    Price range
  Used TF-IDF Vectorization and Cosine Similarity to find similar restaurants based on user preferences.
  Built an interactive UI using ipywidgets where users can:
    Select cuisine
    Choose experience level
    Adjust price range
  Delivered personalized, ranked restaurant suggestions.

**✅ Task 3 – Restaurant Cuisine Classification**
  Developed a classification model to predict the primary cuisine of a restaurant.
  Preprocessed data by handling missing values and encoding categorical features.
  Split the dataset into training and testing sets for robust model evaluation.
  Trained multiple classification models, including:
    Logistic Regression
    Random Forest Classifier
  Evaluated model performance using:
    Accuracy
    Precision
    Recall
  Analyzed key challenges and biases in the classification process:
    Multi-cuisine confusion
    Highly imbalanced data
    Regional bias
    Visual overcrowding in large datasets

**🛠️ Technologies Used**
  Python
  Pandas
  Scikit-learn (Machine Learning models, TF-IDF, Cosine Similarity)
  IPython Widgets (ipywidgets) for interactive UI
  Matplotlib, Seaborn for data visualization

**📊 How to Run the Project**
Clone this repository.

Install required libraries:
    pip install pandas scikit-learn matplotlib seaborn ipywidgets

Load the respective notebooks or Python scripts for each task.
Ensure the Dataset.csv is placed in the appropriate project directory.
Run the code sections to view model predictions, recommendations, and visualizations.

**📈 Future Enhancements**
Improve the cuisine classification by handling multi-label classification.
Enhance the recommendation engine using hybrid models.
Deploy models as live applications using Flask or Streamlit.
Apply collaborative filtering methods for advanced recommendations.

**📚 Acknowledgements**
A special thanks to Cognifyz Technologies for this invaluable  opportunity to work on real-world machine learning challenges.
The open-source community and contributors behind Python libraries that made this work possible.
