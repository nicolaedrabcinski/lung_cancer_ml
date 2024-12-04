import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Установка названия вкладки и иконки
st.set_page_config(
    page_title="Lung Cancer Analysis",  # Название вкладки браузера
    page_icon="🩺",                     # Иконка вкладки
    layout="wide"                       # Макет страницы (можно также использовать 'centered')
)

# Application title
st.title("Lung Cancer Data Analysis and Model Training")

# File uploader for CSV data
uploaded_file = st.file_uploader("Upload a CSV file containing lung cancer data", type="csv")

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    # Apply LabelEncoder to categorical columns
    le = preprocessing.LabelEncoder()
    categorical_columns = ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                           'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ',
                           'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                           'SWALLOWING DIFFICULTY', 'CHEST PAIN']
    
    for col in categorical_columns + ['LUNG_CANCER']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # Target Distribution Plot
    st.write("Target Distribution (LUNG_CANCER):")
    fig, ax = plt.subplots()
    sns.countplot(x='LUNG_CANCER', data=df, ax=ax)
    ax.set_title('Target Distribution')
    st.pyplot(fig)

    # Select feature to visualize lung cancer rates
    st.sidebar.subheader("Visualize Feature Impact on Lung Cancer")
    feature = st.sidebar.selectbox("Select a feature for bar plot:", 
                                   [col for col in df.columns if col != 'LUNG_CANCER'])
    
    if feature:
        st.write(f"Lung Cancer Rates by {feature}:")
        fig, ax = plt.subplots()
        df.groupby(feature)['LUNG_CANCER'].value_counts(normalize=True).unstack().plot(kind='bar', ax=ax)
        ax.set_ylabel('Proportion')
        st.pyplot(fig)

    # Drop selected columns and create a new DataFrame
    df_new = df.drop(columns=['GENDER', 'AGE', 'SMOKING', 'SHORTNESS OF BREATH'])
    
    # Show correlation matrix
    st.write("Correlation Matrix:")
    cn = df_new.corr()
    fig, ax = plt.subplots(figsize=(18, 18))
    cmap = sns.diverging_palette(260, -10, s=50, l=75, n=6, as_cmap=True)
    sns.heatmap(cn, cmap=cmap, annot=True, square=True, ax=ax)
    st.pyplot(fig)

    # Filter high correlations
    st.write("Filtered Correlation (>= 0.40):")
    kot = cn[cn >= .40]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(kot, cmap="Blues", annot=True, ax=ax)
    st.pyplot(fig)

    # Create a new feature and display updated DataFrame
    df_new['ANXYELFIN'] = df_new['ANXIETY'] * df_new['YELLOW_FINGERS']
    st.write("Updated DataFrame with New Feature:")
    st.dataframe(df_new)

    # Splitting independent and dependent variables
    X = df_new.drop('LUNG_CANCER', axis=1)
    y = df_new['LUNG_CANCER']

    # Apply ADASYN for oversampling
    st.write("Applying ADASYN to balance the dataset...")
    adasyn = ADASYN(random_state=42)
    X, y = adasyn.fit_resample(X, y)

    # Splitting data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Fitting Logistic Regression model
    st.write("Training Logistic Regression Model...")
    lr_model = LogisticRegression(random_state=0)
    lr_model.fit(X_train, y_train)
    
    st.success("Logistic Regression model trained successfully!")

    # Predicting result using testing data
    y_lr_pred = lr_model.predict(X_test)

    # Display classification report and accuracy metrics
    st.write("Logistic Regression Model Evaluation:")
    lr_cr = classification_report(y_test, y_lr_pred, output_dict=True)
    st.write("Logistic Regression Classification Report:")
    st.dataframe(pd.DataFrame(lr_cr).transpose())
    
    accuracy = accuracy_score(y_test, y_lr_pred)
    f1 = f1_score(y_test, y_lr_pred)
    
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # Fitting training data to the DecisionTreeClassifier
    from sklearn.tree import DecisionTreeClassifier
    dt_model = DecisionTreeClassifier(criterion='entropy', random_state=0)  
    dt_model.fit(X_train, y_train)

    st.success("Decision Tree Model trained successfully!")

    # Predicting result using testing data
    y_dt_pred = dt_model.predict(X_test)

    # Display classification report and accuracy metrics
    st.write("Decision Tree Model Evaluation:")
    dt_cr = classification_report(y_test, y_dt_pred, output_dict=True)
    st.write("Decision Tree Model Classification Report:")
    st.dataframe(pd.DataFrame(dt_cr).transpose())

    accuracy = accuracy_score(y_test, y_dt_pred)
    f1 = f1_score(y_test, y_dt_pred)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # Fitting K-NN classifier to the training set  
    from sklearn.neighbors import KNeighborsClassifier  
    knn_model= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
    knn_model.fit(X_train, y_train)

    st.success("K-NN Model trained successfully!")

    # Predicting result using testing data
    y_knn_pred= knn_model.predict(X_test)

    # Display classification report and accuracy metrics
    st.write("K-NN Model Evaluation:")
    knn_cr = classification_report(y_test, y_knn_pred, output_dict=True)
    st.write("K-NN Model Classification Report:")
    st.dataframe(pd.DataFrame(knn_cr).transpose())

    accuracy = accuracy_score(y_test, y_knn_pred)
    f1 = f1_score(y_test, y_knn_pred)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # Fitting Gaussian Naive Bayes classifier to the training set  
    from sklearn.naive_bayes import GaussianNB
    gnb_model = GaussianNB()
    gnb_model.fit(X_train, y_train)

    st.success("Gaussian Naive Bayes Model trained succesfully!")

    # Predicting result using testing data
    y_gnb_pred = gnb_model.predict(X_test)

    # Display classification report and accuracy metrics
    st.write("Gaussian Naive Bayes Model Evaluation:")
    gnb_cr = classification_report(y_test, y_gnb_pred, output_dict=True)
    st.write("Gaussian Naive Bayes Model Classification Report:")
    st.dataframe(pd.DataFrame(gnb_cr).transpose())

    accuracy = accuracy_score(y_test, y_gnb_pred)
    f1 = f1_score(y_test, y_gnb_pred)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # Fitting Multinomial Naive Bayes classifier to the training set  
    from sklearn.naive_bayes import MultinomialNB
    mnb_model = MultinomialNB()
    mnb_model.fit(X_train, y_train)

    st.success("Multinomial Naive Bayes Model trained succesfully!")

    # Predicting result using testing data
    y_mnb_pred= mnb_model.predict(X_test)

    # Display classification report and accuracy metrics
    st.write("Multinomial Naive Bayes Model Evaluation:")
    mnb_cr = classification_report(y_test, y_mnb_pred, output_dict=True)
    st.write("Multinomial Naive Bayes Model Classification Report:")
    st.dataframe(pd.DataFrame(mnb_cr).transpose())

    accuracy = accuracy_score(y_test, y_mnb_pred)
    f1 = f1_score(y_test, y_mnb_pred)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    
    # Fitting SVC to the training set  
    from sklearn.svm import SVC
    svc_model = SVC()
    svc_model.fit(X_train, y_train)

    st.success("SVC Model trained succesfully!")

    # Predicting result using testing data
    y_svc_pred= svc_model.predict(X_test)

    # Display classification report and accuracy metrics
    st.write("SVC Model Evaluation:")
    svc_cr = classification_report(y_test, y_svc_pred, output_dict=True)
    st.write("Multinomial Naive Bayes Model Classification Report:")
    st.dataframe(pd.DataFrame(svc_cr).transpose())

    accuracy = accuracy_score(y_test, y_svc_pred)
    f1 = f1_score(y_test, y_svc_pred)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}")    

    # Training Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    st.success("Random Forest Model trained succesfully!")

    # Predicting result using testing data
    y_rf_pred = rf_model.predict(X_test)

    # Display classification report and accuracy metrics
    st.write("Random Forest Model Evaluation:")
    rf_cr = classification_report(y_test, y_rf_pred, output_dict=True)
    st.write("Random Forest Model Classification Report:")
    st.dataframe(pd.DataFrame(rf_cr).transpose())

    accuracy = accuracy_score(y_test, y_rf_pred)
    f1 = f1_score(y_test, y_rf_pred)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}")   

    # Training XGBoost
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)

    st.success("XGBoost Model trained succesfully!")

    # Predicting result using testing data
    y_xgb_pred= xgb_model.predict(X_test)

    # Display classification report and accuracy metrics
    st.write("XGBoost Model Evaluation:")
    xgb_cr = classification_report(y_test, y_xgb_pred, output_dict=True)
    st.write("XGBoost Model Classification Report:")
    st.dataframe(pd.DataFrame(xgb_cr).transpose())

    accuracy = accuracy_score(y_test, y_xgb_pred)
    f1 = f1_score(y_test, y_xgb_pred)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}") 

    # Training MLP model
    from sklearn.neural_network import MLPClassifier
    mlp_model = MLPClassifier()
    mlp_model.fit(X_train, y_train)

    st.success("MLP Model trained succesfully!")

    # Predicting result using testing data
    y_mlp_pred= mlp_model.predict(X_test)

    # Display classification report and accuracy metrics
    st.write("MLP Model Evaluation:")
    mlp_cr = classification_report(y_test, y_xgb_pred, output_dict=True)
    st.write("MLP Model Classification Report:")
    st.dataframe(pd.DataFrame(mlp_cr).transpose())

    accuracy = accuracy_score(y_test, y_mlp_pred)
    f1 = f1_score(y_test, y_mlp_pred)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}") 

    # Training Gradient Boosting Classifier
    from sklearn.ensemble import GradientBoostingClassifier
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train, y_train)

    st.success("Gradient Boosting Model trained succesfully!")

    # Predicting result using testing data
    y_gb_pred= gb_model.predict(X_test)

    # Display classification report and accuracy metrics
    st.write("MLP Model Evaluation:")
    gb_cr = classification_report(y_test, y_gb_pred, output_dict=True)
    st.write("MLP Model Classification Report:")
    st.dataframe(pd.DataFrame(gb_cr).transpose())

    accuracy = accuracy_score(y_test, y_gb_pred)
    f1 = f1_score(y_test, y_gb_pred)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # K-Fold Cross Validation
    st.subheader("K-Fold Cross Validation for Model Comparison")
    
    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(random_state=0),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Classifier": SVC(random_state=0),
        "Random Forest": RandomForestClassifier(random_state=0),
        "XGBoost": XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss'),
        "Multi-layer Perceptron": MLPClassifier(random_state=0, max_iter=300),
        "Gradient Boost": GradientBoostingClassifier(random_state=0)
    }

    model_scores = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kf)
        model_scores[name] = np.mean(scores)

    # Display model scores
    st.write("Average Accuracy Scores for Models:")
    scores_df = pd.DataFrame(list(model_scores.items()), columns=["Model", "Average Accuracy"])
    st.table(scores_df)

    # Bar plot for model comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Average Accuracy", y="Model", data=scores_df, palette="viridis", ax=ax)
    ax.set_title("Model Accuracy Comparison")
    st.pyplot(fig)








else:
    st.info("Please upload a CSV file for analysis.")
