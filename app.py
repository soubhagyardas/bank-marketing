import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os


def main():
    st.title("Bank Marketing Campaign Analysis")
    st.sidebar.title("Bank Marketing Campaign Analysis")
    st.markdown("Are you going to take the term deposit?")
    st.sidebar.markdown("Are you going to take the term deposit?")

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv(os.getcwd() + "/bank-clean-local.csv", index_col=0)
        return data

    @st.cache_data(persist=True)
    def split(df):
        y = df['y_yes']
        x = df.drop(columns=['y_yes'], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d')
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            fig, ax = plt.subplots()
            plt.plot(fpr, tpr, marker='.')
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            fig, ax = plt.subplots()
            plt.plot(recall, precision, marker='.', label=classifier)
            st.pyplot(fig)

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['yes', 'no']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest", "XGBoost"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio('kernel', ('rbf', 'linear'), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            st.write("F1 Score: ", f1_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in the Forest", 100, 5000, step=25, key='n_estimators')
        max_depth = st.sidebar.number_input("Max Depth of the tree", 1, 20, step=1, key='max_depth')
        # bootstrap = st.sidebar.radio("Bootstrap samples while building trees", ('True', 'False'), key='bootstrap')

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            st.write("F1 Score: ", f1_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            st.write("F1 Score: ", f1_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'XGBoost':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in the Forest", 100, 5000, step=25, key='n_estimators')
        max_depth = st.sidebar.number_input("Max Depth of the tree", 1, 20, step=1, key='max_depth')
        learning_rate = st.sidebar.number_input("Learning rate", 0.1, 10.0, step=0.1, key='learning_rate')

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("XGBoost RF Classifier Results")
            model = XGBRFClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                                    n_jobs=1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            st.write("F1 Score: ", f1_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Bank Marketing Campaign - Classification")
        st.write(df)

if __name__ == '__main__':
    main()