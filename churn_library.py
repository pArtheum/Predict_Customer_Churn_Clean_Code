'''

File Name: churn_library.py

Author(s): Artem Plastinkin

Brief Description: Udacity project for POC to Production Ready Code

-----

Created Date: Tuesday, August 29th 2023, 11:52:00 am

-----

Last Modified: Mon Sep 04 2023
Modified By: Artem Plastinkin

-----
History : DD.MM.YYYY	Author				Description
        : 2023-08-30	Artem Plastinkin	First Release
'''


# import libraries
import os
import seaborn as sns

from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap

from constant import CAT_COLUMNS, KEEP_COLUMNS

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(data_frame, show_flag:bool=True):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''

    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    data_frame['Churn'].hist()
    plt.tight_layout()
    plt.savefig(os.path.join("./images/eda", 'test.png'))
    if show_flag:
        plt.show(block=False)

    plt.figure(figsize=(20, 10))
    data_frame['Customer_Age'].hist()
    plt.tight_layout()
    plt.savefig(os.path.join("./images/eda", 'Customer_Age.png'))
    if show_flag:
        plt.show(block=False)

    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join("./images/eda", 'marital_status.png'))

    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join("./images/eda", 'total_trans_hist.png'))

    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join("./images/eda", 'corr_heatmap.png'))
    if show_flag:
        plt.show()


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
                      naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for category in category_lst:
        gender_groups = data_frame.groupby(category).mean()[response]
        gender_lst = [gender_groups.loc[val] for val in data_frame[category]]
        data_frame[f'{category}_Churn'] = gender_lst
    return data_frame


def perform_feature_engineering(data_frame, response: str = "Churn"):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could be
                        used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    churn_axis = data_frame[response]

    data_frame = encoder_helper(data_frame, CAT_COLUMNS, response)

    proc_dataframe = pd.DataFrame()
    proc_dataframe[KEEP_COLUMNS] = data_frame[KEEP_COLUMNS]

    return train_test_split(proc_dataframe, churn_axis, test_size=0.3, random_state=42)


def generate_classification_report(model_name,
                                   y_train,
                                   y_test,
                                   y_train_preds,
                                   y_test_preds):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder

    input:
                    model_name: (str) name of the model, ie 'Random Forest'
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds: training predictions from model_name
                    y_test_preds: test predictions from model_name

    output:
                     None
    '''
    plt.rc('figure', figsize=(5, 5))

    font_size = {'fontsize': 10}
    # Plot Classification report on Train dataset
    plt.text(0.01, 1.25, f'{model_name} Train', font_size)
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds)),
             font_size)

    # Plot Classification report on Test dataset
    plt.text(0.01, 0.6, f'{model_name} Test', font_size)
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds)),
             font_size
             )

    plt.axis('off')

    # Save figure to ./images folder
    plt.tight_layout()
    plt.savefig(os.path.join("./images/results", f'Classification_report_{model_name}.png'))

    # Display figure
    plt.show()
    plt.close()


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    generate_classification_report('Logistic_Regression',
                                   y_train,
                                   y_test,
                                   y_train_preds_lr,
                                   y_test_preds_lr)

    generate_classification_report('Random_Forest',
                                   y_train,
                                   y_test,
                                   y_train_preds_rf,
                                   y_test_preds_rf)


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save figure to output_pth
    plt.tight_layout()
    plt.savefig(os.path.join(output_pth, 'feature_importance.png'))

    # display feature importance figure
    plt.show()
    plt.close()


def train_models(x_train, y_train):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              y_train: y training data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


def predict_models(model, predict_input):
    '''
    train, store model results: images + scores, and store models
    input:
              model: model with which to do the prediciton
              predict_input: input for prediction
    output:
              prediction
    '''
    return model.predict(predict_input)

def evaluate_models(lrc_model, rfc_model,
                    x_train, x_test,
                    y_train, y_test,
                    y_train_preds_lr, y_train_preds_rf,
                    y_test_preds_lr, y_test_preds_rf):
    '''
    evaluate model results: images + scores
    input:
              lrc_model: logarithmic regression model
              rfc_model: random forest model
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
              y_train_preds_lr: y predicted data by log regression on training
              y_train_preds_rf: y predicted data by log regression on training
              y_test_preds_lr: y predicted data by log regression on test
              y_test_preds_rf: y predicted data by log regression on test
    output:
              None
    '''
    # scores
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    lrc_plot = plot_roc_curve(lrc_model, x_test, y_test)

    plt.figure(figsize=(15, 8))
    ax_var = plt.gca()
    plot_roc_curve(rfc_model, x_test, y_test, ax=ax_var, alpha=0.8)

    lrc_plot.plot(ax=ax_var, alpha=0.8)
    plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join("./images/results", 'ROC_curves.png'))
    plt.close()

    # Display feature importance on train data
    feature_importance_plot(model=rfc_model,
                            x_data=x_train,
                            output_pth="./images/results")

    explainer = shap.TreeExplainer(rfc_model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    plt.savefig('./images/results/SHAP.png')

if __name__ == "__main__":
    print("Importing Data")
    dataset = import_data("./data/bank_data.csv")
    print("Perform EDA")
    perform_eda(dataset, show_flag=False)
    print("Feature Engineering")
    x_train, x_test, y_train, y_test = perform_feature_engineering(dataset, response='Churn')
    print("Train Models")
    train_models(x_train, y_train)

    print("Load Trained Models")
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    print("Predict")
    y_train_preds_rf = predict_models(rfc_model, x_train)
    y_test_preds_rf = predict_models(rfc_model, x_test)

    y_train_preds_lr = predict_models(lr_model, x_train)
    y_test_preds_lr = predict_models(lr_model, x_test)

    print("Evaluate")
    evaluate_models(lr_model, rfc_model,
                    x_train, x_test,
                    y_train, y_test,
                    y_train_preds_lr, y_train_preds_rf,
                    y_test_preds_lr, y_test_preds_rf)


