'''

File Name: churn_script_logging_and_tests.py

Author(s): Artem Plastinkin

Brief Description: Udacity project for POC to Production Ready Code and testing

-----

Created Date: Tuesday, August 29th 2023, 11:52:00 am

-----

Last Modified: Sun Sep 03 2023
Modified By: Artem Plastinkin

-----
History : DD.MM.YYYY	Author				Description
        : 2023-08-30	Artem Plastinkin	First Release
'''

import os
import logging
import churn_library as cls

from constant import CAT_COLUMNS

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    data_frame = None
    try:
        data_frame = import_data("./data/bank_data.csv")
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except FileNotFoundError as err:
        logging.error("Testing import_eda execution: The file wasn't found")
        raise err
    except AssertionError as err:
        logging.error(
            "Testing import_data functionnality: The file doesn't appear to have rows and columns")
        raise err
    else:
        logging.info("Testing import_data functionnality: SUCCESS")
        return data_frame


def test_eda(perform_eda, dataframe):
    '''
    test perform eda function
    '''
    try:
        perform_eda(dataframe, False)
        assert os.path.exists("./images/eda/test.png")
        assert os.path.exists("./images/eda/Customer_Age.png")
    except AssertionError as err:
        logging.error("Testing perform_eda functionnality: Files not created")
        raise err
    except Exception as err:
        logging.error("Testing perform_eda execution: Issue encountered")
        raise err
    else:
        logging.info("Testing perform_eda functionnality: SUCCESS")


def test_encoder_helper(encoder_helper, data_frame, cat_column, response):
    '''
    test encoder helper
    '''
    try:
        data_frame_out = encoder_helper(data_frame, cat_column, response)
        assert len(data_frame) > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper functionnality: output data frame empty")
        raise err
    except Exception as err:
        logging.error("Testing encoder_helper execution: Issue encountered")
        raise err
    else:
        logging.info("Testing encoder_helper functionnality: SUCCESS")


def test_perform_feature_engineering(perform_feature_engineering, dataset):
    '''
    test perform_feature_engineering
    '''
    x_train, x_test, y_train, y_test = None, None, None, None
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataset, response='Churn')
        assert len(x_train) > 0
        assert len(x_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering functionnality: one "
            "of the output data frame is empty")
        raise err
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering execution: Issue encountered")
        raise err
    else:
        logging.info(
            "Testing perform_feature_engineering functionnality: SUCCESS")
        return x_train, x_test, y_train, y_test


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        train_models(x_train, x_test, y_train, y_test)
        assert os.path.exists(
            "./images/results/Classification_report_Logistic_Regression.png")
        assert os.path.exists(
            "./images/results/Classification_report_Random_Forest.png")
        assert os.path.exists("./images/results/ROC_curves.png")
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./models/logistic_model.pkl")
        assert os.path.exists("./images/results/feature_importance.png")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering functionnality: one of the output data "
            "frame is empty")
        raise err
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering execution: Issue encountered")
        raise err
    else:
        logging.info(
            "Testing perform_feature_engineering functionnality: SUCCESS")
        return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    data_frame = test_import(cls.import_data)

    test_eda(cls.perform_eda, data_frame)

    test_encoder_helper(cls.encoder_helper,
                        data_frame,
                        CAT_COLUMNS,
                        "Churn")

    x_train, x_test, y_train, y_test = test_perform_feature_engineering(
        cls.perform_feature_engineering,
        data_frame)

    test_train_models(cls.train_models, x_train, x_test, y_train, y_test)
