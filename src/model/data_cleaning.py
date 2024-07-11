import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union, Tuple
import logging
from abc import ABC, abstractmethod

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self):
        pass

# class DataPreprocessingStrategy(DataStrategy):
#     def __init__(self, scaler=None):
#         self.scaler = scaler if scaler is not None else StandardScaler()
#     def handle_data(self, data: pd.DataFrame) -> pd.DataFrame: #Dùng cho bài toán phân loại
#         try:
#             # Đọc dữ liệu
#             df = pd.read_csv(data)
#             df_t = df.T
#             X = df_t.iloc[:-1,:].values
#             y = df_t.iloc[-1, :].values
#             X_scaled = self.scaler.fit_transform(X)
#             y_scaled = self.scaler.fit_transform(y)
#             return X_scaled, y_scaled
#         except Exception as e:
#             logging.error(e)
#             raise e

# class DataTranspositionStrategy(DataStrategy):
#     def __init__(self, scaler=None):
#         self.scaler = scaler if scaler is not None else StandardScaler()
#     def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
#         try:
#             # Đọc dữ liệu
#             df = pd.read_csv(data)
#             df_t = df.T
#             X = df_t.iloc[:-1,:].values
#             y = df_t.iloc[-1, :].values
#             X_scaled = self.scaler.fit_transform(X)
#             return X_scaled
#         except Exception as e:
#             logging.error(e)
#             raise e

def create_sequences(data, time_steps):
    X = []
    for i in range(data.shape[1] - time_steps + 1):
        X.append(data[:, i:i + time_steps])
    return np.array(X)

class TimeSeriesDataPreparer:
    def __init__(self, time_steps):
        self.time_steps = time_steps

    def handle_data(self, data_path: str):
        try:
            logging.info(f"Preparing time series data with time_steps = {self.time_steps}")
            df = pd.read_csv(data_path, header=None)
            data = df.values

            X = create_sequences(data[:, :-1], self.time_steps)

            y = []
            y_seg = data[:, -1]
            for i in range(X.shape[0]):
                y.append(y_seg)
            y = np.array(y)
            return X, y
        except Exception as e:
            logging.error(e)
            raise e

class DataSplitStrategy(DataStrategy):
    def __init__(self, test_size: float, random_state: int):
        self.test_size = test_size
        self.random_state = random_state

    def handle_data(self, X, y):
        try:
            logging.info(f"Splitting data with test_size = {self.test_size} and random_state = {self.random_state}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            logging.info(f"Shape of X_train: {X_train.shape}")
            logging.info(f"Shape of X_test: {X_test.shape}")
            logging.info(f"Shape of y_train: {y_train.shape}")
            logging.info(f"Shape of y_test: {y_test.shape}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(e)
            raise e

class DataCleaning:
    """
    Data cleaning class which preprocesses the data.
    """

    def __init__(self, strategy: DataStrategy):
        """Initializes the DataCleaning class with a specific strategy."""
        self.strategy = strategy

    def handle_data(self, data: str = None, X: pd.DataFrame = None, y: pd.Series = None) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        if data:
            return self.strategy.handle_data(data)
        else:
            return self.strategy.handle_data(X, y)
