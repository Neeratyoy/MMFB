import time
import openml
import numpy as np
import pandas as pd
import ConfigSpace as CS

from sklearn.impute import SimpleImputer
from sklearn.utils import check_random_state
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score, make_scorer


class RandomForestBenchmark():

    def __init__(self, task_id=None, valid_size=0.33, seed=None):
        self.task_id = task_id
        self.valid_size = valid_size
        self.seed = seed
        self.rand_state = check_random_state(self.seed)
        self.cs = self.get_param_space()
        self.f_cs = self.get_fidelity_space()

    def get_param_space(self):
        cs = CS.ConfigurationSpace(seed=self.seed)

        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('max_depth', lower=1, upper=15,
                                            default_value=5, log=False),
            CS.UniformFloatHyperparameter('min_samples_split', lower=0.01,
                                          upper=0.99, default_value=0.01, log=True),
            CS.UniformFloatHyperparameter('max_features', lower=0.01, upper=0.99,
                                          default_value=0.33, log=True),
            # TODO: check variance in performance with these parameters included
            # CS.UniformFloatHyperparameter('min_samples_leaf', lower=0.01,
            #                                upper=0.49, default_value=0.01, log=True),
            # CS.UniformFloatHyperparameter('min_weight_fraction_leaf', lower=0.01,
            #                                upper=0.49, default_value=0.01, log=True),
            # CS.UniformFloatHyperparameter('min_impurity_decrease', lower=0.0,
            #                                upper=0.5, default_value=0.0, log=False)
        ])
        return cs

    def get_fidelity_space(self):
        f_cs = CS.ConfigurationSpace(seed=self.seed)

        f_cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('n_estimators', lower=2, upper=100,
                                            default_value=10, log=False),
            CS.UniformFloatHyperparameter('subsample', lower=0.1,
                                          upper=1, default_value=0.33, log=False)
        ])
        return f_cs

    def get_config(self, size=None):
        if size is None:
            return self.cs.sample_configuration()
        return [self.cs.sample_configuration() for i in range(size)]

    def get_fidelity(self, size=None):
        if size is None:
            return self.f_cs.sample_configuration()
        return [self.f_cs.sample_configuration() for i in range(size)]

    def load_data_automl(self, verbose=False):

        # loads AutoML benchmark
        self.automl_benchmark = openml.study.get_suite(218)
        # if self.task_id is None:
        #     self.task_id = self.automl_benchmark.tasks[
        #         np.random.randint(len(automl_benchmark.tasks))
        #     ]
        self.task = openml.tasks.get_task(self.task_id, download_data=False)
        self.dataset = openml.datasets.get_dataset(self.task.dataset_id, download_data=False)
        if verbose:
            print(self.task, '\n')
            print(self.dataset, '\n')

        # loads full data
        X, y, categorical_ind, feature_names = self.dataset.get_data(
            target=self.task.target_name, dataset_format="dataframe"
        )
        categorical_ind = np.array(categorical_ind)
        (cat_idx,) = np.where(categorical_ind)
        (cont_idx,) = np.where(~categorical_ind)

        # splitting dataset into train and test (10% test)
        train_idx, test_idx = self.task.get_train_test_split_indices()
        train_X = X.iloc[train_idx]
        train_y = y.iloc[train_idx]
        self.test_X = X.iloc[test_idx]
        self.test_y = y.iloc[test_idx]

        # splitting training into training and validation
        self.train_X, self.valid_X, self.train_y, self.valid_y = train_test_split(
            train_X, train_y, test_size=self.valid_size,
            shuffle=True, stratify=train_y, random_state=self.rand_state
        )

        # preprocessor to handle missing values, categorical columns encodings,
        #   and scaling numeric columns
        self.preprocessor = make_pipeline(
            ColumnTransformer([
                (
                    "cat",
                    make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(sparse=False,
                                                handle_unknown="ignore")),
                    cat_idx.tolist(),
                ),
                (
                    "cont",
                    make_pipeline(SimpleImputer(strategy="median"),
                                  StandardScaler()),
                    cont_idx.tolist(),
                )
            ])
        )
        # preprocessing the training set
        if verbose:
            print("Shape of data pre-preprocessing: {}".format(train_X.shape))
        self.train_X = self.preprocessor.fit_transform(self.train_X)
        self.valid_X = self.preprocessor.transform(self.valid_X)
        self.test_X = self.preprocessor.transform(self.test_X)
        if verbose:
            print("Shape of data post-preprocessing: {}".format(train_X.shape), "\n")

        if verbose:
            print("\nTraining data (X, y): ({}, {})".format(self.train_X.shape,
                                                            self.train_y.shape))
            print("Validation data (X, y): ({}, {})".format(self.valid_X.shape,
                                                            self.valid_y.shape))
            print("Test data (X, y): ({}, {})".format(self.test_X.shape,
                                                      self.test_y.shape))

            print("\nData loading complete!\n")

    def objective(self, config, fidelity):
        start = time.time()

        model = RandomForestClassifier(
            **config.get_dictionary(),
            n_estimators=fidelity['n_estimators'],
            # max_samples=fidelities['subsample'],
            bootstrap=True,
            random_state=self.rand_state
        )
        # subsample here
        train_idx = self.rand_state.choice(
            np.arange(len(self.train_X)), size=int(
                fidelity['subsample'] * len(self.train_X)
            )
        )
        model.fit(self.train_X[train_idx], self.train_y.iloc[train_idx])
        accuracy_scorer = make_scorer(accuracy_score)

        val_loss = 1 - accuracy_scorer(model, self.valid_X, self.valid_y)
        # TODO: should training loss be on the subsampled data ??
        train_loss = 1 - accuracy_scorer(model, self.train_X, self.train_y)
        test_loss = 1 - accuracy_scorer(model, self.test_X, self.test_y)

        del model
        end = time.time()

        return {
            'function_value': val_loss,
            'info': {
                'train_loss': train_loss,
                'test_loss': test_loss,
                'cost': end - start,
                'fidelity': fidelity,
                'config': config
            }
        }

    def objective_test(self, config, fidelity):
        start = time.time()

        model = RandomForestClassifier(
            **config.get_dictionary(),
            n_estimators=fidelity['n_estimators'],
            # max_samples=fidelities['subsample'],
            bootstrap=True,
            random_state=self.rand_state
        )
        # no subsampling here
        # train_idx = self.rand_state.choice(
        #     np.arange(len(self.train_X)), size=int(
        #         fidelity['subsample'] * len(self.train_X)
        #     )
        # )
        training_X = np.concatenate((self.train_X, self.valid_X), axis=0)
        training_y = np.concatenate((self.train_y, self.valid_y), axis=0)
        model.fit(training_X, training_y)
        accuracy_scorer = make_scorer(accuracy_score)

        val_loss = 1 - accuracy_scorer(model, self.valid_X, self.valid_y)
        # TODO: should training loss be on the subsampled data ??
        train_loss = 1 - accuracy_scorer(model, training_X, training_y)
        test_loss = 1 - accuracy_scorer(model, self.test_X, self.test_y)

        del model
        end = time.time()

        return {
            'function_value': test_loss,
            'info': {
                'train_loss': train_loss,
                'cost': end - start,
                'fidelity': fidelity,
                'config': config
            }
        }