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


class RandomForestBenchmark:
    _issue_tasks = [3917, 3945]

    def __init__(self, task_id=None, valid_size=0.33, seed=None, fidelity_choice=0):
        self.task_id = task_id
        self.valid_size = valid_size
        self.seed = seed
        self.rand_state = check_random_state(self.seed)
        self.fidelity_choice = fidelity_choice
        self.z_cs = self.get_fidelity_space(self.fidelity_choice)
        self.x_cs = self.get_param_space()
        # data variables
        self.train_X = None
        self.valid_X = None
        self.test_X = None
        self.train_y = None
        self.valid_y = None
        self.test_y = None

    def get_param_space(self):
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=self.seed)

        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter(
                'max_depth', lower=1, upper=15, default_value=2, log=False
            ),
            CS.UniformIntegerHyperparameter(
                'min_samples_split', lower=2, upper=128, default_value=2, log=True
            ),
            CS.UniformFloatHyperparameter(
                'max_features', lower=0.1, upper=0.9, default_value=0.5, log=False
            ),
            CS.UniformIntegerHyperparameter(
                'min_samples_leaf', lower=1, upper=64, default_value=1, log=True
            ),
        ])
        return cs

    def get_fidelity_space(self, fidelity_choice):
        """Fidelity space available --- specifies the fidelity dimensions
        """
        f_cs = CS.ConfigurationSpace(seed=self.seed)
        if fidelity_choice == 1:
            # only n_estimators as fidelity
            ntrees = CS.UniformIntegerHyperparameter(
                'n_estimators', lower=2, upper=100, default_value=10, log=False
            )
            subsample = CS.Constant('subsample', value=1)
        elif fidelity_choice == 2:
            # only subsample as fidelity
            ntrees = CS.Constant('n_estimators', value=100)
            subsample = CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1, default_value=0.33, log=False
            )
        else:
            # both n_estimators and subsample as fidelities
            ntrees = CS.UniformIntegerHyperparameter(
                'n_estimators', lower=2, upper=100, default_value=10, log=False
            )
            subsample = CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1, default_value=0.33, log=False
            )
        f_cs.add_hyperparameters([ntrees, subsample])
        return f_cs

    def get_config(self, size=None):
        """Samples configuration(s) from the (hyper) parameter space
        """
        if size is None:  # return only one config
            return self.x_cs.sample_configuration()
        return [self.x_cs.sample_configuration() for i in range(size)]

    def get_fidelity(self, size=None):
        """Samples candidate fidelities from the fidelity space
        """
        if size is None:  # return only one config
            return self.f_cs.sample_configuration()
        return [self.f_cs.sample_configuration() for i in range(size)]

    def _convert_labels(self, labels):
        label_types = list(map(lambda x: isinstance(x, bool), labels))
        if np.all(label_types):
            _labels = list(map(lambda x: str(x), labels))
            if isinstance(labels, pd.Series):
                labels = pd.Series(_labels, index=labels.index)
            elif isinstance(labels, np.array):
                labels = np.array(labels)
        return labels

    def load_data_from_openml(self, verbose=False):
        """Fetches data from OpenML and initializes the train-validation-test data splits

        The validation set is fixed till this function is called again or explicitly altered
        """
        # fetches task
        self.task = openml.tasks.get_task(self.task_id, download_data=False)
        # fetches dataset
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
        # train-test split is fixed for a task and its associated dataset
        train_idx, test_idx = self.task.get_train_test_split_indices()
        train_X = X.iloc[train_idx]
        train_y = y.iloc[train_idx]
        self.test_X = X.iloc[test_idx]
        self.test_y = y.iloc[test_idx]

        # splitting training into training and validation
        self.train_X, self.valid_X, self.train_y, self.valid_y = train_test_split(
            train_X, train_y, test_size=self.valid_size,  # validation size set in __init__()
            shuffle=True, stratify=train_y, random_state=self.rand_state
        )  # validation set is fixed till this function is called again or explicitly altered

        # preprocessor to handle missing values, categorical columns encodings,
        #   and scaling numeric columns
        self.preprocessor = make_pipeline(
            ColumnTransformer([
                (
                    "cat",
                    make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(sparse=False, handle_unknown="ignore")),
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

        if verbose:
            print("Shape of data pre-preprocessing: {}".format(train_X.shape))
        # preprocessor fit only on the training set
        self.train_X = self.preprocessor.fit_transform(self.train_X)
        # applying preprocessor built on the training set, across validation and test splits
        self.valid_X = self.preprocessor.transform(self.valid_X)
        self.test_X = self.preprocessor.transform(self.test_X)
        # converting boolean labels to strings
        self.train_y = self._convert_labels(self.train_y)
        self.valid_y = self._convert_labels(self.valid_y)
        self.test_y = self._convert_labels(self.test_y)

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

    def _objective(self, config, fidelity):
        start = time.time()

        # initializing model
        model = RandomForestClassifier(
            **config.get_dictionary(),
            n_estimators=fidelity['n_estimators'],  # a fidelity being used during initialization
            bootstrap=True,
            random_state=self.rand_state
        )
        # subsample here
        # application of the other fidelity to the dataset that the model interfaces
        train_idx = self.rand_state.choice(
            np.arange(len(self.train_X)), size=int(
                fidelity['subsample'] * len(self.train_X)
            )
        )
        # fitting the model with subsampled data
        model.fit(self.train_X[train_idx], self.train_y.iloc[train_idx])
        accuracy_scorer = make_scorer(accuracy_score)

        val_loss = 1 - accuracy_scorer(model, self.valid_X, self.valid_y)
        # TODO: should training loss be on the subsampled data ??
        train_loss = 1 - accuracy_scorer(model, self.train_X, self.train_y)
        test_loss = 1 - accuracy_scorer(model, self.test_X, self.test_y)

        del model
        end = time.time()

        return {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'val_loss': val_loss,
            'cost': end - start,
            # storing as dictionary and not ConfigSpace saves tremendous memory
            'fidelity': fidelity.get_dictionary(),
            'config': config.get_dictionary()
        }

    def objective(self, config, fidelity):
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        info = self._objective(config, fidelity)

        return {
            'function_value': info['val_loss'],
            'info': info
        }

    def objective_test(self, config, fidelity):
        """Function that evaluates a 'config' on a 'fidelity' on the test set
        """
        info = self._objective(config, fidelity)

        return {
            'function_value': info['test_loss'],
            'info': info
        }
