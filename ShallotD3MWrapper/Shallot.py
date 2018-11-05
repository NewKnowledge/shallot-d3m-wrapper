import sys
import os.path
import numpy as np
import pandas
import typing
from json import JSONDecoder
from typing import List

from Sloth import Shapelets
from tslearn.datasets import CachedDatasets

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base, params

__author__ = 'Distil'
__version__ = '1.0.0'

Inputs = container.numpy.ndarray
Outputs = container.numpy.ndarray

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    shapelet_length = hyperparams.LogUniform(lower = 0, upper = 1, default = 0.1, 
        upper_inclusive = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
       description = 'base shapelet length, expressed as fraction of length of time series')
    num_shapelet_lengths = hyperparams.UniformInt(lower = 1, upper = 100, default = 2, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of different shapelet lengths')
    # default epoch size from https://tslearn.readthedocs.io/en/latest/auto_examples/plot_shapelets.html#sphx-glr-auto-examples-plot-shapelets-py
    epochs = hyperparams.UniformInt(lower = 1, upper = sys.maxsize, default = 200, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of training epochs')
    pass


class Shallot(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "d351fcf8-5d6c-48d4-8bf6-a56fe11e62d6",
        'version': __version__,
        'name': "shallot",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series', 'Shapelets'],
        'source': {
            'name': __author__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/shallot-d3m-wrapper",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
         'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/shallot-d3m-wrapper.git@{git_commit}#egg=ShallotD3MWrapper'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.distil.shallot',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            # find algorithm type
            # metadata_base.PrimitiveAlgorithmType.AUTOREGRESSIVE_INTEGRATED_MOVING_AVERAGE,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.TIME_SERIES_CLASSIFICATION,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._decoder = JSONDecoder()
        self._params = {}
        self.X_train = None          # training inputs
        self.y_train = None          # training outputs
        self.shapelets = None        # shapelet classifier

    def fit(self) -> None:
        '''
        fits Shapelet classifier using training data from set_training_data and hyperparameters
        '''
        self.shapelets = Shapelets(self.X_train, self.y_train, self.hyperparams['epochs'], 
            self.hyperparams['shapelet_length'], self.hyperparams['num_shapelet_lengths'])

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
        Sets primitive's training data

        Parameters
        ----------
        inputs: numpy ndarray of size (number_of_time_series, time_series_length, dimension) containing training time series

        outputs: numpy ndarray of size (number_time_series,) containing classes of training time series
        '''
        self.X_train = inputs
        self.y_train = outputs


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's classifications for new time series data

        Parameters
        ----------
        inputs : numpy ndarray of size (number_of_time_series, time_series_length, dimension) containing new time series 

        Returns
        ----------
        Outputs
            The output is a numpy ndarray containing a predicted class for each of the input time series
        """
        classes = self.shapelets.PredictClasses(inputs)
        return CallResult(classes)


if __name__ == '__main__':
    client = Shallot(hyperparams={'shapelet_length':0.1, 'num_shapelet_lengths':2, 'epochs':200})
    
    # test using Trace dataset (Bagnall, Lines, Vickers, Keogh, The UEA & UCR Time Series
    # Classification Repository, www.timeseriesclassification.com)
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    client.set_training_data(X_train, y_train)
    client.fit()
    results = client.produce(inputs = X_test)
    print("Predicted Classes")
    print(results)
    print("Accuracy = ", accuracy_score(y_test, results))
