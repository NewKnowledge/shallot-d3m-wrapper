from distutils.core import setup

setup(name='ShallotD3MWrapper',
    version='1.0.1',
    description='A thin wrapper for interacting with New Knowledge shapelet learning tool Shallot',
    packages=['ShallotD3MWrapper'],
    install_requires=["typing",
        "Sloth==2.0.3"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/sloth@8e2b856617780b9c1527929a0d0be34d81b272fa#egg=Sloth-2.0.3"

    ],
    entry_points = {
        'd3m.primitives': [
            'time_series_classification.shapelet_learning.Shallot = ShallotD3MWrapper:Shallot'
        ],
    },
)
