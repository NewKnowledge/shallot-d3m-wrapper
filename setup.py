from distutils.core import setup

setup(name='ShallotD3MWrapper',
    version='1.0.0',
    description='A thin wrapper for interacting with New Knowledge shapelet learning tool Shallot',
    packages=['ShallotD3MWrapper'],
    install_requires=["typing",
        "Sloth==2.0.2"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/sloth@b43ca6b2197a06f747351c01a5110cf66ff60d0#egg=Sloth-2.0.2"

    ],
    entry_points = {
        'd3m.primitives': [
            'distil.shallot = ShallotD3MWrapper:Shallot'
        ],
    },
)