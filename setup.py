from distutils.core import setup

setup(name='ShallotD3MWrapper',
    version='1.0.0',
    description='A thin wrapper for interacting with New Knowledge shapelet learning tool Shallot',
    packages=['ShallotD3MWrapper'],
    install_requires=["typing",
        "Sloth>=2.0.2"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/sloth@cf0a040300e6dd627d4cfa17dda98d5143110ff4#egg=Sloth-2.0.2"

    ],
    entry_points = {
        'd3m.primitives': [
            'distil.shallot = ShallotD3MWrapper:Shallot'
        ],
    },
)
