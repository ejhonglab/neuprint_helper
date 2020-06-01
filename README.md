
### Dependencies
- `git` (to clone this repository)
- `python3`
- `pip` for `python3`

### Installation
```
git clone git://github.com/ejhonglab/neuprint_helper
cd neuprint_helper

python3 -m venv venv
# Change this path to whatever subdirectory of ./venv contains activate
# (if this doesn't work)
source ./venv/bin/activate

python3 -m pip install .
```

### Usage

To get the PN->KC connectivity, in the form of two CSVs:
```
# Virtual environment with neuprint_helper installed must be active
# This is what the line "source ./venv/bin/activate" above does.
write_neuprint_csvs.py
```

The CSVs contain data very close to what is described in the `fetch_adjacencies`
function from `neuprint-python`. See their documentation 
[here](https://connectome-neuprint.github.io/neuprint-python/docs/queries.html#neuprint.queries.fetch_adjacencies) 
for more details.
