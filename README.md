
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

You must also set the environment variable `NEUPRINT_APPLICATION_CREDENTIALS` to
point to your neuprint authentication token. See their website for how to
generate one. For some ways of setting environment variables, you may need to
start a new terminal before it takes effect.

### Generating CSVs (for use outside of Python)

From a terminal where the environment variable `NEUPRINT_APPLICATION_CREDENTIALS` 
is defined appropriately...

To get the PN->KC connectivity, in the form of two CSVs:
```
cd neuprint_helper
source ./venv/bin/activate
write_neuprint_csvs.py
```

If / when I add more query functions here, I'll also have the above command save
their outputs too, maybe with arguments to select which you want.

The CSVs contain data very close to what is described in the `fetch_adjacencies`
function from `neuprint-python`. See their documentation 
[here](https://connectome-neuprint.github.io/neuprint-python/docs/queries.html#neuprint.queries.fetch_adjacencies) 
for more details.


### Available functions

- `pn_kc_connections`
  Returns:
  - `neuron_df` (`DataFrame`) Merge with `connection_df` if you want.

  - `connection_df` (`DataFrame`) The sum of the `weight` column is equal to the
    total number of synapses between all PNs and KCs.
  
  Arguments (all optional):
  - `properties` (list of `str`) Neuron property names to retrieve.
     By default, I try to return all available properties.

  - `sum_across_rois` (default=`False`) For the second returned
     By default, I try to return all available properties.

  - `checks` (default=`False`) Runs some additional sanity checks.

#### Keyword arguments available to all query functions

- `warn_nullcols` (default=`True`) Warns if any requested properties are fully
   null, in the output we get from `neuprint`.

- `print_time` (default=`False`) Prints the time (in seconds) the query function
  takes to run, at the end of each call.

