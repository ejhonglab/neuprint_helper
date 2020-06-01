
import logging
import os
from os.path import join
from functools import wraps
import warnings
import time

import pandas as pd
import neuprint as nu
from neuprint import NeuronCriteria as NC


neuprint_token_env_var = 'NEUPRINT_APPLICATION_CREDENTIALS'
# TODO maybe avoid need for this later...
if neuprint_token_env_var not in os.environ:
    raise RuntimeError('You must set the environment variable '
        f'{neuprint_token_env_var} before neuprint_helper can work! '
        '\n\nContact Tom if you do not know how / if not possible!'
    )

# TODO delete. for testing that R is setting my env var correctly.
tom_token_file = '/home/tom/src/neuprint_test/.envrc'
if os.path.exists(tom_token_file):
    val = os.environ[neuprint_token_env_var]
    print(neuprint_token_env_var + ':', val)

    with open(tom_token_file, 'r') as f:
        data = f.read()
    envrc_val = data.split('=')[1].rstrip()
    assert val == envrc_val
#

# Now we can just let `neuprint` handle it.
del neuprint_token_env_var


logger = logging.getLogger('neuprint.client')
logger.setLevel(logging.DEBUG)

log_dir = os.path.realpath(join(os.getcwd(), os.path.dirname(__file__)))
log_path = join(log_dir, __file__ + '.log')

# Opens file in 'append' ('a') mode by default.
logging.basicConfig(
    # TODO include some id of thread / process for debugging that type of stuff?
    # TODO what is the 8s?
    format='%(asctime)s %(levelname)-8s %(message)s',
    filename=log_path
)

# According to the neuprint docs, this first creation of a Client sets
# the default client, so set_default_client does not need to be explicitly
# called. I will start by trying to not use the client object returned by
# this call, since it seems *that* might make it harder to use
# multiprocessing or threading to parallelize calls later, as it might make
# neuprint think we are using some non-default client, which it won't copy
# as-needed for new workers.
# TODO should i be using a different dataset, for the second arg?
# does the `hemibrain_Neuron` type in the debug cypher queries indicate
# i am not using the correct dataset (probably not)?
# TODO some value for second argument (default?) to just use the latest?
# TODO TODO probably move creation of a single client to something that
# happens at import / first call of `hong_neuprint`, when i make that
# TODO TODO and then also prompt people to set appropriate env var for token
# when that happens if not (maybe asking people to manually enter it at a
# prompt?  .pth files?)
# TODO maybe unconditionally monkey patch neuprint.Client on init, just to
# add a warning if manually creating clients (saying when it is / isn't
# appropriate) (maybe + kwarg to disable warning, also mentioned in
# warning)? (might need care to not cause same warning when neuprint
# itself copies code... maybe check the __file__ of the calling code
# (possible?)?
# TODO TODO TODO before any of the above, just put a line in a README.md
# explaining that users should prefer to access my client rather than
# making their own, and they should only use it for the list of fns below
# (the ones only the client can do)
# TODO TODO or maybe just monkey patch the client so it can't make the calls
# provided by the others? (unless it breaks internals, which it very well
# might)
# TODO or maybe only instantiate this inside my own fns (like in fetch_function
# wrapped fn part + other places that would need client) and never present it?
client = nu.Client('neuprint.janelia.org', 'hemibrain:v1.0.1')

# This seems to be the most up-to-date documentation on what properties are
# availble: https://neuprint.janelia.org/public/neuprintuserguide.pdf
#maybe_missing = ['primaryNeurite', 'somaLocation', 'somaRadius', 'timestamp']
# (just a hack to hide warning for now. commented above is accurate **wrt PDF**)
maybe_missing = ['somaLocation', 'somaRadius']

# These don't include the "<rois>: boolean" properties from the docs, which seem
# to only be defined when True, or only for sets where some have the True
# value...
# TODO TODO (as some comments say elsewhere) still try to replace this w/
# queries that enumerate all available properties
userguide_neuron_properties = [
    'bodyId', 'cropped', 'instance', 'post', 'pre', 'roiInfo', 'size', 'status',
    'statusLabel', 'type'
] + maybe_missing

# Name of node Label (the type in the db) -> list of str properties available.
userguide_properties = {
    'Neuron': userguide_neuron_properties,
    'Segment': userguide_neuron_properties,
    'SynapseSet': ['timeStamp'],
    # This also doesn't include the "<rois>: boolean" properties
    'Synapse': ['confidence', 'location', 'timeStamp', 'type'],
    # This is the one Relationship that the PDF lists Properties for.
    # It is also the only Relationship between Neurons / Segments.
    'ConnectsTo': ['roiInfo', 'weight', 'weightHp']
}

# >>> nu.queries.NEURON_COLS
# ['bodyId', 'instance', 'type', 'pre', 'post', 'size', 'status', 'cropped',
#  'statusLabel','cellBodyFiber', 'somaRadius', 'somaLocation', 'inputRois',
#  'outputRois', 'roiInfo']
#
# >>> set(nu.queries.NEURON_COLS) - set(userguide_neuron_properties)
# {'outputRois', 'cellBodyFiber', 'inputRois'}
#
# >>> set(userguide_neuron_properties) - set(nu.queries.NEURON_COLS)
# {'timestamp', 'primaryNeurite'}

# TODO delete after figuring out differences / better way to get properties
# inputRois and outputRois were all null in a traced_adjacency query...
# why? need to call those ones through some special way? or am i forced
# to process roiInfo myself?
# cellBodyFiber *is* defined, so at least one thing here is queriable
# despite 
# neither of props only in PDF were non-null w/ same query as above
nc1 = list(nu.queries.NEURON_COLS)
nc2 = list(userguide_neuron_properties)
nc = nc2
#

# TODO add verbose flag
def filter_nontraced_or_cropped(neuron_df, connection_df=None, traced=True,
    no_cropped=True, copy=False):
    """
    Arguments:
    `neuron_df` (`pandas` `DataFrame`): Must have columns 'Traced' and 'cropped'
        and 'cropped' must be of dtype `bool`. Also requires 'bodyId' column,
        as `filter_connections_by_neurons`.

    `connection_df` (optional) (`pandas` `DataFrame`):
        See `filter_connections_by_neurons` for required columns
    """
    if not (traced or no_cropped):
        raise ValueError('at least one of traced or no_cropped arguments must'
            ' be True!'
        )

    mask_parts = []
    # TODO make test cases w/ a mix of cropped and nontraced stuff, hopefully
    # not all in the same rows
    if traced:
        mask_parts.append(neuron_df.status == 'Traced')
    if no_cropped:
        mask_parts.append(~neuron_df.cropped)
    mask = pd.DataFrame(mask_parts).all(axis=0)

    neuron_df = neuron_df[mask]
    if copy:
        neuron_df = neuron_df.copy()

    if connection_df is None:
        return neuron_df
    else:
        # TODO TODO test this w/ something that actually DOES subset
        # neuron_df... (using pn->kc stuff as input will probably cause this to
        # have no effect here)
        connection_df = filter_connections_by_neurons(neuron_df, connection_df,
            copy=copy
        )
        return neuron_df, connection_df


# TODO add verbose flag
def filter_connections_by_neurons(neuron_df, connection_df, copy=False):
    """
    Given `neuron_df` with column 'bodyId' and `connection_df` with columns
    ['bodyId_pre', 'bodyId_post'], returns the subset of `connection_df` where
    BOTH of the ID columns reference neuron in `neuron_df`.

    This is useful, for example, when manually further filtering the `neuron_df`
    returned by `neuprint.fetch_adjacencies`, to then apply this filtering to
    the `connection_df` also returned by the same `fetch_adjacencies` call.
    """
    neuron_ids = neuron_df.bodyId.unique()
    mask = (
        connection_df.bodyId_pre.isin(neuron_ids) &
        connection_df.bodyId_post.isin(neuron_ids)
    )
    connection_df = connection_df[mask]
    if copy:
        connection_df = connection_df.copy()

    return connection_df


def is_dataframe(x):
    return isinstance(x, pd.DataFrame)


neuron_id_col = 'bodyId'
# TODO unit tests testing this on all/many neuprint return values
def describes_neurons(df):
    return neuron_id_col in df.columns


# TODO unit tests testing this on all/many neuprint return values
def describes_connections(df):
    return all([c in df.columns for c in 
        [f'{neuron_id_col}_{s}' for s in ('pre','post')]
    ])


# TODO flag for calling filter_connections_by_neurons? (though would probably
# not make sense for dfs storing data on things other than neurons... not sure)
# TODO will this wrapper break the mechanism neuprint (might) use to inject the
# client in parallel contexts? test (how?)!
# TODO maybe add merge kwarg that calls neuprint.utils.merge_neuron_properties
# if one neuron df and one connection df detected in output (exactly)
def fetch_function(fetch_fn):
    """
    Wraps a function which returns a `pandas` `DataFrame` or a `tuple`
    containing at least one of them, adding the following keyword arguments:

    `warn_nullcols` (`bool`, default=`True`): If `True`, warns if any returned
        columns are completely null. This is to help check that you are querying
        the right column / property names.

    `merge` (`bool`, default=`False`): If `True` and the wrapped function
        returns exactly one DataFrame satisfying `describes_neurons` and exactly
        one satisfying `describes_connections` (order irrelevant), the function
        will instead return *one* dataframe that merges the two. `True` will 
        produce an error on return values not satisfying these conditions.


    `print_time` (`bool`, default=`False`): If `True`, the wrapped function will
        be timed, and this time will be printed before returning.

    `debug` (`bool`, default=`False`): If `True`, the will log at start and end
        of wrapped function. For single threaded code, this can enable you to
        see what cypher queries any enclosed functions are making, if the log
        level is `DEBUG`.
    """
    def warn(_df, i=None):
        # TODO TODO check this is behaves as i want in the empty dataframe case
        nullcols_mask = _df.isnull().all(axis=0)
        if not nullcols_mask.any():
            return
        nullcols = list(_df.columns[nullcols_mask])
        msg = f'{fetch_fn.__name__} output'
        if i is not None:
            msg += f'[{i}]'
        msg += f' is all null in columns {nullcols}'
        warnings.warn(msg)

    # TODO need to check none of the fns we are wrapping already have any of the
    # keyword args we add? how?

    @wraps(fetch_fn)
    def wrapped(*args, warn_nullcols=True, merge=False, print_time=False,
        debug=False, **kwargs):

        if merge:
            # (until i finish testing it)
            raise NotImplementedError

        if debug:
            # TODO also include args / kwargs?
            logger.debug(f'before {fetch_fn.__name__} call')

        # TODO maybe also time all runs of a function in some internal thing for
        # saving atexit / logging? logging would probably be less instrusive...
        # (other good tools for more specifically logging performance data
        # though?)
        if print_time:
            before = time.time()

        outputs = fetch_fn(*args, **kwargs)

        if print_time:
            duration = time.time()
            print(f'{fetch_fn.__name__} took {duration:.1f}s')

        _mp = 'merge=True and '
        if is_dataframe(outputs):
            warn(outputs)
            # TODO TODO exhaustively unit test `merge` behavior; in triggering,
            # intentionally erring, and in correctness here
            if merge:
                raise ValueError(f'{_mp}too few return values to '
                    'merge (expected 2)'
                )
        else:
            if merge and len(outputs) > 2:
                raise ValueError(f'{_mp}too many outputs to know how'
                    ' to order outputs or which to merge'
                )

            # TODO am i not thinking of some case in the merge handling in loop
            # below?
            _ndf = None
            _cdf = None
            for i, out in enumerate(outputs):
                if is_dataframe(out):
                    warn(out, i=i)
                    if merge:
                        # Not short circuiting based on whether `_ndf` is
                        # already `not None`, to also err if multiple of either
                        # type returned, to behave consistent w/ docstrings
                        # claims (and to avoid having to figure out how to pick
                        # which to merge with!!).
                        if describes_neurons(out):
                            assert not describes_connections(out)
                            if _ndf is not None:
                                raise ValueError(f'{_mp}multiple outputs '
                                    'satisfying describes_neurons'
                                )
                            _ndf = out
                        elif describes_connections(out):
                            if _cdf is not None:
                                raise ValueError(f'{_mp}multiple outputs '
                                    'satisfying describes_connections'
                                )
                            _cdf = out
                elif merge:
                    raise ValueError(f'{_mp}unexpected non-dataframe output')
                # end if/elif
            # end for

            if merge:
                # TODO maybe use diff message if both missing?
                if _ndf is None:
                    raise ValueError(f'{_mp}no output satisfying '
                        'describes_neurons'
                    )
                if _cdf is None:
                    raise ValueError(f'{_mp}no output satisfying '
                        'describes_connections'
                    )

        if debug:
            logger.debug(f'after {fetch_fn.__name__} call')

        if merge:
            if debug:
                logger.debug(f'(after {fetch_fn.__name__}) beginning merge')

            # TODO do i really want to expose which properties to merge, as they
            # do? i feel like no... b/c why would you request neuron properties
            # if you don't want to merge them in the first place?
            # (leaning towards not)
            # (wrapped function should expose them if they make sense)

            # Doing it with these two lines to preserve order (as in neuron_df)
            # TODO need to exclude any other columns which could cause problems
            # if we try to merge??? (most likely 'bodyId' would be the problem!)
            to_merge = set(_ndf.columns) - set(_cdf.columns)
            # Would (hopefully) cause an error in the `neuprint` function to
            # leave this in.
            to_merge -= {'bodyId'}

            # TODO TODO TODO test this heavily (and above)
            neuron_props_to_merge = [c for c in _ndf.columns if c in to_merge]

            # TODO delete
            '''
            print('_NDF.COLUMNS:', _ndf.columns)
            print('_CDF.COLUMNS:', _cdf.columns)
            print('NEURON_PROPS_TO_MERGE:', neuron_props_to_merge)
            '''
            #
            # could probably just call merge myself at this point, after all the
            # input validation probably takes up more code...
            merged_df = nu.utils.merge_neuron_properties(_ndf, _cdf,
                properties=neuron_props_to_merge
            )
            #

            if debug:
                logger.debug(f'(after {fetch_fn.__name__}) done with merge')

            # Note that we return something different if `merge=True` and we
            # have satisfied all of the other requirements on the outputs.
            return merged_df

        return outputs

    return wrapped


def write_csvs(*dataframes, names=None, path=None, **kwargs):
    # TODO TODO do *args and single iterable arg as first positional really need
    # different handling??? fix this bit if so
    # TODO handle edge case len 0
    '''
    if len(args) == 1:
        assert is_dataframe(args[0])
        # (probably in write_csv)
        if is_dataframe(dataframes):
            dataframes = [data_frame]
    else:
        dataframes = args
    '''

    # TODO handle edge case when some not dataframes?

    # TODO also check all dataframes have __name__ defined sensibly, and err
    # if not (and `names` not passed)
    if names is not None and len(dataframes) != len(names):
        # TODO test err message looks ok
        raise ValueError('if names passed, must be of same length as dataframes'
            f'!\nlen(dataframes)={len(dataframes)} != len(names)={len(names)}'
        )

    # TODO test
    if names is None:
        names = [x.__name__ for x in dataframes]

    # TODO test
    if path is not None and not os.path.isdir(path):
        raise IOError('path must exist! should be a directory to put CSVs in.')

    for df, name in zip(dataframes, names):
        csv_path = f'{name}.csv'
        if path is not None:
            csv_path = join(path, csv_path)

        print(f'Writing to {csv_path}...', end='', flush=True)
        df.to_csv(csv_path, **kwargs)
        print(' done')


# TODO flag to return diff cnxn format, see the neuprint util conn conversion fn
@fetch_function
def pn_kc_connections(properties=None, sum_across_rois=False,
    checks=True, **kwargs):
    """
    Returns `neuron_df`, `connection_df` as `neuprint.fetch_adjacencies`, but
    only for PN->KC connections.

    See also keyword arguments added by the `@fetch_function` decorator.

    Keywords not covered above are passed to `neuprint.fetch_adjacencies`.
    """
    if properties is None:
        # TODO TODO TODO replace hardcoded properties w/ something enumerated
        # using a cypher query
        properties = nc

    # TODO TODO TODO the docs make this sound like it might only return weights
    # for connections shared by ALL neurons matched. is this true? see also the
    # two kwargs dealing with minimum weights
    # TODO consider using rois=[<appropriate str for calyx>] to just match
    # those. compare results to those from manually filtering output not
    # specifying those rois.
    neuron_df, conn_df = nu.fetch_adjacencies(
        # TODO if i end up factoring some of this fn into something to be shared
        # across fns that get connections from one type to another, maybe just
        # test whether input strs have asterisk in them, and set regex=True if
        # so? first check if * is in any of the types that exist in the db!!!
        NC(type='.*PN.*', regex=True),
        NC(type='.*KC.*', regex=True),
        # Default is just ['type','instance']
        properties=properties,
        **kwargs
    )
    # TODO test the number of things returned above is the same as when doing
    # the equivalent CONTAINS query in the web interface
    # (also check just .* as suffix, not also prefix)

    if checks:
        # There CAN be (pre ID, post ID) duplicates, because weights are
        # reported per ROI, so need to sum across ROIs if we just want
        # the total weight between two neurons, ignoring where the synapses are.
        assert not conn_df.duplicated(
            [c for c in conn_df.columns if c != 'weight']
        ).any()

        # TODO TODO technically only do this check if include_nonprimary=False
        # (so check kwargs) (if it's True, would need to do another check)
        assert (set(conn_df.roi.unique()) - set(nu.fetch_primary_rois())
            == {'NotPrimary'}
        )

    if sum_across_rois:
        conn_df = conn_df.groupby(['bodyId_pre', 'bodyId_post']
            ).weight.sum().reset_index()

    # No need to call filter_nontraced_or_cropped, at least as of 2020-05-31,
    # where it had no effect.

    return neuron_df, conn_df

