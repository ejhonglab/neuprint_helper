#!/usr/bin/env python3

import neuprint_helper.util as util


def main():
    print('Querying neuprint for PN->KC connectivity...')
    ndf, cdf = util.pn_kc_connections()

    # TODO try this w/ list and also w/ *args like this
    # (either make names easier?)
    util.write_csvs(ndf, cdf, names=['pn2kc_neurons', 'pn2kc_connections'])


if __name__ == '__main__':
    main()

