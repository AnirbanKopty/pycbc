import logging
from math import sqrt
import numpy

from pycbc.types import TimeSeries, FrequencySeries, zeros, Array, common_kind, complex128, float64
from pycbc.types import complex_same_precision_as, real_same_precision_as
from pycbc.fft import fft, ifft, IFFT

from .matchedfilter import make_frequency_series, get_cutoff_indices, correlate, sigmasq

logger = logging.getLogger('pycbc.filter.matchedfilter')



def matched_filter_core_different_psds(template, data, psd1=None, psd2=None, low_frequency_cutoff=None,
                  high_frequency_cutoff=None, h_norm=None, out=None, corr_out=None):
    """ Same as `matched_filter_core` but with option for two psds

    Parameters
    ----------
    template : TimeSeries or FrequencySeries
        The template waveform
    data : TimeSeries or FrequencySeries
        The strain data to be filtered.
    psd1 : {FrequencySeries}, optional
        The noise weighting of the filter for template.
    psd2 : {FrequencySeries}, optional
        The noise weighting of the filter for data.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin the filter calculation. If None, begin at the
        first frequency after DC.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop the filter calculation. If None, continue to the
        the nyquist frequency.
    h_norm : {None, float}, optional
        The template normalization. If none, this value is calculated internally.
    out : {None, Array}, optional
        An array to use as memory for snr storage. If None, memory is allocated
        internally.
    corr_out : {None, Array}, optional
        An array to use as memory for correlation storage. If None, memory is allocated
        internally. If provided, management of the vector is handled externally by the
        caller. No zero'ing is done internally.

    Returns
    -------
    snr : TimeSeries
        A time series containing the complex snr.
    correlation: FrequencySeries
        A frequency series containing the correlation vector.
    norm : float
        The normalization of the complex snr.
    """
    htilde = make_frequency_series(template)
    stilde = make_frequency_series(data)

    if len(htilde) != len(stilde):
        raise ValueError("Length of template and data must match")

    N = (len(stilde)-1) * 2
    kmin, kmax = get_cutoff_indices(low_frequency_cutoff,
                                   high_frequency_cutoff, stilde.delta_f, N)

    if corr_out is not None:
        qtilde = corr_out
    else:
        qtilde = zeros(N, dtype=complex_same_precision_as(data))

    if out is None:
        _q = zeros(N, dtype=complex_same_precision_as(data))
    elif (len(out) == N) and type(out) is Array and out.kind =='complex':
        _q = out
    else:
        raise TypeError('Invalid Output Vector: wrong length or dtype')

    correlate(htilde[kmin:kmax], stilde[kmin:kmax], qtilde[kmin:kmax])

    if psd1 is not None and psd2 is not None:
        if isinstance(psd1, FrequencySeries) and isinstance(psd2, FrequencySeries):
            try:
                numpy.testing.assert_almost_equal(stilde.delta_f, psd1.delta_f)
                numpy.testing.assert_almost_equal(stilde.delta_f, psd2.delta_f)
            except AssertionError:
                raise ValueError("PSD delta_f does not match data")
            qtilde[kmin:kmax] /= numpy.sqrt(psd1[kmin:kmax] * psd2[kmin:kmax])
        else:
            raise TypeError("PSD must be a FrequencySeries")
    else:
        logger.warning("psd1 and/or psd2 is None, skipping...")

    ifft(qtilde, _q)

    if h_norm is None:
        h_norm = sigmasq(htilde, psd1, low_frequency_cutoff, high_frequency_cutoff)

    norm = (4.0 * stilde.delta_f) / sqrt( h_norm)

    return (TimeSeries(_q, epoch=stilde._epoch, delta_t=stilde.delta_t, copy=False),
           FrequencySeries(qtilde, epoch=stilde._epoch, delta_f=stilde.delta_f, copy=False),
           norm)


def inner_product_twodet(A, B, psd1, psd2):
    """ Return the inner product of the array with complex conjugation.
    """
    cdtype = common_kind(A.dtype, B.dtype)
    if cdtype.kind == 'c':
        acum_dtype = complex128
    else:
        acum_dtype = float64

    return numpy.sum(A.data.conj() * B.data / (psd1.data * psd2.data), dtype=acum_dtype)



def sigmasq_twodet(psd1=None, psd2=None, low_frequency_cutoff=None, high_frequency_cutoff=None):
    frequencies = psd1.sample_frequencies
    N = (len(psd1)-1) * 2
    norm = 8.0 * psd1.delta_f**2
    kmin, kmax = get_cutoff_indices(low_frequency_cutoff,
                                    high_frequency_cutoff, psd1.delta_f, N)
    f = frequencies[kmin:kmax]**(-7/3)

    if psd1 is None or psd2 is None:
        raise ValueError("psd1 and/or psd2 is not provided")
    try:
        numpy.testing.assert_almost_equal(psd1.delta_f, psd2.delta_f)
    except AssertionError:
        raise ValueError('delta_f are not matching between psd1, psd2')

    sq = inner_product_twodet(f, f, psd1[kmin:kmax], psd2[kmin:kmax])

    return sq.real * norm


def matched_filter_twodet(data1, data2, psd1=None, psd2=None, low_frequency_cutoff=None,
                  high_frequency_cutoff=None):

    stilde1 = make_frequency_series(data1)
    stilde2 = make_frequency_series(data2)

    if len(stilde1) != len(stilde2):
        raise ValueError("Length of template and data must match")

    N = (len(stilde2)-1) * 2
    kmin, kmax = get_cutoff_indices(low_frequency_cutoff,
                                   high_frequency_cutoff, stilde2.delta_f, N)


    qtilde = zeros(N, dtype=complex_same_precision_as(data1))

    _q = zeros(N, dtype=complex_same_precision_as(data1))

    correlate(stilde2[kmin:kmax], stilde1[kmin:kmax], qtilde[kmin:kmax])

    if psd1 is None or psd2 is None:
        raise ValueError("psd1 and/or psd2 is not provided")

    if isinstance(psd1, FrequencySeries) and isinstance(psd2, FrequencySeries):
        try:
            numpy.testing.assert_almost_equal(stilde2.delta_f, psd1.delta_f)
            numpy.testing.assert_almost_equal(stilde2.delta_f, psd2.delta_f)
        except AssertionError:
            raise ValueError("PSD delta_f does not match data")

        f = stilde1.sample_frequencies[kmin:kmax]**(-7/3)
        qtilde[kmin:kmax] *= f/(psd1[kmin:kmax] * psd2[kmin:kmax])
    else:
        raise TypeError("PSD must be a FrequencySeries")

    fft(qtilde, _q)

    norm_twodet = sigmasq_twodet(psd1, psd2, low_frequency_cutoff, high_frequency_cutoff)

    norm = (8.0 * stilde2.delta_f**2) / sqrt(norm_twodet)

    return TimeSeries(_q, epoch=stilde2._epoch, delta_t=stilde2.delta_t, copy=False) * norm
