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



def variance_term(psd1, psd2, template1, relative_amplification):
    term1 = 2 * numpy.abs(template1)**2 * ( relative_amplification * psd1 + psd2 )
    term2 = 1/psd1.delta_f * psd1 * psd2
    return term1 + term2


def inner_product_AB(A, B, psd1, psd2, template1, relative_amplification):
    """ Return the inner product of the array with complex conjugation.
    """
    cdtype = common_kind(A.dtype, B.dtype)
    if cdtype.kind == 'c':
        acum_dtype = complex128
    else:
        acum_dtype = float64

    return numpy.sum(A.data.conj() * B.data / (variance_term(psd1, psd2, template1, relative_amplification).data), dtype=acum_dtype)


def sigmasq_twodet(template1=None, psd1=None, psd2=None, relative_amplification=1.0, low_frequency_cutoff=None, high_frequency_cutoff=None):
    template1 = make_frequency_series(template1)
    N = (len(psd1)-1) * 2
    norm = 8.0 * psd1.delta_f
    kmin, kmax = get_cutoff_indices(low_frequency_cutoff,
                                    high_frequency_cutoff, psd1.delta_f, N)
    ttilde = numpy.abs(template1[kmin:kmax])**2

    if psd1 is None or psd2 is None:
        raise ValueError("psd1 and/or psd2 is not provided")
    try:
        numpy.testing.assert_almost_equal(psd1.delta_f, psd2.delta_f)
    except AssertionError:
        raise ValueError('delta_f are not matching between psd1, psd2')

    sq = inner_product_AB(ttilde, ttilde, psd1[kmin:kmax], psd2[kmin:kmax],
                              template1=template1[kmin:kmax], relative_amplification=relative_amplification)
    sq *= relative_amplification
    return sq.real * norm


def matched_filter_twodet_noslide_core(data1, data2, psd1=None, psd2=None, template1=None, relative_amplification=1.0,
                         low_frequency_cutoff=None, high_frequency_cutoff=None):
    stilde1 = make_frequency_series(data1)
    stilde2 = make_frequency_series(data2)
    template1 = make_frequency_series(template1)

    if len(stilde1) != len(stilde2) or len(stilde1) != len(template1):
        raise ValueError("Length of template and data must match")

    N = (len(stilde2)-1) * 2
    kmin, kmax = get_cutoff_indices(low_frequency_cutoff,
                                   high_frequency_cutoff, stilde2.delta_f, N)

    qtilde = zeros(N, dtype=complex_same_precision_as(data1))

    correlate(stilde1[kmin:kmax], stilde2[kmin:kmax], qtilde[kmin:kmax])

    if psd1 is None or psd2 is None:
        raise ValueError("psd1 and/or psd2 is not provided")

    if isinstance(psd1, FrequencySeries) and isinstance(psd2, FrequencySeries):
        try:
            numpy.testing.assert_almost_equal(stilde2.delta_f, psd1.delta_f)
            numpy.testing.assert_almost_equal(stilde2.delta_f, psd2.delta_f)
        except AssertionError:
            raise ValueError("PSD delta_f does not match data")

        ttilde = numpy.abs(template1[kmin:kmax])**2

        out = inner_product_AB(ttilde, qtilde[kmin:kmax], psd1=psd1[kmin:kmax], psd2=psd2[kmin:kmax],
                               template1=template1[kmin:kmax], relative_amplification=relative_amplification)
        out *= numpy.sqrt(relative_amplification)
    else:
        raise TypeError("PSD must be a FrequencySeries")

    norm_twodet = sigmasq_twodet(template1, psd1, psd2, relative_amplification, low_frequency_cutoff, high_frequency_cutoff)

    norm = (8.0 * stilde2.delta_f) / numpy.sqrt(norm_twodet)

    return (out, norm)


def matched_filter_twodet_noslide(data1, data2, psd1=None, psd2=None, template1=None, relative_amplification=1.0,
                         low_frequency_cutoff=None, high_frequency_cutoff=None):

    snr, norm = matched_filter_twodet_noslide_core(data1, data2, psd1=psd1, psd2=psd2,
            template1=template1, relative_amplification=relative_amplification,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff)
    return snr * norm


def matched_filter_twodet_core(data1, data2, psd1=None, psd2=None, template1=None, relative_amplification=1.0,
                           low_frequency_cutoff=None, high_frequency_cutoff=None):

    stilde1 = make_frequency_series(data1)
    stilde2 = make_frequency_series(data2)
    template1 = make_frequency_series(template1)

    if len(stilde1) != len(stilde2) or len(stilde1) != len(template1):
        raise ValueError("Length of template and data must match")

    N = (len(stilde2)-1) * 2
    kmin, kmax = get_cutoff_indices(low_frequency_cutoff,
                                   high_frequency_cutoff, stilde2.delta_f, N)


    qtilde = zeros(N, dtype=complex_same_precision_as(data1))

    _q = zeros(N, dtype=complex_same_precision_as(data1))

    correlate(stilde1[kmin:kmax], stilde2[kmin:kmax], qtilde[kmin:kmax])

    if psd1 is None or psd2 is None:
        raise ValueError("psd1 and/or psd2 is not provided")

    if isinstance(psd1, FrequencySeries) and isinstance(psd2, FrequencySeries):
        try:
            numpy.testing.assert_almost_equal(stilde2.delta_f, psd1.delta_f)
            numpy.testing.assert_almost_equal(stilde2.delta_f, psd2.delta_f)
        except AssertionError:
            raise ValueError("PSD delta_f does not match data")

        ttilde = numpy.abs(template1[kmin:kmax])**2
        qtilde[kmin:kmax] *= ttilde/(variance_term(psd1[kmin:kmax], psd2[kmin:kmax],
                                              template1[kmin:kmax], relative_amplification=relative_amplification).data)
        qtilde[kmin:kmax] *= numpy.sqrt(relative_amplification)
    else:
        raise TypeError("PSD must be a FrequencySeries")

    ifft(qtilde, _q)

    norm_twodet = sigmasq_twodet(template1, psd1, psd2, relative_amplification, low_frequency_cutoff, high_frequency_cutoff)

    norm = (8.0 * stilde2.delta_f) / numpy.sqrt(norm_twodet)

    return (TimeSeries(_q, epoch=stilde2._epoch, delta_t=stilde2.delta_t, copy=False),
           FrequencySeries(qtilde, epoch=stilde2._epoch, delta_f=stilde2.delta_f, copy=False),
           norm)


def matched_filter_twodet(data1, data2, psd1=None, psd2=None, template1=None, relative_amplification=1.0,
                           low_frequency_cutoff=None, high_frequency_cutoff=None):

    snr, _, norm = matched_filter_twodet_core(data1, data2, psd1=psd1, psd2=psd2,
            template1=template1, relative_amplification=relative_amplification,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff)
    return snr * norm
