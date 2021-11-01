import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp
import numbers

EPS = 1e-10

def calculate_metric(y_true, y_score, method=1, sensi_threshold=None, pos_label=None, sample_weight=None):
    """
    Calculate metric true and false positives per binary classification threshold.
    % Fast method of computing AUC, ROC. Splits into n different thresholds and
    % computes for those thresholds. Works for binary classification. 

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    method : int
        Different methods for choosing a point on the ROC curve for which 
        we compute the F1 scores, sensitivity,specificity and accuracy.
        The methods choose the point on the ROC curve where...
         * 1: ... the F1-score is maximal.
         * 2: ... sensitivity + specificity is maximal.
         * 3: ... specificity under X% percent sensitivity where X is the 
                  sensi_threshold.
         * 4: ... binarize at 0.5 level of the score
         * 5: ... binarize with ostu's method
    sensi_threshold: float, the percentage of sensitivity where we set the
        threshold to select the point in method 3
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    FPs : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    TPs : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    y_true_binary : 

    Final Return:
    --------------
    auroc,F1,sensitivity,specificity,accuracy,precision values

    """
    FPs, TPs, y_true_binary, y_score_trimmed = _binary_clf_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    # Given TPs and FPs, get FNs and TNs
    # negative = TN + FP
    # positive = TP + FN
    pos_total = sum(y_true_binary)
    neg_total = y_true_binary.size - pos_total
    FNs = pos_total - TPs
    TNs = neg_total - FPs
    # Calculate the metric
    F1_array = (2*TPs)/(FNs + 2*TPs + FPs + EPS) 
    sens_array = TPs/(TPs + FNs + EPS) 
    spec_array = TNs/(TNs + FPs + EPS) 
    acc_array  = (TPs + TNs)/(TPs + TNs + FPs + FNs + EPS)
    preci_array = TPs / (TPs + FPs + EPS)
    
    # Obtain the thresholding index baed on different method
    if method == 1:
        t = np.argmax(F1_array)
    elif method == 2:
        t = np.argmax(sens_array+spec_array)
    elif method == 3 and sensi_threshold is not None:
        idx_array = np.where(sens_array > sensi_threshold)[0]
        if _increasing(sens_array):
            t = idx_array[0]
        else:
            t = idx_array[-1]
    elif method == 4:
        # Set threshold of 0.5
        t = int(np.where(np.diff(y_score_trimmed>0.5))[0])
    elif method == 5:
        # find the threshold with Ostu's method
        thores = otsu(y_score)
        t = int(np.where(np.diff(y_score_trimmed>thores))[0])
   
    F1 = F1_array[t]
    sensitivity = sens_array[t]
    specificity = spec_array[t]
    accuracy = acc_array[t]
    precision = preci_array[t]
    
    # Obtain the auc score
    auroc = roc_auc_score(y_true, y_score)
    
    return precision,accuracy,sensitivity,specificity,F1,auroc

def otsu(input_vector):
    """
    The value of input_vector range from 0 to 1
    """
    input_vector = input_vector*255
    pixel_number = len(input_vector)
    mean_weight = 1.0/pixel_number
    his, bins = np.histogram(input_vector, np.arange(0,257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]: 
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        if pcb == 0 or pcf == 0:
            continue
        Wb = pcb * mean_weight
        Wf = pcf * mean_weight
        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)        
        value = Wb * Wf * (mub - muf) ** 2
        if value > final_value:
            final_thresh = t
            final_value = value
    final_thresh = final_thresh/255
    return final_thresh

def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    # # Check to make sure y_true is valid
    # y_type = type_of_target(y_true)
    # if not (y_type == "binary" or
    #         (y_type == "multiclass" and pos_label is not None)):
    #     raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (
            classes.dtype.kind in ('O', 'U', 'S') or
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError("y_true takes value in {{{classes_repr}}} and "
                         "pos_label is not specified: either make y_true "
                         "take value in {{0, 1}} or {{-1, 1}} or "
                         "pass pos_label explicitly.".format(
                             classes_repr=classes_repr))
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_true, y_score[threshold_idxs]


def _assert_all_finite(X, allow_nan=False, msg_dtype=None):
    """Like assert_all_finite, but only for ndarray."""
    # validation is also imported in extmath

    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method. The sum is also calculated
    # safely to reduce dtype induced overflows.
    is_float = X.dtype.kind in 'fc'
    if is_float and (np.isfinite(_safe_accumulator_op(np.sum, X))):
        pass
    elif is_float:
        msg_err = "Input contains {} or a value too large for {!r}."
        if (allow_nan and np.isinf(X).any() or
                not allow_nan and not np.isfinite(X).all()):
            type_err = 'infinity' if allow_nan else 'NaN, infinity'
            raise ValueError(
                    msg_err.format
                    (type_err,
                     msg_dtype if msg_dtype is not None else X.dtype)
            )
    # for object dtype data, we only check for NaNs (GH-13254)
    elif X.dtype == np.dtype('object') and not allow_nan:
        if _object_dtype_isnan(X).any():
            raise ValueError("Input contains NaN")

def _object_dtype_isnan(X):
    return X != X

def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.
    Parameters
    ----------
    op : function
        A numpy accumulator function such as np.mean or np.sum.
    x : ndarray
        A numpy array to apply the accumulator function.
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x.
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function.
    Returns
    -------
    result
        The output of the accumulator function passed to this function.
    """
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result 

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(np.isclose(out.take(-1, axis=axis), expected, rtol=rtol,
                             atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out

def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])

def column_or_1d(y, *, warn=False):
    """ Ravel column or 1d numpy array, else raises an error
    Parameters
    ----------
    y : array-like
    warn : boolean, default False
       To control display of warnings.
    Returns
    -------
    y : array
    """
    y = np.asarray(y)
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          DataConversionWarning, stacklevel=2)
        return np.ravel(y)
    raise ValueError(
        "y should be a 1d array, "
        "got an array of shape {} instead.".format(shape))

def assert_all_finite(X, *, allow_nan=False):
    """Throw a ValueError if X contains NaN or infinity.
    Parameters
    ----------
    X : array or sparse matrix
    allow_nan : bool
    """
    _assert_all_finite(X.data if sp.issparse(X) else X, allow_nan)

def _num_samples(x):
    """Return number of samples in array-like x."""
    message = 'Expected sequence or array-like, got %s' % type(x)
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, 'shape') and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error

def _increasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

if __name__ == '__main__':
    y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.35, 0.567, 0.43, 0.68, 0.32, 0.6])
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    fps, tps, thresholds = _binary_clf_curve(y_true, y_scores)
    print(fps, tps, thresholds)