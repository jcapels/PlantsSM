
def process_slices(_columns, _slice):

    if _slice.start is not None:
        start = _slice.start
    else:
        start = 0

    if _slice.stop is not None and _slice.stop != -1:
        stop = _slice.stop
    else:
        stop = _columns.size

    if _slice.step is not None:
        step = _slice.step
    else:
        step = 1

    indexes_list = list(range(start, stop, step))
    return indexes_list
