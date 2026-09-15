"""Killable, spawn-compatible per-fit workers. No workers without a time limit."""
import math
import multiprocessing as mp
import time


class FitTimeout(TimeoutError):
    def __init__(self, seconds, elapsed):
        self.elapsed = elapsed
        super().__init__(f'Fit exceeded {seconds:g} seconds (worker startup included); worker terminated')


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('fit timeout must be finite and positive')
    return seconds


def _worker(connection, function, args, kwargs):
    try:
        result = function(*args, **kwargs)
        connection.send(('ok', result))
    except BaseException as exc:
        connection.send(('error', f'{type(exc).__name__}: {exc}'))
    finally:
        connection.close()


def run_fit(function, *args, fit_timeout=None, **kwargs):
    """Limit the complete call, including startup, all starts, and fit diagnostics.

    Process startup/termination overhead may extend observed wall time slightly.
    The worker only computes a fit; result-file writes remain in the parent.
    """
    if fit_timeout is None:
        return function(*args, **kwargs)
    seconds = positive_seconds(fit_timeout)
    context = mp.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, function, args, kwargs))
    started = time.perf_counter()
    expired = False
    try:
        process.start()
        sender.close()
        remaining = max(0., seconds-(time.perf_counter()-started))
        if not receiver.poll(remaining):
            expired = True
        else:
            try:
                status, result = receiver.recv()
            except EOFError as exc:
                raise ValueError('Fit worker exited without returning a result') from exc
            if status != 'ok': raise ValueError('Fit worker: '+result)
            return result
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            if process.is_alive(): process.terminate()
            process.join(timeout=1.)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.)
            if process.is_alive():
                raise RuntimeError('Could not stop fit worker')
        process.close()
    if expired:
        raise FitTimeout(seconds, time.perf_counter()-started)
