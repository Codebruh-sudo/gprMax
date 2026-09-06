"""Failure records must distinguish an exception from a successful None result."""

import pickle
from types import SimpleNamespace

import pytest

pytest.importorskip("mpi4py")
from gprMax.taskfarm import TaskFailure, TaskfarmError, TaskfarmExecutor


@pytest.mark.unit
def test_worker_none_is_success_not_a_failure():
    worker = TaskfarmExecutor.__new__(TaskfarmExecutor)
    worker.rank, worker.master = 1, 0
    worker.func = lambda: None
    assert worker._TaskfarmExecutor__guarded_work({}) is None


@pytest.mark.unit
def test_worker_exception_is_pickle_safe_and_retains_other_results():
    worker = TaskfarmExecutor.__new__(TaskfarmExecutor)
    worker.rank, worker.master = 1, 0

    def fail():
        raise ValueError("bad input")

    worker.func = fail
    failure = pickle.loads(pickle.dumps(worker._TaskfarmExecutor__guarded_work({})))
    assert failure == TaskFailure("ValueError", "bad input")
    error = TaskfarmError([None, failure, {"output": "completed"}])
    assert list(error.failures) == [1]
    assert error.results[2] == {"output": "completed"}
    assert "job 2: ValueError: bad input" in str(error)


@pytest.mark.unit
def test_executor_context_joins_even_when_submit_raises():
    joined = []
    executor = SimpleNamespace(join=lambda: joined.append(True))
    assert TaskfarmExecutor.__exit__(executor, RuntimeError, RuntimeError("failed"), None) is False
    assert joined == [True]
