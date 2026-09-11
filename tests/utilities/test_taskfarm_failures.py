"""Failure records must distinguish an exception from a successful None result."""

import pickle
from types import SimpleNamespace

import pytest

pytest.importorskip("mpi4py")
from gprMax.taskfarm import TaskFailure, TaskfarmError, TaskfarmExecutor
from gprMax.taskfarm import Tags


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


@pytest.mark.unit
@pytest.mark.parametrize("exception", [None, "jobs", "master", "interrupt"])
def test_collective_batch_joins_before_reporting_on_all_ranks(exception):
    events = []
    broadcasts = []
    results = [None, {"output": "completed"}]
    failure = TaskFailure("ValueError", "bad input")
    error = (
        TaskfarmError([failure, results[1]])
        if exception == "jobs"
        else ValueError("master failure")
        if exception == "master"
        else KeyboardInterrupt()
        if exception == "interrupt"
        else None
    )

    def submit(jobs):
        if error is not None:
            raise error
        return results

    def broadcast(value, root):
        assert events[-1] == "joined"
        broadcasts.append(value)
        return value

    master = SimpleNamespace(
        master=0,
        start=lambda: events.append("started"),
        join=lambda: events.append("joined"),
        is_master=lambda: True,
        submit=submit,
        comm=SimpleNamespace(bcast=broadcast),
    )
    if error is None:
        assert TaskfarmExecutor.run_collective(master, [{}, {}]) is results
    else:
        with pytest.raises(type(error)) as caught:
            TaskfarmExecutor.run_collective(master, [{}, {}])
        assert caught.value is error
        if exception == "jobs":
            assert caught.value.results[1] == results[1]
    assert events == ["started", "joined"]
    worker = SimpleNamespace(
        master=0,
        start=lambda: None,
        join=lambda: None,
        is_master=lambda: False,
        comm=SimpleNamespace(bcast=lambda value, root: broadcasts[0]),
    )
    if error is None:
        assert TaskfarmExecutor.run_collective(worker, [{}, {}]) is None
    elif exception == "jobs":
        with pytest.raises(TaskfarmError) as caught:
            TaskfarmExecutor.run_collective(worker, [{}, {}])
        assert caught.value.failures == {0: failure}
    else:
        with pytest.raises(RuntimeError, match="Task-farm master failed"):
            TaskfarmExecutor.run_collective(worker, [{}, {}])


@pytest.mark.unit
def test_join_drains_ready_and_inflight_done_before_exit():
    pending = {1: [Tags.READY, Tags.EXIT], 2: [Tags.DONE, Tags.READY, Tags.EXIT]}
    sent = []

    class Comm:
        name = "test"

        def send(self, value, dest, tag):
            sent.append((dest, tag))

        def Iprobe(self, source, tag, status):
            if not pending[source]:
                return False
            status.Set_tag(pending[source][0])
            return True

        def recv(self, source, tag):
            assert pending[source].pop(0) == tag

    executor = SimpleNamespace(
        is_master=lambda: True,
        comm=Comm(),
        workers=(1, 2),
        busy=[False, True],
        _up=True,
    )
    TaskfarmExecutor.join(executor)
    assert sent == [(1, Tags.EXIT), (2, Tags.EXIT)]
    assert pending == {1: [], 2: []}
    assert executor.busy == [False, False]
    assert not executor._up
