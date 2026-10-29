"""Check the experiment queue's jobs, order, and failure handling without running them."""

import threading
import time

import run_queue


def names(jobs):
    return [job.name for job in jobs]


def test_jobs_run_the_stage_commands_and_comparisons_wait_for_their_tuning():
    jobs = run_queue.build_jobs(
        ["tune_random", "best_uq", "supervised", "random_search"],
        ["quishpi"],
        ["bert_base"],
        ["--seed-workers", "2"],
    )
    by_name = {job.name: job for job in jobs}
    assert set(by_name) == {
        "tune_random/quishpi/bert_base",
        "best_uq/quishpi/bert_base",
        "supervised/quishpi/bert_base",
        "random_search/quishpi/bert_base",
    }
    assert by_name["tune_random/quishpi/bert_base"].commands == [
        [
            "uv",
            "run",
            "scripts/run_search.py",
            "--config",
            "configs/quishpi/tune_random/bert_base.json",
            "--seed-workers",
            "2",
        ]
    ]
    best = by_name["best_uq/quishpi/bert_base"]
    assert best.needs == ["tune_random/quishpi/bert_base"]
    assert str(best.freeze) == "configs/quishpi/tune_random/bert_base.json"
    assert [command[-3:] for command in best.commands] == [
        ["configs/quishpi/best_uq/bert_base.json", "--seed-workers", "2"],
        ["--gumbel-noise", "--seed-workers", "2"],
    ]
    assert not by_name["supervised/quishpi/bert_base"].needs
    assert not by_name["random_search/quishpi/bert_base"].needs
    # Without the tuning stage, comparisons freeze whatever winner tuning already saved.
    alone = run_queue.build_jobs(["best_uq"], ["quishpi"], ["bert_base"], [])
    assert alone[0].needs == [] and alone[0].freeze is not None


def test_pilot_jobs_cover_both_variants_of_every_checked_in_config():
    jobs = run_queue.build_jobs(["pilot"], ["pet", "quishpi"], ["bert_base", "modernbert_base"], [])
    assert len(jobs) == 8
    assert "pilot/quishpi/modernbert_base_split_avg" in names(jobs)
    assert not run_queue.build_jobs(["pilot"], ["medical"], ["bert_base"], [])


def test_longest_jobs_and_tuning_with_waiting_comparisons_start_first():
    jobs = run_queue.build_jobs(
        ["tune_random", "best_uq", "random_search"],
        ["pet", "quishpi"],
        ["bert_base", "modernbert_base"],
        [],
    )
    started = []
    run_queue.run_queue(jobs, 1, lambda job: started.append(job.name) or "done")
    assert started[0] == "tune_random/pet/modernbert_base"
    assert started.index("best_uq/pet/modernbert_base") > started.index(
        "tune_random/pet/modernbert_base"
    )
    assert started.index("tune_random/quishpi/bert_base") > started.index(
        "tune_random/pet/bert_base"
    )


def test_failure_blocks_only_its_dependents_and_slots_are_respected():
    jobs = run_queue.build_jobs(
        ["tune_random", "best_uq", "supervised"], ["pet", "quishpi"], ["bert_base"], []
    )
    lock = threading.Lock()
    running = peak = 0

    def execute(job):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.01)
        with lock:
            running -= 1
        return "failed" if job.name == "tune_random/pet/bert_base" else "done"

    status = run_queue.run_queue(jobs, 2, execute)
    assert peak == 2
    assert status == {
        "tune_random/pet/bert_base": "failed",
        "best_uq/pet/bert_base": "blocked",
        "supervised/pet/bert_base": "done",
        "tune_random/quishpi/bert_base": "done",
        "best_uq/quishpi/bert_base": "done",
        "supervised/quishpi/bert_base": "done",
    }


def test_a_skipped_comparison_counts_as_finished_and_an_error_as_a_failure():
    jobs = run_queue.build_jobs(["tune_random", "best_uq"], ["pet"], ["bert_base"], [])

    def execute(job):
        if job.stage == "tune_random":
            raise RuntimeError("crashed")
        return "skipped"

    assert run_queue.run_queue(jobs, 2, execute) == {
        "tune_random/pet/bert_base": "failed",
        "best_uq/pet/bert_base": "blocked",
    }
    assert run_queue.run_queue(jobs[1:], 2, execute) == {"best_uq/pet/bert_base": "skipped"}


def test_jobs_share_the_cores_unless_threads_are_already_set(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(run_queue, "available_cores", lambda: 32)
    assert run_queue.job_environment(3)["OMP_NUM_THREADS"] == "10"
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert run_queue.job_environment(3)["OMP_NUM_THREADS"] == "4"
