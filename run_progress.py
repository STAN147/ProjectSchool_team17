"""Periodic training logs and a read-only status file for a benchmark run."""

import atexit
import csv
import json
import math
import os
import socket
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc)


def duration(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class RunProgress:
    def __init__(self, user, config, interval=60, synchronize=lambda: None):
        self.user, self.interval, self.synchronize = user, interval, synchronize
        self.started_at, self.started = now(), time.monotonic()
        self.status, self.error, self.current = "running", None, None
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.rates = defaultdict(list)
        self.history_loaded = set()
        self.pending = {}
        self.trial_limit = config["n_trials"]
        self.weights = {}
        for dataset in config["datasets"]:
            info = json.loads((Path("data") / dataset / "info.json").read_text(encoding="utf-8"))
            n = sum(info[key] for key in ("train_size", "val_size", "test_size"))
            temporary = math.ceil(.36 * n)
            train, test = n - temporary, math.ceil((20 / 36) * temporary)
            val, batch = temporary - test, min(config["batch_size"], train)
            train_pairs = 0
            for start in range(0, train, batch):
                b = min(batch, train - start)
                if b >= 2:
                    train_pairs += b * (b + int(.325 * (train - b)))
            pairs = train_pairs + val * min(train, 100000)
            for model in config["models"]:
                key = (dataset, model)
                self.pending[key] = {"trials": config["n_trials"], "final": len(config["seeds"])}
                self.weights[key] = {
                    "train": max(1, pairs if model == "modernnca" else n),
                    "predict": max(1, test * min(train, 1000000) if model == "modernnca" else test),
                }
        folder = Path(config["results_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"progress_{user}.json"
        self.timings_path = folder / f"timings_{user}_{self.started_at:%Y%m%d_%H%M%S_%fZ}.csv"
        self.columns = ["dataset", "model", "phase", "operation", "trial", "seed", "started_at", "finished_at", "seconds", "status", "error"]
        with self.timings_path.open("x", encoding="utf-8", newline="") as output:
            csv.DictWriter(output, fieldnames=self.columns).writeheader()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.previous_excepthook = sys.excepthook
        sys.excepthook = self._exception
        atexit.register(self._at_exit)
        self.report("START")
        self.thread.start()

    def _estimate(self):
        if self.status != "running":
            return (0.0 if self.status == "finished" else None), []
        remaining, unknown = 0.0, set()
        for key, pending in self.pending.items():
            model = key[1]
            fits = pending["trials"] + pending["final"]
            if not fits:
                continue
            train_rates = self.rates[(model, "train")]
            if not train_rates:
                unknown.add(model)
                continue
            remaining += fits * self.weights[key]["train"] * sum(train_rates) / len(train_rates)
            predict_rates = self.rates[(model, "predict")]
            if predict_rates:
                remaining += pending["final"] * self.weights[key]["predict"] * sum(predict_rates) / len(predict_rates)
        if unknown:
            return None, sorted(unknown)
        if self.current and self.current["operation"] in ("train", "predict"):
            key = (self.current["dataset"], self.current["model"])
            operation = self.current["operation"]
            rates = self.rates[(key[1], operation)]
            if rates:
                expected = self.weights[key][operation] * sum(rates) / len(rates)
                remaining -= min(expected, time.monotonic() - self.current["clock"])
        return max(0.0, remaining), []

    def _write(self):
        remaining, unknown = self._estimate()
        timestamp = now()
        finish = timestamp + timedelta(seconds=remaining) if remaining is not None else None
        current = {key: value for key, value in self.current.items() if key != "clock"} if self.current else None
        if current:
            current["elapsed_seconds"] = round(time.monotonic() - self.current["clock"], 3)
        state = {
            "user": self.user, "pid": os.getpid(), "host": socket.gethostname(),
            "status": self.status, "error": self.error,
            "started_at": self.started_at.isoformat(), "updated_at": timestamp.isoformat(),
            "elapsed_seconds": round(time.monotonic() - self.started, 3), "current": current,
            "remaining_trials_limit": sum(job["trials"] for job in self.pending.values()),
            "remaining_final_runs": sum(job["final"] for job in self.pending.values()),
            "estimated_remaining_seconds": round(remaining, 3) if remaining is not None else None,
            "estimated_finish_utc": finish.isoformat() if finish else None,
            "estimated_finish_moscow": finish.astimezone(timezone(timedelta(hours=3))).isoformat() if finish else None,
            "waiting_for_model_timings": unknown,
            "eta_note": "Rough estimate for this user, using measured fit times and remaining trial cap. NCA scales by train/validation pair counts; other models by rows. Early stopping, parameters and features can change it. Test cost is added after its first measurement; loading/preprocessing is not forecast.",
            "timings_file": str(self.timings_path),
        }
        temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
        return state

    def report(self, event):
        with self.lock:
            state = self._write()
            stage = state["current"] or {}
            finish = state["estimated_finish_moscow"] or ("unavailable: run stopped" if self.status != "running" else "waiting for first model timings")
            print(f"[RUN {now():%Y-%m-%dT%H:%M:%SZ}] {event} user={self.user} "
                  f"dataset={stage.get('dataset', '-')} model={stage.get('model', '-')} "
                  f"phase={stage.get('phase', '-')} operation={stage.get('operation', '-')} "
                  f"trial={stage.get('trial', '-')} seed={stage.get('seed', '-')} "
                  f"stage_elapsed={duration(stage.get('elapsed_seconds', 0))} "
                  f"run_elapsed={duration(state['elapsed_seconds'])} "
                  f"remaining~={duration(state['estimated_remaining_seconds'])} finish~={finish}", flush=True)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self.report("HEARTBEAT")

    @contextmanager
    def measure(self, dataset, model="-", phase="final", operation="train", trial=None, seed=None):
        self.synchronize()
        started, clock = now(), time.monotonic()
        timing = {"dataset": dataset, "model": model, "phase": phase, "operation": operation,
                  "trial": trial, "seed": seed, "started_at": started.isoformat(),
                  "status": "finished", "error": None}
        with self.lock:
            self.current = {**timing, "clock": clock}
            self.report("TRAIN_START" if operation == "train" else "STAGE_START")
        try:
            yield timing
            self.synchronize()
        except BaseException as error:
            timing.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            timing.update(finished_at=now().isoformat(), seconds=round(time.monotonic() - clock, 6))
            with self.lock:
                if timing["status"] == "finished" and operation in ("train", "predict"):
                    self.rates[(model, operation)].append(timing["seconds"] / self.weights[(dataset, model)][operation])
                with self.timings_path.open("a", encoding="utf-8", newline="") as output:
                    csv.DictWriter(output, fieldnames=self.columns).writerow(timing)
                self.report(f"{operation.upper()}_{timing['status'].upper()} seconds={timing['seconds']:.3f}")
                self.current = None
                self._write()

    def register_study(self, dataset, model, study):
        with self.lock:
            complete = [trial for trial in study.get_trials(deepcopy=False) if trial.state.name == "COMPLETE"]
            self.pending[(dataset, model)]["trials"] = max(0, self.trial_limit - len(complete))
            if study.study_name not in self.history_loaded:
                for trial in complete:
                    seconds = trial.user_attrs.get("training_seconds")
                    if seconds is None and trial.duration is not None:
                        seconds = trial.duration.total_seconds()
                    if seconds is not None:
                        self.rates[(model, "train")].append(seconds / self.weights[(dataset, model)]["train"])
                self.history_loaded.add(study.study_name)
            self._write()

    def trial_finished(self, dataset, model, study, trial):
        with self.lock:
            if trial.state.name == "COMPLETE":
                job = self.pending[(dataset, model)]
                job["trials"] = max(0, job["trials"] - 1)
            self._write()

    def tuning_finished(self, dataset, model):
        with self.lock:
            self.pending[(dataset, model)]["trials"] = 0
            self.report(f"TUNING_FINISHED dataset={dataset} model={model}")

    def final_finished(self, dataset, model):
        with self.lock:
            job = self.pending[(dataset, model)]
            job["final"] = max(0, job["final"] - 1)
            self._write()

    def close(self, status="finished", error=None):
        with self.lock:
            if self.status != "running":
                return
            self.status, self.error, self.current = status, error, None
            if status == "finished":
                for job in self.pending.values():
                    job.update(trials=0, final=0)
            self.stop.set()
            self.report(status.upper())
        self.thread.join()
        sys.excepthook = self.previous_excepthook

    def _exception(self, kind, value, traceback):
        self.close("interrupted" if issubclass(kind, KeyboardInterrupt) else "failed", f"{kind.__name__}: {value}")
        self.previous_excepthook(kind, value, traceback)

    def _at_exit(self):
        if self.status == "running":
            self.close("interrupted", "Process exited before the run completed")
