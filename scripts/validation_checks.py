"""Four-state diagnostics with explicit axes; this module never relaxes physics."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import numpy as np

STATUSES = ("PASS", "FAIL", "UNSUPPORTED", "NOT_RUN")


class Checks:
    def __init__(self, *, max_failure_locations: int = 5) -> None:
        if isinstance(max_failure_locations,bool) or not isinstance(max_failure_locations,int) or not 0 <= max_failure_locations <= 1000:
            raise ValueError("max_failure_locations must be an integer in [0,1000]")
        self.max_failure_locations = max_failure_locations
        self.rows: list[dict[str,object]] = []
        self._context = {"stage":"unspecified","fields":[],"axes":None,"timestamps":None,
                         "time_support":"not_applicable_unannotated_or_aggregate","relation_class":"P",
                         "engineering_simplification":"unspecified; constraint validation does not establish empirical realism"}

    @contextmanager
    def context(self, **annotations):
        previous = self._context
        self._context = self._annotation(annotations)
        try:
            yield self
        finally:
            self._context = previous

    def _annotation(self, overrides):
        allowed = set(self._context) | {"time_point"}
        if set(overrides)-allowed:
            raise ValueError(f"Unknown validation annotation: {sorted(set(overrides)-allowed)}")
        result = self._context | overrides
        if result["relation_class"] not in {"P","E","S"}:
            raise ValueError("relation_class must be P, E or S")
        result["fields"] = [result["fields"]] if isinstance(result["fields"],str) else list(result["fields"])
        return result

    def set_context(self, **annotations):
        """Set domain defaults; per-check annotations take precedence."""
        self._context = self._annotation(annotations)

    def _base(self,name,status,note,annotations):
        context = self._annotation(annotations)
        return {"name":name,"status":status,"passed":True if status=="PASS" else False if status=="FAIL" else None,
                "max_residual":None,"raw_max_signed":None,"max_violation":None,"max_error":None,
                "location":None,"raw_max_location":None,"max_violation_location":None,"failure_locations":[],"time":{"status":"not_applicable","support":context["time_support"],"value":None},
                "fields":context["fields"],"stage":context["stage"],"relation_class":context["relation_class"],
                "engineering_simplification":context["engineering_simplification"],"note":note},context

    def _location(self,index,shape,context):
        axes = context["axes"]
        if axes is not None and len(axes) != len(shape):
            raise ValueError("Declared check axes must match residual rank; never infer time from shape")
        location = {"array_index":list(index),"axes":list(axes) if axes is not None else None}
        time = {"status":"not_applicable","support":context["time_support"],"value":None}
        if context.get("time_point") is not None:
            time.update(status="explicit",value=_scalar(context["time_point"]))
        elif axes is not None and any(axis in {"time","state_time"} for axis in axes):
            temporal = [i for i,axis in enumerate(axes) if axis in {"time","state_time"}]
            if len(temporal) != 1 or context["timestamps"] is None:
                raise ValueError("One explicit time axis requires matching timestamps")
            dimension = temporal[0]
            stamps = np.asarray(context["timestamps"])
            if stamps.shape != (shape[dimension],):
                raise ValueError("Declared timestamps must match the explicitly annotated time axis")
            time.update(status="explicit",value=_scalar(stamps[index[dimension]]),index=int(index[dimension]))
        return location,time

    def _numeric(self,name,values,allowed,tolerance,unit,note,kind,annotations,relative_tolerance=0.0):
        values = np.asarray(values,dtype=np.float64)
        allowed = np.broadcast_to(np.asarray(allowed,dtype=np.float64),values.shape)
        if not np.isfinite(allowed).all() or np.any(allowed < 0):
            raise ValueError("Check tolerances and scales must be finite and nonnegative")
        row,context = self._base(name,"NOT_RUN" if values.size==0 else "PASS",note,annotations)
        row.update(unit=unit,tolerance=float(tolerance),relative_tolerance=float(relative_tolerance),sample_count=int(values.size))
        if values.size == 0:
            row["note"] = (note+"; " if note else "")+"No applicable array elements; constraint was not exercised"
            self.rows.append(row)
            return
        metric = np.abs(values) if kind=="equal" else values
        finite = np.isfinite(values)
        violation = np.maximum(metric-allowed,0)
        failed = ~finite | (violation > 0)
        passed = not failed.any()
        row.update(status="PASS" if passed else "FAIL",passed=passed,nonfinite_count=int((~finite).sum()))
        row["max_residual"] = float(np.max(np.abs(values))) if finite.all() else None
        row["raw_max_signed"] = float(np.max(values)) if finite.all() else None
        row["max_violation"] = float(np.max(violation)) if finite.all() else None
        row["max_error"] = float(np.max(np.abs(values))) if kind=="equal" and finite.all() else float(max(0,np.max(values))) if finite.all() else None
        ranking = np.where(finite,violation,np.inf) if not passed else np.abs(values)
        worst = tuple(int(x) for x in np.unravel_index(np.argmax(ranking),values.shape))
        row["location"],row["time"] = self._location(worst,values.shape,context)
        raw_index = tuple(int(x) for x in np.unravel_index(np.argmax(np.where(finite,np.abs(values),np.inf)),values.shape))
        violation_index = tuple(int(x) for x in np.unravel_index(np.argmax(np.where(finite,violation,np.inf)),values.shape))
        row["raw_max_location"],row["raw_max_time"] = self._location(raw_index,values.shape,context)
        row["max_violation_location"],row["max_violation_time"] = self._location(violation_index,values.shape,context)
        row["location_selection"] = "maximum_violation" if not passed else "maximum_absolute_raw_residual"
        row["residual_statistic"] = "maximum_absolute_raw_residual"
        row["residual_at_location"] = float(values[worst]) if finite[worst] else None
        if failed.any() and self.max_failure_locations:
            indices = np.flatnonzero(failed)
            order = np.argsort(-ranking.ravel()[indices],kind="stable")[:self.max_failure_locations]
            for flat in indices[order]:
                index = tuple(int(x) for x in np.unravel_index(flat,values.shape))
                location,time = self._location(index,values.shape,context)
                row["failure_locations"].append({"location":location,"time":time,
                    "residual":float(values[index]) if finite[index] else None,"violation":float(violation[index]) if finite[index] else None})
        self.rows.append(row)

    def equal(self,name,residual,tolerance,unit="",note="",*,relative_tolerance=0.0,scale=0.0,**annotations):
        if not np.isfinite(tolerance) or not np.isfinite(relative_tolerance) or tolerance < 0 or relative_tolerance < 0:
            raise ValueError("Check tolerances must be finite and nonnegative")
        allowed = tolerance+relative_tolerance*np.abs(np.asarray(scale,dtype=np.float64))
        self._numeric(name,residual,allowed,tolerance,unit,note,"equal",annotations,relative_tolerance)

    def upper(self,name,values,upper,tolerance,unit="",note="",**annotations):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Check tolerance must be finite and nonnegative")
        residual = np.asarray(values,dtype=np.float64)-np.asarray(upper,dtype=np.float64)
        self._numeric(name,residual,tolerance,tolerance,unit,note,"upper",annotations)

    def condition(self,name,passed,note="",*,sample_count=None,**annotations):
        value = np.asarray(passed)
        if value.dtype.kind != "b":
            raise ValueError("Condition checks require booleans; numeric/NaN truthiness is not evidence")
        if sample_count is not None and (isinstance(sample_count,bool) or not isinstance(sample_count,(int,np.integer)) or sample_count < 0):
            raise ValueError("Condition sample_count must be a nonnegative integer")
        empty = value.size==0 or sample_count==0
        status = "NOT_RUN" if empty else "PASS" if bool(np.all(value)) else "FAIL"
        row,context = self._base(name,status,note,annotations)
        row["sample_count"] = int(value.size if sample_count is None else sample_count)
        if not empty:
            row["max_violation"] = 0.0 if status=="PASS" else 1.0
        self.rows.append(row)

    def status(self,name,status,note="",**annotations):
        if status not in {"UNSUPPORTED","NOT_RUN"}:
            raise ValueError("Use equal/upper/condition for evaluated PASS or FAIL checks")
        row,_ = self._base(name,status,note,annotations)
        self.rows.append(row)

    def summary(self):
        counts = {status:sum(row["status"]==status for row in self.rows) for status in STATUSES}
        supported = counts["PASS"]+counts["FAIL"]
        return {"status":"FAIL" if counts["FAIL"] else "PASS" if supported else "NOT_RUN",
                "passed":False if counts["FAIL"] else True if supported else None,
                "counts":counts,"supported_check_count":supported,
                "scope":"Evaluated supported constraints only; UNSUPPORTED and NOT_RUN do not certify physical coverage"}


def _scalar(value):
    item = value.item() if isinstance(value,np.generic) else value
    return item if isinstance(item,(str,int,float,bool)) or item is None else str(item)


def input_error_report(world,exc):
    checks = Checks()
    checks.condition("read_and_align_inputs",False,str(exc),stage="input_alignment",fields=["exported_artifacts"],
                     relation_class="S",engineering_simplification="not_applicable_input_error")
    checks.rows[0]["error_category"] = "INPUT_OR_GENERATION_ERROR"
    return {"world":world,"status":"FAIL","passed":False,"error_category":"INPUT_OR_GENERATION_ERROR",
            "checks":checks.rows,"validation_scope":checks.summary(),"limitations":"Input/generation/alignment error; physical feasibility was not evaluated."}


def validation_context(**annotations):
    def decorate(function):
        @wraps(function)
        def wrapped(checks,*args,**kwargs):
            with checks.context(**annotations):
                return function(checks,*args,**kwargs)
        return wrapped
    return decorate
