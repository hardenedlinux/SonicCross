#!/usr/bin/env python3
"""Compare canonical differential dumps and report semantic fields."""

import sys


def records(path):
    result = []
    current = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "record-begin":
            current = {"dispatch": []}
        elif line == "record-end":
            if current is None:
                raise ValueError("record-end without record-begin")
            result.append(current)
            current = None
        elif current is not None and line.startswith("dispatch: "):
            current["dispatch"].append(line[10:])
        elif current is not None and ": " in line:
            key, value = line.split(": ", 1)
            current[key] = value
    if current is not None:
        raise ValueError("unterminated record")
    return result


def view_groups(path):
    result = []
    current = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "view-group-begin":
            current = {}
        elif line == "view-group-end":
            if current is None:
                raise ValueError("view-group-end without view-group-begin")
            result.append(current)
            current = None
        elif current is not None and ": " in line:
            key, value = line.split(": ", 1)
            current[key] = value
    if current is not None:
        raise ValueError("unterminated view-group")
    return result


def mismatch(fixture, index, field, python_value, sonic_value, operator):
    print("Mismatch:", file=sys.stderr)
    print("  fixture: %s" % fixture, file=sys.stderr)
    print("  record: %d" % index, file=sys.stderr)
    print("  operator: %s" % operator, file=sys.stderr)
    print("  field: %s" % field, file=sys.stderr)
    print("  Python value: %s" % python_value, file=sys.stderr)
    print("  SonicCross value: %s" % sonic_value, file=sys.stderr)


if len(sys.argv) != 4:
    raise SystemExit("usage: compare.py FIXTURE PYTHON_DUMP SONIC_DUMP")

fixture, python_path, sonic_path = sys.argv[1:]
python_records = records(python_path)
sonic_records = records(sonic_path)
if len(python_records) != len(sonic_records):
    mismatch(fixture, -1, "record-count", len(python_records),
             len(sonic_records), "<record-count>")
    limit = min(len(python_records), len(sonic_records))
    for index in range(limit):
        left = python_records[index].get("operator", "<missing>")
        right = sonic_records[index].get("operator", "<missing>")
        if left != right:
            mismatch(fixture, index, "operator", left, right, left)
            break
    else:
        if len(python_records) > limit:
            mismatch(fixture, limit, "record", python_records[limit],
                     "<missing>", python_records[limit].get("operator", "<unknown>"))
        elif len(sonic_records) > limit:
            mismatch(fixture, limit, "record", "<missing>",
                     sonic_records[limit], sonic_records[limit].get("operator", "<unknown>"))
    raise SystemExit(1)

ignored = {"index", "dispatch"}
for index, (python_record, sonic_record) in enumerate(
        zip(python_records, sonic_records)):
    operator = python_record.get("operator", "<unknown>")
    for field in sorted(set(python_record) | set(sonic_record)):
        if field in ignored:
            continue
        left = python_record.get(field, "<missing>")
        right = sonic_record.get(field, "<missing>")
        if left != right:
            mismatch(fixture, index, field, left, right, operator)
            raise SystemExit(1)
    if python_record.get("dispatch", []) != sonic_record.get("dispatch", []):
        mismatch(fixture, index, "dispatch", python_record.get("dispatch", []),
                 sonic_record.get("dispatch", []), operator)
        raise SystemExit(1)

python_groups = view_groups(python_path)
sonic_groups = view_groups(sonic_path)
if len(python_groups) != len(sonic_groups):
    mismatch(fixture, -1, "view-group-count", len(python_groups),
             len(sonic_groups), "<view-group-count>")
    raise SystemExit(1)
for index, (python_group, sonic_group) in enumerate(zip(python_groups, sonic_groups)):
    for field in sorted(set(python_group) | set(sonic_group)):
        left = python_group.get(field, "<missing>")
        right = sonic_group.get(field, "<missing>")
        if left != right:
            mismatch(fixture, index, field, left, right,
                     python_group.get("root", "<unknown>"))
            raise SystemExit(1)

print("Python torchgen == SonicCross")
