"""
Aggregate MO-CRS evaluation JSON reports into a thesis-ready CSV table.

Usage example (run from src/):
  python aggregate_eval_reports.py --input_dir ../logs --output ../logs/thesis_eval_summary.csv
"""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List


CSV_COLUMNS = [
    'report_file',
    'dataset',
    'mode',
    'split',
    'checkpoint',
    'ips',
    'snips',
    'dr',
    'dr_ci_low',
    'dr_ci_high',
    'dm',
    'dm_ci_low',
    'dm_ci_high',
    'logged_reward_mean',
    'logged_reward_std',
    'behavior_recommend_rate',
    'num_samples',
    'bootstrap_samples',
    'online_success_rate',
    'online_avg_turns',
    'online_diversity',
    'online_fairness',
]


def _to_float(value, default=0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _rows_from_report(report_path: str, report: Dict) -> List[Dict]:
    if not isinstance(report, dict):
        return []

    ope = report.get('ope', {})
    if not isinstance(ope, dict) or not ope:
        return []

    dataset = str(report.get('dataset', 'unknown'))
    mode = str(report.get('mode', 'unknown'))
    checkpoint = str(report.get('checkpoint', '')) if report.get('checkpoint') is not None else ''

    online = report.get('online_simulator', {})
    if not isinstance(online, dict):
        online = {}

    rows = []
    for split_name in ['validation', 'test']:
        split_metrics = ope.get(split_name)
        if not isinstance(split_metrics, dict):
            continue

        row = {
            'report_file': os.path.basename(report_path),
            'dataset': dataset,
            'mode': mode,
            'split': split_name,
            'checkpoint': checkpoint,
            'ips': _to_float(split_metrics.get('ips')),
            'snips': _to_float(split_metrics.get('snips')),
            'dr': _to_float(split_metrics.get('dr')),
            'dr_ci_low': _to_float(split_metrics.get('dr_ci_low')),
            'dr_ci_high': _to_float(split_metrics.get('dr_ci_high')),
            'dm': _to_float(split_metrics.get('dm')),
            'dm_ci_low': _to_float(split_metrics.get('dm_ci_low')),
            'dm_ci_high': _to_float(split_metrics.get('dm_ci_high')),
            'logged_reward_mean': _to_float(split_metrics.get('logged_reward_mean')),
            'logged_reward_std': _to_float(split_metrics.get('logged_reward_std')),
            'behavior_recommend_rate': _to_float(split_metrics.get('behavior_recommend_rate')),
            'num_samples': _to_float(split_metrics.get('num_samples')),
            'bootstrap_samples': _to_float(split_metrics.get('bootstrap_samples')),
            'online_success_rate': _to_float(online.get('success_rate')),
            'online_avg_turns': _to_float(online.get('avg_turns')),
            'online_diversity': _to_float(online.get('diversity')),
            'online_fairness': _to_float(online.get('fairness')),
        }
        rows.append(row)

    return rows


def aggregate_reports(input_dir: str, pattern: str, recursive: bool) -> List[Dict]:
    search_pattern = os.path.join(input_dir, '**', pattern) if recursive else os.path.join(input_dir, pattern)
    report_paths = sorted(glob.glob(search_pattern, recursive=recursive))

    all_rows = []
    for path in report_paths:
        if not path.lower().endswith('.json'):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        rows = _rows_from_report(path, payload)
        all_rows.extend(rows)

    return all_rows


def write_csv(rows: List[Dict], output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Aggregate per-dataset eval JSON reports into one CSV table')
    parser.add_argument('--input_dir', type=str, default='../logs', help='Directory containing eval JSON files')
    parser.add_argument('--pattern', type=str, default='*.json', help='File pattern to match (default: *.json)')
    parser.add_argument('--output', type=str, default='../logs/thesis_eval_summary.csv', help='Output CSV path')
    parser.add_argument('--recursive', action='store_true', help='Search input_dir recursively')
    args = parser.parse_args()

    rows = aggregate_reports(args.input_dir, args.pattern, args.recursive)
    write_csv(rows, args.output)

    print(f'Aggregated rows: {len(rows)}')
    print(f'CSV output: {os.path.abspath(args.output)}')


if __name__ == '__main__':
    main()
