"""History aggregation excludes missing/error clocks and keeps mode/factor identity."""
from copy import deepcopy
import unittest

from costmodel_tests.dashboard_metrics import (PROFILER, actual_route, primary_result,
                                              summarize_modes, ten_day_trend)


def metric(value=5, mode="auto", factor=1, **extra):
    return {"mode": mode, "factor": factor, "precision": True,
            "measurement_source": PROFILER, "timing_gate_applied": True,
            "ape_percent": value, "status": "PASS" if value <= 10 else "TIME_FAIL", **extra}


def batch(day="2026-10-29", hour="08:00:00", **extra):
    return {"id": day + '_' + hour, "started_at": day + 'T' + hour + '+08:00',
            "finished_at": day + 'T09:00:00+08:00', "status": "FAIL", "jobs": 4,
            "rows": [{"name": "first", "mode_results": [metric(20)]}], **extra}


class DashboardMetricsTests(unittest.TestCase):
    def test_summary_includes_over_limit_errors_and_real_zero(self):
        rows = [{"mode_results": [metric(v)]} for v in (0, 10, 20)]
        summary = summarize_modes(rows)
        self.assertEqual(summary['auto']['mape_percent'], 10)
        self.assertEqual(summary['auto']['median_ape_percent'], 10)
        self.assertEqual(summary['auto']['max_ape_percent'], 20)
        self.assertEqual(summary['auto']['over_limit_count'], 1)
        self.assertIsNone(summary['simd']['mape_percent'])

    def test_event_bad_precision_and_disabled_gate_are_excluded(self):
        values = [metric(5), metric(100, precision=False),
                  metric(100, measurement_source='device_event_median_us'),
                  metric(100, timing_gate_applied=False), metric(float('inf'))]
        result = summarize_modes([{"mode_results": [m]} for m in values])['auto']
        self.assertEqual(result['valid_count'], 1)
        self.assertEqual(result['mape_percent'], 5)

    def test_mix_uses_declared_factor_without_becoming_simt_factor_best_of(self):
        row = {'mode_results': [metric(0, 'mixed', status='NOT_RUN', precision=None),
                                metric(30, 'mixed', 4), metric(20, 'simt_only'),
                                metric(1, 'simt_only', 2)]}
        self.assertEqual(primary_result(row, 'mixed')['factor'], 4)
        self.assertEqual(summarize_modes([row])['mixed']['mape_percent'], 30)
        self.assertEqual(summarize_modes([row])['simt_only']['mape_percent'], 20)

    def test_actual_route_does_not_infer_execution_from_recommendation(self):
        self.assertIsNone(actual_route({'recommended_decision_kind': 'all_simd'})['actual_route'])
        result = actual_route({'effective_decision_kind': 'mixed_simd_simt',
                               'effective_runtime_factor': 4, 'recommended_decision_kind': 'all_simd'})
        self.assertEqual(result['actual_route'], 'Mix')
        self.assertEqual(result['actual_factor'], 4)

    def test_ten_calendar_days_and_missing_day_gaps(self):
        anchor = batch()
        points = ten_day_trend([batch('2026-10-19'), batch('2026-10-20'), anchor], anchor)
        self.assertEqual(len(points), 10)
        self.assertEqual(points[0]['date'], '2026-10-20')
        self.assertEqual(points[-1]['date'], '2026-10-29')
        self.assertIsNone(points[1]['run_id'])
        self.assertEqual(points[1]['modes'], {})
        self.assertEqual(points[-1]['cases']['first']['auto'], 20)

    def test_latest_completed_batch_not_best_error_and_running_does_not_replace_it(self):
        early = batch(hour='06:00:00')
        early['rows'][0]['mode_results'] = [metric(1)]
        later = batch(hour='08:00:00')
        blocked = batch(hour='10:00:00', status='BLOCKED')
        running = batch(hour='11:00:00', status='RUNNING', finished_at=None)
        point = ten_day_trend([early, later, blocked, running], running)[-1]
        self.assertEqual(point['run_id'], later['id'])
        self.assertEqual(point['modes']['auto']['mape_percent'], 20)

    def test_different_case_sets_and_concurrency_are_not_mixed(self):
        anchor = batch()
        other = batch(hour='10:00:00', jobs=1)
        different_cases = deepcopy(anchor)
        different_cases.update(id='other_cases', started_at='2026-10-29T12:00:00+08:00')
        different_cases['rows'][0]['name'] = 'different'
        self.assertEqual(ten_day_trend([anchor, other, different_cases], anchor)[-1]['run_id'], anchor['id'])

    def test_beijing_day_and_timestamp_order(self):
        utc = batch(id='utc-latest', started_at='2026-10-29T23:00:00Z')
        earlier = batch('2026-10-30', '06:00:00')
        point = ten_day_trend([utc, earlier], earlier)[-1]
        self.assertEqual(point['date'], '2026-10-30')
        self.assertEqual(point['run_id'], 'utc-latest')


if __name__ == '__main__':
    unittest.main()
