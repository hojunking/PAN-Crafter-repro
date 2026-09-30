"""Independent evaluation of the bridge's emitted Student formulas (synthetic)."""
import math
import re
import statistics
import unittest

from reporting_bridge import rb_b01_contract as c
from reporting_bridge.rb_b01_summary import summarize
from reporting_bridge.rb_b01_sheets import SheetsBridge, SHEET_ID, column
from reporting_bridge.tests.test_rb_b01_contract_summary import synthetic_entry, synthetic_curve


def split_arguments(text):
    pieces = []
    depth = 0
    start = 0
    quoted = False
    for i, char in enumerate(text):
        if char == '"': quoted = not quoted
        if quoted: continue
        if char == '(': depth += 1
        elif char == ')': depth -= 1
        elif char == ',' and depth == 0:
            pieces.append(text[start:i]); start = i + 1
    pieces.append(text[start:])
    return pieces


def evaluate(expression, cells):
    """Small non-eval interpreter for this renderer's auditable formula subset."""
    text = expression.removeprefix('=')
    if text == '""': return ''
    if text in cells: return cells[text]
    try: return float(text)
    except ValueError: pass
    depth = 0
    quoted = False
    for i, char in enumerate(text):
        if char == '"': quoted = not quoted
        if quoted: continue
        if char == '(': depth += 1
        elif char == ')': depth -= 1
        elif depth == 0 and char in ('=', '-'):
            left, right = evaluate(text[:i], cells), evaluate(text[i+1:], cells)
            return left == right if char == '=' else left - right
    match = re.fullmatch(r'(IF|COUNT|AVERAGE|STDEV)\((.*)\)', text)
    if not match: raise AssertionError('Unsupported formula token: ' + text)
    name, args = match.groups()
    parts = split_arguments(args)
    if name == 'IF':
        return evaluate(parts[1] if evaluate(parts[0], cells) else parts[2], cells)
    values = [evaluate(part, cells) for part in parts]
    numeric = [value for value in values if type(value) in (int, float)]
    if name == 'COUNT': return len(numeric)
    if name == 'AVERAGE': return statistics.mean(numeric)
    return statistics.stdev(numeric)


class SummaryFormulaTests(unittest.TestCase):
    def test_source_formulas_equal_python_student_statistics(self):
        native, stress, cells = [], [], {}
        for case in c.registry():
            if case['server'] != 's1': continue
            entry = synthetic_entry(case)
            base = 2. + .1*c.CASES.index(case['case_id']) + .3*case['repeat']
            for selection in entry['report']['selections'].values():
                selection['rr']['ergas'] = base
            native.extend(c.canonical_native(entry))
            for mode in c.MODES:
                curve = synthetic_curve(entry, mode, base + (.2 if mode == 'A_ON' else 0.))
                if case['repeat'] == 1 and case['case_id'] == 'QMEAN':
                    curve['report']['curve'][1]['coverage_all_pan_paths'] = .98
                stress.extend(c.canonical_stress(curve))
        prepared = dict(native_records=native, stress_records=stress,
                        summary=summarize(native, stress))
        for record in native:
            row = c.build_native_row(record)
            for i, value in enumerate(row, 1):
                cells[f"'_rb01_{record['server']}'!{column(i)}{row[57]}"] = value
        for record in stress:
            number = c.stress_source_row(record)
            for i, value in enumerate(c.build_stress_row(record), 1):
                cells[f"'_rb02_points'!{column(i)}{number}"] = value
        adapter = type('NoNetworkAdapter', (), {'spreadsheet_id': SHEET_ID})()
        rendered, expected = SheetsBridge(adapter)._summary_rows(prepared)
        checked = 0
        for row, values in zip(rendered, expected):
            if values[0] not in ('native', 'paired', 'stress_radius'): continue
            for i, cell in enumerate(row['values']):
                formula = cell['userEnteredValue'].get('formulaValue')
                if not formula: continue
                actual = evaluate(formula, cells)
                desired = values[i]
                if desired is None: self.assertEqual(actual, '')
                else: self.assertTrue(math.isclose(actual, desired, rel_tol=1e-12, abs_tol=1e-12),
                                      (values[:5], i, actual, desired))
                checked += 1
        self.assertGreater(checked, 1000)

    def test_missing_cells_are_not_imputed_zero(self):
        cells = {"'_test'!A2": 4., "'_test'!A3": ''}
        result = evaluate('=IF(COUNT(\'_test\'!A2,\'_test\'!A3)=2,AVERAGE(\'_test\'!A2,\'_test\'!A3),"")', cells)
        self.assertEqual(result, '')


if __name__ == '__main__':
    unittest.main()
