"""Deterministic scientific plots from complete, unfiltered paired DEV panels."""
from pathlib import Path
from statistics import mean, stdev
import math
import os
import tempfile

from fh12.common import object_sha, read_json, sha256
from ablr2.common import immutable_json
from ablr2.plan import MAIN_CASES


def plot_data(report):
    """Pure plotting data; never select seeds/cases or change the table order."""
    if (report.get('schema') != 'ABLR2_BALANCED_PANEL_v1' or report.get('complete') is not True
            or report.get('sweep_count') != 5 or report.get('student_runs') != 85
            or report.get('phase') not in ('BOOT5', 'REFRESH5', 'VERIFY5')):
        raise ValueError('Plots require a complete registered five-sweep balanced panel')
    rows = report['panelrows']
    lookup = {(row['case_id'], row['sweep']): row for row in rows}
    sweeps = sorted({row['sweep'] for row in rows})
    if (len(rows) != 85 or len(lookup) != 85 or len(sweeps) != 5
            or set(lookup) != {(case, sweep) for case in MAIN_CASES for sweep in sweeps}):
        raise ValueError('Incomplete/duplicate case or seed; plotting may not drop an unfavorable point')
    seeds = {}
    for sweep in sweeps:
        seeds_in_sweep = {lookup[case, sweep]['student_seed'] for case in MAIN_CASES}
        if len(seeds_in_sweep) != 1 or None in seeds_in_sweep:
            raise ValueError('Every plotted component must share the same sweep Student seed')
        seeds[sweep] = next(iter(seeds_in_sweep))
    for row in rows:
        if (row.get('sensor') != report['sensor'] or row.get('data_sha256') != report['data_sha256']
                or row.get('source_identity') != report['source_identity']):
            raise ValueError('Plot observation and panel source/sensor/data identity differ')
        for metric in ('HQNR', 'ERGAS'):
            value = row['VAL'][metric]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError('Nonfinite or missing plot metric')
    panels = {}
    for metric in ('HQNR', 'ERGAS'):
        for kind in ('ladder', 'removal'):
            cases = tuple(f'C{i:02}' for i in (range(8) if kind == 'ladder' else range(8, 17)))
            series = []
            for case in cases:
                values = [lookup[case, sweep]['VAL'][metric] for sweep in sweeps]
                if kind == 'removal':
                    values = [lookup['C07', sweep]['VAL'][metric] - value for sweep, value in zip(sweeps, values)]
                series.append(dict(case_id=case, values=values, mean=mean(values), sample_sd=stdev(values),
                                   run_ids=[lookup[case, sweep]['run_id'] for sweep in sweeps]))
            panels[kind + '_' + metric.lower()] = dict(kind=kind, metric=metric, series=series,
                definition='native VAL-selected value' if kind == 'ladder' else 'C07 FULL minus ablation, paired within each sweep',
                improvement_direction='higher' if metric == 'HQNR' else 'lower')
    return dict(schema='ABLR2_PLOT_DATA_v1', source_report_sha256=object_sha(report), sensor=report['sensor'],
                recipe_id=report['recipe_id'], recipe_revision=report['recipe_revision'], phase=report['phase'], wave=report['wave'],
                source_identity=report['source_identity'], data_sha256=report['data_sha256'],
                selection='RR_VAL_SELECTED', sweeps=sweeps, student_seeds=seeds, panels=panels,
                statistic='mean +/- sample SD (ddof=1), all five raw matched pipeline repeats',
                independent_test=False, test_aware=True, filtered_observations=0, reordered_cases=False)


def render_wave(report, output_dir):
    """Render four PNGs with Agg. Missing plotting support is a report error only."""
    data = plot_data(report)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / 'plot_data.json', data)
    manifest_path = output / 'plot_manifest.json'
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get('plot_data_sha256') != object_sha(data) or set(manifest.get('figures', {})) != set(data['panels']):
            raise ValueError('Existing plot manifest belongs to different complete-panel data')
        for item in manifest['figures'].values():
            if Path(item['file']).name != item['file'] or sha256(output / item['file']) != item['sha256']:
                raise ValueError('An archived panel figure changed after publication')
        return manifest
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figures = {}
    for name, panel in data['panels'].items():
        figure, axis = plt.subplots(figsize=(9.0, 5.1))
        try:
            series = panel['series']
            xs = list(range(len(series)))
            for index, sweep in enumerate(data['sweeps']):
                offset = (index - 2) * .075
                axis.scatter([x + offset for x in xs], [row['values'][index] for row in series],
                             s=28, alpha=.82, color=f'C{index}',
                             label=f'{sweep} / seed {data["student_seeds"][sweep]}', zorder=3)
            axis.errorbar(xs, [row['mean'] for row in series], yerr=[row['sample_sd'] for row in series],
                          color='black', fmt='D', markersize=4, capsize=4, linewidth=1.1,
                          label='Mean +/- sample SD', zorder=4)
            if panel['kind'] == 'removal':
                axis.axhline(0, color='.4', linewidth=.9, linestyle='--', zorder=1)
                axis.set_ylabel(f'Delta {panel["metric"]}: FULL C07 - ablation')
            else:
                axis.set_ylabel(panel['metric'])
            axis.set_xticks(xs)
            axis.set_xticklabels([row['case_id'] for row in series])
            axis.set_xlabel('Fixed component order' if panel['kind'] == 'ladder' else 'Fixed FULL-removal order')
            axis.set_title(f'{data["sensor"]} / {data["recipe_id"]} / {data["phase"]} {data["wave"]}\n'
                           f'{panel["metric"]}: {"cumulative components" if panel["kind"] == "ladder" else "paired removal effects"}')
            axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
            axis.ticklabel_format(axis='y', style='plain', useOffset=False)
            axis.grid(axis='y', alpha=.2, zorder=0)
            axis.legend(loc='best', fontsize=7, ncol=2)
            figure.text(.5, .015, 'VAL-selected | all 5 matched pipeline repeats | TEST_AWARE_DEV | error bars: sample SD',
                        ha='center', fontsize=8)
            figure.tight_layout(rect=(0, .04, 1, 1))
            fd, temporary = tempfile.mkstemp(prefix='.' + name + '-', suffix='.png', dir=output)
            os.close(fd)
            temporary = Path(temporary)
            destination = output / (name + '.png')
            try:
                figure.savefig(temporary, dpi=180, facecolor='white', metadata={'Software': 'ABLR2 matplotlib ' + matplotlib.__version__})
                if destination.exists():
                    if sha256(destination) != sha256(temporary):
                        raise ValueError('Refusing to overwrite a different existing panel figure')
                else:
                    os.replace(temporary, destination)
                figures[name] = dict(file=destination.name, sha256=sha256(destination))
            finally:
                if temporary.exists():
                    temporary.unlink()
        finally:
            plt.close(figure)
    manifest = dict(schema='ABLR2_PLOT_MANIFEST_v1', plot_data_sha256=object_sha(data),
                    source_report_sha256=data['source_report_sha256'], figures=figures,
                    renderer='matplotlib', renderer_version=matplotlib.__version__, backend='Agg',
                    all_seed_points=True, uncertainty='sample SD, not confidence interval',
                    negative_and_reversed_effects_retained=True)
    immutable_json(manifest_path, manifest)
    return manifest
