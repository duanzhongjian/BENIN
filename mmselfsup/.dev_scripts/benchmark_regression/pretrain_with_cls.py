import argparse
import json
import os
import os.path as osp
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import yaml
from modelindex.load_model_index import load
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

console = Console()
MMSELFSUP_ROOT = Path(__file__).absolute().parents[2]
CYCLE_LEVELS = ['month', 'quarter', 'half-year', 'no-training']
METRICS_MAP = {
    'Top 1 Accuracy': 'accuracy/top1',
    'Top 5 Accuracy': 'accuracy/top5'
}


class RangeAction(argparse.Action):

    def __call__(self, parser, namespace, values: str, option_string):
        matches = re.match(r'([><=]*)([-\w]+)', values)
        if matches is None:
            raise ValueError(f'Unavailable range option {values}')
        symbol, range_str = matches.groups()
        assert range_str in CYCLE_LEVELS, \
            f'{range_str} are not in {CYCLE_LEVELS}.'
        level = CYCLE_LEVELS.index(range_str)
        symbol = symbol or '<='
        ranges = set()
        if '=' in symbol:
            ranges.add(level)
        if '>' in symbol:
            ranges.update(range(level + 1, len(CYCLE_LEVELS)))
        if '<' in symbol:
            ranges.update(range(level))
        assert len(ranges) > 0, 'No range are selected.'
        setattr(namespace, self.dest, ranges)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train models (in models.yml) and compare accuracy.')
    parser.add_argument(
        'partition', type=str, help='Cluster partition to use.')
    parser.add_argument(
        '--job-name',
        type=str,
        default='selfsup-benchmark',
        help='Slurm job name prefix')
    parser.add_argument('--port', type=int, default=29777, help='dist port')
    parser.add_argument(
        '--models', nargs='+', type=str, help='Specify model names to run.')
    parser.add_argument(
        '--range',
        type=str,
        default={0},
        action=RangeAction,
        metavar='{month,quarter,half-year,no-training}',
        help='The training benchmark range, "no-training" means all models '
        "including those we haven't trained.")
    parser.add_argument(
        '--work-dir',
        default='work_dirs/benchmark_pretrain_cls',
        help='the dir to save train log')
    parser.add_argument(
        '--run', action='store_true', help='run script directly')
    parser.add_argument(
        '--local',
        action='store_true',
        help='run at local instead of cluster.')
    parser.add_argument(
        '--mail', type=str, help='Mail address to watch train status.')
    parser.add_argument(
        '--mail-type',
        nargs='+',
        default=['BEGIN', 'END', 'FAIL'],
        choices=['NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'],
        help='Mail address to watch train status.')
    parser.add_argument(
        '--quotatype',
        default=None,
        choices=['reserved', 'auto', 'spot'],
        help='Quota type, only available for phoenix-slurm>=0.2')
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Summarize benchmark train results.')
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save the summary and archive log files.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        type=str,
        default=[],
        help='Config options for all config files.')

    args = parser.parse_args()
    return args


def get_gpu_number(model_info):
    config = osp.basename(model_info.config)
    matches = re.match(r'.*[-_](\d+)xb(\d+).*', config)
    if matches is None:
        raise ValueError(
            'Cannot get gpu numbers from the config name {config}')
    gpus = int(matches.groups()[0])
    return gpus


def create_train_job_batch(commands, model_info, args, port, script_name):

    fname = model_info.name

    gpus = get_gpu_number(model_info)
    gpus_per_node = min(gpus, 8)

    config = Path(model_info.config)
    assert config.exists(), f'"{fname}": {config} not found.'

    work_dir = Path(args.work_dir) / fname
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.quotatype is not None:
        quota_cfg = f'--quotatype {args.quotatype} '
    else:
        quota_cfg = ''

    launcher = 'none' if args.local else 'slurm'

    job_name = f'{args.job_name}_{fname}'
    job_script = (f'#!/bin/bash\n'
                  f'srun -p {args.partition} '
                  f'--job-name {job_name} '
                  f'--gres=gpu:{gpus_per_node} '
                  f'{quota_cfg}'
                  f'--ntasks-per-node={gpus_per_node} '
                  f'--ntasks={gpus} '
                  f'--cpus-per-task=12 '
                  f'--kill-on-bad-exit=1 '
                  f'python -u {script_name} {config} '
                  f'--work-dir={work_dir} '
                  f'--cfg-option env_cfg.dist_cfg.port={port} '
                  f'{" ".join(args.cfg_options)} '
                  f'default_hooks.checkpoint.max_keep_ckpts=1 '
                  f'--launcher={launcher}\n')

    commands.append(f'echo "{config}"')

    # downstream classification task
    cls_config = getattr(model_info, 'cls_config', None)
    if cls_config:
        fname = model_info.name

        gpus = get_gpu_number(model_info)
        gpus_per_node = min(gpus, 8)

        cls_config_path = Path(model_info.cls_config)
        assert cls_config_path.exists(), f'"{fname}": {cls_config} not found.'

        job_name = f'{args.job_name}_{fname}'

        cls_work_dir = work_dir / Path(
            cls_config.split('/')[-1].replace('.py', ''))
        cls_work_dir.mkdir(parents=True, exist_ok=True)

        srun_args = ''
        if args.quotatype is not None:
            srun_args = srun_args.join(f'--quotatype {args.quotatype}')

        # get pretrain weights
        ckpt_path_file = work_dir / 'last_checkpoint'
        with open(ckpt_path_file, 'r') as f:
            ckpt = f.readlines()[0]

        launcher = 'none' if args.local else 'slurm'

        cls_job_script = (
            f'\n'
            f'mim train mmcls {cls_config} '
            f'--launcher {launcher} '
            f'-G {gpus} '
            f'--gpus-per-node {gpus_per_node} '
            f'--cpus-per-task 12 '
            f'--partition {args.partition} '
            f'--srun-args "{srun_args}" '
            f'--work-dir {cls_work_dir} '
            f'--cfg-option model.backbone.init_cfg.type=Pretrained '
            f'model.backbone.init_cfg.checkpoint={ckpt} '
            f'model.backbone.init_cfg.prefix=backbone. '
            f'default_hooks.checkpoint.max_keep_ckpts=1 '
            f'{" ".join(args.cfg_options)}\n')

        commands.append(f'echo "{cls_config}"')

    with open(work_dir / 'job.sh', 'w') as f:
        f.write(job_script)
        if cls_config:
            f.write(cls_job_script)

    commands.append(
        f'nohup bash {work_dir}/job.sh > {work_dir}/out.log 2>&1 &')

    return work_dir / 'job.sh'


def train(models, args):
    script_name = osp.join('tools', 'train.py')
    port = args.port

    commands = []

    for model_info in models.values():
        script_path = create_train_job_batch(commands, model_info, args, port,
                                             script_name)
        port += 1

    command_str = '\n'.join(commands)

    preview = Table()
    preview.add_column(str(script_path))
    preview.add_column('Shell command preview')
    preview.add_row(
        Syntax.from_path(
            script_path,
            background_color='default',
            line_numbers=True,
            word_wrap=True),
        Syntax(
            command_str,
            'bash',
            background_color='default',
            line_numbers=True,
            word_wrap=True))
    console.print(preview)

    if args.run:
        os.system(command_str)
    else:
        console.print('Please set "--run" to start the job')


def save_summary(summary_data, models_map, work_dir):
    date = datetime.now().strftime('%Y%m%d-%H%M%S')
    zip_path = work_dir / f'archive-{date}.zip'
    zip_file = ZipFile(zip_path, 'w')
    summary_path = work_dir / 'benchmark_summary.md'
    file = open(summary_path, 'w')
    headers = [
        'Model', 'Top-1 Expected(%)', 'Top-1 (%)', 'Top-1 best(%)',
        'best epoch', 'Top-5 Expected (%)', 'Top-5 (%)', 'Config', 'Log'
    ]
    file.write('# Train Benchmark Regression Summary\n')
    file.write('| ' + ' | '.join(headers) + ' |\n')
    file.write('|:' + ':|:'.join(['---'] * len(headers)) + ':|\n')
    for model_name, summary in summary_data.items():
        if len(summary) == 0:
            # Skip models without results
            continue
        row = [model_name]
        if 'Top 1 Accuracy' in summary:
            metric = summary['Top 1 Accuracy']
            row.append(f"{metric['expect']:.2f}")
            row.append(f"{metric['last']:.2f}")
            row.append(f"{metric['best']:.2f}")
            row.append(f"{metric['best_epoch']:.2f}")
        else:
            row.extend([''] * 4)
        if 'Top 5 Accuracy' in summary:
            metric = summary['Top 5 Accuracy']
            row.append(f"{metric['expect']:.2f}")
            row.append(f"{metric['last']:.2f}")
        else:
            row.extend([''] * 2)

        model_info = models_map[model_name]
        row.append(model_info.config)
        row.append(str(summary['log_file'].relative_to(work_dir)))
        zip_file.write(summary['log_file'])
        file.write('| ' + ' | '.join(row) + ' |\n')
    file.close()
    zip_file.write(summary_path)
    zip_file.close()
    print('Summary file saved at ' + str(summary_path))
    print('Log files archived at ' + str(zip_path))


def show_summary(summary_data):
    table = Table(title='Train Benchmark Regression Summary')
    table.add_column('Model')
    for metric in METRICS_MAP:
        table.add_column(f'{metric} (expect)')
        table.add_column(f'{metric}')
        table.add_column(f'{metric} (best)')

    def set_color(value, expect):
        if value > expect:
            return 'green'
        elif value > expect - 0.2:
            return 'white'
        else:
            return 'red'

    for model_name, summary in summary_data.items():
        row = [model_name]
        for metric_key in METRICS_MAP:
            if metric_key in summary:
                metric = summary[metric_key]
                expect = metric['expect']
                last = metric['last']
                last_epoch = metric['last_epoch']
                last_color = set_color(last, expect)
                best = metric['best']
                best_color = set_color(best, expect)
                best_epoch = metric['best_epoch']
                row.append(f'{expect:.2f}')
                row.append(
                    f'[{last_color}]{last:.2f}[/{last_color}] ({last_epoch})')
                row.append(
                    f'[{best_color}]{best:.2f}[/{best_color}] ({best_epoch})')
        table.add_row(*row)

    console.print(table)


def summary(models, args):

    work_dir = Path(args.work_dir)
    dir_map = {p.name: p for p in work_dir.iterdir() if p.is_dir()}

    summary_data = {}
    for model_name, model_info in models.items():

        summary_data[model_name] = {}

        if model_name not in dir_map:
            continue

        # Skip if not found any vis_data folder.
        sub_dir = dir_map[model_name]
        log_files = [f for f in sub_dir.glob('*/*/vis_data/scalars.json')]
        if len(log_files) == 0:
            continue
        log_file = sorted(log_files)[-1]

        # parse train log
        with open(log_file) as f:
            json_logs = [json.loads(s) for s in f.readlines()]
            val_logs = [
                log for log in json_logs
                # TODO: need a better method to extract validate log
                if 'loss' not in log and 'accuracy/top1' in log
            ]

        if len(val_logs) == 0:
            continue

        expect_metrics = model_info.results[0].metrics

        # extract metrics
        summary = {'log_file': log_file}
        for key_yml, key_res in METRICS_MAP.items():
            if key_yml in expect_metrics:
                assert key_res in val_logs[-1], \
                    f'{model_name}: No metric "{key_res}"'
                expect_result = float(expect_metrics[key_yml])
                last = float(val_logs[-1][key_res])
                best_log, best_epoch = sorted(
                    zip(val_logs, range(len(val_logs))),
                    key=lambda x: x[0][key_res])[-1]
                best = float(best_log[key_res])

                summary[key_yml] = dict(
                    expect=expect_result,
                    last=last,
                    last_epoch=len(val_logs),
                    best=best,
                    best_epoch=best_epoch + 1)
        summary_data[model_name].update(summary)

    show_summary(summary_data)
    if args.save:
        save_summary(summary_data, models, work_dir)


def main():
    args = parse_args()

    model_index_file = MMSELFSUP_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    all_models = {model.name: model for model in model_index.models}

    with open(Path(__file__).parent / 'models.yml', 'r') as f:
        train_items = yaml.safe_load(f)
    models = OrderedDict()
    for item in train_items:
        name = item['Name']
        model_info = all_models[item['Name']]
        model_info.cycle = item.get('Cycle', None)
        model_info.cls_config = item.get('ClsConfig', None)
        cycle = getattr(model_info, 'cycle', 'month')
        cycle_level = CYCLE_LEVELS.index(cycle)
        if cycle_level in args.range:
            models[name] = model_info

    if args.models:
        patterns = [re.compile(pattern) for pattern in args.models]
        filter_models = {}
        for k, v in models.items():
            if any([re.match(pattern, k) for pattern in patterns]):
                filter_models[k] = v
        if len(filter_models) == 0:
            print('No model found, please specify models in:')
            print('\n'.join(models.keys()))
            return
        models = filter_models

    if args.summary:
        summary(models, args)
    else:
        train(models, args)


if __name__ == '__main__':
    main()
