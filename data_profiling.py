import pathlib
from os import cpu_count

import matplotlib
import pandas as pd
import yaml
from tqdm import tqdm

from pandas_profiling import ProfileReport

matplotlib.use('Agg')


def run(f_path):
    title = f_path.stem
    out_dir = pathlib.Path('D:/kaggle/m5/res/profiles/')
    if not out_dir.exists():
        pathlib.Path.mkdir(out_dir)
    res_p = out_dir/f'{title}.html'
    with open('pandas_profiling_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['title'] = title
    config['pool_size'] = cpu_count()
    config['html']['style']['full_width'] = True

    df = pd.read_csv(f_path)
    prof = ProfileReport(df, **config)
    prof.to_file(output_file=res_p)


if __name__ == "__main__":
    data_p = pathlib.Path('D:/kaggle/m5/m5-forecasting-accuracy')
    f_li = sorted(data_p.glob('*.csv'))
    for f in tqdm(f_li, desc='Data_profiling', total=len(f_li)):
        run(f)
