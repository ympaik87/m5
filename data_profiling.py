from os import cpu_count
import pathlib
import yaml
import pandas as pd
from pandas_profiling import ProfileReport
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')


def run(f_path):
    title = f_path.stem
    res_p = f'D:/kaggle/m5/res/{title}.html'
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
