import pandas as pd
import numpy as  np

def parse_APT_range_file(path: str) -> pd.DataFrame:
    '''Reads & parses APT metadata range .rrng files at the specified path.'''
    column_lables = ['lower', 'upper', 'vol', 'element']
    data = pd.read_csv(path, names=column_lables, delim_whitespace=True, usecols=range(4))
    data = data[data['element'].isna() == False] #ignore header rows
    data['lower'] = data['lower'].apply(lambda entry: float(entry[entry.find('=')+1:]))
    data['vol'] = data['vol'].apply(lambda i: float(i[i.find(':')+1:]))
    data['element'] = data['element'].apply(lambda i: str(i[:i.find(':')]))
    return data.reset_index()

def get_element_labels(mzratio: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    '''
    Create a vector of element labels from a vector of mass-charge ratios.

    Metadata dataframe should have a 'lower', 'upper', and 'element' columns.

    Returns an array of <U12 strings with 'not assigned' where mz values did not
    fit into any of the ranges in the metadata.
    '''
    elements = np.full(fill_value='not assigned', shape=mzratio.shape)
    for i in metadata.index:
        upper = metadata['upper'][i]
        lower = metadata['lower'][i]
        element = metadata['element'][i]
        elements[(mzratio<upper) & (mzratio>lower)] = element
    return elements