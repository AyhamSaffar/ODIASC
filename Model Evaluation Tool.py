'''
To launch page, paste the following into cmd prompt:

streamlit run "Model Evaluation Tool.py"

'''

from sklearn.metrics import precision_recall_curve, auc, recall_score, precision_score, f1_score
from utils.apt_range import parse_APT_range_file, get_element_labels
from utils.simulators import Steel_APT_Dataset
from utils.models import models
import streamlit as st
import pathlib as pl
import numpy as np
import pandas as pd
from dbscan import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
from pythresh.thresholds.karch import KARCH

THRESHOLDER = KARCH
APT_RANGE_FILE_PATH = pl.Path(__file__).resolve().parent.joinpath(r'APT Range Files/M18.rrng')

st.set_page_config(layout="wide")

def labelled_model_output_histogram(Y_true: np.ndarray, Y_pred: np.ndarray) -> go.Figure:
    fig = go.Figure(layout={"template": "simple_white"})
    fig.add_trace(go.Histogram(x=Y_pred[Y_true == 1], name="Clusters", marker_color="blue"))
    fig.add_trace(go.Histogram(x=Y_pred[Y_true == 0], name="Not Clusters", marker_color="red"))
    fig.add_vline(ss['auto threshold'], line_width=2, line_dash="solid", opacity=1)
    fig.update_layout(xaxis_title="Threshold", yaxis_title="Count")
    return fig

def unlabelled_model_output_histogram(Y_pred) -> go.Figure:
    fig = go.Figure(layout={"template": "simple_white"})
    fig.add_trace(go.Histogram(x=Y_pred, marker_color="purple"))
    fig.add_vline(ss['auto threshold'], line_width=2, line_dash="solid", opacity=1)
    fig.update_layout(xaxis_title="Threshold", yaxis_title="Count")
    return fig

def model_precision_recall_graph(Y_true: np.ndarray, Y_pred: np.ndarray) -> go.Figure:
    precision, recall, threshold = precision_recall_curve(Y_true, Y_pred)
    fig = px.scatter(y=precision, x=recall)
    auto_threshold_index = np.abs(threshold-ss['auto threshold']).argmin()
    fig.add_hline(precision[auto_threshold_index], line_width=2, line_dash="solid", opacity=1)
    fig.add_vline(recall[auto_threshold_index], line_width=2, line_dash="solid", opacity=1)
    fig.update_traces(mode="lines")
    fig.update_layout(
        template="simple_white",
        yaxis_title="Precision", xaxis_title="Recall",
        yaxis_range=[0, 1.1], xaxis_range=[0, 1.1]
    )
    return fig

def DBSCAN_precision_recall_graph(DBSCAN_results: pd.DataFrame) -> go.Figure:
    fig = px.scatter(DBSCAN_results, y='precision', x='recall', color='cluster_count',
                      hover_data=['min_samples', 'eps'], color_continuous_scale='plasma')
    auto_index = DBSCAN_results[DBSCAN_results['min_samples']==ss['auto min samples']].index[0]
    fig.add_hline(DBSCAN_results.loc[auto_index, 'precision'], line_width=2, line_dash="solid", opacity=1)
    fig.add_vline(DBSCAN_results.loc[auto_index, 'recall'], line_width=2, line_dash="solid", opacity=1)
    fig.update_traces(mode="markers+lines", line_color='light blue')
    fig.update_layout(template="simple_white", yaxis_title="Precision", xaxis_title="Recall",
                        yaxis_range=[0, 1.1], xaxis_range=[0, 1.1])
    return fig  

def find_optimal_DBSCAN_params(no_cluster_dataset: pd.DataFrame, max_min_samples: int = 30, eps_resolution: float = 0.05) -> dict[int: float]:
    '''
    finds the maximum eps with zero DBSCAN clusters for each min_samples under max_min_samples
    returns a dict of {min sample: optimal eps}
    '''
    clusterless_datapoints = no_cluster_dataset[['X', 'Y', 'Z']]
    min_samples_values = np.arange(2, max_min_samples+1)
    eps = eps_resolution
    max_eps = dict()
    for min_samples in min_samples_values:
        while True:
            labels, _ = DBSCAN(clusterless_datapoints, eps=eps, min_samples=min_samples) 
            num_clusters = max(labels) + 1 #not cluster is labelled as -1 then each cluster is labelled 0, 1, 2...
            if num_clusters == 0:
                eps += eps_resolution
            else:
                eps = max(eps_resolution, eps-eps_resolution)
                max_eps[min_samples] = np.round(eps, 2)
                break
    return max_eps

def find_DBSCAN_stats(min_samples_to_eps_dict: dict[int: float], clustered_dataset: pd.DataFrame, data_mode: str) -> pd.DataFrame:
    results = pd.DataFrame()
    results['min_samples'] = list(min_samples_to_eps_dict.keys())
    results['eps'] = list(min_samples_to_eps_dict.values())
    precision_list, recall_list, cluster_counts = [], [], []
    for min_samples, eps in min_samples_to_eps_dict.items():
        cluster_labels, _ = DBSCAN(clustered_dataset[['X', 'Y', 'Z']], eps=eps, min_samples=min_samples)
        cluster_counts.append(np.max(cluster_labels)+1)
        is_cluster_labels = cluster_labels != -1
        if data_mode == 'simulated':
            precision_list.append(precision_score(clustered_dataset['is cluster'], is_cluster_labels, zero_division=0.0))
            recall_list.append(recall_score(clustered_dataset['is cluster'], is_cluster_labels))
    results['cluster_count'] = cluster_counts
    if data_mode == 'read in':
        return results
    results['precision'] = precision_list
    results['recall'] = recall_list
    return results.sort_values(by='recall')

def find_params_with_most_clusters(DBSCAN_stats: pd.DataFrame) -> list[int, float]:
    '''finds the min samples: eps pair with the most DBSCAN clusters'''
    max_cluster_index = DBSCAN_stats['cluster_count'].argmax()
    return DBSCAN_stats['min_samples'][max_cluster_index], DBSCAN_stats['eps'][max_cluster_index]

def generate_lattice_data() -> list[pd.DataFrame, pd.DataFrame]:
    '''
    generates simulated data with parameters in the st.session_state
    returns cluster dataframe and clusterless dataframe for DBSCAN hyperparamter tuning
    '''
    ss['data mode'] = 'simulated'
    dataset = Steel_APT_Dataset(
        unit_cells_per_side=ss['unit cell length'],
        cluster_relative_density=ss['cluster relative density'],
        cluster_atom_counts=np.full(shape=ss['number of clusters'], fill_value=ss['atoms per cluster']),
    )
    return dataset.data[dataset.data['Element']!='Fe'], dataset.DBScan_baseline

def read_lattice_data() -> pd.DataFrame:
    APT_range_metadata = parse_APT_range_file(APT_RANGE_FILE_PATH)
    data = pd.read_csv(ss['current dataset path'], names=['X', 'Y', 'Z', 'MCR'])
    data['Element'] = get_element_labels(data['MCR'].to_numpy(), APT_range_metadata)
    return data[data['Element'].apply(lambda element: element not in ['Fe', 'not assigned'])]

# ---Manipulate Session Data---

ss = st.session_state

def generate_model_labels() -> np.ndarray:
    '''generates model labels with parameters in the st.session_state '''
    model = ss['models'][ss['current model index']]
    return model.get_outputs(ss['cluster data'], ss['clusterless data'], ss['model n neighbours'])

def find_automatic_threshold() -> float:
    model_labels = ss['cluster data']['model labels']
    max_label, min_label = model_labels.max(), model_labels.min()
    if max_label - min_label <= 0.0001:
        return np.mean((max_label, min_label))
    if len(model_labels) > 10_000:
        model_labels = model_labels.sample(10_000)
    normalized_model_labels = model_labels - min_label / (max_label - min_label)
    thresholder = THRESHOLDER()
    thresholder.eval(normalized_model_labels) #thresolder only works with probability (0-1) labels
    normalized_threshold = thresholder.thresh_
    return (normalized_threshold * (max_label - min_label)) + min_label

def reset_threshold_ranges():
    max_model_value = ss['cluster data']['model labels'].max()
    min_model_value = ss['cluster data']['model labels'].min()
    ss['threshold'] = ss['auto threshold']
    ss['min threshold'] = min_model_value
    ss['max threshold'] = max_model_value
    ss['threshold step'] = (max_model_value-min_model_value)/100

def generate_DBSCAN_labels() -> np.ndarray:
    '''generates DBSCAN labels with parameters in st.session_state'''
    labels, _ = DBSCAN(ss['cluster data'][["X", "Y", "Z"]], eps=ss['eps'], min_samples=ss['min samples'])
    return labels != -1

def generate_summary_stats() -> pd.DataFrame:
    summary = pd.DataFrame()

    DBSCAN_data = ss['DBSCAN result stats']
    auto_index = DBSCAN_data[DBSCAN_data['min_samples'] == ss['auto min samples']].index[0]
    summary.loc['precision', 'DBSCAN'] = DBSCAN_data.loc[auto_index, 'precision']
    summary.loc['recall', 'DBSCAN'] = DBSCAN_data.loc[auto_index, 'recall']
    summary.loc['F1 Score', 'DBSCAN'] = (2*summary['DBSCAN']['precision']*summary['DBSCAN']['recall'])/(summary['DBSCAN']['precision']+summary['DBSCAN']['recall'])
    summary.loc['AUPRC', 'DBSCAN'] = auc(x=DBSCAN_data['recall'], y=DBSCAN_data['precision'])
    DBSCAN_95R_index = (DBSCAN_data['recall']-0.95).abs().argmin()
    summary.loc['P95R', 'DBSCAN'] = DBSCAN_data['precision'][DBSCAN_95R_index]

    true_labels = ss['cluster data']['is cluster']
    model_labels = ss['cluster data']['model labels']
    summary.loc['precision', 'Model'] = precision_score(true_labels, model_labels>=ss['auto threshold'])
    summary.loc['recall', 'Model'] = recall_score(true_labels, model_labels>=ss['auto threshold'])
    summary.loc['F1 Score', 'Model'] = f1_score(true_labels, model_labels>=ss['auto threshold'])
    precision_list, recall_list, _ = precision_recall_curve(true_labels, model_labels)
    summary.loc['AUPRC', 'Model'] = auc(x=recall_list, y=precision_list)
    model_95R_index = np.abs(recall_list-0.95).argmin()
    summary.loc['P95R', 'Model'] = precision_list[model_95R_index]

    summary.loc[:, 'Improvement'] = summary['Model'] - summary['DBSCAN']

    return summary

def update_DBSCAN_data():
    ss['DBSCAN result stats'] = find_DBSCAN_stats(ss['candidate DBSCAN params'], ss['cluster data'], ss['data mode'])
    ss['auto min samples'], ss['auto eps'] = find_params_with_most_clusters(ss['DBSCAN result stats'])
    ss['min samples'], ss['eps'] = ss['auto min samples'], ss['auto eps']
    ss['cluster data']['DBSCAN labels'] = generate_DBSCAN_labels()

def change_min_samples():
    ss['eps'] = ss['candidate DBSCAN params'][ss['min samples']]
    ss['cluster data']['DBSCAN labels'] = generate_DBSCAN_labels()

def update_model_labels():
    ss['cluster data']['model labels'] = generate_model_labels()
    ss['auto threshold'] = find_automatic_threshold()
    reset_threshold_ranges()
    if ss['data mode'] == 'simulated':
        ss['summary stats'] = generate_summary_stats()

def update_lattice_data():
    try:
        ss['cluster data'], ss['clusterless data'] = generate_lattice_data()
    except AssertionError as error:
        st.sidebar.error(error)
        ss['unit cell length'] = 50
        ss['cluster relative density'] = 200
        ss['number of clusters'] = 10
        ss['atoms per cluster'] = 50
    update_DBSCAN_data()
    update_model_labels()

def update_loaded_data():
    ss['data mode'] = 'read in'
    ss['cluster data'] = read_lattice_data()
    update_DBSCAN_data()
    update_model_labels()

if 'has_started' not in ss:
    ss['has_started'] = True
    ss['data mode'] = 'simulated'

    ss['unit cell length'] = 50
    ss['cluster relative density'] = 200
    ss['number of clusters'] = 10
    ss['atoms per cluster'] = 50
    ss['cluster data'], ss['clusterless data'] = generate_lattice_data()

    ss['datasets file path'] = pl.Path(__file__).resolve().parent.joinpath(r"Real Datasets")
    ss['dataset paths'] = [path for path in  ss['datasets file path'].glob(pattern="*.csv")]
    ss['current dataset path'] = ss['dataset paths'][0]

    ss['models'] = models
    ss['model n neighbours'] = 10
    ss['current model index'] = 0
    ss['cluster data']['model labels'] = generate_model_labels()
    ss['auto threshold'] = find_automatic_threshold()

    ss['DBSCAN result stats'] = pd.DataFrame()
    ss['min samples'], ss['eps'] = 10, 1
    ss['auto min samples'], ss['auto eps'] = 10, 1
    ss['candidate DBSCAN params'] = find_optimal_DBSCAN_params(ss['clusterless data'])
    update_DBSCAN_data()
    ss['cluster data']['DBSCAN labels'] = generate_DBSCAN_labels()

    ss['threshold'] = 0.5
    ss['min threshold'] = 0.0
    ss['max threshold'] = 1.0
    ss['threshold step'] = 0.01
    reset_threshold_ranges()

    ss['summary stats'] = generate_summary_stats()


# ---Creating Plots---

if ss['data mode'] == 'simulated':
    histogram = labelled_model_output_histogram(ss['cluster data']['is cluster'], ss['cluster data']['model labels'])
else:
    histogram = unlabelled_model_output_histogram(ss['cluster data']['model labels'])
histogram.add_vline(ss['threshold'], line_width=2, line_dash="dash", opacity=1)

kwargs = {'color': 'is cluster', 'color_continuous_scale':['red', 'blue']} if ss['data mode'] == 'simulated' else {'color': 'Element'}
model_result_plot = px.scatter_3d(ss['cluster data'][ss['cluster data']['model labels']>=ss['threshold']], x='X', y='Y', z='Z', **kwargs)
DBSCAN_plot = px.scatter_3d(ss['cluster data'][ss['cluster data']['DBSCAN labels']==True], x='X', y='Y', z='Z', **kwargs)

for plot in [model_result_plot, DBSCAN_plot]:
    plot.update_traces(marker={'size': 2.5})
    plot.update(layout_coloraxis_showscale=False)

if ss['data mode'] == 'simulated':
    model_precision = precision_score(ss['cluster data']['is cluster'], ss['cluster data']['model labels']>ss['threshold'])
    DBSCAN_precision = precision_score(ss['cluster data']['is cluster'], ss['cluster data']['DBSCAN labels'])
    model_recall = recall_score(ss['cluster data']['is cluster'], ss['cluster data']['model labels']>ss['threshold'])
    DBSCAN_recall = recall_score(ss['cluster data']['is cluster'], ss['cluster data']['DBSCAN labels'])

    model_scatter = model_precision_recall_graph(ss['cluster data']['is cluster'], ss['cluster data']['model labels'])
    model_scatter.add_vline(model_recall, line_width=2, line_dash="dash", opacity=1)
    model_scatter.add_hline(model_precision, line_width=2, line_dash="dash", opacity=1)

    DBSCAN_scatter = DBSCAN_precision_recall_graph(ss['DBSCAN result stats'])
    DBSCAN_scatter.add_vline(DBSCAN_recall, line_width=2, line_dash="dash", opacity=1)
    DBSCAN_scatter.add_hline(DBSCAN_precision, line_width=2, line_dash="dash", opacity=1)


# ---Frontend---

st.markdown("""
        <style>
            footer {visibility: hidden;}
            .block-container {
                padding-top: 0.5rem;
                padding-bottom: 0.5rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        </style>
        """, unsafe_allow_html=True)


with st.sidebar:

    st.header('Test Dataset')

    tab1, tab2 = st.tabs(['Data Simulator', 'Data Loader'])

    with tab1:
        with st.form(key='form'):
            st.slider(label='unit cells per dimension',
                        min_value=10, max_value=150, step=5, key='unit cell length')
            st.slider(label='relative density of atoms in cluster',
                        min_value=1, max_value=1500, key='cluster relative density')
            st.slider(label='number of clusters',
                        min_value=5, max_value=100, key='number of clusters')
            st.slider(label='atoms per cluster',
                        min_value=5, max_value=100, step=5, key='atoms per cluster')
            st.form_submit_button('generate data', on_click=update_lattice_data)
        
        if ss['data mode'] == 'simulated':
            st.text(f'{np.sum(ss["cluster data"]["is cluster"]): .0f} cluster data points')
            st.text(f'{np.sum(ss["cluster data"]["is cluster"]==False)} non cluster data points')

    with tab2:
        with st.form(key='form 2'):
            st.markdown('File must be a csv with X, Y, Z, and M/Z ratio columns')
            st.radio('Datasets Available', options=ss['dataset paths'], key='current dataset path',
                    format_func=lambda path: path.stem)
            st.form_submit_button('read data', on_click=update_loaded_data)
    
    st.header('Models Available')
    if 'LUNOT' not in ss['models'][ss['current model index']].name:
        st.slider('Number of Neighbours', min_value=1, max_value=100, key='model n neighbours', on_change=update_model_labels)
    else:
        ss['model n neighbours'] = 20 if ss['models'][ss['current model index']].name == 'LUNOT Classifier' else 50
        st.slider('Number of Neighbours', min_value=1, max_value=100, key='model n neighbours', disabled=True)
    
    st.radio('hidden', options=range(len(ss['models'])), key='current model index', format_func=lambda i: ss['models'][i].name,
             on_change=update_model_labels, label_visibility='hidden')

if ss['data mode'] == 'simulated':
    with st.container():
        model_performance_col, model_seperation_col, DBSCAN_performance_col = st.columns(3, gap='large')
else:
    with st.container():
        model_performance_col, DBSCAN_performance_col = st.columns(2, gap='large')
    with st.container():
        _, model_seperation_col, _ = st.columns([0.2, 0.6, 0.2])

with model_performance_col:
    st.header("Model Performance")
    middle, _ = st.columns([0.9, 0.1])
    with middle:
        st.slider(label='Threshold', min_value=float(ss['min threshold']), max_value=float(ss['max threshold']),
                    step=ss['threshold step'], key='threshold', format="%g")
    st.plotly_chart(model_result_plot, use_container_width=True, theme=None)

with model_seperation_col:
    st.header("Model Seperation")
    st.plotly_chart(histogram, use_container_width=True)

with DBSCAN_performance_col:
    st.header("DBSCAN Performance")
    left, right = st.columns([0.8, 0.2])
    with left:
        st.slider(label='Min Samples', min_value=(ss['DBSCAN result stats']['min_samples']).min(), max_value=(ss['DBSCAN result stats']['min_samples']).max(),
                  key='min samples', on_change=change_min_samples)
    with right:
        st.metric('Epsilon', value=ss['eps'])
    st.plotly_chart(DBSCAN_plot, use_container_width=True, theme=None)

if ss['data mode'] == 'simulated':
    with st.container():
        model_overall_performance_col, summary_stats_col, DBSCAN_overall_performance_col = st.columns(3, gap='large')

    with model_overall_performance_col:
        st.header("Model Overall Performance")
        st.plotly_chart(model_scatter, use_container_width=True)

    with summary_stats_col:
        st.header('Summary')

        st.markdown('Automatic Model Setting Metrics')
        resuls_metrics = ss['summary stats'].iloc[:3,:]
        resuls_metrics = resuls_metrics.style.format("{:.2%}")
        st.dataframe(resuls_metrics, use_container_width=True)

        st.markdown('Overall Model Metrics')
        model_metrics = ss['summary stats'].iloc[3:,:]
        model_metrics = model_metrics.style.format("{:.2%}")
        st.dataframe(model_metrics, use_container_width=True)

    with DBSCAN_overall_performance_col:
        st.header("DBSCAN Overall Performance")
        st.plotly_chart(DBSCAN_scatter, use_container_width=True)