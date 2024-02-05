import numpy as np 
from matplotlib.colors import Normalize
import pandas as pd
from datetime import datetime
import scipy 
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sklearn
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, mean_squared_error, r2_score, make_scorer, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm, neighbors
from sklearn.svm import SVR
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, RationalQuadratic, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import pandas as pd
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D 
from tqdm import tqdm
import warnings
import plotly.graph_objects as go
from itertools import product

    
    

def uncertainty_sampler(df):
    
    # find the row with highest gpr_std in df: 
    max_std = df['gpr_std'].max()
    max_std_row = df[df['gpr_std']==max_std]
    
    # select the data with highest uncertainty(gpr standard deviation)
    df_new = df.copy()
    df_new['labeled'].iloc[max_std_row.index] = True
    labeled_data = df_new.iloc[max_std_row.index].copy()
    
    return df_new, labeled_data

def random_sampler(df, random_seed=None):
    
    if random_seed is not None:
        random.seed(random_seed)
    
    random_1 = random.sample(range(0, len(df)), 1)
    
    while df['labeled'].iloc[random_1].values[0] == True:
        random_1 = random.sample(range(0, len(df)), 1)
    
    df_new = df.copy()
    df_new['labeled'].iloc[random_1] = True
    labeled_data = df_new.iloc[random_1].copy()
    
    return df_new, labeled_data

def gpr_labeled_data(df):
        
        # split into training and test data
        X_train = df[df['labeled']==True][['init_C', 'init_N', 'init_Li']].copy()
        
        y_train = df[df['labeled']==True]['yield'].copy()
        
        X_test = df[['init_C', 'init_N', 'init_Li']].copy()
        
        # scale training data
        scaler = StandardScaler()
        scale = scaler.fit(X_train)
        
        X_train_scaled = scale.transform(X_train)
        X_test_scaled = scale.transform(X_test)
        
        gpr_model = GaussianProcessRegressor(kernel=Matern(length_scale= 3, nu=1.5),
                                            random_state=42, 
                                            n_restarts_optimizer=5)
        
        gpr_model.fit(X_train_scaled, y_train)
        
        y_pred, std = gpr_model.predict(X_test_scaled, return_std=True)
        
        new_df = df.copy()
        new_df['gpr_prediction'] = y_pred
        new_df['gpr_std'] = std
        
        return new_df, gpr_model, scale     


def sample_gaussian(mean, std_dev, n_points):
    samples = np.random.normal(mean, std_dev, n_points)
    return samples.tolist()


def plot_gaussian(mean, std_dev, n_points):
    # Generate random samples
    samples = sample_gaussian(mean, std_dev, n_points)

    # Generate Gaussian distribution
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
    y = norm.pdf(x, mean, std_dev)

    # Plot the Gaussian distribution
    plt.plot(x, y, label='Gaussian Distribution')

    # Plot the randomly sampled points
    y_samples = norm.pdf(samples, mean, std_dev)
    plt.scatter(samples, y_samples, color='red', label='Sampled Points')

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()

    # Display the plot
    plt.show()

def evaluate_predictions(df):
    y_true = df[df['labeled']==False]['yield']
    y_pred = df[df['labeled']==False]['gpr_prediction']

    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    results = pd.DataFrame({'MSE': [mse], 'MAPE': [mape], 'MAE': [mae], 'R2': [r2]})
    return results

def load_from_dict(data_dict_address):
    '''This function loads dataframes from excel files and returns a dictionary of dataframes.
    
    Parameters:
    ----------
    data_dict_address: dictionary of addresses of excel files
        dictionary
        
    Returns:
    -------
    data_dict: dictionary of dataframes
        dictionary'''
        
    data_dict = {}
    
    # load dataframes from excel files:
    
    for key in data_dict_address.keys():
        
        data_dict[key] = pd.read_excel(data_dict_address[key])
        
    return data_dict


def dataframe_format_correction(data_dict, label1, label2):
    '''
    This function changes the column names/labels to the same format for all dataframes.
    
    Parameters:
    ----------
    data_dict: dictionary of dataframes
        dictionary of pd.DataFrame
        
    label1: list of old column names
        list
        
    label2: list of new column names
        list
        
    Returns:
    -------
    data_corrected_labels: dictionary of dataframes with corrected labels
        dictionary of pd.DataFrame'''
    
    data_corrected_labels = {}
    
    # for each dataframe in the dictionary:
    # change the column names/labels to the same format for all dataframes
    # label1 list of old column names and label2 list of new column names
    
    for key in data_dict.keys():
        
        data_corrected_labels[key] = data_dict[key].rename(columns=dict(zip(label1, label2)))
    
    return data_corrected_labels


def basic_clean(data_dict, eps=0.01):
    '''
    This function cleans the dataframes by removing NaN values and replacing 0 values with eps.
    
    Parameters:
    ----------
    data_dict: dictionary of dataframes
        dictionary of pd.DataFrame
    
    eps: small number to replace 0 values
        float
    
    Returns:
    -------
    data_cleaned: dictionary of cleaned dataframes
        dictionary of pd.DataFrame
    '''
    data_cleaned = {}
    
    for key in data_dict.keys():
        data_cleaned[key] = data_dict[key].dropna()
        data_cleaned[key]['yield'] = data_dict[key]['yield'].apply(lambda x: x+eps if x==0 else x)
        
    return data_cleaned

def data_integration(data_dict):
    ''' 
    This function integrates the dataframes in the dictionary into one dataframe.
    
    Parameter:
    ----------
    data_dict: dictionary of dataframes
        dictionary of pd.DataFrame
    
    Return:
    -------
    data_integrated: integrated dataframe
        pd.DataFrame
    '''
    
    data_integrated = pd.DataFrame()
    
    for key in data_dict.keys():
        data_integrated = pd.concat([data_integrated, data_dict[key]])
    
    data_integrated = data_integrated.reset_index(drop=True)
    
    return data_integrated


def fact_check(combination):
    '''
    Checking if the nominated data (suggested experiment) satisfies acquisition guidlines or not
    
    Parameter:
    ----------
    combination: an array containing the experiment combinations. 
                     The combination of data must follow below order: 
                     [{C} , {N}, {Li}, {T}]
    Returens: 
    ----------
    check: True if all the criteria are satisfied, False if not 
    
    '''
    #print(combination)
    
    c = combination[0]
    n = combination[1]
    l = combination[2]
    
    c_total_limit = 0.5 * n
    
    
    if n > 6:
        return False
    
    if c > c_total_limit:
        return False
    
    if c > 2.5: 
        return False
    
    if l > 6 or l <= 0:
        return False
    
    if c < (l/2):
        return False
    
    #print('n: ', n, ' || n_total: ',n_total, ' || c: ', c, " || c_total_limit: ", c_total_limit, " || Li: ", l)
    return True 

def df_scaler(fit_df, transform_df, labels_list = ['init_C', 'init_N', 'init_Li', 'T']):
    '''
    Recieves two dataframes and scales the second dataframe using the first dataframe
    Parameters:
    ----------
    fit_df: the dataframe to be used for fitting the scaler
        pd.DataFrame
        
    transform_df: the dataframe to be scaled
        pd.DataFrame
        
    labels_list: the list of labels to be used for scaling
        list
    
    Returns:
    ----------
    scaled_df: the scaled dataframe
    
    '''
    
    # using the batch containing all the data scale the experimental grid
    scaler = preprocessing.StandardScaler().fit(fit_df[labels_list])
    scaled_df = scaler.transform(transform_df[labels_list])
    
    return scaled_df

def entropy_measure(std_list):
    '''
    The function is used to calculate the entropy of the standard deviation of the model prediction
    
    Parameters:
    ------------
        std_list: the list of the standard deviation of the model prediction
            list
            
    Returns:
    ------------
        entropy: the entropy of the standard deviation of the model prediction
            float
    '''
    
    std_list = np.array(std_list)
    # entropy = ln(std*root(2*pi*e))
    entropy = np.log(std_list * np.sqrt(2*np.pi*np.e))
    
    return entropy


def binary_model_plot_T(model, scaler, range_a, range_b, val_c, label_a, label_b , label_c, entropy_contour=False, std_contour=True, yield_contour=True):
    '''
    The function is used to plot the binary contour plot of the model prediction

    Parameters:
    ------------
        model: the model to predict the yield
            SKlearn model object
            
        scaler: the scaler used to scale the data
            SKlearn scaler object
            
        range_a: the range of the first parameter
            tuple
            
        range_b: the range of the second parameter
            tuple
            
        val_c: the value of the third parameter
            float
            
        label_a: the label of the first parameter
            string
        
        label_b: the label of the second parameter
            string
            
        label_c: the label of the third parameter
            string
            
        T: the temperature of the experiment
            float
            
        entropy_contour: if True, the entropy contour will be plotted
            bool
            
        std_contour: if True, the standard deviation contour will be plotted
            bool
        
        yield_contour: if True, the yield contour will be plotted
            bool
        
    Returns:
    ------------
    none
        
        '''
    
    a_list = np.linspace(range_a[0], range_a[1], 50)
    b_list = np.linspace(range_b[0], range_b[1], 50)

    a_mesh, b_mesh = np.meshgrid(a_list, b_list)
    yield_mesh = a_mesh.copy()
    std_mesh = a_mesh.copy()
    entropy_mesh = a_mesh.copy()
    
    for i in tqdm(range(len(a_list))):
        for j in range(len(b_list)):
            x_df = pd.DataFrame({label_a: a_mesh[i,j],
                                 label_b : b_mesh[i,j],
                                 label_c : [val_c]})
            # it is crucial to rearrange the columns in the same order as the training data!!!
            x_df = x_df[['init_C', 'init_N', 'init_Li']]
            
            scaled_grid = scaler.transform(x_df)
            yield_mesh[i,j], std_mesh[i,j] = model.predict(scaled_grid, return_std=True)
            if entropy_contour:
                entropy_mesh[i,j] = entropy_measure(std_mesh[i,j])
            
    plt.figure()
    
    plt.contourf(a_mesh, b_mesh, yield_mesh, 100, cmap = 'viridis')
    cbar = plt.colorbar()
    
    if std_contour:
        contour1 = plt.contour(a_mesh, b_mesh, std_mesh, colors = 'red')
        plt.clabel(contour1, inline=True, fontsize=8)
    
    if yield_contour:
        contour2 = plt.contour(a_mesh, b_mesh, yield_mesh, colors = 'black')
        plt.clabel(contour2, inline=True, fontsize=8)
    
    if entropy_contour:
        contour3 = plt.contour(a_mesh, b_mesh, entropy_mesh, colors = 'blue')
        plt.clabel(contour3, inline=True, fontsize=8)
    
        
    
    plt.xlabel(label_a)
    plt.ylabel(label_b)
    cbar.ax.set_ylabel('Yield')
    plt.title(label_c + ": " + str(val_c))
    plt.show()
    plt.tight_layout()
    
    return


def model_scaler_setup(model, train_df, labels_list = ['init_C', 'init_N', 'init_Li', 'T'], target = 'yield'):
    '''
    This function takes in a model, training data, and the labels and target of the training data and returns the trained model and scaler.
    
    Parameters:
    ----------
    model: sklearn model
    
    train_df: pandas dataframe
    
    labels_list: list of strings
    
    target: string
    
    Returns:
    -------
    model: fitted sklearn model 
    
    scaler: fitted sklearn scaler
    
    '''
    # Define the scaler 
    scaler = preprocessing.StandardScaler().fit(train_df[labels_list])
    scaled_batch = scaler.transform(train_df[labels_list])
    
    # train the model on scaled data
    model.fit(scaled_batch, train_df[target])
    
    return model, scaler

# Create a grid for the prediction space:

def create_exp_grid(label_a = "init_C", label_b = "init_N", label_c = "init_Li", label_d= "T",
                    range_a = (0,2.5), range_b = (0,6), range_c = (0,6), range_d = (66,66)
                    ):
    '''
    This function creates a grid of experimental conditions for the model to predict.
    
    Parameters:
    ----------
    label_a: string
    
    label_b: string
    
    label_c: string
    
    label_d: string
    
    range_a: tuple
    
    range_b: tuple
    
    range_c: tuple
    
    range_d: tuple
    
    Returns:
    -------
    exp_grid: pandas dataframe
    
    '''
    
    exp_grid = pd.DataFrame({label_a:[], label_b : [], label_c : [], label_d: []})
    
    for a in tqdm(np.linspace(range_a[0], range_a[1], 50)):
        for b in np.linspace(range_b[0],range_b[1],50):
            for c in np.linspace(range_c[0],range_c[1],50):
                for d in np.linspace(range_d[0],range_d[1],1):
                    x_df = pd.DataFrame({label_a:[a],
                                         label_b : [b],
                                         label_c : [c],
                                         label_d: [d]})
                    exp_grid = exp_grid.append(x_df)
    
    return exp_grid
    
    
def grid_feasibility(data_df, label_a = "init_C", label_b = "init_N", label_c = "init_Li", label_d= "T"):
    '''This function takes in a dataframe of experimental conditions and returns a list of booleans indicating whether the conditions are feasible or not.
    Parameters:
    ----------
    data_df: pandas dataframe
    
    label_a: string
    
    label_b: string
    
    label_c: string
    
    label_d: string
    
    Returns:
    -------
    data_df: pandas dataframe'''    
    fact_check_list = []
    
    for i in tqdm(range(len(data_df))):
        fact_check_list.append(fact_check([data_df[label_a][i],
                                           data_df[label_b][i],
                                           data_df[label_c][i],
                                           data_df[label_d][i]]))
    
    data_df['fact_check'] = fact_check_list
    
    return data_df


def scale_dataframe(exp_grid, train_df, labels_list = ['init_C', 'init_N', 'init_Li', 'T']):
    '''This function takes in a dataframe of experimental conditions and returns the scaled dataframe.
    Parameters:
    ----------
    exp_grid: pandas dataframe
    
    train_df: pandas dataframe
    
    labels_list: list of strings
    
    Returns:
    -------
    exp_grid: pandas dataframe with scaled columns'''
    
    # Define the scaler 
    scaler = preprocessing.StandardScaler().fit(train_df[labels_list])
    scaled_batch = scaler.transform(train_df[labels_list])
    
    # scale the experimental grid
    scaled_exp_grid = scaler.transform(exp_grid[labels_list])
    
    exp_grid['scl_'+labels_list[0]] = scaled_exp_grid[:,0]
    exp_grid['scl_'+labels_list[1]] = scaled_exp_grid[:,1]
    exp_grid['scl_'+labels_list[2]] = scaled_exp_grid[:,2]
    exp_grid['scl_'+labels_list[3]] = scaled_exp_grid[:,3]
    
    return exp_grid



def euclidean_distance(input_batch, vector, n_components = 3):
    '''
    Finds the minimum euclidean distance between a batch of vector (batch, np.arrays) and a new vector (experiment, np.array)
    
    Parameters:
    -----------
    input_batch: 2D np array (a set of vectors)
    vector: 1D np array (a vector)
    
    Returne:
    -----------
    min_distance: minimum Euclidean distance of the vector to the batch 
    
    '''
    
    batch_np = np.array(input_batch)
    vector_np = np.array(vector)
    

    dislocation = batch_np[0, 0:n_components] - vector_np[0:n_components]
    min_distance = np.linalg.norm(dislocation)

    for i in range(len(batch_np)):
        dislocation = batch_np[i,0:n_components] - vector_np[0:n_components]
        distance = np.linalg.norm(dislocation)
        
        if min_distance > distance:
            min_distance = distance
    
    return min_distance




def tierd_greedy_acquisition(exp_grid_df, train_df, tier1_label = 'std', tier2_label = 'gpr_yield', 
                             tier1_acquisition = 12, tier2_acquisition = 6, tier3_acquisition = 6, min_distance = 0.5):
    """
    data_df: dataframe with columns 'std' and 'gpr_yield'
    tier1_label: column name for the first tier of acquisition
    tier2_label: column name for the second tier of acquisition
    n: number of samples to acquire
    """
    acquisition_batch = []
    control_batch = train_df[['init_C', 'init_N', 'init_Li']].values
    temp_df = exp_grid_df.copy()
    # sort the dataframe by the first tier
    temp_df = temp_df.sort_values(by=tier1_label, ascending=False)
    temp_df.reset_index(drop=True)
    
    acquisition_counter = 0 
    row_counter = 0
    while acquisition_counter < tier1_acquisition:
        if euclidean_distance(control_batch, temp_df.iloc[row_counter, 0:3].values) > min_distance and temp_df.iloc[row_counter]["fact_check"]==True:
            acquisition_batch.append(temp_df.iloc[row_counter])
            control_batch = np.append(control_batch, [temp_df.iloc[row_counter, 0:3]], axis=0)
            acquisition_counter += 1
        
        row_counter += 1
    
    acquisition_counter = 0 
    row_counter = 0
    temp_df = temp_df.sort_values(by=tier2_label, ascending=False)
    temp_df.reset_index(drop=True)
    while acquisition_counter < tier2_acquisition:
        if euclidean_distance(control_batch, temp_df.iloc[row_counter, 0:3].values) > min_distance and temp_df.iloc[row_counter]["fact_check"]==True:
            acquisition_batch.append(temp_df.iloc[row_counter])
            control_batch = np.append(control_batch, [temp_df.iloc[row_counter, 0:3]], axis=0)
            acquisition_counter += 1
        
        row_counter += 1
    
    acquisition_counter = 0 
    while acquisition_counter < tier3_acquisition:
        row_counter = random.randint(0,len(exp_grid_df)-1)
        if euclidean_distance(control_batch, temp_df.iloc[row_counter, 0:3].values) > min_distance and temp_df.iloc[row_counter]["fact_check"]==True:
            acquisition_batch.append(temp_df.iloc[row_counter])
            control_batch = np.append(control_batch, [temp_df.iloc[row_counter, 0:3]], axis=0)
            acquisition_counter += 1
        
    acquisition_df = pd.DataFrame(acquisition_batch, columns = ['init_C', 'init_N', 'init_Li', 'T', 'gpr_yield', 'std', 'fact_check'])
    
    acquisition_type = []
    for i in range(len(acquisition_df)):
        if i < tier1_acquisition:
            acquisition_type.append(tier1_label)
        elif i < tier1_acquisition + tier2_acquisition:
            acquisition_type.append(tier2_label)
        else:
            acquisition_type.append('random')
        
    acquisition_df['acquisition'] = acquisition_type
    
    return acquisition_df
    

def binary_model_plot(model, scaler, range_a, range_b, val_c,
                      label_a, label_b , label_c , T = 66,
                      entropy_contour=False, std_contour=True, yield_contour=True):
    '''
    The function is used to plot the binary contour plot of the model prediction

    Parameters:
    ------------
        model: the model to predict the yield
            SKlearn model object
            
        scaler: the scaler used to scale the data
            SKlearn scaler object
            
        range_a: the range of the first parameter
            tuple
            
        range_b: the range of the second parameter
            tuple
            
        val_c: the value of the third parameter
            float
            
        label_a: the label of the first parameter
            string
            
        label_b: the label of the second parameter
            string
            
        label_c: the label of the third parameter
            string
            
        T: the temperature of the experiment
            float
            
        entropy_contour: if True, the entropy contour will be plotted
            bool
            
        std_contour: if True, the standard deviation contour will be plotted
            bool
        
        yield_contour: if True, the yield contour will be plotted
            bool
        
    Returns:
    ------------
    none
        
        '''
    
    a_list = np.linspace(range_a[0], range_a[1], 50)
    b_list = np.linspace(range_b[0], range_b[1], 50)

    a_mesh, b_mesh = np.meshgrid(a_list, b_list)
    yield_mesh = a_mesh.copy()
    std_mesh = a_mesh.copy()
    entropy_mesh = a_mesh.copy()
    
    for i in tqdm(range(len(a_list))):
        for j in range(len(b_list)):
            x_df = pd.DataFrame({label_a: a_mesh[i,j],
                                 label_b : b_mesh[i,j],
                                 label_c : [val_c],
                                 'T':[T]})
            # it is crucial to rearrange the columns in the same order as the training data!!!
            x_df = x_df[['init_C', 'init_N', 'init_Li', 'T']]
            
            scaled_grid = scaler.transform(x_df)
            yield_mesh[i,j], std_mesh[i,j] = model.predict(scaled_grid, return_std=True)
            if entropy_contour:
                entropy_mesh[i,j] = entropy_measure(std_mesh[i,j])
            
    plt.figure()
    
    plt.contourf(a_mesh, b_mesh, yield_mesh, 100, cmap = 'viridis')
    cbar = plt.colorbar()
    
    if std_contour:
        contour1 = plt.contour(a_mesh, b_mesh, std_mesh, colors = 'red')
        plt.clabel(contour1, inline=True, fontsize=8)
    
    if yield_contour:
        contour2 = plt.contour(a_mesh, b_mesh, yield_mesh, colors = 'black')
        plt.clabel(contour2, inline=True, fontsize=8)
    
    if entropy_contour:
        contour3 = plt.contour(a_mesh, b_mesh, entropy_mesh, colors = 'blue')
        plt.clabel(contour3, inline=True, fontsize=8)
    
        
    
    plt.xlabel(label_a)
    plt.ylabel(label_b)
    cbar.ax.set_ylabel('Yield')
    plt.title(label_c + ": " + str(val_c))
    plt.show()
    plt.tight_layout()
    
    return


def ternary_dataframe_plot(df, col_a, col_b, col_c, c_label):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    plot_data = ax.scatter(df[col_a], df[col_b], df[col_c], c=df[c_label], cmap='viridis')
    cbar = fig.colorbar(plot_data)
    
    ax.set_xlabel(col_a)
    ax.set_ylabel(col_b)
    ax.set_zlabel(col_c)
    cbar.ax.set_ylabel(c_label)
    plt.show()
    plt.tight_layout()
    
    
def ternary_model_plot(model, scaler, range_a, range_b, range_c,
                      label_a, label_b, label_c):
    
    a_list = np.linspace(range_a[0], range_a[1], 30)
    b_list = np.linspace(range_b[0], range_b[1], 30)
    c_list = np.linspace(range_c[0], range_c[1], 30)

    a_mesh, b_mesh, c_mesh = np.meshgrid(a_list, b_list, c_list, indexing='ij')
    yield_mesh = np.zeros_like(a_mesh)
    
    for i in tqdm(range(len(a_list))):
        for j in range(len(b_list)):
            for k in range(len(c_list)):
                x_df = pd.DataFrame({label_a: [a_mesh[i, j, k]],
                                     label_b: [b_mesh[i, j, k]],
                                     label_c: [c_mesh[i, j, k]]})
                x_df = x_df[['init_C', 'init_N', 'init_Li']]
                
                scaled_grid = scaler.transform(x_df)
                yield_mesh[i, j, k] = model.predict(scaled_grid)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    plot_data = ax.scatter(a_mesh, b_mesh, c_mesh, c=yield_mesh.flatten(), cmap='viridis')
    cbar = fig.colorbar(plot_data)
    
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    ax.set_zlabel(label_c)
    cbar.ax.set_ylabel('Yield')
    plt.show()
    plt.tight_layout()

    return


def plotly_ternary_model_plot(model, scaler, range_a, range_b, range_c,
                      label_a, label_b, label_c):
    
    a_list = np.linspace(range_a[0], range_a[1], 30)
    b_list = np.linspace(range_b[0], range_b[1], 30)
    c_list = np.linspace(range_c[0], range_c[1], 30)

    a_mesh, b_mesh, c_mesh = np.meshgrid(a_list, b_list, c_list, indexing='ij')
    yield_mesh = np.zeros_like(a_mesh)
    
    for i in tqdm(range(len(a_list))):
        for j in range(len(b_list)):
            for k in range(len(c_list)):
                x_df = pd.DataFrame({label_a: [a_mesh[i, j, k]],
                                     label_b: [b_mesh[i, j, k]],
                                     label_c: [c_mesh[i, j, k]]})
                x_df = x_df[['init_C', 'init_N', 'init_Li']]
                
                scaled_grid = scaler.transform(x_df)
                yield_mesh[i, j, k] = model.predict(scaled_grid)

    fig = go.Figure(data=[go.Scatter3d(
        x=a_mesh.flatten(),
        y=b_mesh.flatten(),
        z=c_mesh.flatten(),
        mode='markers',
        marker=dict(
            size=3,
            color=yield_mesh.flatten(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Yield')
        ),
        text=yield_mesh.flatten()
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title=label_a,
            yaxis_title=label_b,
            zaxis_title=label_c
        )
    )

    fig.show()

    return



def create_multidimensional_grid(dimensions):
    '''
    This function creates a grid of experimental conditions for the model to predict.
    
    Parameters:
    ----------
    dimensions: dict
        A dictionary with keys as dimension labels and values as (range, steps) tuples.
    
    Returns:
    -------
    exp_grid: pandas dataframe
    '''
    
    # Generate a list of arrays for each dimension
    dimension_arrays = []
    for label, (range_, steps) in dimensions.items():
        dimension_arrays.append(np.linspace(range_[0], range_[1], steps))
    
    # Generate a Cartesian product of the dimension arrays
    grid_points = list(product(*dimension_arrays))
    
    # Create the DataFrame
    exp_grid = pd.DataFrame(grid_points, columns=dimensions.keys())
    
    return exp_grid



def grid_feasibility_v2(data_df, label_a = "init_C", label_b = "init_N", label_c = "init_Li"):

    fact_check_list = []
    
    for i in tqdm(range(len(data_df))):
        
        fact_check_list.append(fact_check_v2([data_df[label_a].iloc[i],
                                              data_df[label_b].iloc[i],
                                              data_df[label_c].iloc[i]]))
    
    data_df['fact_check'] = fact_check_list
    
    return data_df




def fact_check_v2(combination):
    '''
    Checking if the nominated data (suggested experiment) satisfies acquisition guidlines or not
    
    Parameter:
    ----------
    combination: an array containing the experiment combinations. 
                     The combination of data must follow below order: 
                     [{C} , {N}, {Li}, {T}]
    Returens: 
    ----------
    check: True if all the criteria are satisfied, False if not 
    
    '''
    #print(combination)
    
    c = combination[0]
    n = combination[1]
    l = combination[2]
    
    c_total_limit = 0.5 * n
    
    
    if (n > 6):
        return False
    
    if c > c_total_limit:
        return False
    
    if c > 2.5: 
        return False    
    
    if l > 6 or l <= 0:
        return False
    
    if c < (l/2):
        return False
    
    #print('n: ', n, ' || n_total: ',n_total, ' || c: ', c, " || c_total_limit: ", c_total_limit, " || Li: ", l)
    return True 


def create_exp_grid_v2(label_a = "init_C", label_b = "init_N", label_c = "init_Li",
                    range_a = (0,2.5), range_b = (0,6), range_c = (0,6)):

    exp_grid = pd.DataFrame({label_a:[], label_b : [], label_c : []})
    
    for a in tqdm(np.linspace(range_a[0], range_a[1], 50)):
        for b in np.linspace(range_b[0],range_b[1],50):
            for c in np.linspace(range_c[0],range_c[1],50):
                    
                x_df = pd.DataFrame({label_a:[a],
                                        label_b : [b],
                                        label_c : [c]})
                exp_grid = exp_grid.append(x_df)
    
    return exp_grid



