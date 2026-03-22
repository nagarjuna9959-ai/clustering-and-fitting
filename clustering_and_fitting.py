"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def plot_relational_plot(df):
    #fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='budget', y='worlwide_gross_income', alpha=0.5)
    plt.title('Movie Budget vs. Worldwide Gross Income')
    plt.xlabel('Budget ($)')
    plt.ylabel('Worldwide Gross ($)')
    plt.savefig('relational_plot.png')
    return


def plot_categorical_plot(df):
    #fig, ax = plt.subplots()
    df['primary_genre'] = df['genre'].str.split(',').str[0]
    df['primary_genre'].value_counts().head(10).plot(kind='barh', color='teal')
    plt.title('Top 10 Movie Genres by Frequency')
    plt.xlabel('Count')
    plt.savefig('categorical_plot.png')
    return


def plot_statistical_plot(df):
    #fig, ax = plt.subplots()
    corr = df[['duration', 'avg_vote', 'votes', 'metascore', 'budget']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Movie Attributes')
    plt.savefig('statistical_plot.png')
    return


def statistical_analysis(df, col: str):
    mean = df[col].mean()
    stddev = df[col].std()  
    skew = ss.skew(df[col].dropna())
    excess_kurtosis = ss.kurtosis(df[col].dropna())
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    # You should preprocess your data in this function and
    # make use of quick features such as 'describe', 'head/tail' and 'corr'.
    # Convert currency strings to floats
    for col in ['budget', 'worlwide_gross_income']:
        if col in df.columns:
            df[col] = df[col].replace(r'[\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaNs in key columns for analysis
    cols_to_fix = ['avg_vote', 'metascore', 'duration', 'budget']
    df = df.dropna(subset=cols_to_fix)
    
    print(df.describe())
    return df


def writing(moments, col):
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    # Delete the following options as appropriate for your data.
    # Not skewed and mesokurtic can be defined with asymmetries <-2 or >2.
    skew_type = "right skewed" if moments[2] > 0 else "left skewed"
    kurt_type = "leptokurtic" if moments[3] > 0 else "platykurtic"
    print(f'The data was {skew_type} and {kurt_type}.')

    return


def perform_clustering(df, col1, col2):
    data = df[[col1, col2]].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    def plot_elbow_method():
        #fig, ax = plt.subplots()
        inertia = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertia.append(kmeans.inertia_)
        plt.plot(range(1, 11), inertia, marker='o')
        plt.title('Elbow Method')
        plt.savefig('elbow_plot.png')
        return

    def one_silhouette_inertia():
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_data)
        _score = silhouette_score(scaled_data, labels)
        _inertia = km.inertia_
        return _score, _inertia,labels, km

    # Gather data and scale

    # Find best number of clusters
    one_silhouette_inertia()
    plot_elbow_method()
    score, inertia, labels, kmeans_obj = one_silhouette_inertia()
    cenlabels = scaler.inverse_transform(kmeans_obj.cluster_centers_)
    xkmeans, ykmeans = cenlabels[:, 0], cenlabels[:, 1]

    # Get cluster centers
    return labels, data, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    #fig, ax = plt.subplots()
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(xkmeans, ykmeans, c='red', marker='X', s=200, label='Centroids')
    plt.title('K-Means Clustering: Movie Ratings')
    plt.xlabel('Avg Vote')
    plt.ylabel('Metascore')
    plt.legend()
    plt.savefig('clustering.png')
    return


def perform_fitting(df, col1, col2):
    # Gather data and prepare for fitting
    x_data = df[col1].values
    y_data = df[col2].values

    # Fit model
    params = np.polyfit(x_data, y_data, 1)
    line_x = np.linspace(x_data.min(), x_data.max(), 100)
    line_y = np.polyval(params, line_x)

    # Predict across x
    return (x_data, y_data), line_x, line_y


def plot_fitted_data(data, x, y):
    #fig, ax = plt.subplots()
    plt.figure(figsize=(8, 6))
    plt.scatter(data[0], data[1], alpha=0.3, label='Data')
    plt.plot(x, y, color='red', linewidth=3, label='Fit')
    plt.title('Linear Fit: Duration vs. Average Vote')
    plt.xlabel('Duration (min)')
    plt.ylabel('Avg Vote')
    plt.legend()
    plt.savefig('fitting.png')
    return


def main():
    df = pd.read_csv('data.csv',low_memory=False)
    df = preprocessing(df)
    col = 'avg_vote'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    clustering_results = perform_clustering(df, 'avg_vote', 'metascore')
    plot_clustered_data(*clustering_results)
    fitting_results = perform_fitting(df, 'duration', 'avg_vote')
    plot_fitted_data(*fitting_results)
    return


if __name__ == '__main__':
    main()
    
