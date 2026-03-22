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
    """Plot a relational plot of budget vs. worldwide gross income."""
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x='budget',
        y='worlwide_gross_income',
        alpha=0.5,
        ax=ax)
    ax.set_title('Movie Budget vs. Worldwide Gross Income')
    ax.set_xlabel('Budget ($)')
    ax.set_ylabel('Worldwide Gross ($)')
    plt.savefig('relational_plot.png')
    return


def plot_categorical_plot(df):
    """Plot a categorical plot of the top 10 movie genres by frequency."""
    fig, ax = plt.subplots()
    temp = df.copy()
    temp['primary_genre'] = temp['genre'].str.split(',').str[0]
    temp['primary_genre'].value_counts().head(
        10).plot(kind='barh', color='teal', ax=ax)
    ax.set_title('Top 10 Movie Genres by Frequency')
    ax.set_xlabel('Count')
    plt.savefig('categorical_plot.png')
    return


def plot_statistical_plot(df):
    """Plot a heatmap of the correlation between key numerical attributes."""
    fig, ax = plt.subplots()
    corr = df[['duration', 'avg_vote', 'votes', 'metascore', 'budget']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Heatmap of Movie Attributes')
    plt.savefig('statistical_plot.png')
    return


def statistical_analysis(df, col: str):
    """Calculate and return the statistics of a column."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col].dropna())
    excess_kurtosis = ss.kurtosis(df[col].dropna())
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Preprocess the data by cleaning and transforming it as necessary."""
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
    print(df.head())
    print(df.select_dtypes(include=[np.number]).corr())
    return df


def writing(moments, col):
    """Write a summary of the statistical analysis results."""
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    # Delete the following options as appropriate for your data.
    # Not skewed and mesokurtic can be defined with asymmetries <-2 or >2.

    if moments[2] > 0.5:
        skew_type = "right skewed"
    elif moments[2] < -0.5:
        skew_type = "left skewed"
    else:
        skew_type = "not skewed"

    if moments[3] > 0.5:
        kurt_type = "leptokurtic"
    elif moments[3] < -0.5:
        kurt_type = "platykurtic"
    else:
        kurt_type = "mesokurtic"

    print(f'The data was {skew_type} and {kurt_type}.')
    return


def perform_clustering(df, col1, col2):
    """Perform K-Means clustering on two columns and return the results."""
    data = df[[col1, col2]].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    def plot_elbow_method():
        fig, ax = plt.subplots(figsize=(8, 6))
        inertia = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, random_state=42, n_init=10)
            km.fit(scaled_data)
            inertia.append(km.inertia_)
        ax.plot(range(1, 11), inertia, marker='o')
        ax.set_title('Elbow Method')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Inertia')
        plt.savefig('elbow_plot.png')
        return

    def one_silhouette_inertia():
        """Calculate silhouette score and inertia for clusters."""
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_data)
        _score = silhouette_score(scaled_data, labels)
        _inertia = km.inertia_
        return _score, _inertia

    # Gather data and scale

    # Find best number of clusters
    one_silhouette_inertia()
    plot_elbow_method()
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_data)

    cenlabels = scaler.inverse_transform(km.cluster_centers_)
    xkmeans, ykmeans = cenlabels[:, 0], cenlabels[:, 1]
    # Get cluster centers
    return labels, data, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Plot the clustered data with centroids."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    ax.scatter(xkmeans, ykmeans, c='red', marker='X', s=200, label='Centroids')
    ax.set_title('K-Means Clustering: Movie Ratings')
    ax.set_xlabel('Avg Vote')
    ax.set_ylabel('Metascore')
    ax.legend()
    plt.savefig('clustering.png')
    return


def perform_fitting(df, col1, col2):
    """Perform linear fitting between two columns and return data, x, y"""
    # Gather data and prepare for fitting
    x_data = df[[col1]].values
    y_data = df[[col2]].values

    # Scale both X and Y
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_scaled = scaler_x.fit_transform(x_data)
    y_scaled = scaler_y.fit_transform(y_data)

    # Fit model on scaled data
    params = np.polyfit(x_scaled.flatten(), y_scaled.flatten(), 1)

    # Generate smooth line in scaled space
    x_scaled_line = np.linspace(x_scaled.min(), x_scaled.max(), 100)
    y_scaled_line = np.polyval(params, x_scaled_line)

    # Inverse transform back to original scale
    x = scaler_x.inverse_transform(x_scaled_line.reshape(-1, 1)).flatten()
    y = scaler_y.inverse_transform(y_scaled_line.reshape(-1, 1)).flatten()

    data = (x_data.flatten(), y_data.flatten())

    # Predict across x
    return data, x, y


def plot_fitted_data(data, x, y):
    """Plot the original data and the fitted line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[0], data[1], alpha=0.3, label='Data')
    ax.plot(x, y, color='red', linewidth=3, label='Fit')
    ax.set_title('Linear Fit: Duration vs. Average Vote')
    ax.set_xlabel('Duration (min)')
    ax.set_ylabel('Avg Vote')
    ax.legend()
    plt.savefig('fitting.png')
    return


def main():
    """Main function to execute the data analysis, clustering, and fitting."""
    df = pd.read_csv('data.csv')
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
