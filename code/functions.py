import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.covariance import MinCovDet
from sklearn.mixture import GaussianMixture
import prince

def load_and_prepare_data(file_path):
    """
    Load and preprocess the dataset.
    - Reads CSV
    - Drops 'Field of Study' column
    - Maps categorical variables to numeric values
    - Scales selected columns using MinMaxScaler
    - Filters and renames columns
    
    Parameters:
        file_path (str): path to the CSV file.

    Returns:
        df (pd.DataFrame): original DataFrame after preprocessing
        filtered_df (pd.DataFrame): filtered and renamed DataFrame
        target_column (str): name of the target column
    """
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Field of Study'])

    # Mappings
    education_mapping = {'High School': 1, 'PhD': 4, "Master's": 3, "Bachelor's": 2}
    growth_mapping = {'High': 3, 'Low': 1, "Medium": 2}
    influence_mapping = {'High': 3, 'Low': 1, "Medium": 2}
    gender_mapping = {'Male': 0, 'Female': 1}

    df['Education Level'] = df['Education Level'].map(education_mapping)
    df['Industry Growth Rate'] = df['Industry Growth Rate'].map(growth_mapping)
    df['Family Influence'] = df['Family Influence'].map(influence_mapping).fillna(0).astype(int)
    df['Gender'] = df['Gender'].map(gender_mapping)

    # Scale certain columns
    scaler = MinMaxScaler()
    ordinal_col = ['Skills Gap', 'Job Satisfaction', 'Work-Life Balance', 'Job Security', 
                   'Professional Networks', 'Technology Adoption']
    df[ordinal_col] = scaler.fit_transform(df[ordinal_col])

    # Filter columns and rename
    target_column = 'Likely_to_Change_Occupation'
    filtered_columns = df.columns.difference(['Current Occupation', 'Career Change Interest'])
    filtered_df = df.loc[:, filtered_columns]
    filtered_df.columns = filtered_df.columns.str.replace(' ', '_').str.replace('-', '_')

    return df, filtered_df, target_column

def fit_logistic_regression(formula, train_df):
    """
    Fit a logistic regression model using statsmodels.

    Parameters:
        formula (str): formula specifying the model
        train_df (pd.DataFrame): training data

    Returns:
        model (statsmodels.discrete.discrete_model.BinaryResults): fitted logistic regression model
    """
    model = smf.logit(formula, data=train_df).fit()
    return model

def evaluate_model(model, train_df, test_df, target_column):
    """
    Evaluate the given model on train and test sets.

    Parameters:
        model: trained model
        train_df (pd.DataFrame): training data
        test_df (pd.DataFrame): test data
        target_column (str): name of the target variable

    Returns:
        dict: dictionary with train_accuracy, train_auc, test_accuracy, test_auc
    """
    # Training predictions
    train_predictions = model.predict(train_df)
    train_preds_class = (train_predictions > 0.5).astype(int)
    train_accuracy = accuracy_score(train_df[target_column], train_preds_class)
    train_auc = roc_auc_score(train_df[target_column], train_predictions)

    # Testing predictions
    test_predictions = model.predict(test_df)
    test_preds_class = (test_predictions > 0.5).astype(int)
    test_accuracy = accuracy_score(test_df[target_column], test_preds_class)
    test_auc = roc_auc_score(test_df[target_column], test_predictions)

    return {
        'train_accuracy': train_accuracy,
        'train_auc': train_auc,
        'test_accuracy': test_accuracy,
        'test_auc': test_auc
    }

def fit_random_forest(X_train, y_train):
    """
    Fit a random forest classifier.
    
    Parameters:
        X_train (pd.DataFrame or np.ndarray): training features
        y_train (pd.Series or np.ndarray): training targets

    Returns:
        rf_model (RandomForestClassifier): trained Random Forest model
    """
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_random_forest(rf_model, X_test, y_test):
    """
    Evaluate the random forest model.

    Parameters:
        rf_model (RandomForestClassifier): trained random forest model
        X_test (pd.DataFrame or np.ndarray): test features
        y_test (pd.Series or np.ndarray): test targets

    Returns:
        rf_accuracy (float)
        rf_auc (float)
    """
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_preds = (rf_probs > 0.5).astype(int)

    rf_accuracy = accuracy_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf_probs)
    return rf_accuracy, rf_auc

def detect_outliers(train_df, formula, target_column):
    """
    Detect outliers using Cook's distance, Mahalanobis distance, and MCD-based Mahalanobis distance.
    Plot results with Plotly and choose the best method based on training accuracy.

    Parameters:
        train_df (pd.DataFrame): training data
        formula (str): formula specifying the model
        target_column (str): name of the target variable

    Returns:
        chosen_outliers (np.ndarray): indices of chosen outliers
    """
    # Fit the model once on the full training set
    outlier_model = sm.Logit.from_formula(formula, data=train_df).fit(disp=0)

    # Cook's Distance
    influence = outlier_model.get_influence()
    cooks_d = influence.cooks_distance[0]
    cooks_threshold = 4 / len(train_df)
    cooks_outliers = np.where(cooks_d > cooks_threshold)[0]

    # Mahalanobis Distance
    X_train_values = train_df.drop(columns=[target_column]).values
    mean_vec = np.mean(X_train_values, axis=0)
    cov_mat = np.cov(X_train_values, rowvar=False)
    inv_cov_mat = np.linalg.inv(cov_mat)
    diff = X_train_values - mean_vec
    mahal_dist = np.sqrt(np.diag(diff @ inv_cov_mat @ diff.T))

    p = X_train_values.shape[1]
    cutoff = chi2.ppf(0.975, p)
    mahal_threshold = np.sqrt(cutoff)
    mahal_outliers = np.where(mahal_dist > mahal_threshold)[0]

    # MCD-Based Mahalanobis Distance
    mcd = MinCovDet().fit(X_train_values)
    mala_dis = mcd.mahalanobis(X_train_values)
    mcd_threshold = np.percentile(mala_dis, 97.5)
    mcd_outliers = np.where(mala_dis > mcd_threshold)[0]

    # Plot Cook's Distance
    fig_cooks = px.scatter(
        x=np.arange(len(cooks_d)),
        y=cooks_d,
        color=np.isin(np.arange(len(cooks_d)), cooks_outliers),
        title="Cook's Distance per Observation",
        labels={"x": "Observation Index", "y": "Cook's Distance", "color": "Is Outlier"}
    )
    fig_cooks.add_hline(y=cooks_threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig_cooks.show()

    # Plot Mahalanobis Distance
    fig_mahal = px.scatter(
        x=np.arange(len(mahal_dist)),
        y=mahal_dist,
        color=np.isin(np.arange(len(mahal_dist)), mahal_outliers),
        title="Mahalanobis Distance per Observation",
        labels={"x": "Observation Index", "y": "Mahalanobis Distance", "color": "Is Outlier"}
    )
    fig_mahal.add_hline(y=mahal_threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig_mahal.show()

    # Plot MCD Mahalanobis Distance
    fig_mcd = px.scatter(
        x=np.arange(len(mala_dis)),
        y=mala_dis,
        color=np.isin(np.arange(len(mala_dis)), mcd_outliers),
        title="MCD Mahalanobis Distance per Observation",
        labels={"x": "Observation Index", "y": "MCD Mahalanobis Distance", "color": "Is Outlier"}
    )
    fig_mcd.add_hline(y=mcd_threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig_mcd.show()

    # Function to retrain and score after removing outliers
    def retrain_and_score(removed_indices):
        clean_train_df = train_df.drop(index=train_df.index[removed_indices]).reset_index(drop=True)
        model = sm.Logit.from_formula(formula, data=clean_train_df).fit(disp=0)
        train_preds = (model.predict(clean_train_df) > 0.5).astype(int)
        acc = accuracy_score(clean_train_df[target_column], train_preds)
        return acc

    cooks_acc = retrain_and_score(cooks_outliers)
    mahal_acc = retrain_and_score(mahal_outliers)
    mcd_acc = retrain_and_score(mcd_outliers)

    print(f"Accuracy after removing Cook's outliers: {cooks_acc:.4f}")
    print(f"Accuracy after removing Mahalanobis outliers: {mahal_acc:.4f}")
    print(f"Accuracy after removing MCD outliers: {mcd_acc:.4f}")

    # Choose the best method
    accuracies = {"Cook's": cooks_acc, "Mahalanobis": mahal_acc, "MCD": mcd_acc}
    chosen_method = max(accuracies, key=accuracies.get)
    print(f"Chosen method: {chosen_method}")

    if chosen_method == "Cook's":
        chosen_outliers = cooks_outliers
    elif chosen_method == "Mahalanobis":
        chosen_outliers = mahal_outliers
    else:
        chosen_outliers = mcd_outliers

    print(f"Number of chosen outliers: {len(chosen_outliers)}")

    return chosen_outliers

def backward_elimination(data, formula, significance_level=0.1, t_stat_threshold=2):
    """
    Perform backward elimination for logistic regression, removing one variable per iteration
    until all remaining variables meet the significance criteria.

    Parameters:
        data (pd.DataFrame): dataset
        formula (str): initial model formula
        significance_level (float): p-value threshold for removal
        t_stat_threshold (float): t-statistic threshold for removal

    Returns:
        model (statsmodels.discrete.discrete_model.BinaryResults): final fitted model
        formula (str): updated formula after elimination
    """
    while True:
        model = smf.logit(formula, data=data).fit(disp=False)
        
        # Extract statistics
        coefficients = model.params
        std_errors = model.bse
        p_values = model.pvalues
        t_stats = (coefficients / std_errors).abs()

        stats_df = p_values.to_frame(name='p_value')
        stats_df['coeff'] = coefficients
        stats_df['stderr'] = std_errors
        stats_df['t_stat'] = t_stats

        # Exclude the intercept from consideration
        stats_df = stats_df.drop('Intercept', errors='ignore')

        # Identify variables that don't meet the criteria
        stats_to_remove = stats_df[(stats_df['p_value'] > significance_level) & 
                                   (stats_df['t_stat'] <= t_stat_threshold)]
        
        if stats_to_remove.empty:
            # Stop if no variables need to be removed
            break
        
        # Identify the variable with the highest p-value to remove
        feature_to_remove = stats_to_remove['p_value'].idxmax()
        
        print(f"Removing feature '{feature_to_remove}' with p-value {stats_df.loc[feature_to_remove, 'p_value']:.4f} "
              f"and t-stat {stats_df.loc[feature_to_remove, 't_stat']:.4f}")

        # Update the formula by removing the identified feature
        formula = formula.replace(f"+ {feature_to_remove}", "").replace(f"{feature_to_remove} +", "").replace(feature_to_remove, "")

    return model, formula

def data_process_cluster_v2(df):
    """
    Preprocess the dataset for clustering analysis:
    - Maps categorical variables to numeric values
    - Scales ordinal and continuous columns
    - Performs one-hot encoding of categorical columns

    Parameters:
        df (pd.DataFrame): original DataFrame

    Returns:
        df_encoded (pd.DataFrame): preprocessed DataFrame
    """
    # Mappings
    education_mapping = {'High School': 1, 'PhD': 4, "Master's": 3, "Bachelor's": 2}
    growth_mapping = {'High': 3, 'Low': 1, "Medium": 2}
    influence_mapping = {'High': 3, 'Low': 1, "Medium": 2}

    df['Education Level'] = df['Education Level'].map(education_mapping)
    df['Industry Growth Rate'] = df['Industry Growth Rate'].map(growth_mapping)
    df['Family Influence'] = df['Family Influence'].map(influence_mapping).fillna(0).astype(int)

    # Scaling ordinal columns
    ordinal_columns = ['Skills Gap', 'Job Satisfaction', 'Work-Life Balance', 'Job Security', 
                       'Professional Networks', 'Technology Adoption', 'Education Level', 
                       'Industry Growth Rate', 'Family Influence']
    scaler = MinMaxScaler()
    df[ordinal_columns] = scaler.fit_transform(df[ordinal_columns])

    # Scaling continuous columns
    numerical_col = ['Age', 'Years of Experience', 'Job Opportunities', 'Salary', 'Career Change Events']
    stscaler = StandardScaler()
    df[numerical_col] = stscaler.fit_transform(df[numerical_col])

    # One-hot encoding of categorical columns
    categorical_col = ['Field of Study', 'Current Occupation', 'Gender']
    df_encoded = pd.get_dummies(df, columns=categorical_col, drop_first=True)

    return df_encoded

def apply_pca_and_cluster(df_encoded, n_components=2, max_clusters=10):
    """
    Perform PCA for dimensionality reduction and Gaussian Mixture Model clustering,
    choosing the optimal number of clusters based on BIC.

    Parameters:
        df_encoded (pd.DataFrame): preprocessed DataFrame
        n_components (int): number of principal components
        max_clusters (int): maximum number of clusters to consider

    Returns:
        pca_data (np.ndarray): PCA-transformed data
        cluster_labels (np.ndarray): cluster labels
        optimal_clusters (int): optimal number of clusters
        gmm_optimal (GaussianMixture): GMM model with optimal number of clusters
        bic_scores (list): BIC scores for each number of clusters
    """
    # PCA
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df_encoded)

    # Calculate BIC and choose the number of clusters
    bic_scores = []
    clusters_range = range(1, max_clusters + 1)

    for n in clusters_range:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(pca_data)
        bic_scores.append(gmm.bic(pca_data))

    optimal_clusters = np.argmin(bic_scores) + 1
    print(f"Number of optimal cluster: {optimal_clusters}")

    # Clustering with optimal number of clusters
    gmm_optimal = GaussianMixture(n_components=optimal_clusters, random_state=42)
    gmm_optimal.fit(pca_data)
    cluster_labels = gmm_optimal.predict(pca_data)

    return pca_data, cluster_labels, optimal_clusters, gmm_optimal, bic_scores

def visualize_explained_variance(pca):
    """
    Visualize the cumulative explained variance.

    Parameters:
        pca (PCA): fitted PCA model
    """
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance by Number of Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.show()

def visualize_clusters(pca_data, cluster_labels):
    """
    Visualize the resulting clusters (requires 2 principal components).

    Parameters:
        pca_data (np.ndarray): PCA-transformed data (2D)
        cluster_labels (np.ndarray): cluster labels
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis', s=50)
    plt.title('Clustering with GMM (after PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.grid()
    plt.show()

def apply_pca_and_kmeans(df_encoded, df, n_components=2, n_clusters=3):
    """
    Perform PCA and then KMeans clustering.

    Parameters:
        df_encoded (pd.DataFrame): preprocessed data
        df (pd.DataFrame): original data
        n_components (int): number of PCA components
        n_clusters (int): number of clusters

    Returns:
        df (pd.DataFrame): original DataFrame with cluster labels and principal components
    """
    # PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_encoded)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(pca_result)

    # Add principal components to the DataFrame
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    return df

def visualize_kmeans_clusters(df):
    """
    Visualize clusters obtained by PCA and KMeans.

    Parameters:
        df (pd.DataFrame): DataFrame with cluster labels and principal components
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="PCA1", y="PCA2", hue="Cluster", style="Field of Study", palette="viridis", data=df, s=100
    )
    plt.title("K-means Clustering applying PCA")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend()
    plt.show()

def apply_tclust(df_encoded, n_clusters=3, alpha=0.1, n_components=2):
    """
    Apply a trimmed clustering approach after PCA.
    In this simplified approach, points are trimmed based on their distance 
    from the mean, and then KMeans is applied on the remaining points.

    Parameters:
        df_encoded (pd.DataFrame): encoded and preprocessed dataset
        n_clusters (int): number of clusters
        alpha (float): proportion of points to trim
        n_components (int): number of PCA components

    Returns:
        cluster_labels (np.ndarray): cluster labels for each observation
        trimmed_indices (list): indices of trimmed observations
        pca_result (np.ndarray): PCA-transformed dataset
        centers (np.ndarray): cluster centers
    """
    # PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_encoded)

    # Distance from the mean
    distances = np.linalg.norm(pca_result - np.mean(pca_result, axis=0), axis=1)
    threshold = np.percentile(distances, 100 * (1 - alpha))
    trimmed_indices = np.where(distances > threshold)[0]
    non_trimmed_indices = np.where(distances <= threshold)[0]
    pca_non_trimmed = pca_result[non_trimmed_indices]

    # KMeans on the non-trimmed data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pca_non_trimmed)

    cluster_labels = np.full(len(pca_result), -1)
    cluster_labels[non_trimmed_indices] = kmeans.labels_

    return cluster_labels, trimmed_indices, pca_result, kmeans.cluster_centers_

def visualize_tclust_clusters(pca_result, cluster_labels, trimmed_indices):
    """
    Visualize clusters and trimmed points from the trimmed clustering approach.

    Parameters:
        pca_result (np.ndarray): PCA-transformed data
        cluster_labels (np.ndarray): cluster labels
        trimmed_indices (list): indices of trimmed observations
    """
    plt.figure(figsize=(10, 8))

    # Plot non-trimmed clusters
    for cluster_idx in np.unique(cluster_labels):
        if cluster_idx == -1:  # trimmed points
            continue
        cluster_points = pca_result[cluster_labels == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_idx}")

    # Highlight trimmed observations
    trimmed_points = pca_result[trimmed_indices]
    plt.scatter(trimmed_points[:, 0], trimmed_points[:, 1], c='red', label="Trimmed Points", alpha=0.6)

    plt.title("Trimmed Clustering (t-Clust) with Trimmed Observations")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend()
    plt.grid()
    plt.show()

def trimmed_kmeans(X, k, alpha=0.1, max_iter=100, tol=1e-4):
    """
    Implementation of a simplified Trimmed k-means approach.
    The algorithm excludes a fraction of the points (alpha) that are farthest from the centroids.

    Parameters:
        X (np.ndarray): data array of shape (n_samples, n_features)
        k (int): number of clusters
        alpha (float): fraction of points to exclude
        max_iter (int): maximum number of iterations
        tol (float): tolerance for centroid convergence

    Returns:
        centroids (np.ndarray): final cluster centroids
        closest_cluster (np.ndarray): cluster assignments for each point
        excluded_points (np.ndarray): indices of excluded points
    """
    n_samples = X.shape[0]
    n_trim = int(alpha * n_samples)  # number of points to exclude

    np.random.seed(42)
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for iteration in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        closest_cluster = np.argmin(distances, axis=1)
        sorted_indices = np.argsort(np.min(distances, axis=1))

        # Exclude the farthest points
        trimmed_indices = sorted_indices[:-n_trim]

        new_centroids = np.array([
            X[trimmed_indices][closest_cluster[trimmed_indices] == cluster].mean(axis=0)
            for cluster in range(k)
        ])

        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids

    excluded_points = sorted_indices[-n_trim:]

    return centroids, closest_cluster, excluded_points

def evaluate_model_performance(models, accuracy, auc):
    """
    Visualize the comparison of Accuracy and AUC metrics across different models.

    Parameters:
        models (list): list of model names
        accuracy (list): list of accuracy values
        auc (list): list of AUC values
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 14), sharex=True)

    # Accuracy plot
    axs[0].bar(models, accuracy, color='cornflowerblue', alpha=0.8)
    axs[0].set_title('Model Accuracy Comparison', fontsize=16)
    axs[0].set_ylabel('Accuracy', fontsize=14)
    axs[0].set_ylim(0.7, 1.0)
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)

    # AUC plot
    axs[1].bar(models, auc, color='mediumseagreen', alpha=0.8)
    axs[1].set_title('Model AUC Comparison', fontsize=16)
    axs[1].set_ylabel('AUC', fontsize=14)
    axs[1].set_ylim(0.7, 1.0)
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    axs[1].set_xlabel('Models', fontsize=14)

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.show()

def apply_famd(data, n_components=2, random_state=42):
    """
    Applies Factorial Analysis of Mixed Data (FAMD) to the dataset.

    Parameters:
        data (pd.DataFrame): The input dataset.
        n_components (int): Number of components for dimensionality reduction.
        random_state (int): Random state for reproducibility.

    Returns:
        famd (prince.FAMD): Trained FAMD instance.
        X_famd (pd.DataFrame): Transformed data with reduced dimensions.
    """
    famd = prince.FAMD(
        n_components=n_components,
        n_iter=10,
        copy=True,
        check_input=True,
        random_state=random_state,
        engine="sklearn",
        handle_unknown="error"  # Same parameter as sklearn.preprocessing.OneHotEncoder
    )
    famd = famd.fit(data)
    X_famd = famd.transform(data)
    return famd, X_famd


def visualize_famd_clusters(X_famd, cluster_labels):
    """
    Visualizes clusters on the first two FAMD components.

    Parameters:
        X_famd (pd.DataFrame): Transformed data with reduced dimensions.
        cluster_labels (np.ndarray): Cluster labels obtained from clustering.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_famd.iloc[:, 0],
        X_famd.iloc[:, 1],
        c=cluster_labels,
        cmap='viridis',
        alpha=0.7
    )
    plt.title("Clusters visualized after FAMD dimensionality reduction")
    plt.xlabel("FAMD Component 1")
    plt.ylabel("FAMD Component 2")
    plt.colorbar(label='Cluster')
    plt.show()


def apply_kmeans_clustering(X, n_clusters=6, random_state=42):
    """
    Applies KMeans clustering to the data.

    Parameters:
        X (pd.DataFrame or np.ndarray): Input data for clustering.
        n_clusters (int): Number of clusters.
        random_state (int): Random state for reproducibility.

    Returns:
        cluster_labels (np.ndarray): Cluster labels for each data point.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X)
    return cluster_labels

