# MSD Artist Clustering Analysis Notebook Prompt

You are tasked with creating a comprehensive Jupyter notebook (`msd_clustering_analysis.ipynb`) that performs artist-level clustering and popularity analysis on the extracted Million Song Dataset features.

## Objective

Build a clean, well-documented analysis notebook that:
1. Loads the extracted CSV files.
2. Performs exploratory data analysis (EDA).
3. Implements K-Means and Gaussian Mixture Model (GMM) clustering.
4. Analyzes cluster-popularity relationships using ANOVA and t-tests.
5. Generates visualizations and interpretable insights.

---

## Notebook Structure

Organize the notebook into **7 main sections** with clear markdown headers and code cells. Each section should be self-contained and runnable independently (with proper setup).

### Section 1: Setup & Data Loading

**Goal:** Import libraries, set random seeds, configure display options, load data.

**Content:**
- Import statements:
  - Data: `pandas`, `numpy`
  - ML: `sklearn` (preprocessing, cluster, metrics, decomposition)
  - Stats: `scipy.stats` (f_oneway, ttest_ind, etc.)
  - Viz: `matplotlib`, `seaborn`
  - Utils: `warnings`, `logging`
  
- Configuration:
  - Random seed (e.g., `np.random.seed(42)`)
  - Matplotlib style (e.g., `sns.set_style("whitegrid")`)
  - Pandas display options (max columns, etc.)

- Data loading:
  ```python
  songs_df = pd.read_csv('output/songs_10k_features.csv')
  artists_df = pd.read_csv('output/artists_1500_features.csv')
  segments_df = pd.read_csv('output/segments_long.csv')  # optional for now
  beats_df = pd.read_csv('output/beats_long.csv')  # optional for now
  ```

- Basic shape/info logging:
  ```
  Songs shape: (10000, 81)
  Artists shape: (1500, 150)
  ```

---

### Section 2: Data Exploration & Cleaning

**Goal:** Understand data, identify issues, prepare for clustering.

**Content:**

1. **Missing value analysis:**
   - Show counts/percentages of NaN per column in artists_df.
   - Identify zero-variance or near-zero-variance columns.
   - Display summary: "X columns are >50% missing, Y columns have zero variance."

2. **Distribution overview:**
   - For key features (tempo, loudness, duration), show:
     - Summary stats (mean, std, min, max).
     - Histograms/distribution plots (tempo, loudness, duration).
   - Check for outliers (e.g., artists with song_count = 1).

3. **Popularity distribution:**
   - Plot histogram of `artist_familiarity_mean` and `artist_hotttnesss_mean`.
   - Show correlation matrix snippet (familiarity vs hotttnesss).

4. **Handling missing data in artists_df:**
   - For columns with <10% missing: impute with column median.
   - For columns with >10% missing (e.g., all `_std` for single-song artists): drop or impute with 0.
   - Log: "Imputed X values; dropped Y columns."

5. **Year analysis (optional):**
   - Count songs with `year != 0` vs `year == 0`.
   - If useful, create age variable: `2026 - year` (only for non-zero years).
   - Note in markdown: "Year is heavily missing; will not use as clustering feature."

---

### Section 3: Feature Selection & Preprocessing

**Goal:** Choose features for clustering, standardize, reduce dimensionality.

**Content:**

1. **Feature selection:**
   - Define clustering features explicitly:
     - Scalar audio: tempo, time_signature, key, mode, loudness, duration, etc.
     - Counts: num_bars, num_beats, num_sections, num_tatums, num_segments.
     - Aggregated timbre means (12 features): artist_timbre_mean_i_mean.
     - Aggregated timbre stds (12 features): artist_timbre_std_i_mean.
     - Aggregated pitch means (12 features): artist_pitch_mean_i_mean.
     - Aggregated pitch stds (12 features): artist_pitch_std_i_mean.
     - Aggregated loudness (6 features): segments_loudness_max_mean_mean, etc.
   - **Exclude:**
     - artist_id, artist_name, song_count (identifiers).
     - artist_familiarity_mean, artist_hotttnesss_mean (evaluation only).
     - Any column with >50% missing or zero variance.
     - danceability_* and energy_* (all zeros in MSD).

   - Code:
   ```python
   # Define clustering feature columns explicitly
   clustering_features = [
       'artist_tempo_mean', 'artist_tempo_std',
       'artist_loudness_mean', 'artist_loudness_std',
       # ... add all relevant columns
   ]
   
   # Verify all exist
   missing_cols = [c for c in clustering_features if c not in artists_df.columns]
   if missing_cols:
       print(f"Warning: Missing columns: {missing_cols}")
   
   X = artists_df[clustering_features].copy()
   print(f"Clustering feature matrix shape: {X.shape}")
   ```

2. **Standardization:**
   - Use `sklearn.preprocessing.StandardScaler`.
   - Fit on X, transform.
   - Preserve the original X for later interpretation.
   
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
   print(f"Scaled features - mean: {X_scaled.mean().mean():.2e}, std: {X_scaled.std().mean():.3f}")
   ```

3. **Dimensionality reduction (PCA):**
   - Apply PCA to reduce dimensionality.
   - Retain 80-90% of variance (typical threshold).
   - Show scree plot and cumulative explained variance.
   
   ```python
   from sklearn.decomposition import PCA
   pca = PCA()
   X_pca = pca.fit_transform(X_scaled)
   
   # Find n_components for 85% variance
   cumsum = np.cumsum(pca.explained_variance_ratio_)
   n_components_85 = np.argmax(cumsum >= 0.85) + 1
   print(f"Variance retained with {n_components_85} PCs: {cumsum[n_components_85-1]:.2%}")
   
   # Plot scree plot
   plt.figure(figsize=(10, 5))
   plt.plot(cumsum)
   plt.axhline(0.85, color='r', linestyle='--', label='85% variance')
   plt.xlabel('Number of PCs')
   plt.ylabel('Cumulative Explained Variance')
   plt.legend()
   plt.title('PCA Scree Plot')
   plt.show()
   
   # Reduce to optimal n_components
   pca_optimal = PCA(n_components=n_components_85)
   X_pca_reduced = pca_optimal.fit_transform(X_scaled)
   ```

---

### Section 4: K-Means Clustering

**Goal:** Find optimal number of clusters, train K-Means, analyze results.

**Content:**

1. **Elbow method & silhouette analysis:**
   - For k = 2 to 15, compute:
     - Inertia (within-cluster sum of squares).
     - Silhouette score.
   - Plot both and identify elbow.
   
   ```python
   from sklearn.cluster import KMeans
   from sklearn.metrics import silhouette_score
   
   inertias = []
   silhouette_scores = []
   K_range = range(2, 16)
   
   for k in K_range:
       kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
       labels = kmeans.fit_predict(X_pca_reduced)
       inertias.append(kmeans.inertia_)
       silhouette_scores.append(silhouette_score(X_pca_reduced, labels))
   
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
   ax1.plot(K_range, inertias, 'bo-')
   ax1.set_xlabel('k')
   ax1.set_ylabel('Inertia')
   ax1.set_title('Elbow Method')
   
   ax2.plot(K_range, silhouette_scores, 'ro-')
   ax2.set_xlabel('k')
   ax2.set_ylabel('Silhouette Score')
   ax2.set_title('Silhouette Analysis')
   plt.tight_layout()
   plt.show()
   
   optimal_k = K_range[np.argmax(silhouette_scores)]
   print(f"Optimal k (silhouette): {optimal_k}")
   ```

2. **Train final K-Means model:**
   - Use optimal k.
   - Assign cluster labels.
   - Add to artists_df.
   
   ```python
   kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
   artists_df['kmeans_cluster'] = kmeans_final.fit_predict(X_pca_reduced)
   print(f"Cluster sizes:\n{artists_df['kmeans_cluster'].value_counts().sort_index()}")
   ```

3. **Cluster validation:**
   - Silhouette score for final model.
   - Davies-Bouldin index.
   - Summary: "Final model quality: silhouette = X, Davies-Bouldin = Y"

---

### Section 5: Gaussian Mixture Model (GMM)

**Goal:** Compare clustering with a probabilistic model, soft assignments.

**Content:**

1. **BIC/AIC for model selection:**
   - For k = 2 to 15, fit GMM, compute BIC and AIC.
   - Plot both, identify elbow.
   
   ```python
   from sklearn.mixture import GaussianMixture
   
   bics = []
   aics = []
   K_range = range(2, 16)
   
   for k in K_range:
       gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
       gmm.fit(X_pca_reduced)
       bics.append(gmm.bic(X_pca_reduced))
       aics.append(gmm.aic(X_pca_reduced))
   
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
   ax1.plot(K_range, bics, 'bo-')
   ax1.set_xlabel('k')
   ax1.set_ylabel('BIC')
   ax1.set_title('BIC Criterion')
   
   ax2.plot(K_range, aics, 'ro-')
   ax2.set_xlabel('k')
   ax2.set_ylabel('AIC')
   ax2.set_title('AIC Criterion')
   plt.tight_layout()
   plt.show()
   
   optimal_k_gmm = K_range[np.argmin(bics)]
   print(f"Optimal k (BIC): {optimal_k_gmm}")
   ```

2. **Train final GMM model:**
   - Use optimal k.
   - Assign hard labels (argmax of responsibilities).
   - Add to artists_df.
   
   ```python
   gmm_final = GaussianMixture(n_components=optimal_k_gmm, random_state=42, n_init=10)
   artists_df['gmm_cluster'] = gmm_final.fit_predict(X_pca_reduced)
   print(f"Cluster sizes:\n{artists_df['gmm_cluster'].value_counts().sort_index()}")
   ```

3. **Soft assignments (optional):**
   - Compute responsibilities (probability of each sample in each cluster).
   - Show average responsibility for best cluster per artist.

---

### Section 6: Cluster Analysis & Interpretation

**Goal:** Understand what each cluster represents based on audio features and popularity.

**Content:**

1. **Cluster profiles (K-Means):**
   - For each cluster, compute mean values of key audio features:
   
   ```python
   profile_features = ['artist_tempo_mean', 'artist_loudness_mean', 
                       'artist_duration_mean', 'artist_timbre_mean_1_mean', ...]
   
   for cluster in sorted(artists_df['kmeans_cluster'].unique()):
       cluster_data = artists_df[artists_df['kmeans_cluster'] == cluster]
       print(f"\n=== Cluster {cluster} (n={len(cluster_data)}) ===")
       print(cluster_data[profile_features].mean())
   ```

2. **Popularity by cluster:**
   - Compute mean familiarity/hotttnesss per cluster.
   - Create boxplots showing distribution of popularity per cluster.
   
   ```python
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
   
   artists_df.boxplot(column='artist_familiarity_mean', by='kmeans_cluster', ax=ax1)
   ax1.set_title('Artist Familiarity by K-Means Cluster')
   ax1.set_xlabel('Cluster')
   ax1.set_ylabel('Familiarity')
   
   artists_df.boxplot(column='artist_hotttnesss_mean', by='kmeans_cluster', ax=ax2)
   ax2.set_title('Artist Hotttnesss by K-Means Cluster')
   ax2.set_xlabel('Cluster')
   ax2.set_ylabel('Hotttnesss')
   
   plt.tight_layout()
   plt.show()
   ```

3. **Cluster naming (optional):**
   - Create a human-readable name for each cluster based on its audio signature.
   - Example:
     - Cluster 0: "Mainstream Power" (high loudness, high tempo, consistent timbre).
     - Cluster 1: "Experimental Niche" (varied loudness, varied timbre, lower familiarity).
   
   ```python
   cluster_names = {
       0: "Mainstream Power",
       1: "Experimental Niche",
       # ... etc.
   }
   artists_df['cluster_name'] = artists_df['kmeans_cluster'].map(cluster_names)
   ```

---

### Section 7: Statistical Hypothesis Testing

**Goal:** Formally test whether audio features and popularity differ significantly across clusters.

**Content:**

1. **ANOVA - Cluster differences in key features:**
   - For each audio feature, test: "Does this feature differ significantly across clusters?"
   - Use one-way ANOVA.
   - Report F-statistic and p-value.
   
   ```python
   from scipy.stats import f_oneway
   
   anova_results = []
   test_features = ['artist_tempo_mean', 'artist_loudness_mean', 
                    'artist_duration_mean', 'artist_timbre_mean_1_mean', ...]
   
   for feature in test_features:
       groups = [artists_df[artists_df['kmeans_cluster'] == c][feature].dropna() 
                 for c in sorted(artists_df['kmeans_cluster'].unique())]
       f_stat, p_value = f_oneway(*groups)
       anova_results.append({
           'Feature': feature,
           'F-Statistic': f_stat,
           'p-value': p_value,
           'Significant': 'Yes' if p_value < 0.05 else 'No'
       })
   
   anova_df = pd.DataFrame(anova_results).sort_values('F-Statistic', ascending=False)
   print(anova_df.to_string(index=False))
   ```

2. **ANOVA - Popularity by cluster:**
   - Test: "Does artist popularity differ significantly across clusters?"
   
   ```python
   for popularity_metric in ['artist_familiarity_mean', 'artist_hotttnesss_mean']:
       groups = [artists_df[artists_df['kmeans_cluster'] == c][popularity_metric].dropna() 
                 for c in sorted(artists_df['kmeans_cluster'].unique())]
       f_stat, p_value = f_oneway(*groups)
       print(f"\n{popularity_metric}:")
       print(f"  F-Statistic: {f_stat:.4f}")
       print(f"  p-value: {p_value:.2e}")
       print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
   ```

3. **t-tests - Popular vs Niche artists:**
   - Define "popular" as artists with familiarity >= 75th percentile.
   - Define "niche" as artists with familiarity < 25th percentile.
   - Compare audio features between these two groups.
   
   ```python
   familiarity_75 = artists_df['artist_familiarity_mean'].quantile(0.75)
   familiarity_25 = artists_df['artist_familiarity_mean'].quantile(0.25)
   
   popular = artists_df[artists_df['artist_familiarity_mean'] >= familiarity_75]
   niche = artists_df[artists_df['artist_familiarity_mean'] <= familiarity_25]
   
   print(f"Popular artists (n={len(popular)}), Niche artists (n={len(niche)})\n")
   
   ttest_results = []
   for feature in test_features:
       pop_values = popular[feature].dropna()
       niche_values = niche[feature].dropna()
       
       if len(pop_values) > 1 and len(niche_values) > 1:
           t_stat, p_value = ttest_ind(pop_values, niche_values)
           cohens_d = (pop_values.mean() - niche_values.mean()) / \
                      np.sqrt((pop_values.std()**2 + niche_values.std()**2) / 2)
           
           ttest_results.append({
               'Feature': feature,
               't-Statistic': t_stat,
               'p-value': p_value,
               "Cohen's d": cohens_d,
               'Significant': 'Yes' if p_value < 0.05 else 'No'
           })
   
   ttest_df = pd.DataFrame(ttest_results).sort_values('t-Statistic', ascending=False)
   print(ttest_df.to_string(index=False))
   ```

4. **Summary of key discriminative features:**
   - List top features that best separate popular from niche artists.
   - Show effect sizes (Cohen's d).

---

## Code Structure Guidelines

1. **Markdown clarity:**
   - Every section starts with a markdown header (`## Section Name`).
   - Include subsection explanations (markdown cells).
   - Use code comments sparingly; let variable names be self-documenting.

2. **Function organization:**
   - Define utility functions (for repeated tasks) in dedicated cells.
   - Example:
     ```python
     def cluster_profile_summary(df, feature_list, cluster_col='kmeans_cluster'):
         """Compute mean feature values per cluster."""
         # ... implementation
     ```

3. **Output clarity:**
   - Every analysis cell should end with a clear print statement or visualization.
   - Example: `print(f"Optimal k: {optimal_k} with silhouette score: {silhouette_scores[optimal_k-2]:.3f}")`

4. **Variable naming:**
   - Use descriptive names: `X_scaled`, `X_pca_reduced`, `kmeans_final`, `anova_results`.
   - Avoid single-letter variables except in loops.

5. **Reproducibility:**
   - Set random seeds at the start.
   - Document assumptions (e.g., "Cluster means computed excluding NaN values").
   - Save intermediate results if needed: `artists_df.to_csv('artists_with_clusters.csv', index=False)`

---

## Visualizations to Include

1. **PCA scree plot** (explained variance).
2. **Elbow plot & silhouette plot** (K-Means optimal k).
3. **BIC/AIC plot** (GMM optimal k).
4. **Cluster size bar charts** (K-Means and GMM).
5. **Boxplots of popularity by cluster** (familiarity and hotttnesss).
6. **ANOVA results table** (sorted by F-statistic).
7. **t-test results table** (sorted by t-statistic).
8. **Heatmap of cluster profiles** (mean feature values per cluster).

---

## Execution & Output

- The notebook should run end-to-end without errors.
- Each section should be independently interpretable.
- Final outputs:
  - Optimal cluster count (K-Means and GMM).
  - Cluster profiles and interpretations.
  - Statistical significance of cluster differences.
  - List of top discriminative features for popularity.

---

## Notes

- All code should be simple, readable, and avoid complex libraries beyond the specified ones.
- Use `pd.DataFrame` operations over loops where possible.
- Include markdown cells explaining each analysis step and interpretation.
- Save the final `artists_with_clusters.csv` for later reporting.
