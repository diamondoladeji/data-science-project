#import "./template/lib.typ": *
#import "@preview/tablem:0.3.0": tablem, three-line-table

#show: project.with(
  title: "Analyzing Research Trends in Computer Science Using the DBLP Dataset",
  author: "Olutoba \"Toba\" Sanyaolu (2135924), Diamond Oladeji (2164567), Diego Coronado (2303693)",
  date: "December 05, 2025",
  remove-hpi-logo: true,
  chair: "Data Science I Final Report",
)

= Introduction

Scientific publishing continues to expand rapidly, making it essential to develop scalable methods for understanding research dynamics. In this project, we analyze a large corpus of academic papers through preprocessing, text mining, clustering, temporal trend analysis, predictive modeling, and network analysis. Text features are extracted using TF‑IDF and dimensionality reduction, enabling us to group papers into coherent topic clusters. Temporal analysis then reveals how these clusters evolve over time, highlighting both emerging areas such as machine learning and declining paradigms. Predictive models including Logistic Regression, Random Forest, and LightGBM are trained to classify papers by cluster, with performance evaluated using accuracy and F1 metrics.

In addition, we construct citation networks to measure influence through PageRank and centrality, identifying key papers and communities. Together, these methods provide both technical rigor and actionable insights into how ideas spread, venues specialize, and fields evolve. This report serves as a methodological guide and analytical narrative, bridging machine learning, network science, and interpretive clarity.


= Dataset Description

The dataset for this project comes from the DBLP Computer Science Bibliography, a large collection of metadata on computer science publications. We worked with JSON files (`dblp-ref-0.json` through `dblp-ref-3.json`) containing records with titles, abstracts, authors, publication years, venues, references, and citation counts. After loading the files into a single pandas DataFrame, we cleaned the data by dropping entries missing critical fields, filling missing values, and restricting the year range to 1950–2017.

For prototyping, we sampled subsets of 50,000–100,000 papers, while the full pipeline scales to millions of records. This dataset is valuable because it combines textual content for clustering and trend analysis with citation networks for influence and community detection, providing a rich foundation for exploring how research topics evolve and spread over time.

= Task 1: Data Preprocessing and Feature Generation
== Data Cleaning
The four DBLP JSON files were merged into a single dataset before any sampling or feature extraction. Papers missing a title or publication year were removed at the start, since both fields are required for text analysis and for understanding changes in research output over time


=== Venue Filtering and Missing Venue Removal

Because the DBLP dataset included thousand of venues, many of which appear only a handful of times, we first calculated venue frequencies and kept only venues with at least fifty papers. During this step, we discovered that more than 506,699 papers had an empty venue value (`""`). Because this missing entry appeared so frequently, it passed the $gt.eq 50$ threshold and would have shown up as the most common venue, which was not meaningful information.

To correct this, we removed all papers with missing venue metadata before sampling. After this fix, the most frequent venues in the cleaned dataset were recognizable and legitimate publication outlets such as _Communications of the ACM, Journal of the ACM, and IEEE Transactions on Information Theory_, confirming that the venue field was now reliable.

=== Removing a Citation Artifact

When we later plotted the citation distribution using a log-scaled histogram, we observed a large spike at exactly 50 citations, which did not align with the expected heavy tailed distribution that is observed in bibliometric datasets. Because the spike was evidence of a possible data artifact, we removed all papers with `n_citation = 50` before sampling. After this correction, the updated histogram (see 3.6) displayed a smooth decline in the upper tail, with:

- median = 7 citations
- 75th percentile = 52
- 95th percentile = 245
- 99th percentile = 808
- maximum = 17,064

These values match what we expect from large citation datasets.

== Stratified Sampling

After cleaning the venues and removing the citation artifact, we drew a stratified sample from the dataset by selecting up to 2,000 papers per publication year. We sampled per year to prevent the sample from being dominated by modern papers and in order to preserve long-term temporal structure. Sampling was intentionally done after cleaning the dataset, in order to ensure that the sample did not include missing venues or distorted citation values.

== Text Normalization

After sampling, text fields were normalized:

- missing abstracts were replaced with empty strings
- titles and abstracts were converted to lowercase for consistent tokenisation
- a combined text field was created (text = title + " " + abstract)

This combined field served as the input for TF-IDF.

== Black-Box Feature Generation

=== TF-IDF Representation

Once the combined text field (title + abstract) was prepared, we transformed each paper into a numerical representation using TF-IDF. To keep representation focused and manageable, we limited the vocabulary to the 5,000 most informative terms and removed standard English stop words. This allowed the model to capture meaningful short phrases (e.g. "neural network", "data mining") rather than only being able to capture isolated keywords.

This produced a 5,000 dimensional matrix, where each paper is represented by a term of weights. Although TF-IDF preserves important distinctions between documents, the feature space is large and not ideal for clustering or visualization, which motivated the next step.

=== Choosing the PCA Dimensionality

In order to choose an appropriate number of components, we evaluated cluster quality across a range of PCA sizes. For each value in ${50, 75, 100, 125, ..., 300}$, we reduced the TF-IDF matrix to k components and computed the silhouette score after running K-means. The silhouette score consistently decreased as dimensionality increased, and the highest score was obtained at 50 components. Although the silhouette scores and cumulative explained variance values are low this is expected for text datasets since TF-IDF variance is extremely spread out across thousands of components and text datasets also do not form sharply separated clusters.

#figure(
  image("./images/pca_ncomponent_tradeoff.png", width: 65%),
  caption: "PCA n_dimension tradeoff",
)





=== Venue and Author Embeddings

The PCA features were also aggregated to represent entities larger than individual papers.

For venues we took all papers that appeared in the same venue and averaged their 100-dimensional PCA vectors. This gives each venue a single vector that captures the thematic profile of that venue. Since empty venues were removed earlier every venue embedding corresponds to a real publication source.

For authors, we expanded the dataset so that each author-paper pair became an individual row. We then averaged the PCA vectors for all papers written by the same author. This produces on vector per author that summarizes the kind of topics they commonly work on.

These aggregated embeddings allow us to compare authors, venues, and individual papers using a consistent feature representation, which is useful for clustering, similarity analysis, and studying broader patterns in the DBLP dataset.

== Interpretable Feature Engineering

We also generated several interpretable metadata features:

- Number of authors: This measures collaboration size. When plotted across years, it shows a clear upward trend, increasing from roughly one author per paper in the early decades to about 3.8 authors by 2018.- title length in characters
- Title length: Recorded as the number of characters in the title. This offers a simple structural indicator of how concise or descriptive titles are.

- Number of references: The count of cited papers included in each record. This is useful for examining how referencing depth changes over time or across venues.

- Citation velocity: Defined as citations divided by the paper’s age $(2018−"year"+1)$. This normalizes citation counts so older papers do not automatically appear more influential.


=== Metadata Correlation Check

A correlation heatmap (see 3.6) confirmed that these engineered features are not redundant. For example:

- number of authors increases with year $(r approx 0.48)$
- number of references also increases with year $(r approx 0.42)$
- citation velocity is almost perfectly correlated with citation $(r approx 0.90)$

Other relationships were weak, and indicates that the metadata features capture different aspects of each paper.

== Exploratory Validation

The EDA plots were used to validate that the preprocessing produced a clean well-behaved sample:

=== Citation Distribution Plot

#figure(
  image("./images/eda_citation_distribution.png", width: 65%),
  caption: "Citation Distribution",
)

The log-scaled histogram of citation counts show a heavy-tailed shape after removing `n_citation = 50`. There were no remaining artificial spikes, and the upper tail decreased smoothly.

=== Collaboration Trend Plot

#figure(
  image("./images/eda_collaboration_trend.png", width: 65%),
  caption: "Collaboration Trend Plot",
)

The line plot of the average number of authors per year showed a clear upward trend, confirming that collaboration in computer science has increased over time. This pattern is held consistently from the 1970s through 2018.

=== Venue Frequency Plot

#figure(
  image("./images/eda_top_venues.png", width: 65%),
  caption: "Venue Frequency Plot",
)

After removing empty venues, the top-20 venue plot consisted of real recognized conferences and journals. No placeholder or missing values appeared.

=== Metadata Correlation Plot

#figure(
  image("./images/eda_corr_matrix.png", width: 65%),
  caption: "Metadata Correlation Plot",
)

The correlation heatmap showed reasonable relationships among metadata features, with no signs of duplicated or bad features.

Together these diagnostic plots confirm that the dataset is clean, internally consistent, and suitable for the modeling and trend analysis performed in later sections.


= Task 2: Topic Clustering

== Overview

The cleaned and feature-engineered dataset from Task 1 was used to identify broad research themes within the DBLP corpus. Because DBLP papers do not come with ground-truth topic labels, we used unsupervised learning to discover natural groupings of papers based on their textual similarity. The analysis followed the pipeline:
1. TF-IDF vectorization
2. PCA reduction to 50 components
3. K-means clustering
4. Keyword extraction to interpret clusters
5. PCA and t-SNE visualizations

The pipeline produced eight research topics and revealed one unexpected cluster of non English papers, which we removed before re-fitting the model.

== Selecting the Number of Clusters

We explored several values of k and inspected the resulting keyword sets and cluseter coherence. Although DBLP contains many subfields, we found that using a large number of clusters leads to overly fragmented groups. Through experimentation we found $k = 8$ to be the best value because it offered the best balance as it was enough to capture big fields in Computer Science (like Theory, Systems, and AI) without splitting the data into overly specific niches.


== Initial Clustering and Removal of Non-English Papers



The first run of K-means revealed a small but distinct cluster dominated by German language papers (194 papers). Its top keywords ('der', 'die', 'und', ...) were because considered "noise" in our analysis. Retaining them would introduce irrelevant vocabulary into the feature space and intefere with downstream supervised classification modeling (Task 4), as the model might learn to classify based on language rather than topic. So we removed these papers and re-fitted the entire TF-IDF $arrow.r$ PCA $arrow.r$ K-Means pipeline. This resulted in sharper topic boundaries and more coherent keyword sets in the remaining clusters.

== Final Topic Clusters and Interpretations

After removing the German-language documents, the re-fitted model identified eight clear research themes. We assigned labels to each cluster by examining their top-10 TF-IDF keywords:

- _Logic & Formal Methods:_ ("logic", "symbolic", "semantics", "intuitionistic", "modal")
- _Numerical Computing:_ ("matrix", "polynomial", "inversion", "equation", "solution")
- _Algorithms & Theory:_ ("graph", "linear", "finite", "set", "algorithms")
- _Compilers & Languages:_ ("languages", "grammars", "context-free", "programming")
- _Coding Theory:_ ("error", "codes", "decoding", "cyclic", "binary")
- _Information Retrieval:_ ("retrieval", "document", "model", "search", "user")
- _Systems & Networks:_ ("network", "performance", "models", "control", "traffic")
- _General / Applied CS":_ ("information", "research", "computer", "design", "software")

These labels serve as the ground truth targers for our predictive modeling tasks.


== Visualization with PCA

We visualized the clusters using the first two PCA components.

#figure(
  image("./images/"),
  caption: "Topic clusters projected onto the first two PCA components.",
)
