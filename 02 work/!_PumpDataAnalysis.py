import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

GRAVITY_CONSTANT = 9.8


class PumpDataAnalysis:
    def __init__(self, data):
        self.data = data
        self.features = ["total_head", "current", "output", "shaft_power", "pump_eff"]
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.nn = NearestNeighbors(
            n_neighbors=4
        )  # The first nearest neighbor is the point itself
        self.kmeans = KMeans(n_clusters=3, random_state=42)

    def calculate_theoretical_power_and_efficiency(self):
        self.data["theoretical_power"] = (
            self.data["capacity"] * self.data["total_head"] * GRAVITY_CONSTANT / 60
        )
        self.data["pump_eff"] = (
            self.data["theoretical_power"] / self.data["shaft_power"] * 100
        )

    def plot_change_in_variables(self):
        plt.figure(figsize=(8, 10))

        for i, variable in enumerate(self.features):
            plt.subplot(len(self.features), 1, i + 1)

            for test_date in self.data["TESTDATE"].unique():
                subset = self.data[self.data["TESTDATE"] == test_date]
                sns.lineplot(x="capacity", y=variable, data=subset)

            plt.title(f"Change in {variable} with respect to capacity")

        plt.tight_layout()
        plt.show()

    def plot_feature_distributions(self):
        for feature in self.features:
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))

            sns.histplot(
                self.data,
                x=feature,
                hue="test_no",
                multiple="stack",
                kde=True,
                ax=ax[0],
            )
            ax[0].set_title(f"Histogram of {feature} by test_no")

            sns.boxplot(x="test_no", y=feature, data=self.data, ax=ax[1])
            ax[1].set_title(f"Box plot of {feature} by test_no")

            sns.scatterplot(
                x="capacity", y=feature, hue="test_no", data=self.data, ax=ax[2]
            )
            ax[2].set_title(f"Scatter plot of {feature} vs capacity by test_no")

            sns.lineplot(
                x="TESTDATE",
                y=feature,
                hue="test_no",
                data=self.data.sort_values(["TESTDATE", "test_no"]),
                ax=ax[3],
            )
            ax[3].set_title(f"Time series plot of {feature} by test_no")

            plt.tight_layout()
            plt.show()

    def perform_regression_analysis(self):
        X = self.data[["total_head", "current", "shaft_power"]]
        Y = self.data["output"]
        X = sm.add_constant(X)  # Add a constant column to the predictors

        model = sm.OLS(Y, X)
        results = model.fit()

        print(results.summary())

    def calculate_outliers_zscore(self):
        for test_no in self.data["test_no"].unique():
            self.data.loc[self.data["test_no"] == test_no, self.features] = self.data[
                self.data["test_no"] == test_no
            ][self.features].apply(stats.zscore)

        self.data["outlier"] = self.data[self.features].apply(
            lambda x: (abs(x) > 3).any(), axis=1
        )

    def plot_pca_outliers_zscore(self):
        for test_no in self.data["test_no"].unique():
            data_test = self.data[self.data["test_no"] == test_no][self.features]
            outliers_test = self.data[self.data["test_no"] == test_no]["outlier"]

            data_scaled = self.scaler.fit_transform(data_test)
            data_pca = self.pca.fit_transform(data_scaled)

            z_scores = np.abs(stats.zscore(data_test)).max(axis=1)

            plt.scatter(
                data_pca[:, 0], data_pca[:, 1], c=z_scores, marker="o", cmap="viridis"
            )
            plt.scatter(
                data_pca[outliers_test, 0],
                data_pca[outliers_test, 1],
                c="red",
                marker="x",
            )
            plt.title(f"PCA plot for test_no {test_no}")
            plt.xlabel("First principal component")
            plt.ylabel("Second principal component")
            plt.colorbar(label="Maximum z-score")
            plt.show()

    def calculate_outliers_kmeans(self):
        self.data = self.data.dropna()

        for test_no in self.data["test_no"].unique():
            data_test = self.data[self.data["test_no"] == test_no][self.features]

            data_scaled = self.scaler.fit_transform(data_test)

            self.kmeans.fit(data_scaled)

            distances = self.kmeans.transform(data_scaled).min(axis=1)

            threshold = distances.mean() + 3 * distances.std()

            self.data.loc[self.data["test_no"] == test_no, "outlier"] = (
                distances > threshold
            )

    def plot_pca_outliers_kmeans(self):
        for test_no in self.data["test_no"].unique():
            data_test = self.data[self.data["test_no"] == test_no][self.features]
            outliers_test = self.data[self.data["test_no"] == test_no]["outlier"]

            data_scaled = self.scaler.fit_transform(data_test)

            data_pca = self.pca.fit_transform(data_scaled)

            plt.scatter(
                data_pca[:, 0],
                data_pca[:, 1],
                c=self.kmeans.predict(data_scaled),
                marker="o",
                cmap="viridis",
            )
            plt.scatter(
                data_pca[outliers_test, 0],
                data_pca[outliers_test, 1],
                c="red",
                marker="x",
            )
            plt.title(f"PCA plot for test_no {test_no}")
            plt.xlabel("First principal component")
            plt.ylabel("Second principal component")
            plt.show()

    def calculate_outliers_nearestneighbors(self):
        for test_no in self.data["test_no"].unique():
            data_test = self.data[self.data["test_no"] == test_no][self.features]

            data_scaled = self.scaler.fit_transform(data_test)

            self.nn.fit(data_scaled)

            distances, _ = self.nn.kneighbors(data_scaled)
            distances = distances[:, -1]

            threshold = distances.mean() + 3 * distances.std()

            self.data.loc[self.data["test_no"] == test_no, "outlier"] = (
                distances > threshold
            )

    def plot_pca_outliers_nearestneighbors(self):
        for test_no in self.data["test_no"].unique():
            data_test = self.data[self.data["test_no"] == test_no][self.features]
            outliers_test = self.data[self.data["test_no"] == test_no]["outlier"]

            data_scaled = self.scaler.fit_transform(data_test)

            data_pca = self.pca.fit_transform(data_scaled)

            plt.scatter(
                data_pca[:, 0],
                data_pca[:, 1],
                c=outliers_test,
                marker="o",
                cmap="viridis",
            )
            plt.scatter(
                data_pca[outliers_test, 0],
                data_pca[outliers_test, 1],
                c="red",
                marker="x",
            )
            plt.title(f"PCA plot for test_no {test_no}")
            plt.xlabel("First principal component")
            plt.ylabel("Second principal component")
            plt.show()
