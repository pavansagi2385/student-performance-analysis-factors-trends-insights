# -----------------------------------------
# 📦 Import Libraries
# -----------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# -----------------------------------------
# 📂 Load & Preprocess Data + Visualizations
# -----------------------------------------
class Visualizations:
    def __init__(self):
        # Ensure 'static' directory exists
        if not os.path.exists('static'):
            os.makedirs('static')

        # Load the dataset
        self.data = pd.read_csv("StudentsPerformance.csv")

        # Rename columns for consistency
        self.data.rename(columns={
            'race/ethnicity': 'race_ethnicity',
            'parental level of education': 'parent_education',
            'test preparation course': 'test_prep',
            'math score': 'math_score',
            'reading score': 'reading_score',
            'writing score': 'writing_score'
        }, inplace=True)

        # Convert score columns to numeric
        self.score_cols = ['math_score', 'reading_score', 'writing_score']
        self.data[self.score_cols] = self.data[self.score_cols].apply(pd.to_numeric, errors='coerce')

        # Add average score & grade
        self.data['average_score'] = self.data[self.score_cols].mean(axis=1)
        self.data['grade'] = self.data['average_score'].apply(self.assign_grade)

        # Generate visualizations
        self.plot_correlation_heatmap()
        self.plot_pairplot_by_gender()
        self.plot_violin_parent_education()
        self.plot_3d_scatter()
        self.plot_boxplot_lunch()
        self.plot_test_prep_impact()

        print("✅ All visualizations saved successfully in the 'static/' folder!")

    def assign_grade(self, score):
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

    # -----------------------------------------
    # 📊 Visualization Functions
    # -----------------------------------------

    # 1. Correlation Heatmap (Seaborn)
    def plot_correlation_heatmap(self):
        plt.figure(figsize=(6, 4))
        sns.heatmap(self.data[self.score_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap of Subject Scores")
        plt.tight_layout()
        plt.savefig("static/viz1_correlation_heatmap.png")
        plt.close()

    # 2. Pairplot by Gender (Seaborn)
    def plot_pairplot_by_gender(self):
        sns.pairplot(self.data, hue="gender", vars=self.score_cols)
        plt.suptitle("Pairplot of Scores by Gender", y=1.02)
        plt.savefig("static/viz2_pairplot_gender.png")
        plt.close()

    # 3. Violin Plot: Parental Education vs Math Score (Seaborn)
    def plot_violin_parent_education(self):
        plt.figure(figsize=(8, 5))
        sns.violinplot(x='parent_education', y='math_score', data=self.data, inner='quartile')
        plt.xticks(rotation=45)
        plt.title("Math Score by Parental Education Level")
        plt.tight_layout()
        plt.savefig("static/viz3_violin_parent_education.png")
        plt.close()

    # 4. 3D Scatter Plot: Math vs Reading vs Writing (Plotly)
    def plot_3d_scatter(self):
        fig = px.scatter_3d(
            self.data,
            x='math_score', y='reading_score', z='writing_score',
            color='gender',
            title='3D Scatter Plot of Exam Scores'
        )
        fig.write_html("static/viz4_3d_scatter.html")

    # 5. Box Plot: Lunch Type vs Reading Score (Seaborn)
    def plot_boxplot_lunch(self):
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='lunch', y='reading_score', data=self.data)
        plt.title("Reading Score by Lunch Type")
        plt.tight_layout()
        plt.savefig("static/viz5_boxplot_lunch.png")
        plt.close()

    # 6. Interactive Bar Chart: Test Prep Course Impact (Plotly)
    def plot_test_prep_impact(self):
        avg_scores = self.data.groupby('test_prep')[self.score_cols].mean().reset_index()
        fig = px.bar(
            avg_scores,
            x='test_prep',
            y=self.score_cols,
            barmode='group',
            title='Average Scores by Test Preparation Course'
        )
        fig.write_html("static/viz6_test_prep_bar.html")


# -----------------------------------------
# 🚀 Run
# -----------------------------------------
if __name__ == "__main__":
    Visualizations()
