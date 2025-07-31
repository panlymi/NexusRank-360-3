
import streamlit as st
import pandas as pd
import numpy as np

# Function to normalize the decision matrix using vector normalization
def normalize_matrix(matrix):
    return matrix / np.linalg.norm(matrix, axis=0)

# Function to apply the MOORA method
def moora_method(matrix, weights, impacts):
    normalized_matrix = normalize_matrix(matrix)
    weighted_matrix = normalized_matrix * weights
    aggregated_scores = weighted_matrix.sum(axis=1)

    # Apply the impact direction
    for i in range(len(impacts)):
        if impacts[i] == "negative":
            aggregated_scores = -aggregated_scores

    return aggregated_scores

# Streamlit UI elements
st.title("NexusRank 360: MOORA-Based Ranking System")

# File upload
uploaded_file = st.file_uploader("Upload your decision matrix CSV file", type=["csv"])
if uploaded_file is not None:
    # Load the decision matrix
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Decision Matrix:")
    st.write(df)

    # Request weights and impacts input
    num_criteria = len(df.columns) - 1  # excluding the alternatives column
    weights = st.multiselect("Enter weights for each criterion", [float(i) for i in range(1, num_criteria+1)])
    impacts = st.multiselect("Enter impact (positive/negative) for each criterion", ["positive", "negative"] * num_criteria)

    if len(weights) == num_criteria and len(impacts) == num_criteria:
        # Apply MOORA method and display rankings
        rankings = moora_method(df.iloc[:, 1:].values, weights, impacts)
        df['Ranking'] = rankings.argsort().argsort() + 1  # Rank the alternatives

        st.write("Ranking of Alternatives:")
        st.write(df[['Alternative', 'Ranking']])

# Additional features or calculations can be added here
