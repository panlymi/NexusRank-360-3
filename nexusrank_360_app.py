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
    
    # Apply the impact direction (cost or benefit)
    for i in range(len(impacts)):
        if impacts[i] == "cost":
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

    # Request weights and impacts input for each criterion
    num_criteria = len(df.columns) - 1  # excluding the alternatives column
    weights = []
    impacts = []
    
    for i in range(1, num_criteria+1):
        # Get weight and impact type for each criterion
        weight = st.number_input(f"Enter weight for Criterion {i} (0 to 1)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
        impact = st.selectbox(f"Select impact for Criterion {i} (Cost/Benefit)", ["benefit", "cost"], key=f"impact_{i}")
        
        weights.append(weight)
        impacts.append(impact)
    
    if st.button('Rank Alternatives'):
        # Apply MOORA method and display rankings
        rankings = moora_method(df.iloc[:, 1:].values, weights, impacts)
        df['Ranking'] = rankings.argsort().argsort() + 1  # Rank the alternatives
        
        st.write("Ranking of Alternatives:")
        st.write(df[['Alternative', 'Ranking']])

# Additional features or calculations can be added here
