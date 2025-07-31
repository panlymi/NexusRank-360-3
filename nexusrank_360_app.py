import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to normalize the decision matrix using vector normalization
def normalize_matrix(matrix):
    return matrix / np.linalg.norm(matrix, axis=0)

# Function to apply the MOORA method
def moora_method(matrix, weights, impacts):
    # Step 1: Normalize the matrix
    normalized_matrix = normalize_matrix(matrix)
    
    # Display normalized matrix
    st.write("Normalized Decision Matrix:")
    st.write(normalized_matrix)
    
    # Step 2: Apply weights to the normalized matrix
    weighted_matrix = normalized_matrix * weights
    st.write("Weighted Matrix:")
    st.write(weighted_matrix)
    
    # Step 3: Aggregate the scores
    aggregated_scores = weighted_matrix.sum(axis=1)
    st.write("Aggregated Scores (Before Impact Adjustment):")
    st.write(aggregated_scores)
    
    # Step 4: Apply impact adjustment (cost or benefit)
    for i in range(len(impacts)):
        if impacts[i] == "cost":
            aggregated_scores = -aggregated_scores

    # Display final aggregated scores
    st.write("Final Aggregated Scores (After Impact Adjustment):")
    st.write(aggregated_scores)
    
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
        
        # Rank the alternatives
        df['Ranking'] = rankings.argsort().argsort() + 1
        
        # Display Rankings
        st.write("Ranking of Alternatives:")
        st.write(df[['Alternative', 'Ranking']])
        
        # Step 5: Plot ranking graph (Bar chart)
        fig, ax = plt.subplots()
        ax.bar(df['Alternative'], df['Ranking'], color='skyblue')
        ax.set_xlabel('Alternatives')
        ax.set_ylabel('Rank')
        ax.set_title('Ranking of Alternatives Based on MOORA Method')
        st.pyplot(fig)
