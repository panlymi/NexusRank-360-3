import streamlit as st
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="NexusRank 360", page_icon="🌐", layout="centered")

# --- Helper Function for MOORA (Unchanged) ---
def moora_method(df, weights, criteria_types):
    """
    Performs the MOORA method calculation.
    """
    norm_df = df.copy()
    for col in df.columns:
        norm_df[col] = df[col] / np.sqrt((df[col]**2).sum())
    
    weighted_df = norm_df.copy()
    for col in weighted_df.columns:
        weighted_df[col] *= weights[col]
        
    scores = []
    for i in range(len(weighted_df)):
        benefit_score = sum(weighted_df.iloc[i][col] for col, c_type in criteria_types.items() if c_type == 'positive')
        cost_score = sum(weighted_df.iloc[i][col] for col, c_type in criteria_types.items() if c_type == 'negative')
        scores.append(benefit_score - cost_score)
        
    result_df = pd.DataFrame({'MOORA Score (Yi)': scores}, index=df.index)
    result_df['Rank'] = result_df['MOORA Score (Yi)'].rank(ascending=False).astype(int)
    return result_df.sort_values(by='Rank')

# --- Main Application UI ---
st.title("NexusRank 360: MOORA-Based Ranking System")

uploaded_file = st.file_uploader(
    "Upload your decision matrix CSV or Excel file",
    type=["csv", "xlsx"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)
        
        alt_col = df_input.columns[0]
        criteria_cols = df_input.columns[1:]
        df_criteria = df_input.set_index(alt_col)

        # Display uploaded file info and data matrix
        # Use st.code to display the filename, looks like the image
        st.write("**Uploaded Decision Matrix:**")
        st.code(f"{uploaded_file.name}")
        st.dataframe(df_input)
        st.markdown("---")

        # --- NEW & IMPROVED: Inputs for Weights and Impacts ---
        st.subheader("Enter Weights for each criterion")
        weights_input = st.text_input(
            "Enter weights separated by commas (e.g., 0.4, 0.3, 0.2, 0.1)",
            placeholder="e.g., 0.25, 0.25, 0.25, 0.25"
        )
        
        st.subheader("Enter Impact (positive/negative) for each criterion")
        
        # Create a dictionary to hold the impact for each criterion
        impacts = {}
        # Use columns to create a clean horizontal layout
        cols = st.columns(len(criteria_cols))

        for i, col_name in enumerate(criteria_cols):
            with cols[i]:
                # Use st.selectbox for a compact choice
                impacts[col_name] = st.selectbox(
                    f"**{col_name}**", # The label is the criterion name
                    options=["positive", "negative"],
                    key=f"impact_{col_name}"
                )
        
        st.markdown("---")
        
        if st.button("🚀 Calculate Final Ranks", type="primary", use_container_width=True):
            try:
                # 1. Process and validate weights
                weights_list = [float(w.strip()) for w in weights_input.split(',')]
                if len(weights_list) != len(criteria_cols):
                    st.error(f"Error: You entered {len(weights_list)} weights, but there are {len(criteria_cols)} criteria. Please provide one weight for each criterion.")
                else:
                    # If weights are valid, proceed
                    weights = dict(zip(criteria_cols, weights_list))
                    
                    if not np.isclose(sum(weights.values()), 1.0):
                        st.warning(f"The sum of weights is {sum(weights.values()):.2f}. It's recommended that weights sum to 1.0.")
                    
                    final_ranking = moora_method(df_criteria, weights, impacts)

                    st.header("🏆 Final Ranking")
                    st.dataframe(
                        final_ranking.style.format({'MOORA Score (Yi)': "{:.4f}"})
                                  .background_gradient(cmap='viridis_r', subset=['Rank']),
                        use_container_width=True
                    )
                    
                    st.subheader("Visual Comparison of MOORA Scores")
                    chart_data = final_ranking[['MOORA Score (Yi)']].sort_values(by='MOORA Score (Yi)', ascending=False)
                    st.bar_chart(chart_data)

            except ValueError:
                st.error("Invalid input for weights. Please enter numbers separated by commas only (e.g., 0.5, 0.3, 0.2).")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
