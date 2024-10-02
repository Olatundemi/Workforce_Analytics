# Function to plot metrics
@st.cache_data
def plot_metrics(_best_model):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(_best_model.evals_result_['validation_0']['rmse'], label='Training RMSE')
    ax.plot(_best_model.evals_result_['validation_1']['rmse'], label='Validation RMSE')
    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel('RMSE')
    ax.set_title('Training and Validation RMSE over Boosting Rounds')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    pass