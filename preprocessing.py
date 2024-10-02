# Function to load data
@st.cache_data
def load_data(file):
    if file is not None:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format")
        return data
    else:
        return None

# Function to rename columns automatically
@st.cache_data
def rename_columns_auto(df_current, df_leavers):
    column_mappings = {}
    used_names = set()

    for current_column in df_current.columns:
        best_match, score = process.extractOne(current_column, df_leavers.columns)
        
        if score >= 50:  
            if best_match not in df_current.columns and best_match not in used_names:
                column_mappings[current_column] = best_match
                used_names.add(best_match)
            else:
                alt_best_match, alt_score = process.extractOne(current_column, df_leavers.columns)
                if alt_best_match not in df_current.columns and alt_best_match not in used_names:
                    column_mappings[current_column] = alt_best_match
                    used_names.add(alt_best_match)
    df_current = df_current.rename(columns=column_mappings)
    return df_current

# Function to drop columns not common in both datasets
@st.cache_data
def drop_columns_not_common(df_current, df_leavers):
    common_columns = list(set(df_current.columns) & set(df_leavers.columns))
    df_current = df_current[common_columns]
    df_leavers = df_leavers[common_columns]
    return df_current, df_leavers

# Function to preprocess data and encode it
@st.cache_data
def preprocess_data(cleaned_data):
    encoded_cleaned_data = pd.DataFrame()
    all_columns_dict = {}

    if not cleaned_data.empty:
        # Storing Values of data to encode
        encoded_cleaned_data = cleaned_data.copy()
        # Label Encoding
        for col in sorted(list(encoded_cleaned_data.select_dtypes(include='object').columns)):
            label = LabelEncoder()
            encoded_cleaned_data[col] = label.fit_transform(encoded_cleaned_data[col].astype(str))
            all_columns_dict[col] = dict(zip(label.classes_, label.transform(label.classes_)))
    return encoded_cleaned_data, all_columns_dict

# Function to handle missing values
def handle_missing_values(df):
    if df.isnull().sum().sum() > 0:
        st.error("Missing values detected")
        option = st.radio("Choose an option to handle missing values:", ('Fill data', 'Drop Null Values'))
        if option == 'Fill data':
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Fill categorical columns with modal value
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    # Fill numerical columns with mean
                    df[col] = df[col].fillna(df[col].mean())
        elif option == 'Drop Null Values':
            # Drop rows with null values
            df = df.dropna()
            df.reset_index(drop=True, inplace=True)
        
    return df