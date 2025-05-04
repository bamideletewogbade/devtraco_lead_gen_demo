import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="DevTraco Lead AI",
    page_icon="🏠",
    layout="wide"
)

# Load artifacts
@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    encoder = joblib.load("encoder.joblib")
    return model, encoder

try:
    model, encoder = load_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Load Ghana-specific lead data
@st.cache_data
def load_data():
    df = pd.read_csv("devtraco_leads.csv")
    # Convert Yes/No to 1/0 for numeric operations
    df['converted_numeric'] = df['converted'].apply(lambda x: 1 if x == "Yes" else 0)
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")
selected_channels = st.sidebar.multiselect(
    "Acquisition Channels",
    options=sorted(df['source'].unique()),
    default=sorted(df['source'].unique())
)

selected_locations = st.sidebar.multiselect(
    "Locations",
    options=sorted(df['location_pref'].unique()),
    default=sorted(df['location_pref'].unique())
)

# Filter data based on selections
filtered_df = df[
    (df['source'].isin(selected_channels)) & 
    (df['location_pref'].isin(selected_locations))
]

# --- Main Dashboard ---
st.title("🏠 DevTraco Ghana Lead Intelligence")
st.write("AI-powered lead scoring for Ghana real estate developments")

# --- Row 1: Key Metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Leads", len(filtered_df))
with col2:
    st.metric("Conversion Rate", f"{filtered_df['converted_numeric'].mean():.1%}")
with col3:
    st.metric("Avg Budget (GHS)", f"{filtered_df['budget'].mean():,.0f}")
with col4:
    st.metric("Premium Leads", f"{filtered_df[filtered_df['budget'] > 3000000].shape[0]}")

# --- Row 2: Lead Scoring Simulator ---
st.header("🔍 Lead Scoring Simulator")

with st.expander("Simulate New Lead", expanded=True):
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        source = st.selectbox("Acquisition Channel", options=sorted(df['source'].unique()))
        budget = st.slider("Budget (GHS)", 200000, 10000000, 1500000, step=50000)
        location = st.selectbox("Location Preference", options=sorted(df['location_pref'].unique()))
    
    with sim_col2:
        property_type = st.selectbox("Property Type", options=sorted(df['property_type'].unique()))
        time_on_website = st.slider("Website Time (seconds)", 0, 1800, 120) if source in ['GhanaNewsOnline', 'GoogleAds', 'Facebook', 'PropertyWebsite'] else 0
        contacted = st.radio("Contact Status", ["Yes", "No"])

    # Predict
    try:
        source_encoded = encoder.transform([source])[0]
        prob = model.predict_proba([[source_encoded, budget, time_on_website]])[0][1]
        score = int(prob * 100)
        priority = "High" if score > 70 else "Medium" if score > 30 else "Low"
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        score = 0
        priority = "Error"

    # Display results
    st.subheader("AI Recommendation")
    score_col1, score_col2, score_col3 = st.columns(3)
    with score_col1:
        st.metric("Conversion Score", f"{score}%", help="Likelihood to convert based on similar historical leads")
    with score_col2:
        st.metric("Priority Tier", priority, 
                 help="High: Immediate follow-up\nMedium: Schedule call\nLow: Nurture campaign")
    with score_col3:
        action = "🚀 Hot Lead - Call Now" if priority == "High" else \
                "📅 Warm Lead - Follow Up" if priority == "Medium" else \
                "✉️ Nurture Sequence"
        st.metric("Recommended Action", action)

# --- Row 3: Channel Performance ---
st.header("📊 Channel Insights")

tab1, tab2, tab3 = st.tabs(["Channel ROI", "Location Analysis", "Lead Quality"])

with tab1:
    try:
        channel_stats = filtered_df.groupby('source').agg({
            'converted_numeric': 'mean',
            'budget': 'mean',
            'lead_id': 'count'
        }).rename(columns={
            'converted_numeric': 'Conversion Rate',
            'budget': 'Avg Budget (GHS)',
            'lead_id': 'Lead Count'
        }).sort_values('Conversion Rate', ascending=False)
        
        fig = px.bar(channel_stats, 
                     x=channel_stats.index, 
                     y='Conversion Rate',
                     color='Avg Budget (GHS)',
                     title='Conversion Rate by Channel',
                     color_continuous_scale='Bluered')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to generate channel stats: {str(e)}")

with tab2:
    try:
        loc_analysis = filtered_df.groupby('location_pref').agg({
            'converted_numeric': 'mean',
            'budget': 'mean'
        }).sort_values('converted_numeric', ascending=False)
        
        fig = px.scatter(loc_analysis,
                         x='budget',
                         y='converted_numeric',
                         size=filtered_df['location_pref'].value_counts(),
                         color=loc_analysis.index,
                         title='Location Performance (Bubble Size = Lead Volume)',
                         labels={'converted_numeric': 'Conversion Rate', 'budget': 'Average Budget (GHS)'})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to generate location analysis: {str(e)}")

with tab3:
    try:
        fig = px.box(filtered_df, 
                     x='source', 
                     y='budget',
                     color='converted',
                     title='Budget Distribution by Channel and Conversion Status')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to generate quality analysis: {str(e)}")

# --- Row 4: Raw Data ---
st.header("📋 Lead Database")
with st.expander("View All Leads"):
    st.dataframe(filtered_df.sort_values('budget', ascending=False), use_container_width=True)

# --- Footer ---
st.markdown("---")
st.caption("DevTraco Properties AI Lead Scoring System | v1.0")