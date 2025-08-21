import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# Set page configuration:
st.set_page_config(page_title="UK Rail Dashboard", page_icon="🚆", layout="wide")

# --- Data Loading and Caching ---
# Use st.cache_data to load and process data only once.
@st.cache_data
def load_data(file_path):
    """
    Loads and preprocesses the UK rail data from a CSV file.
    - Fills missing values.
    - Converts data types.
    - Removes outliers based on price.
    - Creates new features for analysis.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Display a clear error and stop the app if the data file is not found.
        st.error(f"Data file not found. Please ensure '{Path(file_path).name}' is in the same directory as the script.")
        st.stop()
    
    # --- Data Cleaning (from the notebook) ---
    # 1. Handle Missing Values
    df['Railcard'].fillna('None', inplace=True)
    # For 'Reason for Delay', fill NaNs with 'On Time' for clarity
    df['Reason for Delay'].fillna('On Time', inplace=True)
    
    # 2. Correct Data Types
    # Use format='mixed' to handle both 'YYYY-MM-DD' and 'MM/DD/YYYY'
    df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], format='mixed')
    df['Date of Journey'] = pd.to_datetime(df['Date of Journey'], format='mixed')
    
    # Combine date and time more concisely
    for col in ['Departure', 'Arrival', 'Actual Arrival']:
        df[f'{col} Datetime'] = pd.to_datetime(df['Date of Journey'].astype(str) + ' ' + df[f'{col} Time'], errors='coerce')

    # 3. Feature Engineering
    df['Route'] = df['Departure Station'] + ' to ' + df['Arrival Destination']
    df['Departure Hour'] = df['Departure Datetime'].dt.hour
    df['Delay in Mins'] = (df['Actual Arrival Datetime'] - df['Arrival Datetime']).dt.total_seconds() / 60
    
    # 4. Outlier Removal (as per the notebook's logic)
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)].copy()
    
    return df_cleaned

# --- Path Configuration and Data Loading ---
# Get the directory of the current script to build a robust, absolute path.
script_dir = Path(__file__).resolve().parent
data_path = script_dir / "cleaned_UK_Rides.csv"
body_image_path = script_dir / "mm_body.jpg"
sidebar_image_path = script_dir / "mm_s.jpg"

df = load_data(data_path)

# --- Sidebar ---
st.sidebar.markdown("<h1 style='text-align: left; font-weight: 900; font-size: 28px; padding-left: 10px;'>UK Rail Tickets Data Analysis</h1>", unsafe_allow_html=True)
st.sidebar.image(str(sidebar_image_path))

st.sidebar.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
st.sidebar.markdown("**Objective**")
st.sidebar.markdown("Analyze National Rail’s ticket sales data to generate insights that support passenger train operators across England, Scotland, and Wales.")

st.sidebar.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
st.sidebar.markdown("**Dataset Briefing**")
st.sidebar.markdown("The dataset covers UK National Rail ticket sales (Jan–Apr 2024), including ticket type, journey date/time, stations, price, and related details.")

st.sidebar.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
st.sidebar.markdown("**Business Requirements**")
st.sidebar.markdown("""
The dashboard should provide insights into:
- The most popular routes
- Highlight peak travel times
- Analyze revenue by ticket type and class
- Evaluate on-time performance along with its contributing factors
""")

st.sidebar.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
st.sidebar.markdown("**Prepared by**")
st.sidebar.markdown("Mohammed Abdullah AlMahel")
st.sidebar.markdown("Contact: +966 503 409 483")
st.sidebar.markdown("Role: AI Engineer")
st.sidebar.markdown("Date: 22-Aug-2025")

# --- Main Page Layout ----
st.image(str(body_image_path))
st.markdown("✨ Start exploring — use the tabs below to seamlessly navigate the app’s insights!")

# --- Helper function for filters to avoid code duplication ---
def create_interactive_filters(df, key_prefix, exclude_filters=None):
    """
    Creates a row of selectbox filters and returns the filtered DataFrame.
    - key_prefix: A unique string to avoid widget key collisions.
    - exclude_filters: A list of filter names to omit (e.g., ['status']).
    """
    if exclude_filters is None:
        exclude_filters = []
        
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        class_filter = st.selectbox("Ticket Class", options=['None'] + list(df['Ticket Class'].unique()), key=f'class_{key_prefix}')
    with col2:
        type_filter = st.selectbox("Ticket Type", options=['None'] + list(df['Ticket Type'].unique()), key=f'type_{key_prefix}')
    with col3:
        purchase_filter = st.selectbox("Purchase Type", options=['None'] + list(df['Purchase Type'].unique()), key=f'purchase_{key_prefix}')
    
    status_filter = 'None' # Default value
    if 'status' not in exclude_filters:
        with col4:
            status_filter = st.selectbox("Journey Status", options=['None'] + list(df['Journey Status'].unique()), key=f'status_{key_prefix}')

    # Apply filters without creating an unnecessary copy of the dataframe
    filtered_df = df
    if class_filter != 'None':
        filtered_df = filtered_df[filtered_df['Ticket Class'] == class_filter]
    if type_filter != 'None':
        filtered_df = filtered_df[filtered_df['Ticket Type'] == type_filter]
    if purchase_filter != 'None':
        filtered_df = filtered_df[filtered_df['Purchase Type'] == purchase_filter]
    if status_filter != 'None':
        filtered_df = filtered_df[filtered_df['Journey Status'] == status_filter]
        
    return filtered_df

# Create tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🗺️ Route Analysis", "💰 Revenue Analysis", "⏱️ Performance Analysis", "👤 Customer Behavior"])

# --- Tab 1: Dashboard ---
with tab1:
    st.header("Key Performance Indicators")
    st.write("Use the filters below to see how the main KPIs change for different segments.")

    # --- KPI Filters ---
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        kpi_class = st.selectbox("Filter by Ticket Class", options=['All'] + list(df['Ticket Class'].unique()), key='kpi_class')
    with kpi_col2:
        kpi_type = st.selectbox("Filter by Ticket Type", options=['All'] + list(df['Ticket Type'].unique()), key='kpi_type')
    with kpi_col3:
        kpi_status = st.selectbox("Filter by Journey Status", options=['All'] + list(df['Journey Status'].unique()), key='kpi_status')

    # Apply KPI filters
    df_kpi_selection = df.copy()
    if kpi_class != 'All': df_kpi_selection = df_kpi_selection[df_kpi_selection['Ticket Class'] == kpi_class]
    if kpi_type != 'All': df_kpi_selection = df_kpi_selection[df_kpi_selection['Ticket Type'] == kpi_type]
    if kpi_status != 'All': df_kpi_selection = df_kpi_selection[df_kpi_selection['Journey Status'] == kpi_status]
    
    # Top-level metrics
    total_journeys = int(df_kpi_selection.shape[0])
    total_revenue = int(df_kpi_selection['Price'].sum())
    avg_price = round(df_kpi_selection['Price'].mean(), 2) if not df_kpi_selection.empty else 0
    
    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Total Journeys")
        st.subheader(f"{total_journeys:,}")
    with middle_column:
        st.subheader("Total Revenue")
        st.subheader(f"£ {total_revenue:,}")
    with right_column:
        st.subheader("Average Price")
        st.subheader(f"£ {avg_price}")
        
    st.markdown("---")
    
    # Display a sample of the filtered data
    st.write("Data Sample for Selected KPIs:")
    st.dataframe(df_kpi_selection.head())

# --- Tab 2: Route & Time Analysis ---
with tab2:
    st.header("Popular Routes & Travel Times")

    # Q1: What are the most popular routes?
    st.subheader("Top 10 Most Popular Routes")
    
    # Use the helper function to create filters and get the filtered data
    filtered_df_route = create_interactive_filters(df, key_prefix='route')

    if filtered_df_route.empty:
        st.warning("No data available for the selected filters.")
    else:
        top_10_routes = filtered_df_route['Route'].value_counts().nlargest(10).index
        plot_df_routes = filtered_df_route[filtered_df_route['Route'].isin(top_10_routes)]
        
        fig_routes = px.bar(
            plot_df_routes,
            y='Route',
            color='Ticket Class',
            orientation='h',
            labels={'count': 'Number of Journeys', 'Route': 'Route', 'Ticket Class': 'Ticket Class'},
            title='Top 10 Most Popular Routes by Ticket Class',
            category_orders={'Route': top_10_routes} # Ensures order is maintained
        )
        fig_routes.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_routes, use_container_width=True)
        
        # Generate dynamic insights based on the filtered data
        top_route_name = top_10_routes[0]
        st.info(f"""
        **Insight:** The chart displays the most frequently traveled routes for your current selection.
        - The **{top_route_name}** route is currently the most popular.
        - Routes originating from London are prominent, highlighting its role as a central hub.
        - The color breakdown shows the proportion of First and Standard class tickets for each route.
        """)

    # Q2: What are the peak travel times?
    st.subheader("Journeys by Hour of Day")

    # This chart shows the overall trend across all data, without filters.
    journeys_by_hour = df['Departure Hour'].value_counts().sort_index()
    fig_hours = px.bar(
        journeys_by_hour,
        x=journeys_by_hour.index,
        y=journeys_by_hour.values,
        labels={'x': 'Hour of Day (24-hour format)', 'y': 'Number of Journeys'},
        title='Total Number of Journeys by Hour'
    )
    st.plotly_chart(fig_hours, use_container_width=True)
    st.info("""
    **Insight:** This chart reveals the overall travel peaks across all journeys.
    - **Morning Rush:** A sharp increase in travel around 6-8 AM, typical for morning commutes.
    - **Evening Rush:** Another peak occurs between 4 PM and 6 PM (16:00-18:00), corresponding to the evening commute.
    - Travel is significantly lower during off-peak hours and overnight.
    """)

# --- Tab 3: Revenue Analysis ---
with tab3:
    st.header("Revenue Insights")

    # Q3: How does revenue vary by ticket types and classes?
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Ticket Type")
        revenue_by_type = df.groupby('Ticket Type')['Price'].sum().sort_values(ascending=False)
        fig_rev_type = px.bar(
            revenue_by_type,
            x=revenue_by_type.index,
            y=revenue_by_type.values,
            labels={'x': 'Ticket Type', 'y': 'Total Revenue (£)'},
            title='Total Revenue by Ticket Type'
        )
        st.plotly_chart(fig_rev_type, use_container_width=True)
        st.info("""
        **Insight:** 'Advance' tickets generate the most revenue, likely due to their high volume. 'Anytime' tickets, while having a higher average price, contribute less to the total revenue, suggesting fewer are sold.
        """)

    with col2:
        st.subheader("Revenue by Ticket Class")
        revenue_by_class = df.groupby('Ticket Class')['Price'].sum()
        fig_rev_class = px.pie(
            values=revenue_by_class.values,
            names=revenue_by_class.index,
            title='Revenue Share by Ticket Class',
            hole=0.4
        )
        st.plotly_chart(fig_rev_class, use_container_width=True)
        st.info("""
        **Insight:** Standard Class tickets account for the vast majority of revenue, which is expected given they represent the bulk of sales. First Class, while having a higher price per ticket, contributes a smaller portion to the overall revenue.
        """)

# --- Tab 4: Performance Analysis ---
with tab4:
    st.header("Journey Performance")

    # Q4: What is the on-time performance?
    st.subheader("Journey Status Overview")
    status_counts = df['Journey Status'].value_counts()
    fig_status = px.pie(
        status_counts,
        values=status_counts.values,
        names=status_counts.index,
        title='Journey Status Distribution',
        hole=0.4,
        color_discrete_map={'On Time':'green', 'Delayed':'orange', 'Cancelled':'red'}
    )
    st.plotly_chart(fig_status, use_container_width=True)
    st.info("""
    **Insight:** The majority of journeys are 'On Time'. Delays and cancellations, while less frequent, still impact a notable portion of trips. This highlights a generally reliable service but with room for improvement in punctuality and service continuity.
    """)

    # Q5: What are the main contributing factors to delays?
    st.subheader("Reasons for Delays")

    # Use the main helper function, but exclude the 'Journey Status' filter for this chart
    filtered_df_delays = create_interactive_filters(df, key_prefix='del', exclude_filters=['status'])

    # Filter for only delayed journeys
    delayed_journeys = filtered_df_delays[filtered_df_delays['Journey Status'] == 'Delayed']

    if delayed_journeys.empty:
        st.info("No delayed journeys in the selected data to analyze.")
    else:
        top_delay_reasons = delayed_journeys['Reason for Delay'].value_counts().index
        fig_delays = px.bar(
            delayed_journeys,
            y='Reason for Delay',
            color='Ticket Class',
            orientation='h',
            labels={'count': 'Number of Incidents', 'Reason for Delay': 'Reason for Delay'},
            title='Top Reasons for Delays by Ticket Class',
            category_orders={'Reason for Delay': top_delay_reasons}
        )
        fig_delays.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_delays, use_container_width=True)
        
        # Generate dynamic insight for the top delay reason
        top_delay_reason = top_delay_reasons[0]
        st.info(f"""
        **Insight:** For your current selection, **'{top_delay_reason}'** is the most common cause of delays. Use the filters to see if certain ticket types or classes are more affected by specific issues.
        """)

# --- Tab 5: Purchase Behavior ---
with tab5:
    st.header("Purchase Behavior")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Purchase Type Distribution")
        purchase_counts = df['Purchase Type'].value_counts()
        fig_purchase = px.pie(
            values=purchase_counts.values,
            names=purchase_counts.index,
            title='Tickets by Purchase Type',
            hole=0.4
        )
        st.plotly_chart(fig_purchase, use_container_width=True)
        st.info("""
        **Insight:** The vast majority of tickets are purchased online, highlighting the importance of the digital sales channel. Station purchases are the second most common method.
        """)

    with col2:
        st.subheader("Railcard Usage")
        railcard_counts = df['Railcard'].value_counts()
        fig_railcard = px.pie(
            values=railcard_counts.values,
            names=railcard_counts.index,
            title='Railcard Usage Distribution',
            hole=0.4
        )
        st.plotly_chart(fig_railcard, use_container_width=True)
        st.info("""
        **Insight:** A significant portion of travelers (over 60%) do not use a railcard. For those who do, the 'Adult' railcard is the most frequently used, suggesting its popularity among regular commuters.
        """)

        
