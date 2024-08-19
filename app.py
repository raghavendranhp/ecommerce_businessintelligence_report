import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import seaborn as sns
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from cachetools import cached, TTLCache
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.preprocessing import StandardScaler

"""
# Initialize geocoder and cache
geolocator = Nominatim(user_agent="geoapiExercises")
cache = TTLCache(maxsize=1000, ttl=3600)  # Cache with 1-hour expiration

@cached(cache)
def geocode_city(city, state, country):
    try:
        location = geolocator.geocode(f"{city}, {state}, {country}")
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None"""
# Load the dataset
file_path = r"C:\Users\ksund\Music\denrite\DA assignment\DA assignment\ecommerce_data.csv"
data = pd.read_csv(file_path, encoding='ISO-8859-1')


# Convert relevant columns to appropriate types
data['order_date'] = pd.to_datetime(data['order_date'], format='%d-%m-%Y')
data['sales_per_order'] = data['sales_per_order'].astype(float)
data['profit_per_order'] = data['profit_per_order'].astype(float)
data['order_id'] = data['order_id'].astype(str)
data['customer_city'] = data['customer_city'].astype(str)
data['customer_state'] = data['customer_state'].astype(str)
data['customer_country'] = data['customer_country'].astype(str)
data['customer_segment'] = data['customer_segment'].astype(str)
data['customer_id'] = data['customer_id'].astype(str)
# Convert relevant columns to appropriate types
data['order_item_discount'] = data['order_item_discount'].astype(float)
# Add latitude and longitude columns based on the geocoding function
#data[['latitude', 'longitude']] = data.apply(lambda row: pd.Series(geocode_city(row['customer_city'], row['customer_state'], row['customer_country'])), axis=1)

# Convert relevant columns to appropriate types
data['order_date'] = pd.to_datetime(data['order_date'], format='%d-%m-%Y')
data['ship_date'] = pd.to_datetime(data['ship_date'], format='%d-%m-%Y')
data['days_for_shipment_scheduled'] = data['days_for_shipment_scheduled'].astype(int)
data['days_for_shipment_real'] = data['days_for_shipment_real'].astype(int)
data['shipping_type'] = data['shipping_type'].astype(str)

# Calculate shipment delays
data['shipment_delay'] = data['days_for_shipment_real'] - data['days_for_shipment_scheduled']


# Page 1: Executive Summary
def executive_summary():
    st.title("Executive Summary")

    # Overview KPIs
    total_sales = data['sales_per_order'].sum()
    total_profit = data['profit_per_order'].sum()
    number_of_orders = data['order_id'].nunique()

    st.header("Overview")
    st.write(f"**Total Sales:** ${total_sales:,.2f}")
    st.write(f"**Total Profit:** ${total_profit:,.2f}")
    st.write(f"**Number of Orders:** {number_of_orders}")

    # Key Highlights
    st.header("Key Highlights")

    # Monthly Sales Trends
    data['order_month'] = data['order_date'].dt.to_period('M')
    monthly_sales = data.groupby('order_month')['sales_per_order'].sum()

    st.subheader("Monthly Sales Trends")
    st.line_chart(monthly_sales)

    # Top-Performing Categories
    top_categories = data.groupby('category_name')['sales_per_order'].sum().sort_values(ascending=False)
    
    st.subheader("Top-Performing Categories")
    st.bar_chart(top_categories)


# Page 2: Sales and Category Analysis
def sales_and_category_analysis():
    st.title("Sales and Category Analysis")

    # Monthly Sales Trends
    data['order_month'] = data['order_date'].dt.to_period('M')
    monthly_sales = data.groupby('order_month')['sales_per_order'].sum()

    st.header("Monthly Sales Trends")
    st.line_chart(monthly_sales)

    # Product Categories
    st.header("Sales by Category")
    sales_by_category = data.groupby('category_name')['sales_per_order'].sum().sort_values(ascending=False)

    # Display as bar chart
    st.subheader("Sales by Category (Bar Chart)")
    st.bar_chart(sales_by_category)

    # Display as pie chart
    st.subheader("Sales by Category (Pie Chart)")
    st.pyplot(pie_chart(sales_by_category))

    # Top and Bottom Products
    st.header("Top and Bottom Products")
    top_products = data.groupby('product_name')['sales_per_order'].sum().sort_values(ascending=False).head(10)
    bottom_products = data.groupby('product_name')['sales_per_order'].sum().sort_values(ascending=False).tail(10)

    st.subheader("Top 10 Products")
    st.write(top_products)

    st.subheader("Bottom 10 Products")
    st.write(bottom_products)

def pie_chart(sales_by_category):
    fig, ax = plt.subplots()
    ax.pie(sales_by_category, labels=sales_by_category.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return fig

# Page 3: Customer Segmentation
def customer_segmentation():
    st.title("Customer Segmentation")

    # Customer Segments
    st.header("Customer Segments")
    sales_by_segment = data.groupby('customer_segment')['sales_per_order'].sum()

    st.subheader("Sales Distribution by Segment")
    st.pyplot(pie_chart1(sales_by_segment))

    # Geographic Sales Analysis
    st.header("Geographic Sales Analysis")
    geographic_sales = data[['customer_city', 'customer_state', 'customer_country', 'latitude', 'longitude', 'sales_per_order']].dropna()

    # Create a map centered around the average latitude and longitude
    avg_lat = geographic_sales['latitude'].mean()
    avg_lon = geographic_sales['longitude'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5)

    # Add data points to the map
    for _, row in geographic_sales.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['customer_city']}, {row['customer_state']} - Sales: ${row['sales_per_order']:.2f}",
            icon=folium.Icon(color='blue')
        ).add_to(m)

    st.subheader("Sales Distribution Map")
    st_folium(m, width=700, height=500)

    # Customer Behavior
    st.header("Customer Behavior")

    # Repeat Purchase Behavior
    customer_order_counts = data['customer_id'].value_counts()
    repeat_customers = customer_order_counts[customer_order_counts > 1].count()
    total_customers = customer_order_counts.count()
    retention_rate = repeat_customers / total_customers

    st.subheader("Customer Retention")
    st.write(f"Total Customers: {total_customers}")
    st.write(f"Repeat Customers: {repeat_customers}")
    st.write(f"Customer Retention Rate: {retention_rate:.2%}")

    # Top Products Ordered by Customer Segment
    st.subheader("Top Products by Customer Segment")
    top_products_by_segment = data.groupby(['customer_segment', 'product_name'])['sales_per_order'].sum().reset_index()
    top_products_by_segment = top_products_by_segment.sort_values(by=['customer_segment', 'sales_per_order'], ascending=[True, False])
    top_products_by_segment = top_products_by_segment.groupby('customer_segment').head(5)

    st.write(top_products_by_segment)

def pie_chart1(sales_by_segment):
    fig, ax = plt.subplots()
    ax.pie(sales_by_segment, labels=sales_by_segment.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return fig

# Page 4: Shipping and Logistics Analysis
def shipping_logistics_analysis():
    st.title("Shipping and Logistics Analysis")

    # Shipment Type Performance
    st.header("Shipment Type Performance")
    shipment_performance = data.groupby('shipping_type').agg({
        'days_for_shipment_real': ['mean', 'std'],
        'shipment_delay': ['mean', 'std']
    })
    shipment_performance.columns = ['_'.join(col).strip() for col in shipment_performance.columns.values]
    shipment_performance = shipment_performance.reset_index()

    st.subheader("Average Delivery Time by Shipping Type")
    st.write(shipment_performance[['shipping_type', 'days_for_shipment_real_mean']])

    # Efficiency Analysis
    st.header("Efficiency Analysis")
    heatmap_data = data.pivot_table(index='shipping_type', columns='order_date', values='days_for_shipment_real', aggfunc='mean')

    st.subheader("Heatmap of Scheduled vs. Real Shipment Days")
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".1f")
    st.pyplot()

    # Impact on Customer Satisfaction
    st.header("Impact on Customer Satisfaction")
    # Retention Rate
    customer_order_counts = data['customer_id'].value_counts()
    repeat_customers = customer_order_counts[customer_order_counts > 1].count()
    total_customers = customer_order_counts.count()
    retention_rate = repeat_customers / total_customers

    # Average shipment delay
    average_delay = data['shipment_delay'].mean()

    st.subheader("Correlation Between Shipping Efficiency and Retention Rate")
    st.write(f"Customer Retention Rate: {retention_rate:.2%}")
    st.write(f"Average Shipment Delay (Days): {average_delay:.2f}")

    # Correlation analysis
    shipping_efficiency = data.groupby('shipping_type')['shipment_delay'].mean().reset_index()
    shipping_efficiency.columns = ['Shipping Type', 'Average Delay']
    shipping_efficiency = shipping_efficiency.sort_values(by='Average Delay')

    st.write(shipping_efficiency)
# Page 5: Retention and Cohort Analysis
def retention_cohort_analysis():
    st.title("Retention and Cohort Analysis")

    # Retention Rate
    st.header("Retention Rate")
    
    # Creating a cohort group based on the first purchase month
    data['order_month'] = data['order_date'].dt.to_period('M')
    cohort_data = data.groupby(['customer_id', 'order_month']).size().reset_index(name='order_count')
    cohort_data['cohort_month'] = cohort_data.groupby('customer_id')['order_month'].transform('min')
    
    cohort_pivot = cohort_data.pivot_table(index='cohort_month', columns='order_month', values='order_count', aggfunc='sum').fillna(0)
    retention = cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0) * 100

    st.subheader("Cohort Analysis Retention Table")
    st.write(retention)

    plt.figure(figsize=(12, 8))
    sns.heatmap(retention, annot=True, fmt=".1f", cmap="YlGnBu")
    st.subheader("Cohort Analysis Heatmap")
    st.pyplot()

    # Customer Lifetime Value (CLV)
    st.header("Customer Lifetime Value (CLV)")
    
    # Calculate total sales per customer
    customer_sales = data.groupby('customer_id')['sales_per_order'].sum().reset_index()
    customer_sales.columns = ['customer_id', 'total_sales']
    
    # Predict CLV using a simple linear regression model
    X = np.array(range(len(customer_sales))).reshape(-1, 1)
    y = customer_sales['total_sales'].values
    model = LinearRegression()
    model.fit(X, y)
    customer_sales['predicted_clv'] = model.predict(X)
    
    st.subheader("Customer Sales and Predicted CLV")
    st.write(customer_sales.sort_values(by='total_sales', ascending=False))

    plt.figure(figsize=(12, 6))
    plt.scatter(customer_sales.index, customer_sales['total_sales'], label='Actual Sales', color='blue')
    plt.plot(customer_sales.index, customer_sales['predicted_clv'], color='red', linestyle='--', label='Predicted CLV')
    plt.xlabel('Customer Index')
    plt.ylabel('Sales / CLV')
    plt.title('Customer Sales vs Predicted CLV')
    plt.legend()
    st.pyplot()

    # Insights on Retention Strategies
    st.header("Insights on Retention Strategies")
    
    retention_rate = cohort_data['customer_id'].nunique() / data['customer_id'].nunique()
    
    st.write(f"Overall Customer Retention Rate: {retention_rate:.2%}")
    st.write("### Recommendations to Boost Retention:")
    st.write("""
        1. **Personalized Marketing:** Use customer purchase history to send targeted offers.
        2. **Loyalty Programs:** Implement programs that reward repeat purchases.
        3. **Customer Feedback:** Regularly collect feedback to improve products and services.
        4. **Engagement:** Increase engagement through personalized communication and promotions.
        5. **Quality Customer Service:** Provide exceptional customer service to enhance satisfaction and loyalty.
    """)
# Page 6: Profitability Insights
def profitability_insights():
    st.title("Profitability Insights")

    # Profit vs. Sales Volume
    st.header("Profit vs. Sales Volume")
    
    # Aggregate sales and profit by product
    product_performance = data.groupby('product_name').agg({
        'sales_per_order': 'sum',
        'profit_per_order': 'sum'
    }).reset_index()

    # Correlation Analysis
    st.subheader("Correlation Analysis")
    corr = product_performance[['sales_per_order', 'profit_per_order']].corr().iloc[0, 1]
    st.write(f"Correlation between Sales and Profit: {corr:.2f}")

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='sales_per_order', y='profit_per_order', data=product_performance)
    plt.title('Profit vs. Sales Volume')
    plt.xlabel('Total Sales')
    plt.ylabel('Total Profit')
    st.pyplot()

    # Discount Impact
    st.header("Discount Impact")
    
    # Calculate the relationship between discounts and profit margins
    data['profit_margin'] = data['profit_per_order'] / data['sales_per_order']
    discount_impact = data.groupby('order_item_discount').agg({
        'profit_margin': 'mean',
        'sales_per_order': 'mean'
    }).reset_index()
    
    # Scatter Plot for Discounts vs Profit Margin
    st.subheader("Discounts vs Profit Margin")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='order_item_discount', y='profit_margin', data=discount_impact)
    plt.title('Impact of Discounts on Profit Margin')
    plt.xlabel('Discount (%)')
    plt.ylabel('Profit Margin')
    st.pyplot()

    # Scatter Plot for Discounts vs Sales
    st.subheader("Discounts vs Sales")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='order_item_discount', y='sales_per_order', data=discount_impact)
    plt.title('Impact of Discounts on Sales')
    plt.xlabel('Discount (%)')
    plt.ylabel('Sales')
    st.pyplot()

# Page 7: Future Trends and Recommendations
def future_trends_and_recommendations():
    st.title("Future Trends and Recommendations")

    # Forecasting
    st.header("Forecasting")

    # Prepare data for forecasting
    data['order_month'] = data['order_date'].dt.to_period('M').dt.to_timestamp()
    monthly_sales = data.groupby('order_month')['sales_per_order'].sum().reset_index()

    # Linear Regression Model for Forecasting
    monthly_sales['month_num'] = np.arange(len(monthly_sales))
    X = monthly_sales[['month_num']]
    y = monthly_sales['sales_per_order']
    model = LinearRegression()
    model.fit(X, y)
    
    future_months = np.arange(len(monthly_sales), len(monthly_sales) + 12).reshape(-1, 1)
    future_sales = model.predict(future_months)
    future_dates = pd.date_range(start=monthly_sales['order_month'].max() + pd.DateOffset(months=1), periods=12, freq='M')

    # Plot Forecast
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales['order_month'], monthly_sales['sales_per_order'], label='Historical Sales', marker='o')
    plt.plot(future_dates, future_sales, label='Forecasted Sales', marker='o', linestyle='--')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.title('Sales Forecast')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot()

    # Strategic Recommendations
    st.header("Strategic Recommendations")

    st.write("""
    Based on the sales forecast and recent trends, here are some strategic recommendations to consider:
    
    1. **Enhance Marketing Efforts:** Increase marketing campaigns for high-performing periods to capitalize on peak sales times.
    2. **Optimize Inventory Management:** Adjust inventory levels based on forecasted sales to reduce overstock and stockouts.
    3. **Expand Product Lines:** Consider introducing new products or expanding existing lines that have shown strong growth trends.
    4. **Improve Customer Engagement:** Implement targeted promotions and loyalty programs during forecasted high-sales periods to boost customer retention.
    5. **Focus on High-Margin Products:** Prioritize the promotion of products with higher profit margins to improve overall profitability.
    6. **Monitor and Adapt:** Continuously monitor actual sales against forecasts and adjust strategies as necessary to respond to market changes.
    """)


# Main function for Streamlit app
def main():
    # Sidebar with navigation menu
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Executive Summary", "Sales and Category Analysis", "Customer Segmentation", 
                     "Shipping and Logistics Analysis", "Retention and Cohort Analysis", 
                     "Profitability Insights", "Future Trends and Recommendations"],
            icons=["house", "bar-chart", "person", "truck", "clock", "dollar", "lightbulb"],
            default_index=0
        )
    
    if selected == "Executive Summary":
        st.title("Executive Summary")
        executive_summary()
        # Code for Executive Summary page (Page 1) goes here
        
    elif selected == "Sales and Category Analysis":
        sales_and_category_analysis()
    elif selected == "Customer Segmentation":
        # Code for Customer Segmentation page (Page 3) goes here
        #customer_segmentation()
        pass
    elif selected == "Shipping and Logistics Analysis":
        # Code for Shipping and Logistics Analysis page (Page 4) goes here
        # ...
        shipping_logistics_analysis()
    elif selected == "Retention and Cohort Analysis":
        # Code for Retention and Cohort Analysis page (Page 5) goes here
        # ...
        retention_cohort_analysis()
    elif selected == "Profitability Insights":
        # Code for Profitability Insights page (Page 6) goes here
        # ...
        profitability_insights()
    elif selected == "Future Trends and Recommendations":
        # Code for Future Trends and Recommendations page (Page 7) goes here
        # ...
        future_trends_and_recommendations()
if __name__ == "__main__":
    main()


