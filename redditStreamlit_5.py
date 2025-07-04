import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

st.set_page_config(layout="wide")
st.title("🧵 AI-Driven Fashion Trend Forecasting for SMEs")
st.write("📈 Real-time insights from Google Trends to inform your fashion retail strategy")

# === Folder Config ===
data_root = "GoogleTrends"
if not os.path.exists(data_root):
    st.error(f"❌ Folder '{data_root}' not found.")
    st.stop()

# === Product to Category Mapping ===
product_to_category = {
    "CheckSuit": "Suits", "BlackTuxedo": "Suits",
    "WoolOvercoat": "Coats", "TrenchCoat": "Coats", "PufferJacket": "Coats",
    "LinenBlazer": "Blazers", "VelvetBlazer": "Blazers",
    "OversizedGraphicTee": "T-shirts", "CrewNeckTee": "T-shirts",
    "AviatorSunglasses": "Accessories", "WoolScarf": "Accessories",
    "CrewNeckSweatshirt": "Sweatshirts", "OversizedSweatshirt": "Sweatshirts",
    "Cardigans": "Knitwear", "Pullovers": "Knitwear",
    "DenimShirt": "Shirts",
    "CargoJoggers": "Pants", "Palazzo": "Pants", "BlackJeans": "Pants",
    "FloralMaxiDress": "Dress", "SatinDress": "Dress",
    "GymLeggings": "Sportswear", "SportsBra": "Sportswear",
    "BabyBodysuit": "Kidswear",
    "WoolSweater": "Sweaters", "FuzzySweater": "Sweaters",
    "CropTop": "Tops", "OffShoulderTop": "Tops",
    "CottonPolo": "Polos",
    "Jumpsuit": "Jumpsuits"
}

city_aliases = {
    "UK": ["London", "Birmingham"],
    "Spain": ["Madrid", "Barcelona"]
}

@st.cache_data
def load_all_data():
    all_data = []

    for folder in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path, skiprows=2)
                df.columns = ["Date", "Mentions"]
                df["Date"] = pd.to_datetime(df["Date"])

                name_parts = file.replace(".csv", "").split("_")
                product = name_parts[0]
                location_tag = name_parts[1]

                df["Product"] = product
                df["Category"] = product_to_category.get(product, "Unknown")

                if location_tag in city_aliases:
                    for city in city_aliases[location_tag]:
                        temp_df = df.copy()
                        temp_df["Location"] = city
                        all_data.append(temp_df)
                else:
                    df["Location"] = location_tag
                    all_data.append(df)

            except Exception as e:
                st.warning(f"⚠️ Error reading {file_path}: {e}")

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Load Data
with st.spinner("📥 Loading Google Trends data..."):
    df = load_all_data()

if df.empty:
    st.warning("⚠️ No data available.")
    st.stop()

# Sidebar Filters
st.sidebar.header("🔍 Filters")
unique_products = sorted(df["Product"].unique())
unique_categories = sorted(df["Category"].unique())
unique_locations = sorted(df["Location"].unique())

selected_products = st.sidebar.multiselect("Select Products", unique_products, default=unique_products[:5])
selected_categories = st.sidebar.multiselect("Select Categories", unique_categories, default=unique_categories)
selected_locations = st.sidebar.multiselect("Select Locations", unique_locations, default=unique_locations[:5])

date_range = st.sidebar.slider(
    "Select Date Range:",
    min_value=df["Date"].min().date(),
    max_value=df["Date"].max().date(),
    value=(df["Date"].min().date(), df["Date"].max().date())
)

# Filter dataframe based on selections
df_filtered = df[
    df["Product"].isin(selected_products) &
    df["Category"].isin(selected_categories) &
    df["Location"].isin(selected_locations) &
    df["Date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
]

if df_filtered.empty:
    st.warning("⚠️ No data matches the selected filters.")
    st.stop()

# --- KPI Calculations ---

# Aggregate mentions per product over first and last date (for % change)
first_date = df_filtered["Date"].min()
last_date = df_filtered["Date"].max()

agg_start = df_filtered[df_filtered["Date"] == first_date].groupby("Product")["Mentions"].sum()
agg_end = df_filtered[df_filtered["Date"] == last_date].groupby("Product")["Mentions"].sum()

growth_df = pd.DataFrame({"Start": agg_start, "End": agg_end}).fillna(0)

# Avoid divide by zero and extreme % changes
def capped_growth(row):
    start = row["Start"]
    end = row["End"]
    if start < 20:
        return 0  # Ignore very low base counts to avoid huge misleading %
    if start == 0:
        return 100.0 if end > 0 else 0
    change = ((end - start) / start) * 100
    # Cap between -1000% and +1000%
    if change > 1000:
        change = 1000
    elif change < -1000:
        change = -1000
    return change

growth_df["Growth %"] = growth_df.apply(capped_growth, axis=1)

# Top growing and declining products (exclude 0 growth)
growth_nonzero = growth_df[growth_df["Growth %"] != 0]

if not growth_nonzero.empty:
    top_growing_product = growth_nonzero["Growth %"].idxmax()
    top_growing_value = growth_nonzero.loc[top_growing_product, "Growth %"]

    top_declining_product = growth_nonzero["Growth %"].idxmin()
    top_declining_value = growth_nonzero.loc[top_declining_product, "Growth %"]
else:
    top_growing_product = None
    top_growing_value = None
    top_declining_product = None
    top_declining_value = None

# --- KPIs Display ---

st.subheader("📌 Trend Overview")
col1, col2, col3, col4 = st.columns(4)

total_mentions = int(df_filtered["Mentions"].sum())
col1.metric("Total Mentions", total_mentions)

def styled_metric(label, value, growth, is_growth=True):
    # Show ↑ for growth, ↓ for decline
    arrow = "↑" if growth > 0 else "↓"
    color = "green" if (growth > 0 and is_growth) or (growth < 0 and not is_growth) else "red"
    formatted_value = f"{abs(growth):.1f}% {arrow}"
    col = st.columns(1)[0]
    col.markdown(f"**{label}**")
    col.markdown(f"<h2 style='color:{color}; margin:0'>{value} {formatted_value}</h2>", unsafe_allow_html=True)

# For top growing product
with col2:
    if top_growing_product is not None:
        arrow = "↑"
        color = "green"
        val_str = f"{abs(top_growing_value):.1f}% {arrow}"
        st.markdown(f"**Top Growing Product**")
        st.markdown(f"<h2 style='color:{color}; margin:0'>{top_growing_product} ({val_str})</h2>", unsafe_allow_html=True)
    else:
        st.markdown("**Top Growing Product**")
        st.write("N/A")

# For top declining product
with col3:
    if top_declining_product is not None:
        arrow = "↓"
        color = "red"
        val_str = f"{abs(top_declining_value):.1f}% {arrow}"
        st.markdown(f"**Top Declining Product**")
        st.markdown(f"<h2 style='color:{color}; margin:0'>{top_declining_product} ({val_str})</h2>", unsafe_allow_html=True)
    else:
        st.markdown("**Top Declining Product**")
        st.write("N/A")

# Top Location by total mentions
top_location = df_filtered.groupby("Location")["Mentions"].sum().idxmax()
col4.metric("Top Location", top_location)

# --- Product popularity distribution across cities ---
st.subheader("🏙️ Product Popularity Distribution Across Cities")

pop_dist = df_filtered.groupby(["Location", "Product"])["Mentions"].sum().unstack(fill_value=0)

plt.figure(figsize=(14, 6))
sns.heatmap(pop_dist, annot=True, fmt="d", cmap="Blues")
plt.title("Total Mentions of Products by City")
plt.xlabel("Product")
plt.ylabel("City")
plt.xticks(rotation=45)
st.pyplot(plt)

# --- Category share by city ---
st.subheader("📦 Category Share by City")

cat_city = df_filtered.groupby(["Location", "Category"])["Mentions"].sum().unstack(fill_value=0)
cat_city_pct = cat_city.div(cat_city.sum(axis=1), axis=0) * 100

st.dataframe(cat_city_pct.style.format("{:.1f}%"))

# --- Top 3 Products in Each City ---

st.subheader("🏆 Top 3 Products in Each City")

top3_per_city = (
    df_filtered.groupby(["Location", "Product"])["Mentions"]
    .sum()
    .reset_index()
)

# For each city get top 3 products
def format_top3(city_df):
    city = city_df.name
    df_sorted = city_df.sort_values("Mentions", ascending=False).head(3)
    result = f"**{city}**\n"
    for i, row in enumerate(df_sorted.itertuples(), 1):
        result += f"{i}. {row.Product} — {int(row.Mentions)} mentions\n"
    return result

top3_text = top3_per_city.groupby("Location").apply(format_top3)

for city, text in top3_text.items():
    st.markdown(text)


# --- Weekly Trend Line by Product ---

st.subheader("📈 Weekly Trend Line by Product")

weekly_kw = (
    df_filtered.groupby([pd.Grouper(key="Date", freq="W"), "Product"])["Mentions"]
    .sum()
    .reset_index()
    .pivot(index="Date", columns="Product", values="Mentions")
    .fillna(0)
)

plt.figure(figsize=(12, 5))
for kw in weekly_kw.columns:
    plt.plot(weekly_kw.index, weekly_kw[kw], label=kw, marker='o')
plt.legend()
plt.title("Weekly Google Search Trends")
plt.xlabel("Week")
plt.ylabel("Mentions")
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(plt)
