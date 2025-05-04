import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# DevTraco-specific parameters
num_leads = 100  # Larger dataset for better demo
online_channels = ["GhanaNewsOnline", "GoogleAds", "Facebook", "PropertyWebsite"]
offline_channels = ["MallActivation", "AirportSetup", "SalesOffice", "Event"]
all_channels = online_channels + offline_channels

# Ghana-specific real estate price ranges (in GHS)
budget_ranges = {
    "low": (200000, 800000),    # Affordable housing
    "medium": (800000, 3000000), # Middle-income
    "high": (3000000, 10000000)  # Premium/luxury
}

# Generate mock data
data = {
    "lead_id": range(1, num_leads + 1),
    "source": np.random.choice(all_channels, num_leads, p=[0.2, 0.15, 0.15, 0.1, 0.15, 0.1, 0.1, 0.05]),
    "budget": np.concatenate([
        np.random.randint(*budget_ranges["low"], size=int(num_leads*0.4)),
        np.random.randint(*budget_ranges["medium"], size=int(num_leads*0.5)),
        np.random.randint(*budget_ranges["high"], size=int(num_leads*0.1))
    ]),
    "time_on_website": np.where(
        np.random.random(num_leads) > 0.3,  # 70% online leads have website time
        np.random.exponential(300, num_leads).astype(int),
        0
    ),
    "property_type": np.random.choice(
        ["Studio", "2-Bedroom", "3-Bedroom", "Penthouse", "Commercial"], 
        num_leads,
        p=[0.2, 0.4, 0.3, 0.05, 0.05]
    ),
    "location_pref": np.random.choice(
        ["Accra", "Kumasi", "Takoradi", "Tema", "East Legon"], 
        num_leads
    ),
    "contacted": np.random.choice(["Yes", "No"], num_leads, p=[0.6, 0.4]),
    "converted": np.random.choice(["Yes", "No"], num_leads, p=[0.25, 0.75])
}

# Shuffle and create DataFrame
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)

# Add offline-specific features
df["offline_activity"] = np.where(
    df["source"].isin(offline_channels),
    np.random.choice(["BrochureTaken", "Consultation", "SiteVisit", "None"], size=num_leads),
    "N/A"
)

# Save to CSV
df.to_csv("devtraco_leads.csv", index=False)
print(f"Generated {len(df)} leads with Ghana-specific channels:")
print(df["source"].value_counts())