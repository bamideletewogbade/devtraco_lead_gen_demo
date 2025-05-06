# import pandas as pd
# import numpy as np

# # Set random seed for reproducibility
# np.random.seed(42)

# # DevTraco-specific parameters
# num_leads = 100  # Larger dataset for better demo
# online_channels = ["GhanaNewsOnline", "GoogleAds", "Facebook", "PropertyWebsite"]
# offline_channels = ["MallActivation", "AirportSetup", "SalesOffice", "Event"]
# all_channels = online_channels + offline_channels

# # Ghana-specific real estate price ranges (in GHS)
# budget_ranges = {
#     "low": (200000, 800000),    # Affordable housing
#     "medium": (800000, 3000000), # Middle-income
#     "high": (3000000, 10000000)  # Premium/luxury
# }

# # Generate mock data
# data = {
#     "lead_id": range(1, num_leads + 1),
#     "source": np.random.choice(all_channels, num_leads, p=[0.2, 0.15, 0.15, 0.1, 0.15, 0.1, 0.1, 0.05]),
#     "budget": np.concatenate([
#         np.random.randint(*budget_ranges["low"], size=int(num_leads*0.4)),
#         np.random.randint(*budget_ranges["medium"], size=int(num_leads*0.5)),
#         np.random.randint(*budget_ranges["high"], size=int(num_leads*0.1))
#     ]),
#     "time_on_website": np.where(
#         np.random.random(num_leads) > 0.3,  # 70% online leads have website time
#         np.random.exponential(300, num_leads).astype(int),
#         0
#     ),
#     "property_type": np.random.choice(
#         ["Studio", "2-Bedroom", "3-Bedroom", "Penthouse", "Commercial"], 
#         num_leads,
#         p=[0.2, 0.4, 0.3, 0.05, 0.05]
#     ),
#     "location_pref": np.random.choice(
#         ["Accra", "Kumasi", "Takoradi", "Tema", "East Legon"], 
#         num_leads
#     ),
#     "contacted": np.random.choice(["Yes", "No"], num_leads, p=[0.6, 0.4]),
#     "converted": np.random.choice(["Yes", "No"], num_leads, p=[0.25, 0.75])
# }

# # Shuffle and create DataFrame
# df = pd.DataFrame(data)
# df = df.sample(frac=1).reset_index(drop=True)

# # Add offline-specific features
# df["offline_activity"] = np.where(
#     df["source"].isin(offline_channels),
#     np.random.choice(["BrochureTaken", "Consultation", "SiteVisit", "None"], size=num_leads),
#     "N/A"
# )

# # Save to CSV
# df.to_csv("devtraco_leads.csv", index=False)
# print(f"Generated {len(df)} leads with Ghana-specific channels:")
# print(df["source"].value_counts())

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# DevTraco-specific parameters
num_leads = 100  # Larger dataset for better demo
online_channels = ["GhanaNewsOnline", "GoogleAds", "Facebook", "PropertyWebsite"]
offline_channels = ["MallActivation", "AirportActivation", "SalesOffice", "Event"]
all_channels = online_channels + offline_channels

# Consultant names (using common Ghanaian names)
consultants = ["Seun", "Tyler", "David", "Tola", "Tope", "Paul", "Abena", "Jackie"]

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
    "consultant": np.random.choice(consultants, num_leads),
    "contacted": np.random.choice(["Yes", "No"], num_leads, p=[0.6, 0.4]),
}

# Create the DataFrame
df = pd.DataFrame(data)

# Create conversion rates that vary by consultant and channel to show patterns
# This helps demonstrate consultant-channel effectiveness
conversion_probabilities = np.zeros(num_leads)

# Base conversion rate of 25%
base_rate = 0.25

# Set consultant-specific effectiveness (some consultants are better than others)
consultant_factors = {
    "Seun": 1.5,  # Kwame is great overall
    "Tola": 1.3,    # Ama is good overall
    "Tope": 0.9,   # Kofi is below average
    "Abena": 1.2,  # Abena is good
    "David": 1.0,   # Kojo is average
    "Paul": 1.1, # Akosua is slightly above average
    "Tyler": 0.8,    # Yaw is below average
    "Jackie": 1.4     # Esi is very good
}

# Set channel-specific effectiveness
channel_factors = {
    "GhanaNewsOnline": 1.2,
    "GoogleAds": 1.3,
    "Facebook": 1.1,
    "PropertyWebsite": 1.4,
    "MallActivation": 1.0,
    "AirportActivation": 0.9,
    "SalesOffice": 1.2,
    "Event": 1.1
}

# Some consultants have specialties with certain channels
consultant_channel_specialties = {
    ("Jackie", "GoogleAds"): 1.7,
    ("Tyler", "Facebook"): 1.8,
    ("Bamidele", "PropertyWebsite"): 1.6,
    ("Tola", "MallActivation"): 1.9,
    ("Jackie", "SalesOffice"): 1.5,
    ("David", "GhanaNewsOnline"): 1.6
}

# Calculate conversion probabilities
for i in range(num_leads):
    consultant = df.loc[i, "consultant"]
    channel = df.loc[i, "source"]
    
    # Start with base conversion rate
    prob = base_rate
    
    # Apply consultant factor
    prob *= consultant_factors[consultant]
    
    # Apply channel factor
    prob *= channel_factors[channel]
    
    # Apply specialty bonus if applicable
    if (consultant, channel) in consultant_channel_specialties:
        prob *= consultant_channel_specialties[(consultant, channel)]
    
    # Budget factor - higher budgets are harder to convert
    if df.loc[i, "budget"] > budget_ranges["medium"][0]:
        prob *= 0.8
    if df.loc[i, "budget"] > budget_ranges["high"][0]:
        prob *= 0.7
    
    # Website time factor - more time = more interested
    if df.loc[i, "time_on_website"] > 300:
        prob *= 1.2
        
    # Cap probability at 95%
    prob = min(prob, 0.95)
    
    conversion_probabilities[i] = prob

# Determine if converted based on calculated probabilities
df["converted"] = np.random.random(num_leads) < conversion_probabilities
df["converted"] = df["converted"].map({True: "Yes", False: "No"})

# Add column for conversion probability (useful for model evaluation)
df["conversion_probability"] = conversion_probabilities

# Add offline-specific features
df["offline_activity"] = np.where(
    df["source"].isin(offline_channels),
    np.random.choice(["BrochureTaken", "Consultation", "SiteVisit", "None"], size=num_leads),
    "N/A"
)

# Shuffle and reset index
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
df.to_csv("devtraco_leads2.csv", index=False)
print(f"Generated {len(df)} leads with Ghana-specific channels:")
print(df["source"].value_counts())
print("\nConsultant distribution:")
print(df["consultant"].value_counts())
print("\nOverall conversion rate: {:.2%}".format(df["converted"].value_counts(normalize=True)["Yes"]))
print("\nConversion rates by consultant:")
print(df.groupby("consultant")["converted"].apply(lambda x: (x == "Yes").mean()).sort_values(ascending=False))
print("\nConversion rates by channel:")
print(df.groupby("source")["converted"].apply(lambda x: (x == "Yes").mean()).sort_values(ascending=False))