# 🏡 Home Energy Usage Optimizer

**Predict and optimize your home’s electricity consumption using AI!**

---

## Overview

This is a **Streamlit app** that predicts your next hour's energy usage and helps you visualize potential savings by adjusting appliance usage. The app is powered by a **Random Forest Regressor** and offers an interactive, professional dashboard with:

- Historical energy usage visualization  
- Predicted next-hour usage  
- Adjustments for lighting, heating/cooling, and appliances  
- Clear savings summary  
- Subtle animated/faded background for visual appeal

---

## Demo

You can try the app live here:  
[**Streamlit Cloud Demo Link**]https://bmk5nfduseai9apcngmrec.streamlit.app/

---

## Features

- Upload your own electricity usage CSV (`timestamp`, `usage_kWh`) or use the demo data  
- Interactive sliders for appliance usage reduction  
- Dynamic plotly graph with predictive markers and subtle background fade  
- Summary of energy and estimated cost savings  
- Professional, clean UI design suitable for portfolio showcase  

---

## Sample Data

Your CSV should have the following format:

| timestamp           | usage_kWh |
|--------------------|-----------|
| 2026-01-01 00:00:00 | 2.15      |
| 2026-01-01 01:00:00 | 2.30      |
| 2026-01-01 02:00:00 | 2.10      |

- `timestamp` in `YYYY-MM-DD HH:MM:SS` format  
- `usage_kWh` as float representing energy consumed  

A **sample CSV** is included: `sample_data.csv`.

---

## Installation & Running Locally

1. Clone the repo:

```bash
git clone https://github.com/YOUR_USERNAME/your-repo.git
cd your-repo
# home-energy-optimizer
