import random
from datetime import datetime, timedelta
from pathlib import Path

output_path = Path(__file__).resolve().with_name("electric_load_profile_hour.csv")

with output_path.open('a') as f:
    start = datetime(2024, 1, 1)
    for i in range(8760):  # ore in un anno
        t = start + timedelta(minutes=i)
        hour = t.hour
        month = t.month
        # Base load: low at night, higher during day
        base = random.gauss(300, 50) if hour < 6 or hour > 22 else random.gauss(1200, 200)
        # Morning/evening peaks
        if 6 <= hour < 9 or 18 <= hour < 22:
            base += random.gauss(1000, 300)
        # Winter HVAC boost
        if month in [12, 1, 2]:
            base += random.gauss(500, 100)
        # Afternoon HVAC boost
        if 12 <= hour < 16:
            base += random.gauss(500, 200)
        base = max(0, min(base, 4000))
        f.write(f'{base:.2f}\n')
