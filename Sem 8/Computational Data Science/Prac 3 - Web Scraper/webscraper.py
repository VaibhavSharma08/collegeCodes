import requests
from bs4 import BeautifulSoup

page = requests.get("https://forecast.weather.gov/MapClick.php?lat=37.7772&lon=-122.4168")
soup = BeautifulSoup(page.content, 'html.parser')
seven_day = soup.find(id="seven-day-forecast")
forecast_items = seven_day.find_all(class_="tombstone-container")
tonight = forecast_items[0]
print(tonight.prettify())
period = tonight.find(class_="period-name").get_text()
short_desc = tonight.find(class_="short-desc").get_text()
temp = tonight.find(class_="temp").get_text()
print("Period")
print(period)
print("Short Desc")
print(short_desc)
print("Temp")
print(temp)
img = tonight.find("img")
desc = img['title']
print("Desc")
print(desc)
period_tags = seven_day.select(".tombstone-container .period-name")
periods = [pt.get_text() for pt in period_tags]
print("Periods")
print(periods)
short_descs = [sd.get_text() for sd in seven_day.select(".tombstone-container .short-desc")]
temps = [t.get_text() for t in seven_day.select(".tombstone-container .temp")]
descs = [d["title"] for d in seven_day.select(".tombstone-container img")]
print("short descs")
print(short_descs)
print("temps")
print(temps)
print("descs")
print(descs)
import pandas as pd
weather = pd.DataFrame({
    "period": periods,
    "short_desc": short_descs,
    "temp": temps,
    "desc":descs
})
print("Weather")
print(weather)
print(weather["temp"])

# temp_nums = weather["temp"].str.extract("(?Pd+)", expand=False)
temp_nums = weather["temp"].map(lambda x:x[-5:-3])
print(temp_nums)
weather["temp_num"] = temp_nums.astype('int')
print("Temp NUms")
print(temp_nums)

print(weather["temp_num"].mean())
is_night = weather["temp"].str.contains("Low")
weather["is_night"] = is_night
print("Is night")
print(is_night)
print("weather is night")
print(weather[is_night])
weather.to_csv('/Users/vaibhavs/PycharmProjects/collegeCodes/Sem 8/Computational Data Science/Prac 3/scrapedData.csv')

