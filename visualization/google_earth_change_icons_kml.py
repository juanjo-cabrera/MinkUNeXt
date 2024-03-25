
import csv
import simplekml
import pandas as pd
import numpy as np
from pyproj import Proj

def create_kml_from_csv(csv_file, output_kml):
    kml = simplekml.Kml()

    # Define the URL for the circle icon
    circle_icon_url = 'http://maps.google.com/mapfiles/kml/pal4/icon49.png'
    # circle_icon_url = 'http://maps.google.com/mapfiles/kml/paddle/blu-circle-lv.png'
    # circle_icon_url = 'http://maps.google.com/mapfiles/kml/paddle/ltblu-circle.png' #53
    # circle_icon_url = 'http://maps.google.com/mapfiles/kml/paddle/red-circle.png'
    # circle_icon_url = 'http://maps.google.com/mapfiles/kml/paddle/blu-circle-lv.png'

    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            latitude = float(row['latitude'])
            longitude = float(row['longitude'])
            name = row.get('Name', '')  # Assuming the CSV file has a 'Name' column
            description = row.get('Description', '')  # Assuming the CSV file has a 'Description' column

            # Create a placemark for each coordinate with a circle icon
            pnt = kml.newpoint(name=name, coords=[(longitude, latitude)], description=description)
            pnt.style.iconstyle.icon.href = circle_icon_url
            pnt.style.iconstyle.scale = 0.8

            # Save the KML file
    kml.save(output_kml)

def easting_northing_to_lat_lon(easting, northing, zone_number, northern_hemisphere=True):
    if northern_hemisphere:
        proj_str = "+proj=utm +zone={} +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(zone_number)
    else:
        proj_str = "+proj=utm +zone={} +ellps=WGS84 +datum=WGS84 +units=m +no_defs +south".format(zone_number)

    # Define the UTM projection
    utm_proj = Proj(proj_str)

    # Convert from easting/northing to latitude/longitude
    lon, lat = utm_proj(easting, northing, inverse=True)



initial_zoom = 16.99999
zone_number = 30  # UTM zone for Oxford
directory = '/home/arvc/Juanjo/Datasets/benchmark_datasets/oxford/2015-03-10-14-18-10/pointcloud_locations_20m.csv'
df = pd.read_csv(directory)
northing = df['northing'].values.tolist()
easting = df['easting'].values.tolist()
timestamp = df['timestamp'].values.tolist()

lats = []
lons = []
for i in range(0, len(df)):
    lat, lon = easting_northing_to_lat_lon(easting[i], northing[i], zone_number)
    lats.append(lat)
    lons.append(lon)

lats = np.array(lats)
lons = np.array(lons)

import csv

with open('/home/arvc/Desktop/locations.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["latitud", "longitud"]
    writer.writerow(field)
    for i in range(0, len(lats)):
        writer.writerow([lats[i], lons[i]])
# Example usage:
csv_file = '/home/arvc/Desktop/locations.csv'  # Path to your CSV file
# csv_file = '/home/arvc/Desktop/locations_test.csv'  # Path to your CSV file
output_kml = '/home/arvc/Desktop/output.kml'  # Path for the output KML file
# output_kml = '/home/arvc/Desktop/output_test.kml'  # Path for the output KML file
create_kml_from_csv(csv_file, output_kml)