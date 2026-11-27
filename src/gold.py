"""Gold stage: analysis-ready tables (full rebuild from silver + bronze every run => idempotent).

gold.camera_hourly   grain: camera_id x hour_start
    traffic  avg_vehicles / avg_car / avg_motorcycle / avg_bus / avg_truck / avg_box_area / congestion_index
    quality  n_images (used), n_frozen_excluded, dark_share, light_group
    time     service_date, hour_of_day, dow (0=Sun), is_weekend
    weather  precipitation_mm, rain_mm, is_rainy (rain_mm >= 1), temperature_c, humidity_pct, cloud_cover_pct
    wetness  wet_score (mean CLIP p_wet of the hour's images), wet_share (share of images with p_wet >= 0.5);
             image-based, only meaningful in daytime (see silver_wetness.py)
    air      pm2_5, pm10
    place    n_school ... n_traffic_signal within POI_RADIUS_M (NULL when the camera has no coordinates)
gold.camera_profile  grain: camera_id (static facts + day-time traffic summary; used for the map and K-Means)
gold.pipeline_audit  row counts at every stage (bronze -> rejected -> frozen -> detected -> gold) for the dashboard

congestion_index (per image) = n_vehicles / p95(n_vehicles) of the SAME camera AND SAME light group
(day vs dark = twilight + night), because YOLO under-counts after dusk and cameras are not comparable in raw counts.
The hourly value is the mean over the hour's images. Frozen-frame images are excluded from every average.
"""
from __future__ import annotations

from datetime import datetime

from common import BKK, open_catalog
from scrape_environment import GRID

GRID_VALUES = ", ".join(f"('{gid}', {la}, {lo})" for gid, la, lo in GRID)
POI_RADIUS_M = 300
RAIN_THRESHOLD_MM = 1.0

CAMERA_HOURLY_SQL = f"""
CREATE OR REPLACE TABLE bkk.gold.camera_hourly AS
WITH obs AS (
    SELECT d.image_id, f.camera_id, f.captured_at,
           l.is_dark, CASE WHEN l.is_dark THEN 'dark' ELSE 'day' END AS light_group,
           d.n_vehicles, d.n_car, d.n_motorcycle, d.n_bus, d.n_truck, d.box_area_frac, w.p_wet
    FROM bkk.silver.detections d
    JOIN bkk.silver.image_features f USING (image_id)
    JOIN bkk.silver.light_regime l USING (image_id)
    JOIN bkk.silver.wetness w USING (image_id)
    JOIN bkk.silver.image_scan sc USING (image_id)
    WHERE NOT f.is_frozen AND sc.reason IS NULL   -- frozen frames and glitch / test-card / unreadable images are left out
),
p95 AS (
    SELECT camera_id, light_group, quantile_cont(n_vehicles, 0.95) AS p95_vehicles
    FROM obs GROUP BY 1, 2
),
scored AS (
    SELECT o.*, CASE WHEN p.p95_vehicles > 0 THEN o.n_vehicles / p.p95_vehicles END AS ci
    FROM obs o JOIN p95 p USING (camera_id, light_group)
),
hourly AS (
    SELECT camera_id,
           date_trunc('hour', captured_at) AS hour_start,
           count(*) AS n_images,
           avg(is_dark::int) AS dark_share,
           avg(n_vehicles) AS avg_vehicles, avg(n_car) AS avg_car, avg(n_motorcycle) AS avg_motorcycle,
           avg(n_bus) AS avg_bus, avg(n_truck) AS avg_truck, avg(box_area_frac) AS avg_box_area,
           avg(ci) AS congestion_index,
           avg(p_wet) AS wet_score, avg((p_wet >= 0.5)::int) AS wet_share
    FROM scored GROUP BY 1, 2
),
frozen AS (
    SELECT f.camera_id, date_trunc('hour', f.captured_at) AS hour_start, count(*) AS n_frozen_excluded
    FROM bkk.silver.image_features f WHERE f.is_frozen GROUP BY 1, 2
),
cams AS (
    SELECT camera_id, lat, lng FROM bkk.bronze.camera_locations
),
-- display name comes from the image file names (consistent); BMA's name field mixes in direction text
names AS (
    SELECT camera_id, any_value(camera_name) AS camera_name FROM bkk.bronze.cctv_images GROUP BY 1
),
-- nearest weather / air grid point per camera (NULL coordinates -> 'ALL' = mean over the grid)
-- Use the REQUESTED grid coordinates, not the snapped ones the APIs echo back: the air-quality model returns
-- coarse cells (~0.4 deg) so several requested points share identical echoed coordinates (ties -> unstable joins).
wgrid AS (SELECT * FROM (VALUES {GRID_VALUES}) t(grid_id, grid_lat, grid_lng)),
agrid AS (SELECT * FROM (VALUES {GRID_VALUES}) t(grid_id, grid_lat, grid_lng)),
cam_w AS (
    SELECT camera_id, coalesce(max(grid_id) FILTER (WHERE rn = 1), 'ALL') AS grid_id FROM (
        SELECT c.camera_id, g.grid_id,
               row_number() OVER (PARTITION BY c.camera_id
                   ORDER BY power(g.grid_lat - c.lat, 2) + power(g.grid_lng - c.lng, 2), g.grid_id) AS rn
        FROM cams c LEFT JOIN wgrid g ON c.lat IS NOT NULL) GROUP BY 1
),
cam_a AS (
    SELECT camera_id, coalesce(max(grid_id) FILTER (WHERE rn = 1), 'ALL') AS grid_id FROM (
        SELECT c.camera_id, g.grid_id,
               row_number() OVER (PARTITION BY c.camera_id
                   ORDER BY power(g.grid_lat - c.lat, 2) + power(g.grid_lng - c.lng, 2), g.grid_id) AS rn
        FROM cams c LEFT JOIN agrid g ON c.lat IS NOT NULL) GROUP BY 1
),
wx AS (
    SELECT grid_id, ts, precipitation_mm, rain_mm, temperature_c, humidity_pct, cloud_cover_pct
    FROM bkk.bronze.weather_hourly
    UNION ALL
    SELECT 'ALL', ts, avg(precipitation_mm), avg(rain_mm), avg(temperature_c), avg(humidity_pct), avg(cloud_cover_pct)
    FROM bkk.bronze.weather_hourly GROUP BY ts
),
aq AS (
    SELECT grid_id, ts, pm2_5, pm10 FROM bkk.bronze.air_quality_hourly
    UNION ALL
    SELECT 'ALL', ts, avg(pm2_5), avg(pm10) FROM bkk.bronze.air_quality_hourly GROUP BY ts
),
poi AS (
    SELECT c.camera_id,
           count(*) FILTER (WHERE p.category = 'school') AS n_school,
           count(*) FILTER (WHERE p.category = 'university') AS n_university,
           count(*) FILTER (WHERE p.category = 'hospital') AS n_hospital,
           count(*) FILTER (WHERE p.category = 'mall') AS n_mall,
           count(*) FILTER (WHERE p.category = 'market') AS n_market,
           count(*) FILTER (WHERE p.category = 'bus_stop') AS n_bus_stop,
           count(*) FILTER (WHERE p.category = 'rail_station') AS n_rail_station,
           count(*) FILTER (WHERE p.category = 'traffic_signal') AS n_traffic_signal
    FROM cams c
    LEFT JOIN bkk.bronze.osm_poi p
      ON sqrt(power((p.lat - c.lat) * 111000, 2) + power((p.lng - c.lng) * 108000, 2)) <= {POI_RADIUS_M}
    WHERE c.lat IS NOT NULL
    GROUP BY 1
)
SELECT h.camera_id, nm.camera_name AS name_th, cl.camera_code, cl.lat, cl.lng, cl.coord_source,
       h.hour_start,
       -- NB: CAST(timestamptz AS DATE) returns a wrong year in DuckLake here; build the date from text instead
       CAST(strftime(h.hour_start, '%Y-%m-%d') AS DATE) AS service_date,
       hour(h.hour_start) AS hour_of_day,
       dayofweek(h.hour_start) AS dow,
       dayofweek(h.hour_start) IN (0, 6) AS is_weekend,
       h.n_images, coalesce(fz.n_frozen_excluded, 0) AS n_frozen_excluded,
       h.dark_share, CASE WHEN h.dark_share >= 0.5 THEN 'dark' ELSE 'day' END AS light_group,
       h.avg_vehicles, h.avg_car, h.avg_motorcycle, h.avg_bus, h.avg_truck, h.avg_box_area, h.congestion_index,
       h.wet_score, h.wet_share,
       wx.precipitation_mm, wx.rain_mm, wx.rain_mm >= {RAIN_THRESHOLD_MM} AS is_rainy,
       wx.temperature_c, wx.humidity_pct, wx.cloud_cover_pct,
       aq.pm2_5, aq.pm10,
       poi.n_school, poi.n_university, poi.n_hospital, poi.n_mall, poi.n_market,
       poi.n_bus_stop, poi.n_rail_station, poi.n_traffic_signal
FROM hourly h
JOIN bkk.bronze.camera_locations cl USING (camera_id)
JOIN names nm USING (camera_id)
LEFT JOIN frozen fz USING (camera_id, hour_start)
JOIN cam_w USING (camera_id)
JOIN cam_a ca ON ca.camera_id = h.camera_id
LEFT JOIN wx ON wx.grid_id = cam_w.grid_id AND wx.ts = h.hour_start
LEFT JOIN aq ON aq.grid_id = ca.grid_id AND aq.ts = h.hour_start
LEFT JOIN poi ON poi.camera_id = h.camera_id
"""

CAMERA_PROFILE_SQL = """
CREATE OR REPLACE TABLE bkk.gold.camera_profile AS
WITH day_h AS (SELECT * FROM bkk.gold.camera_hourly WHERE light_group = 'day'),
peak AS (
    SELECT camera_id, arg_max(hour_of_day, v) AS peak_hour FROM (
        SELECT camera_id, hour_of_day, avg(avg_vehicles) AS v FROM day_h GROUP BY 1, 2) GROUP BY 1
)
SELECT h.camera_id, any_value(h.name_th) AS name_th, any_value(h.camera_code) AS camera_code, any_value(h.lat) AS lat, any_value(h.lng) AS lng,
       any_value(h.coord_source) AS coord_source,
       count(*) AS n_camera_hours, sum(h.n_images) AS n_images,
       avg(h.avg_vehicles) FILTER (WHERE h.light_group = 'day') AS avg_vehicles_day,
       avg(h.avg_vehicles) FILTER (WHERE h.light_group = 'day' AND NOT h.is_weekend) AS avg_vehicles_weekday,
       avg(h.avg_vehicles) FILTER (WHERE h.light_group = 'day' AND h.is_weekend) AS avg_vehicles_weekend,
       avg(h.congestion_index) FILTER (WHERE h.light_group = 'day') AS avg_ci_day,
       avg(h.avg_motorcycle) FILTER (WHERE h.light_group = 'day') AS avg_motorcycle_day,
       pk.peak_hour AS peak_hour_day,
       any_value(h.n_school) AS n_school, any_value(h.n_university) AS n_university,
       any_value(h.n_hospital) AS n_hospital, any_value(h.n_mall) AS n_mall,
       any_value(h.n_market) AS n_market, any_value(h.n_bus_stop) AS n_bus_stop,
       any_value(h.n_rail_station) AS n_rail_station, any_value(h.n_traffic_signal) AS n_traffic_signal
FROM bkk.gold.camera_hourly h LEFT JOIN peak pk USING (camera_id)
GROUP BY h.camera_id, pk.peak_hour
"""

AUDIT_SQL = """
CREATE OR REPLACE TABLE bkk.gold.pipeline_audit AS
SELECT * FROM (
    SELECT 1 AS step, 'bronze.cctv_images' AS stage, count(*) AS n_rows, 'images received (from file names)' AS note FROM bkk.bronze.cctv_images
    UNION ALL SELECT 2, 'silver.rejected_images', count(*), 'blank / unreadable images, not sent to YOLO' FROM bkk.silver.rejected_images
    UNION ALL SELECT 3, 'silver.image_features', count(*), 'images that passed quality checks' FROM bkk.silver.image_features
    UNION ALL SELECT 4, 'silver.image_features (frozen)', count(*) FILTER (WHERE is_frozen), 'near-identical to previous frame, excluded from gold averages' FROM bkk.silver.image_features
    UNION ALL SELECT 5, 'silver.detections', count(*), 'YOLOv8 vehicle counts' FROM bkk.silver.detections
    UNION ALL SELECT 6, 'silver.light_regime (dark)', count(*) FILTER (WHERE is_dark), 'twilight/night images (YOLO under-counts)' FROM bkk.silver.light_regime
    UNION ALL SELECT 7, 'bronze.camera_locations (with coords)', count(*) FILTER (WHERE lat IS NOT NULL), 'cameras with a map position' FROM bkk.bronze.camera_locations
    UNION ALL SELECT 8, 'bronze.weather_hourly', count(*), 'Open-Meteo archive, 9 grid points' FROM bkk.bronze.weather_hourly
    UNION ALL SELECT 9, 'bronze.air_quality_hourly', count(*), 'Open-Meteo air quality (CAMS model)' FROM bkk.bronze.air_quality_hourly
    UNION ALL SELECT 10, 'bronze.osm_poi', count(*), 'OpenStreetMap POIs' FROM bkk.bronze.osm_poi
    UNION ALL SELECT 11, 'gold.camera_hourly', count(*), 'camera x hour rows' FROM bkk.gold.camera_hourly
    UNION ALL SELECT 12, 'gold.camera_profile', count(*), 'one row per camera' FROM bkk.gold.camera_profile
    UNION ALL SELECT 13, 'silver.image_scan (excluded)', count(*) FILTER (WHERE reason IS NOT NULL), 'glitch / colour-bar test card / unreadable images, excluded from gold' FROM bkk.silver.image_scan
) ORDER BY step
"""


def run() -> dict:
    catalog = open_catalog()
    con = catalog.connection
    con.execute("CREATE SCHEMA IF NOT EXISTS bkk.gold")
    con.execute(CAMERA_HOURLY_SQL)
    con.execute(CAMERA_PROFILE_SQL)
    con.execute(AUDIT_SQL)
    out = dict(con.sql("SELECT stage, n_rows FROM bkk.gold.pipeline_audit ORDER BY step").fetchall())
    print(out)
    return out


if __name__ == "__main__":
    run()
