from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Float, Integer, String, Date

db = SQLAlchemy()

class CityData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    adm3_en = db.Column(String(100), nullable=False)
    adm3_pcode = db.Column(String(20), nullable=False)
    date = db.Column(Date, nullable=False)
    year = db.Column(Integer, nullable=False)
    week = db.Column(Integer, nullable=False)
    doh_pois_count = db.Column(Float)
    ndvi = db.Column(Float)
    pct_area_cropland = db.Column(Float)
    pct_area_flood_hazard_5yr_high = db.Column(Float)
    pct_area_flood_hazard_5yr_low = db.Column(Float)
    pct_area_flood_hazard_5yr_med = db.Column(Float)
    pct_area_herbaceous_wetland = db.Column(Float)
    pct_area_mangroves = db.Column(Float)
    pct_area_permanent_water_bodies = db.Column(Float)
    pnp = db.Column(Float)
    pop_count_mean = db.Column(Float)
    pop_count_stdev = db.Column(Float)
    pop_count_total = db.Column(Float)
    pop_density_mean = db.Column(Float)
    pop_density_stdev = db.Column(Float)
    pr = db.Column(Float)
    rh = db.Column(Float)
    rwi_mean = db.Column(Float)
    rwi_std = db.Column(Float)
    spi3 = db.Column(Float)
    spi6 = db.Column(Float)
    tave = db.Column(Float)
    tmax = db.Column(Float)
    tmin = db.Column(Float)
    case_total_dengue = db.Column(Integer)
    death_total_dengue = db.Column(Integer)

    def __repr__(self):
        return f'<CityData {self.adm3_en} {self.date}>'