# models.py
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class CityData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    city = db.Column(db.String(100), nullable=False)
    date = db.Column(db.Date, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    population = db.Column(db.Float, nullable=False)
    dengue_cases = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<CityData {self.city} {self.date}>'