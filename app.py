import os
import json
import base64
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import requests
from bs4 import BeautifulSoup
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agriculture.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db = SQLAlchemy(app)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY', '')

class Land(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    area = db.Column(db.Float, nullable=False)
    soil_type = db.Column(db.String(50))
    current_crop = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class FinancialRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    type = db.Column(db.String(20), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/land')
def land():
    lands = Land.query.all()
    return render_template('land.html', lands=lands)

@app.route('/land/add', methods=['POST'])
def add_land():
    try:
        land = Land(
            name=request.form['name'],
            location=request.form['location'],
            area=float(request.form['area']),
            soil_type=request.form['soil_type'],
            current_crop=request.form.get('current_crop', '')
        )
        db.session.add(land)
        db.session.commit()
        flash('Land added successfully!', 'success')
    except Exception as e:
        flash(f'Error adding land: {str(e)}', 'error')
    return redirect(url_for('land'))

@app.route('/weather')
def weather():
    return render_template('weather.html')

@app.route('/api/weather/<city>')
def get_weather(city):
    try:
        # Web scraping weather data from wttr.in
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Get current weather
        url = f'https://wttr.in/{city}?format=j1'
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            raise Exception('Failed to fetch weather data')
        
        data = response.json()
        current = data['current_condition'][0]
        
        # Parse forecast
        forecast = []
        for i, day in enumerate(data['weather'][:5]):
            if i == 0:
                day_name = 'Today'
            elif i == 1:
                day_name = 'Tomorrow'
            else:
                day_name = f'Day {i}'
            
            forecast.append({
                'day': day_name,
                'temp': int(day['avgtempC']),
                'condition': day['hourly'][0]['weatherDesc'][0]['value']
            })
        
        return jsonify({
            'city': data['nearest_area'][0]['areaName'][0]['value'],
            'temperature': int(current['temp_C']),
            'condition': current['weatherDesc'][0]['value'],
            'humidity': int(current['humidity']),
            'wind_speed': int(current['windspeedKmph']),
            'forecast': forecast,
            'demo': False
        })
    except Exception as e:
        # Fallback to demo data if scraping fails
        return jsonify({
            'city': city,
            'temperature': 28,
            'condition': 'Partly Cloudy',
            'humidity': 65,
            'wind_speed': 12,
            'forecast': [
                {'day': 'Today', 'temp': 28, 'condition': 'Partly Cloudy'},
                {'day': 'Tomorrow', 'temp': 29, 'condition': 'Sunny'},
                {'day': 'Day 2', 'temp': 27, 'condition': 'Cloudy'},
                {'day': 'Day 3', 'temp': 26, 'condition': 'Rainy'},
                {'day': 'Day 4', 'temp': 28, 'condition': 'Sunny'}
            ],
            'demo': True,
            'error_msg': str(e)
        })

@app.route('/land-preparation')
def land_preparation():
    return render_template('land_preparation.html')

@app.route('/disease-detection')
def disease_detection():
    return render_template('disease_detection.html')

@app.route('/api/detect-disease', methods=['POST'])
def detect_disease():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        from PIL import Image
        import numpy as np
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and analyze image
        img = Image.open(filepath)
        img_array = np.array(img)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Calculate color statistics
        avg_green = np.mean(img_array[:, :, 1])
        avg_red = np.mean(img_array[:, :, 0])
        avg_blue = np.mean(img_array[:, :, 2])
        
        # Calculate color variance
        green_var = np.var(img_array[:, :, 1])
        
        # Calculate brown/yellow spots (potential disease indicators)
        brown_mask = (img_array[:, :, 0] > 100) & (img_array[:, :, 1] > 80) & (img_array[:, :, 2] < 80)
        brown_ratio = np.sum(brown_mask) / (img_array.shape[0] * img_array.shape[1])
        
        # Calculate dark spots
        dark_mask = np.mean(img_array, axis=2) < 80
        dark_ratio = np.sum(dark_mask) / (img_array.shape[0] * img_array.shape[1])
        
        # Detect yellow spots
        yellow_mask = (img_array[:, :, 0] > 180) & (img_array[:, :, 1] > 180) & (img_array[:, :, 2] < 100)
        yellow_ratio = np.sum(yellow_mask) / (img_array.shape[0] * img_array.shape[1])
        
        # Disease detection algorithm
        disease = 'Healthy Plant'
        confidence = 'Medium'
        description = 'Plant appears healthy with normal coloration'
        treatment = [
            'Continue regular watering schedule',
            'Maintain balanced fertilization',
            'Monitor for any changes',
            'Ensure adequate sunlight',
            'Keep area clean and free of debris'
        ]
        prevention = [
            'Regular inspection of plants',
            'Proper spacing for air circulation',
            'Use quality seeds and soil',
            'Practice crop rotation'
        ]
        
        # Leaf Blight detection (brown spots)
        if brown_ratio > 0.15 and green_var > 1000:
            disease = 'Leaf Blight'
            confidence = 'High' if brown_ratio > 0.25 else 'Medium'
            description = 'Fungal disease causing brown spots and lesions on leaves'
            treatment = [
                'Remove and destroy infected leaves immediately',
                'Apply copper-based fungicide spray',
                'Improve air circulation around plants',
                'Reduce overhead watering',
                'Apply neem oil as organic treatment'
            ]
            prevention = [
                'Use disease-resistant varieties',
                'Practice crop rotation every season',
                'Avoid working with wet plants',
                'Water at base of plants only'
            ]
        
        # Bacterial Spot detection (dark spots)
        elif dark_ratio > 0.20 and avg_green < 100:
            disease = 'Bacterial Spot'
            confidence = 'High' if dark_ratio > 0.30 else 'Medium'
            description = 'Bacterial infection causing dark water-soaked spots'
            treatment = [
                'Remove infected plant parts',
                'Apply copper-based bactericide',
                'Avoid overhead irrigation',
                'Disinfect tools between uses',
                'Reduce plant density for better airflow'
            ]
            prevention = [
                'Use certified disease-free seeds',
                'Avoid working in wet conditions',
                'Practice 3-year crop rotation',
                'Maintain proper plant spacing'
            ]
        
        # Nutrient Deficiency (yellowing)
        elif yellow_ratio > 0.20 or (avg_green < 90 and avg_red > 120 and avg_blue < 90):
            disease = 'Nutrient Deficiency (Nitrogen)'
            confidence = 'Medium'
            description = 'Yellowing of leaves indicating nitrogen deficiency'
            treatment = [
                'Apply nitrogen-rich fertilizer',
                'Use organic compost or manure',
                'Consider foliar feeding',
                'Check soil pH and adjust if needed',
                'Ensure proper watering for nutrient uptake'
            ]
            prevention = [
                'Regular soil testing',
                'Balanced fertilization schedule',
                'Use cover crops to add nitrogen',
                'Maintain optimal soil pH'
            ]
        
        # Early Blight
        elif brown_ratio > 0.10 and dark_ratio > 0.10:
            disease = 'Early Blight'
            confidence = 'Medium'
            description = 'Fungal disease with concentric ring patterns on leaves'
            treatment = [
                'Remove lower infected leaves',
                'Apply fungicide containing chlorothalonil',
                'Mulch around plants to prevent splash',
                'Space plants for better air flow',
                'Use drip irrigation instead of overhead'
            ]
            prevention = [
                'Rotate crops annually',
                'Use resistant varieties',
                'Remove plant debris after harvest',
                'Avoid wetting foliage when watering'
            ]
        
        return jsonify({
            'disease': disease,
            'confidence': confidence,
            'description': description,
            'treatment': treatment,
            'prevention': prevention,
            'demo': False
        })
        
    except Exception as e:
        return jsonify({'error': f'Error analyzing image: {str(e)}'}), 500

@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')

@app.route('/schemes')
def schemes():
    return render_template('schemes.html')

@app.route('/crop-rotation')
def crop_rotation():
    return render_template('crop_rotation.html')

@app.route('/crop-calendar')
def crop_calendar():
    return render_template('crop_calendar.html')

@app.route('/market-analysis')
def market_analysis():
    return render_template('market_analysis.html')

@app.route('/financial')
def financial():
    records = FinancialRecord.query.order_by(FinancialRecord.date.desc()).all()
    
    income = sum(r.amount for r in records if r.type == 'income')
    expenses = sum(r.amount for r in records if r.type == 'expense')
    profit = income - expenses
    
    return render_template('financial.html', records=records, 
                         income=income, expenses=expenses, profit=profit, now=datetime.now())

@app.route('/financial/add', methods=['POST'])
def add_financial_record():
    try:
        record = FinancialRecord(
            date=datetime.strptime(request.form['date'], '%Y-%m-%d').date(),
            type=request.form['type'],
            category=request.form['category'],
            amount=float(request.form['amount']),
            description=request.form.get('description', '')
        )
        db.session.add(record)
        db.session.commit()
        flash('Record added successfully!', 'success')
    except Exception as e:
        flash(f'Error adding record: {str(e)}', 'error')
    return redirect(url_for('financial'))

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
