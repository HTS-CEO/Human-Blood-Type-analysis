from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import os
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def analyze_data(df):
    results = {}
    df['AdmissionDate'] = pd.to_datetime(df['AdmissionDate'])
    df['DischargeDate'] = pd.to_datetime(df['DischargeDate'])
    df['LengthOfStay'] = (df['DischargeDate'] - df['AdmissionDate']).dt.days
    
    
    results['total_patients'] = int(len(df))
    results['avg_stay'] = float(round(df['LengthOfStay'].mean(), 2))
    results['median_stay'] = float(round(df['LengthOfStay'].median(), 2))
    results['max_stay'] = int(df['LengthOfStay'].max())
    results['min_stay'] = int(df['LengthOfStay'].min())
    
    blood_type_stats = df.groupby('BloodType')['LengthOfStay'].agg(['mean', 'count', 'median'])
    blood_type_stats['mean'] = blood_type_stats['mean'].round(2)
    blood_type_stats['median'] = blood_type_stats['median'].round(2)
    results['blood_type_stats'] = {
        'mean': blood_type_stats['mean'].astype(float).to_dict(),
        'count': blood_type_stats['count'].astype(int).to_dict(),
        'median': blood_type_stats['median'].astype(float).to_dict()
    }
    
    diagnosis_stats = df.groupby('Diagnosis')['LengthOfStay'].agg(['mean', 'count'])
    diagnosis_stats['mean'] = diagnosis_stats['mean'].round(2)
    results['diagnosis_stats'] = {
        'mean': diagnosis_stats.sort_values('count', ascending=False).head(10)['mean'].astype(float).to_dict(),
        'count': diagnosis_stats.sort_values('count', ascending=False).head(10)['count'].astype(int).to_dict()
    }
    
    blood_diagnosis = df.groupby(['BloodType', 'Diagnosis'])['LengthOfStay'].mean().unstack()
    results['blood_diagnosis_matrix'] = blood_diagnosis.round(2).fillna('-').astype(object).to_dict()
    
    monthly_trend = df.groupby(df['AdmissionDate'].dt.to_period('M')).size()
    results['monthly_trend'] = {str(k): int(v) for k, v in monthly_trend.items()}
    
    return results

def generate_visualizations(df, file_prefix):
    plots = {}
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='BloodType', y='LengthOfStay', data=df)
    plt.title('Hospital Stay Duration by Blood Type')
    plt.xlabel('Blood Type')
    plt.ylabel('Days in Hospital')
    blood_type_path = f"{app.config['UPLOAD_FOLDER']}/{file_prefix}_blood_type.png"
    plt.savefig(blood_type_path)
    plt.close()
    plots['blood_type'] = blood_type_path
    
    plt.figure(figsize=(12, 6))
    top_diagnoses = df['Diagnosis'].value_counts().head(5).index
    df_top = df[df['Diagnosis'].isin(top_diagnoses)]
    sns.boxplot(x='Diagnosis', y='LengthOfStay', hue='BloodType', data=df_top)
    plt.title('Hospital Stay by Diagnosis and Blood Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    blood_diag_path = f"{app.config['UPLOAD_FOLDER']}/{file_prefix}_blood_diag.png"
    plt.savefig(blood_diag_path)
    plt.close()
    plots['blood_diagnosis'] = blood_diag_path
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['LengthOfStay'], bins=20, kde=True)
    plt.title('Distribution of Hospital Stay Duration')
    plt.xlabel('Days')
    plt.ylabel('Number of Patients')
    hist_path = f"{app.config['UPLOAD_FOLDER']}/{file_prefix}_hist.png"
    plt.savefig(hist_path)
    plt.close()
    plots['histogram'] = hist_path
    
    blood_type_counts = df['BloodType'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(blood_type_counts, labels=blood_type_counts.index, autopct='%1.1f%%')
    plt.title('Blood Type Distribution')
    pie_path = f"{app.config['UPLOAD_FOLDER']}/{file_prefix}_pie.png"
    plt.savefig(pie_path)
    plt.close()
    plots['pie'] = pie_path
    
    return plots

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blood Type Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <style>
        body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background-color:#f8f9fa;color:#333}
        .navbar-brand{font-weight:700}
        .upload-container{background-color:white;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,0.1);padding:30px;margin-top:30px}
        .analysis-card{background-color:white;border-radius:10px;box-shadow:0 4px 8px rgba(0,0,0,0.05);padding:20px;margin-bottom:20px;height:100%}
        .stat-card{background-color:#f8f9fa;border-left:4px solid #0d6efd;border-radius:5px;padding:15px;margin-bottom:15px}
        .visualization{background-color:white;border-radius:10px;box-shadow:0 4px 8px rgba(0,0,0,0.05);padding:15px;margin-bottom:20px}
        .visualization img{max-width:100%;height:auto;border-radius:5px}
        .loading-spinner{display:none;text-align:center;margin:20px 0}
        .file-upload{border:2px dashed #dee2e6;border-radius:5px;padding:20px;text-align:center;cursor:pointer;transition:all 0.3s}
        .file-upload:hover{border-color:#0d6efd;background-color:#f8f9fa}
        .file-upload.dragover{border-color:#0d6efd;background-color:#e7f1ff}
        .sample-data-btn{margin-top:10px}
        .tab-content{padding:20px 0}
        .nav-tabs .nav-link.active{font-weight:600}
        .blood-type-table th{background-color:#f8f9fa}
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-danger">
        <div class="container">
            <a class="navbar-brand" href="#">Blood Type & Hospital Stay Analysis</a>
        </div>
    </nav>
    <div class="container">
        <div class="upload-container">
            <h2 class="text-center mb-4">Blood Type Impact Analysis</h2>
            <p class="text-center text-muted mb-4">Upload patient data to analyze associations between blood types and hospital stay duration.</p>
            <div class="file-upload mb-3" id="dropZone">
                <i class="bi bi-droplet fs-1 text-danger"></i>
                <h5>Drag & Drop your patient data file here</h5>
                <p class="text-muted">or</p>
                <input type="file" id="fileInput" class="d-none" accept=".csv,.xlsx,.xls">
                <button class="btn btn-danger" onclick="document.getElementById('fileInput').click()">Select File</button>
                <div class="form-text mt-2">Required columns: PatientID, BloodType, Diagnosis, AdmissionDate, DischargeDate</div>
            </div>
            <div class="text-center">
                <button class="btn btn-outline-danger sample-data-btn" id="useSampleData">
                    <i class="bi bi-file-earmark-medical"></i> Use Sample Data
                </button>
            </div>
            <div class="loading-spinner" id="loadingSpinner">
                <div class="spinner-border text-danger" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Analyzing blood type patterns...</p>
            </div>
            <div class="alert alert-danger mt-3 d-none" id="errorAlert" role="alert"></div>
        </div>
        <div id="analysisResults" class="mt-4 d-none">
            <ul class="nav nav-tabs" id="analysisTabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="summary-tab" data-bs-toggle="tab" data-bs-target="#summary" type="button">Summary</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="bloodtype-tab" data-bs-toggle="tab" data-bs-target="#bloodtype" type="button">Blood Type Analysis</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="visualizations-tab" data-bs-toggle="tab" data-bs-target="#visualizations" type="button">Visualizations</button>
                </li>
            </ul>
            <div class="tab-content" id="analysisTabContent">
                <div class="tab-pane fade show active" id="summary" role="tabpanel">
                    <div class="row mt-3">
                        <div class="col-md-3">
                            <div class="analysis-card">
                                <h5><i class="bi bi-people-fill text-danger"></i> Total Patients</h5>
                                <h2 id="totalPatients" class="text-danger">0</h2>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="analysis-card">
                                <h5><i class="bi bi-calendar-check text-success"></i> Avg. Stay (days)</h5>
                                <h2 id="avgStay" class="text-success">0</h2>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="analysis-card">
                                <h5><i class="bi bi-calendar-range text-primary"></i> Median Stay (days)</h5>
                                <h2 id="medianStay" class="text-primary">0</h2>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="analysis-card">
                                <h5><i class="bi bi-calendar-event text-warning"></i> Stay Range (days)</h5>
                                <h2 id="stayRange" class="text-warning">0 - 0</h2>
                            </div>
                        </div>
                    </div>
                    <div class="row mt-3">
                        <div class="col-md-6">
                            <div class="analysis-card">
                                <h5><i class="bi bi-droplet text-danger"></i> Blood Type Distribution</h5>
                                <div id="bloodTypeDistribution" class="mt-3"></div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="analysis-card">
                                <h5><i class="bi bi-clipboard2-pulse text-danger"></i> Top Diagnoses</h5>
                                <div id="topDiagnoses" class="mt-3"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="tab-pane fade" id="bloodtype" role="tabpanel">
                    <div class="analysis-card">
                        <h5><i class="bi bi-droplet text-danger"></i> Hospital Stay by Blood Type</h5>
                        <div class="table-responsive mt-3">
                            <table class="table table-striped blood-type-table">
                                <thead>
                                    <tr>
                                        <th>Blood Type</th>
                                        <th>Average Stay (days)</th>
                                        <th>Median Stay (days)</th>
                                        <th>Number of Patients</th>
                                    </tr>
                                </thead>
                                <tbody id="bloodTypeTable"></tbody>
                            </table>
                        </div>
                    </div>
                    <div class="analysis-card mt-4">
                        <h5><i class="bi bi-clipboard2-pulse text-danger"></i> Blood Type vs Diagnosis Matrix</h5>
                        <div class="table-responsive mt-3">
                            <table class="table table-striped" id="bloodDiagnosisMatrix">
                            </table>
                        </div>
                    </div>
                </div>
                <div class="tab-pane fade" id="visualizations" role="tabpanel">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="visualization">
                                <h5><i class="bi bi-droplet text-danger"></i> Hospital Stay by Blood Type</h5>
                                <img id="bloodTypeImg" src="" alt="Hospital Stay by Blood Type" class="img-fluid">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="visualization">
                                <h5><i class="bi bi-pie-chart text-danger"></i> Blood Type Distribution</h5>
                                <img id="pieImg" src="" alt="Blood Type Distribution" class="img-fluid">
                            </div>
                        </div>
                    </div>
                    <div class="row mt-4">
                        <div class="col-md-6">
                            <div class="visualization">
                                <h5><i class="bi bi-clipboard2-pulse text-danger"></i> Top Diagnoses by Blood Type</h5>
                                <img id="bloodDiagImg" src="" alt="Top Diagnoses by Blood Type" class="img-fluid">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="visualization">
                                <h5><i class="bi bi-bar-chart text-danger"></i> Hospital Stay Distribution</h5>
                                <img id="histogramImg" src="" alt="Hospital Stay Distribution" class="img-fluid">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <footer class="bg-light mt-5 py-4">
        <div class="container text-center text-muted">
            <p>Blood Type Analysis Tool &copy; 2025</p>
        </div>
    </footer>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const dropZone = document.getElementById('dropZone');
            const fileInput = document.getElementById('fileInput');
            const loadingSpinner = document.getElementById('loadingSpinner');
            const errorAlert = document.getElementById('errorAlert');
            const analysisResults = document.getElementById('analysisResults');
            const useSampleDataBtn = document.getElementById('useSampleData');
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, unhighlight, false);
            });
            
            function highlight() {
                dropZone.classList.add('dragover');
            }
            
            function unhighlight() {
                dropZone.classList.remove('dragover');
            }
            
            dropZone.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                if (files.length) {
                    fileInput.files = files;
                    handleFiles(files);
                }
            }
            
            fileInput.addEventListener('change', function() {
                if (this.files.length) {
                    handleFiles(this.files);
                }
            });
            
            function handleFiles(files) {
                const file = files[0];
                if (!file.name.match(/\.(csv|xlsx|xls)$/i)) {
                    showError('Please upload a CSV or Excel file.');
                    return;
                }
                uploadFile(file);
            }
            
            function uploadFile(file) {
                loadingSpinner.style.display = 'block';
                errorAlert.classList.add('d-none');
                const formData = new FormData();
                formData.append('file', file);
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(err => { throw new Error(err.error); });
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    displayResults(data);
                })
                .catch(error => {
                    showError(error.message);
                })
                .finally(() => {
                    loadingSpinner.style.display = 'none';
                });
            }
            
            useSampleDataBtn.addEventListener('click', function() {
                loadingSpinner.style.display = 'block';
                errorAlert.classList.add('d-none');
                const sampleData = `PatientID,BloodType,Diagnosis,AdmissionDate,DischargeDate
1,A+,Pneumonia,2023-01-05,2023-01-12
2,O-,Appendicitis,2023-01-06,2023-01-09
3,B+,Diabetes,2023-01-07,2023-01-14
4,AB-,Heart Failure,2023-01-08,2023-01-18
5,A-,Stroke,2023-01-09,2023-01-20
6,O+,Pneumonia,2023-01-10,2023-01-15
7,B-,Fracture,2023-01-11,2023-01-14
8,AB+,COVID-19,2023-01-12,2023-01-22
9,A+,Diabetes,2023-01-13,2023-01-19
10,O-,Pneumonia,2023-01-14,2023-01-21
11,B+,Stroke,2023-01-15,2023-01-25
12,AB-,Appendicitis,2023-01-16,2023-01-19
13,A-,Heart Failure,2023-01-17,2023-01-24
14,O+,COVID-19,2023-01-18,2023-01-28
15,B-,Pneumonia,2023-01-19,2023-01-26
16,AB+,Diabetes,2023-01-20,2023-01-27
17,A+,Fracture,2023-01-21,2023-01-24
18,O-,Stroke,2023-01-22,2023-01-30
19,B+,Heart Failure,2023-01-23,2023-01-31
20,AB-,COVID-19,2023-01-24,2023-02-05`;
                const blob = new Blob([sampleData], { type: 'text/csv' });
                const file = new File([blob], 'blood_type_data.csv', { type: 'text/csv' });
                uploadFile(file);
            });
            
            function showError(message) {
                errorAlert.textContent = message;
                errorAlert.classList.remove('d-none');
            }
            
            function displayResults(data) {
                analysisResults.classList.remove('d-none');
                const analysis = data.analysis;
                const visualizations = data.visualizations;
                
                document.getElementById('totalPatients').textContent = analysis.total_patients;
                document.getElementById('avgStay').textContent = analysis.avg_stay;
                document.getElementById('medianStay').textContent = analysis.median_stay;
                document.getElementById('stayRange').textContent = `${analysis.min_stay} - ${analysis.max_stay}`;
                
                document.getElementById('bloodTypeImg').src = visualizations.blood_type;
                document.getElementById('bloodDiagImg').src = visualizations.blood_diagnosis;
                document.getElementById('histogramImg').src = visualizations.histogram;
                document.getElementById('pieImg').src = visualizations.pie;
                
                let bloodTypeTableHtml = '';
                for (const bloodType in analysis.blood_type_stats.mean) {
                    bloodTypeTableHtml += `
                        <tr>
                            <td>${bloodType}</td>
                            <td>${analysis.blood_type_stats.mean[bloodType]}</td>
                            <td>${analysis.blood_type_stats.median[bloodType]}</td>
                            <td>${analysis.blood_type_stats.count[bloodType]}</td>
                        </tr>
                    `;
                }
                document.getElementById('bloodTypeTable').innerHTML = bloodTypeTableHtml;
                
                let bloodTypeDistributionHtml = '';
                for (const bloodType in analysis.blood_type_stats.count) {
                    const count = analysis.blood_type_stats.count[bloodType];
                    const percentage = Math.round(count / analysis.total_patients * 100);
                    bloodTypeDistributionHtml += `
                        <div class="stat-card">
                            <h6>${bloodType}</h6>
                            <h4>${count} <small class="text-muted">(${percentage}%)</small></h4>
                        </div>
                    `;
                }
                document.getElementById('bloodTypeDistribution').innerHTML = bloodTypeDistributionHtml;
                
                let topDiagnosesHtml = '';
                for (const diagnosis in analysis.diagnosis_stats.mean) {
                    topDiagnosesHtml += `
                        <div class="stat-card">
                            <h6>${diagnosis}</h6>
                            <h4>${analysis.diagnosis_stats.mean[diagnosis]} <small class="text-muted">(${analysis.diagnosis_stats.count[diagnosis]} cases)</small></h4>
                        </div>
                    `;
                }
                document.getElementById('topDiagnoses').innerHTML = topDiagnosesHtml;
                
                let bloodDiagnosisMatrixHtml = '<thead><tr><th>Diagnosis</th>';
                const bloodTypes = Object.keys(analysis.blood_type_stats.mean);
                bloodTypes.forEach(bt => bloodDiagnosisMatrixHtml += `<th>${bt}</th>`);
                bloodDiagnosisMatrixHtml += '</tr></thead><tbody>';
                
                for (const diagnosis in analysis.blood_diagnosis_matrix) {
                    bloodDiagnosisMatrixHtml += '<tr><td>' + diagnosis + '</td>';
                    bloodTypes.forEach(bt => {
                        const days = analysis.blood_diagnosis_matrix[diagnosis][bt] || '-';
                        bloodDiagnosisMatrixHtml += `<td>${days}</td>`;
                    });
                    bloodDiagnosisMatrixHtml += '</tr>';
                }
                bloodDiagnosisMatrixHtml += '</tbody>';
                document.getElementById('bloodDiagnosisMatrix').innerHTML = bloodDiagnosisMatrixHtml;
            }
        });
    </script>
</body>
</html>
''')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            file_prefix = str(uuid.uuid4())
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
            
            required_columns = ['PatientID', 'BloodType', 'Diagnosis', 'AdmissionDate', 'DischargeDate']
            if not all(col in df.columns for col in required_columns):
                return jsonify({'error': f'Missing required columns: {", ".join(required_columns)}'}), 400
            
            analysis_results = analyze_data(df)
            visualizations = generate_visualizations(df, file_prefix)
            
            return jsonify({
                'success': True,
                'analysis': analysis_results,
                'visualizations': visualizations
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)