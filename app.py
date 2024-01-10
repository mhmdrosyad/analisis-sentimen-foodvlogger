from flask import Flask, render_template, \
    request, redirect, url_for, session, flash, get_flashed_messages, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from io import StringIO
import joblib
import re

from sqlalchemy import create_engine, text

engine = create_engine('mysql://root:@localhost/db_sentimen_food')

model_filename = 'sentiment_model.joblib'
model = None
vectorizer = CountVectorizer()

application = Flask(__name__)
application.secret_key = "umby_oke"

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/analisis', methods=['GET', 'POST'])
def analisis():
    if request.method == 'POST':
        try:
            # Menerima data dari formulir HTML
            caption = request.form.get('caption')

            # Pastikan caption tidak kosong
            if not caption:
                return redirect('/')

            # Muat kembali model dan vectorizer yang sudah disimpan
            global model, vectorizer
            if model is None or not hasattr(model, 'predict'):
                # Jika model belum dilatih, muat ulang model dan vectorizer
                model = joblib.load('sentiment_model.joblib')
                vectorizer = joblib.load('vectorizer.joblib')  # Gantilah dengan nama yang sesuai saat menyimpan vectorizer

            # Ekstraksi fitur dari caption yang diterima
            caption_vectorized = vectorizer.transform([caption])

            # Lakukan prediksi menggunakan model
            prediction = model.predict(caption_vectorized)

            return render_template('analisis.html', prediction=prediction[0], caption=caption)
        except Exception as e:
            return render_template('error.html', message=f'Error: {str(e)}')
    else:
        # Jika request GET, tampilkan halaman biasa
        return render_template('analisis.html')



@application.route('/data-set')
def data_set():
    with engine.connect() as conn:
        # Menjalankan query SQL untuk mengambil data dari tabel
        query = text("SELECT * FROM dataset")
        result = conn.execute(query)

        # Mendapatkan semua baris hasil query
        rows = result.fetchall()
        
    return render_template('upload-data.html', data=rows )

@application.route('/delete-all')
def delete_all():
    with engine.connect() as conn:
        # Menjalankan query SQL untuk mengambil data dari tabel
        query = text("DELETE FROM dataset")
        conn.execute(query)
        conn.commit()
    return redirect('/data-set')

@application.route('/delete/<int:id>')
def delete(id):
    with engine.connect() as conn:
        # Menjalankan query SQL untuk menghapus baris berdasarkan ID
        query = text("DELETE FROM dataset WHERE id = :id")
        conn.execute(query, {'id': id})
        conn.commit()
    return redirect('/data-set')

@application.route('/upload-data', methods=['POST'])
def upload_data():
    try:
        # Mendapatkan file dari formulir
        file = request.files['file']
        df = pd.read_csv(file)
        df = df.dropna()
        df.to_sql(name='dataset', con=engine, if_exists='append', index=False)
        return redirect('/data-set')
    except Exception as e:
        return f"Error: {str(e)}", 500

@application.route('/train')
def train():
    try:
        # Mengambil data dari database
        with engine.connect() as conn:
            query = text("SELECT * FROM dataset")
            result = conn.execute(query)
            rows = result.fetchall()

        if not rows:
            return jsonify({'message': 'Data not found.'}), 404

        data = pd.DataFrame([(row.caption, row.label) for row in rows], columns=['caption', 'label'])

        # Bagi data menjadi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(data['caption'], data['label'], test_size=0.2, random_state=42)

        # Ekstraksi fitur menggunakan CountVectorizer
        vectorizer = CountVectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train)

        # Latih model Naive Bayes
        global model
        model = MultinomialNB()
        model.fit(X_train_vectorized, y_train)

        # Simpan model dan vectorizer
        joblib.dump(model, 'sentiment_model.joblib')
        joblib.dump(vectorizer, 'vectorizer.joblib')

        # Evaluasi model
        X_test_vectorized = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vectorized)

        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        # Render tampilan dengan informasi akurasi
        return render_template('result.html', accuracy=accuracy, classification_report=classification_rep)
    
    except Exception as e:
        return f"Error: {str(e)}", 500


@application.route('/proses-naive', methods=['POST'])
def proses_naive():
    try:
        file = request.files['file']
        df_upload = pd.read_csv(file)
        df_upload = df_upload.dropna(subset=['caption'])
        df_upload['caption'] = df_upload['caption'].apply(lambda x: re.sub(r'\n', '', str(x)))

        global model, vectorizer
        if model is None or not hasattr(model, 'predict'):
            # Jika model belum dilatih, muat ulang model dan vectorizer
            model = joblib.load('sentiment_model.joblib')
            vectorizer = joblib.load('vectorizer.joblib')  # Gantilah dengan nama yang sesuai saat menyimpan vectorizer

        # Ekstraksi fitur dari caption yang diterima
        captions = df_upload['caption'].tolist()
        captions_vectorized = vectorizer.transform(captions)

        # Lakukan prediksi menggunakan model
        predictions = model.predict(captions_vectorized)

        # Tambahkan kolom label ke DataFrame
        df_upload['label'] = predictions

        total_data = len(df_upload)

        neutral_count = df_upload[df_upload['label'].str.lower() == 'netral'].shape[0]
        positive_count = df_upload[df_upload['label'].str.lower() == 'positif'].shape[0]
        negative_count = df_upload[df_upload['label'].str.lower() == 'negatif'].shape[0]

        # Mengonversi DataFrame ke HTML
        result_html = df_upload.to_html(index=False, classes='table table-striped table-bordered table-hover').replace('\n', '')

        # Rendering template result.html dengan hasil HTML
        return render_template('analyze_result.html', result_html=result_html, neutral_count=neutral_count, positive_count=positive_count, negative_count=negative_count, total_data=total_data)


    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == '__main__':
    application.run(debug=True)