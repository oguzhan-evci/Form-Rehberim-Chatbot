
#!/bin/bash

echo "Ortam değişkenleri yükleniyor..."
# source .flaskenv # Artık Secrets kullandığımız için buna gerek yok

echo "Python bağımlılıkları kuruluyor..."
pip install -r requirements.txt

echo "Flask uygulaması başlatılıyor..."
python app.py
