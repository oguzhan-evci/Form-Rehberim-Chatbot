# Form Rehberi: AI Destekli Egzersiz Kütüphanesi

## 📝 Proje Özeti

Bu proje, kullanıcılara vücut ağırlığı egzersizleri (Squat, Plank, Lunge vb.) hakkında sorulan sorulara, yalnızca sağlanan belirli bir bilgi kümesine (Markdown dosyaları) dayanarak doğru, özgün ve güvenilir cevaplar üreten bir Retrieval-Augmented Generation (RAG) sistemi üzerine kurulu bir sohbet robotu geliştirmeyi amaçlamaktadır.

Geliştirilen sohbet robotu, Python Flask kütüphanesi kullanılarak oluşturulan, kullanıcı dostu ve çok dilli (Türkçe/İngilizce) bir web arayüzü üzerinden sunulmaktadır. Temel hedef, yalnızca sağlanan veri setindeki bilgileri kullanarak yanıt üreten, halüsinasyonları en aza indiren ve doğal bir konuşma akışı sunan bir yapay zeka asistanı yaratmaktır.

Projenin tüm kodları GitHub üzerinde açık kaynak olarak paylaşılmış ve geliştirme süreci bir Kaggle Notebook üzerinde yürütülmüştür.

## 🎯 Hedefler

Projenin temel hedefleri şunlardır:

- **Güvenilir Bilgi Erişimi**: Belirli bir Markdown dosyaları koleksiyonunu tek bilgi kaynağı olarak kullanan bir RAG sistemi kurarak, yalnızca doğrulanmış bilgilere dayalı cevaplar üretmek.

- **Halüsinasyon Önleme**: Sohbet robotunun, bilgi kaynağında bulunmayan konularda spekülasyon yapmasını veya yanlış bilgi üretmesini engelleyerek güvenilirliği artırmak.

- **Niyet Anlama ve Doğal Diyalog**: Kullanıcı girdisinin amacını (sohbet ifadesi mi, spesifik egzersiz sorusu mu, belirsiz istek mi?) anlayarak duruma uygun, akıcı ve doğal yanıtlar vermek.

- **Etkili RAG Pipeline Yönetimi**: LangChain kütüphanesinin yeteneklerinden faydalanarak veri yükleme (DirectoryLoader), metin parçalama (RecursiveCharacterTextSplitter), embedding (HuggingFaceEmbeddings), vektör indeksleme/arama (FAISS) ve cevap üretimi (LLMChain) adımlarını içeren optimize edilmiş bir RAG iş akışı oluşturmak.

- **Sohbet Hafızası Entegrasyonu**: Kullanıcıyla yapılan konuşmanın bağlamını korumak için LangChain hafıza mekanizmalarını (ConversationBufferMemory, create_history_aware_retriever) entegre ederek, takip sorularına tutarlı ve bağlama uygun cevaplar verilmesini sağlamak.

- **Prompt Mühendisliği ile Davranış Kontrolü**: Google Gemini modelinin (gemini-2.5-flash) davranışını (ChatPromptTemplate kullanılarak tasarlanan detaylı sistem prompt'ları aracılığıyla) hassas bir şekilde yönlendirmek; belirlenen "Usta Asistan" personasını (pozitif, teşvik edici, özgün) benimsemesini, yalnızca sağlanan bağlamı kullanmasını, bilgiyi sentezlemesini ve cevapları istenen formatta (kısa, odaklı, basit Markdown) sunmasını sağlamak.

- **Kullanıcı Dostu Web Arayüzü**: Flask ve temel web teknolojileri (HTML/Jinja2, CSS, JS) ile erişilebilir, estetik, duyarlı (responsive) ve çok dilli bir arayüz tasarlamak; sohbet akışını, navigasyonu ve egzersiz listesi gibi ek özellikleri kullanıcıya sunmak.


## 📚 Veri Seti

### Kaynak
Proje için özel olarak hazırlanmış, temel vücut ağırlığı egzersizlerini detaylandıran Markdown (.md) dosyaları koleksiyonu. Bu koleksiyon, Kaggle üzerinde **hareket-ansiklopedisi-dataset** adıyla bir veri seti olarak barındırılmıştır.

### İçerik
Her bir .md dosyası, belirli bir egzersize odaklanarak; egzersizin adını, genel bir açıklamasını, "Nasıl Yapılır" başlığı altında adım adım talimatlarını, hedeflenen ana kas gruplarını ve egzersizin önerilen zorluk seviyesini (örn: Başlangıç) içermektedir.

### Boyut ve Yapı
Veri seti, toplam **45 adet** bağımsız .md egzersiz tanım dosyasından oluşmaktadır. Her dosya, RAG sistemi tarafından tek bir bilgi birimi (chunk) olarak işlenmiştir.

### Kullanım Amacı
Bu Markdown dosyaları, RAG sisteminin bilgi çekirdeğini (knowledge base) oluşturur. LLM, kullanıcı sorularını cevaplarken yalnızca bu dosyalardan retriever tarafından getirilen ilgili metin parçalarını referans alır.

## ⚙️ Çözüm Mimarisi ve Kullanılan Yöntemler

Projenin çözümü, **Retrieval-Augmented Generation (RAG)** mimarisine dayanmaktadır ve **LangChain** kütüphanesi etrafında inşa edilmiştir.

### RAG Mimarisi Akışı

1. **Sohbet Geçmişi Analizi**: Kullanıcının mevcut sorgusu ve sohbet geçmişi, bir `history_aware_retriever` tarafından değerlendirilir. Bu, sadece anahtar kelimeler yerine bağlamı anlayan ve buna göre arama sorgusunu optimize eden bir mekanizmadır.

2. **Bilgi Getirme (Retrieval)**: Optimize edilmiş arama sorgusu, HuggingFaceEmbeddings ile vektörleştirilmiş ve FAISS vektör veritabanında indekslenmiş olan egzersiz ansiklopedisinden (Markdown dosyaları) en alakalı bilgi parçalarını (k=3) getirir.

3. **Prompt Hazırlığı**: Getirilen bilgiler, kullanıcının orijinal sorgusu ve detaylı bir sistem prompt'u, Google Gemini-2.5-Flash modeline gönderilmek üzere hazırlanır.

4. **Yanıt Üretimi (Generation)**: Gemini modeli, aldığı bağlamı, sorguyu ve prompt'taki yönergeleri kullanarak kullanıcının sorusuna özgün, doğru ve bağlama uygun bir yanıt üretir.

5. **Sohbet Geçmişi Güncelleme**: Üretilen yanıt, gelecekteki konuşmalar için sohbet geçmişine eklenir.

### Kullanılan Teknolojiler ve Adımlar

#### Veri Hazırlık ve Vektörleştirme

- .md dosyaları `DirectoryLoader` ve `UnstructuredMarkdownLoader` ile yüklendi.
- Metinler, her dosyanın bütünlüğünü koruyacak şekilde (chunk_size=1500) `RecursiveCharacterTextSplitter` ile parçalandı.
- `HuggingFaceEmbeddings` kütüphanesi ve `all-MiniLM-L6-v2` modeli kullanılarak metin parçaları anlamsal vektörlere dönüştürüldü.
- Bu vektörler, verimli benzerlik araması için **FAISS** kütüphanesi ile bir vektör deposunda indekslendi ve yerel olarak kaydedildi (`faiss_exercise_index` klasörü).

#### Dil Modeli ve Zincir Yapılandırması (LangChain)

- Google'ın **gemini-2.5-flash** modeli, `ChatGoogleGenerativeAI` entegrasyonu ile LLM olarak seçildi. Modelin determinizmi ve yaratıcılığı `temperature` (0.75) ve `top_p` (0.9) parametreleri ile ayarlandı.

- **Prompt Mühendisliği**: Modelin davranışını detaylı bir şekilde kontrol etmek için iki ana `ChatPromptTemplate` tasarlandı:
  - `history_aware_retriever_prompt`: LLM'in, sohbet geçmişini analiz ederek retriever için bağlamdan bağımsız, optimize edilmiş bir arama sorgusu üretmesini sağlar.
  - `exercise_info_prompt`: LLM'in ana sistem prompt'udur. "Usta Asistan" rolünü, niyet algılama adımlarını (sohbet/soru/belirsiz), bağlamı TEK bilgi kaynağı olarak kullanma zorunluluğunu, genel bilgiyi KULLANMAMA kuralını, bilgiyi sentezleyip özgünleştirme beklentisini, sadece sorulana odaklanma ilkesini, basit Markdown formatlama iznini ve persona'ya uygun (pozitif, teşvik edici) üslup gerekliliklerini tanımlar.

- **Hafızalı RAG Zinciri**: `create_history_aware_retriever` (geçmişe duyarlı retriever oluşturur), `create_stuff_documents_chain` (bulunan dokümanları ve prompt'u LLM'e hazırlar) ve `create_retrieval_chain` (tüm akışı birleştirir) fonksiyonları kullanılarak, sohbet bağlamını koruyan ve RAG sürecini yöneten `qa_chain_with_history` oluşturuldu. Retriever, her sorgu için en alakalı 3 dokümanı (k=3) getirecek şekilde ayarlandı.

#### Web Uygulaması Geliştirme (Flask)

- Kullanıcı arayüzü ve arka uç mantığı Python **Flask** framework'ü ile geliştirildi.
- **Jinja2** şablonlama motoru kullanılarak HTML sayfaları (`index.html`, `about.html`, `egzersizler.html`) dinamik olarak oluşturuldu ve `LANG_DATA` sözlüğü aracılığıyla çoklu dil (TR/EN) desteği sağlandı.
- Arayüz tasarımı için temel HTML, CSS (duyarlı tasarım dahil) ve sohbet akışını iyileştirmek için minimal JavaScript (mesaj gönderme, yükleme göstergesi) kullanıldı.
- LLM'den gelen ve basit Markdown içerebilen yanıtlar, Python `markdown` kütüphanesi ile güvenli HTML formatına dönüştürülerek (`convert_markdown_to_html` fonksiyonu) arayüzde doğru şekilde gösterildi.
- Uygulama, ana sohbet sayfası (`/`), egzersiz listesi sayfası (`/egzersizler`) ve hakkında sayfası (`/about`) olmak üzere üç ana bölümden oluşmaktadır.

## ✅ Elde Edilen Sonuçlar

- **Başarılı RAG İmplementasyonu**: Belirlenen .md dosyalarını bilgi kaynağı olarak kullanan, hafızalı ve fonksiyonel bir RAG sohbet robotu başarıyla geliştirilmiş ve Flask ile web arayüzüne entegre edilmiştir.

- **Yüksek Güvenilirlik**: Sohbet robotu, prompt mühendisliği sayesinde bilgi kaynağında olmayan konularda ("Bench Press" örneği) spekülasyon yapmaktan kaçınmakta ve kullanıcıyı mevcut bilgiler dahilinde kalmaya yönlendirerek halüsinasyon riskini minimize etmektedir.

- **Gelişmiş Doğal Dil Etkileşimi**: Hafıza entegrasyonu, sohbet robotunun önceki konuşmaları hatırlamasını ("peki nasıl yapılır?" gibi takip soruları) sağlarken, niyet algılama yeteneği sayesinde basit sohbet ifadelerine ("Merhaba") uygun ve persona'ya özgü yanıtlar verebilmektedir.

- **Özgün ve Odaklı Yanıt Üretimi**: LLM, prompt'larda belirtilen katı kurallara rağmen, bağlamdaki bilgiyi işleyerek kendi kelimeleriyle, daha doğal ve özgün bir üslupla sunmakta; ayrıca kullanıcının sorduğu spesifik detaya (örn: sadece tanım veya sadece yapılış) odaklanarak gereksiz bilgiden kaçınmaktadır.

- **Kullanıcı Dostu ve Erişilebilir Arayüz**: Geliştirilen web arayüzü, temiz bir sohbet ekranı, kolay anlaşılır navigasyon (Ana Sayfa, Egzersiz Listesi, Hakkında), dil seçeneği ve mobil uyumluluk sunarak kullanıcıların sohbet robotu ile rahatça etkileşim kurmasına olanak tanımaktadır. Egzersiz listesi sayfası, uygulamanın kapsamı hakkında kullanıcıya net bir fikir vermektedir.

## 🚀 Kurulum ve Çalıştırma

### Önkoşullar

- Python 3.10 veya üzeri
- pip (Python paket yöneticisi)
- git (depoyu klonlamak için)
- Google Gemini API Anahtarı

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/oguzhan-evci/Form-Rehberim-Chatbot.git
cd Form-Rehberim-Chatbot
```

### 2. Sanal Ortam Oluşturun ve Bağımlılıkları Kurun
```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Google Gemini API Anahtarınızı Ayarlayın

Uygulama, Google Gemini modelini kullanmak için bir API anahtarına ihtiyaç duyar. Bu anahtarı bir ortam değişkeni olarak ayarlamalısınız:
```bash
# Linux/macOS
export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

# Windows (Command Prompt)
set GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

# Windows (PowerShell)
$env:GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

`YOUR_GEMINI_API_KEY` yerine kendi Google Gemini API anahtarınızı yapıştırın.

### 4. FAISS Vektör Veritabanını Hazırlayın

Proje, egzersiz bilgilerini depolamak için bir FAISS indeksi kullanır. `faiss_exercise_index` klasörünün projenizin kök dizininde olduğundan emin olun.

> **Not**: Veri setiniz (Markdown dosyaları) GitHub deponuzda yer almamaktadır, bu nedenle `faiss_exercise_index` klasörünü deponuza yüklemeniz gerekmektedir. Eğer bu klasör eksikse veya yeni egzersizler eklediyseniz, indeksi yeniden oluşturmanız gerekebilir.

### 5. Uygulamayı Çalıştırın
```bash
flask run --host 0.0.0.0 --port 5000
```

Uygulama genellikle `http://127.0.0.1:5000` adresinde çalışacaktır.

## ❓ Kullanım Örnekleri

Uygulama arayüzüne gittikten sonra, sohbet kutusuna aşağıdaki gibi sorular sorabilirsiniz:

**Kullanıcı**: Merhaba!  
**Asistan**: Merhaba! Ben Egzersiz Form Rehberi Asistanı. Hangi egzersiz hakkında bilgi almak istersin?

---

**Kullanıcı**: Squat nasıl yapılır?  
**Asistan**: Squat (Çömelme), vücut ağırlığı ile yapılan, birden fazla kas grubunu çalıştıran temel bir egzersizdir. İşte adım adım nasıl yapıldığına dair bilgi:

- **Başlangıç Pozisyonu**: Ayaklar omuz genişliğinde açık, parmak uçları hafifçe dışa dönük olmalı. Sırt dik, omuzlar geride ve karın kasları sıkı.
- **Aşağı İniş**: Kalçanızı geriye doğru iterken dizlerinizi bükün, sanki bir sandalyeye oturacakmış gibi...
- **Yukarı Kalkış**: Topuklarınızdan güç alarak başlangıç pozisyonuna geri dönün...

---

**Kullanıcı**: Peki Lunge için ne söyleyebilirsin?  
**Asistan**: Lunge (Hamle), bacak kaslarını ayrı ayrı çalıştıran harika bir egzersizdir...

---

**Kullanıcı**: Bench Press hakkında bilgin var mı?  
**Asistan**: Üzgünüm, "Bench Press" hakkında doğrudan bir bilgiye sahip değilim. Ben, yalnızca vücut ağırlığı egzersizleri ansiklopedisindeki bilgilere dayanarak yanıt verebiliyorum. Başka bir vücut ağırlığı egzersizi hakkında bilgi almak ister misin?


## 📜 Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır - daha fazla bilgi için [LICENSE](LICENSE) dosyasına bakın.

## 🌐 Canlı Demo

Uygulamanın çalışan versiyonuna Hugging Face Spaces üzerinden erişebilirsiniz:

🔗 [Form Rehberi App - Hugging Face Space](https://huggingface.co/spaces/oguzhan-evci/FormRehberimApp)
