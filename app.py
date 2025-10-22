
import os
import warnings
import logging
import sys
import contextlib
import time
import re
import traceback
import glob
import markdown # For markdown to HTML conversion

from flask import Flask, render_template, request, session, redirect, url_for
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- SESSİZ MOD AYARLARI (Deployment için ayarlar kaldırılabilir veya basitleştirilebilir) ---
# Genellikle Hugging Face Spaces'de varsayılan loglama yeterlidir, ancak istenirse tutulabilir.
# warnings.filterwarnings('ignore')
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# logging.basicConfig(level=logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
# try:
#     import absl.logging as absl_logging
#     absl_logging.set_verbosity(absl_logging.ERROR)
# except ImportError:
#     pass

# KAGGLENOTES: Hücre 6'da tanımlanan global değişkenler
llm = None
vector_store = None
retriever = None
qa_chain_with_history = None
embeddings = None # embeddings objesini de global olarak tanımlayalım

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'super_secret_key_form_rehberim_v2_exlist_deploy' # Yeni bir key kullanalım

# --- LANG_DATA ---
LANG_DATA = {
    'tr': {
        'title': 'Form Rehberim',
        'chatbot_title': 'Form Rehberim',
        'chatbot_subtitle': 'Hareketlerin nasıl yapıldığını ve inceliklerini sorun.',
        'welcome_message': 'Merhaba! Ben sizin Form Rehberinizim. Squat, Plank, Lunge gibi hareketlerin nasıl yapıldığı, hangi kasları çalıştırdığı gibi konularda bilgi almak için bana spesifik egzersiz adını sorabilirsiniz. Size özel antrenman programları oluşturamam, ancak mevcut egzersizler hakkında detaylı bilgi verebilirim. Hangi hareketi merak ediyorsunuz?',
        'loading_text': 'Cevap Aranıyor...',
        'input_placeholder': 'Örn: Squat nasıl yapılır? / Plank hangi kasları çalıştırır?',
        'send_button': 'Gönder',
        'clear_chat_button': 'Sohbeti Temizle',
        'nav_home': 'Ana Sayfa',
        'nav_exercises': 'Egzersiz Listesi',
        'nav_about': 'Hakkında',
        'about_title': 'Hakkında - Form Rehberim',
        'about_page_heading': 'Form Rehberim Hakkında',
        'about_paragraph_1': 'Bu yapay zeka asistanı, sadece evde veya istediğiniz yerde yapabileceğiniz temel vücut ağırlığı egzersizleri (Squat, Plank, Lunge vb.) hakkında bilgi sağlamak amacıyla tasarlanmıştır. Hareketlerin doğru yapılışı ve temel detayları hakkında sorular sorabilirsiniz.',
        'about_paragraph_2': 'Amacımız, temel egzersizleri doğru formda öğrenmenize yardımcı olarak daha bilinçli hareket etmenizi sağlamaktır. Bu asistan, size özel antrenman programları oluşturmaz veya kişisel fitness tavsiyesi vermez, yalnızca mevcut egzersiz kütüphanesindeki bilgileri sunar.',
        'about_contact_heading': 'Geri Bildirim',
        'about_contact_info': 'Uygulama hakkındaki düşüncelerinizi veya karşılaştığınız sorunları belirtirseniz sevinirim.',
        'back_to_chat': 'Sohbete Geri Dön',
        'error_message': 'Üzgünüm, bir hata oluştu: {error}',
        'chatbot_not_ready_message': 'Chatbot bileşenleri henüz hazır değil. Lütfen bir süre bekleyin veya hata loglarını kontrol edin.',
        'exercise_list_title': 'Egzersiz Listesi',
        'exercise_list_intro': 'Aşağıda hakkında bilgi alabileceğiniz egzersizlerin listesini bulabilirsiniz:'
    },
    'en': {
        'title': 'Form Guide',
        'chatbot_title': 'Form Guide',
        'chatbot_subtitle': 'Ask how exercises are done and their intricacies.',
        'welcome_message': 'Hello! I am your Form Guide Assistant. You can ask me for the name of a specific exercise to get information on topics such as how movements like Squat, Plank, Lunge are performed and which muscles they work. I cannot create personalized training programs for you, but I can provide detailed information about existing exercises. Which movement are you curious about?',
        'loading_text': 'Searching for an answer...',
        'input_placeholder': 'E.g., How to do a Squat? / What muscles does Plank work?',
        'send_button': 'Send',
        'clear_chat_button': 'Clear Chat',
        'nav_home': 'Home',
        'nav_exercises': 'Exercise List',
        'nav_about': 'About',
        'about_title': 'About - Form Guide',
        'about_page_heading': 'About Form Guide',
        'about_paragraph_1': 'This AI assistant is designed only to provide information about basic bodyweight exercises (Squat, Plank, Lunge, etc.) that you can do at home or anywhere. You can ask questions about the correct execution and basic details of the movements.',
        'about_paragraph_2': 'Our aim is to help you perform basic exercises with the correct form, enabling you to move more consciously. This assistant does not create personalized training programs or provide personal fitness advice, it only presents information from the existing exercise library.',
        'about_contact_heading': 'Feedback',
        'about_contact_info': 'I would appreciate it if you could share your thoughts about the application or any issues you encountered.',
        'back_to_chat': 'Back to Chat',
        'error_message': 'Sorry, an error occurred: {error}',
        'chatbot_not_ready_message': 'The chatbot components are not ready yet. Please wait a moment or check the error logs.',
        'exercise_list_title': 'Exercise List',
        'exercise_list_intro': 'Below you can find the list of exercises you can ask about:'
    }
}

def convert_markdown_to_html(md_text):
    try:
        # 'nl2br' extension'ı yeni satırları <br> etiketine çevirir.
        # 'fenced_code' code bloklarını doğru işler.
        # 'safe_mode' artık markdown kütüphanesinde kullanılmıyor, kaldırıldı.
        # Güvenlik için sanitize=True kullanabiliriz (Python Markdown 3.0+ için)
        html = markdown.markdown(md_text, extensions=['fenced_code', 'nl2br'], sanitize=True)
        return html
    except Exception as e:
        print(f"Markdown dönüşüm hatası: {e}")
        return md_text

def get_exercise_list():
    # Hugging Face'de veri setini root dizine kopyalayacağımız için yol değişecek
    ansiklopedi_dir = "hareket_ansiklopedisi" # Hugging Face'de root altında olacak

    exercise_files = []
    if os.path.exists(ansiklopedi_dir):
        pattern = os.path.join(ansiklopedi_dir, "*.md")
        files = glob.glob(pattern)
        for f_path in files:
            f_name = os.path.basename(f_path)
            exercise_name = os.path.splitext(f_name)[0]
            display_name = exercise_name.replace('-', ' ').replace('_', ' ').title()
            # İlk harfin büyük olduğundan emin olalım (örn: "plank" -> "Plank")
            if not display_name and exercise_name: # Boşsa ve orijinal isim varsa
                display_name = exercise_name.title()
            elif display_name and not display_name[0].isupper(): # İlk harf küçükse düzelt
                display_name = display_name[0].upper() + display_name[1:]
            exercise_files.append(display_name)
        exercise_files.sort()
    return exercise_files

# --- BAĞIMLILIK YÜKLEME VE RAG ZİNCİRİ OLUŞTURMA İŞLEMİ ---
# Uygulama başlatıldığında sadece bir kez çalışacak.
# Bu fonksiyon, Flask'ın ilk isteği almasından önce çağrılmalı,
# veya ilk isteğe hazır olması için Flask context'i dışında çağrılmalı.
def initialize_rag_components():
    global llm, vector_store, retriever, qa_chain_with_history, embeddings

    print("RAG bileşenleri başlatılıyor...")
    try:
        api_key = os.getenv("GOOGLE_API_KEY") # Hugging Face Secrets'tan alınacak
        if not api_key:
            print("HATA: GOOGLE_API_KEY ortam değişkeni bulunamadı. Lütfen Hugging Face Secrets'ı ayarlayın.")
            return

        os.environ["GOOGLE_API_KEY"] = api_key # LangChain için de ayarlı kalsın

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.75,
            top_p=0.9
        )
        print("Gemini modeli 'gemini-2.5-flash' olarak ayarlandı.")

        # LangChainDeprecationWarning'i gidermek için yeni paket kullanılmalı
        # Ancak mevcut setup'a göre devam ediyoruz, sadece uyarıyı göz ardı ediyoruz.
        warnings.filterwarnings('ignore', category=UserWarning, module='langchain_community.embeddings.huggingface')
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Embedding modeli yüklendi.")

        # Vektör veritabanının olduğu klasörün yolu (Uygulama root'unda olacak)
        faiss_index_path = "faiss_exercise_index"
        if os.path.exists(faiss_index_path):
            vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            print("Egzersiz FAISS veritabanı başarıyla yüklendi.")
        else:
            print(f"HATA: FAISS indeksi '{faiss_index_path}' bulunamadı. Lütfen yüklediğinizden emin olun.")
            return

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        print("Tek (Egzersiz) retriever başarıyla oluşturuldu (k=3).")

        history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Yukarıdaki konuşmaya dayanarak, sadece son soruyu cevaplamak için vektör veritabanında arama yapmaya uygun, tek başına bir sorgu cümlesi oluştur.")
        ])

        # Hata veren kısım düzeltildi: İç tırnaklar tek tırnakla veya escape edilerek kullanıldı.
        exercise_info_prompt_template = '''**SENİN ROLÜN:** Sen Active Break Egzersiz Asistanısın. **Pozitif, teşvik edici** ve bilgiyi **kendi özgün üslubuyla sentezleyerek** açıklayan bir uzmansın. Amacın sadece bilgi vermek değil, kullanıcıyı motive etmek ve ona net bir anlayış sunmaktır. Sen bir **bilgi kaynağısın**, program oluşturmaz veya egzersiz önermezsin.

**ANA GÖREVİN:** Kullanıcının girdisini analiz et ve aşağıdaki süreci izleyerek yanıtını oluştur:

1.  **NİYETİ BELİRLE:** Kullanıcı sohbet mi ediyor, spesifik bir egzersiz mi soruyor, yoksa belirsiz bir istekte mi bulunuyor?
    * **Sohbetse:** Bağlamı (`Referans Bilgiler`) yok say. Kullanıcının ifadesine uygun, kısa, nazik ve **yardımcı olmaya odaklı** bir yanıt ver. Yanıtında, kullanıcıyı bir egzersiz hakkında soru sormaya **nazikçe teşvik et**. Bunu yaparken **doğal ol ve ifadelerini çeşitlendir**. İşte *bazı* örnekler, bunlara benzer ama **kendi kelimelerinle** yanıtlar üretebilirsin:
        * "Merhaba! Size hangi egzersiz hakkında bilgi verebilirim?"
        * "Selam! Aklınızdaki egzersizi sormaktan çekinmeyin."
        * "Rica ederim! Başka bir egzersizle ilgili sorunuz olursa buradayım."
        * "Ben hazırım! Hangi egzersizi merak ediyorsunuz?"
        -> **Bitir.**
    * **Soru/İstekse:** Devam et.

2.  **REFERANS BİLGİYİ ANALİZ ET (Soru/İstekse):**
    * 'Referans Bilgiler (Bağlam)' senin **TEK BİLGİ KAYNAĞINDIR**.
    * Kullanıcının sorusuyla **doğrudan ilgili** bilgileri **belirle ve özümse**. Alakasızları **filtrele**.

3.  **ÖZGÜN VE DEĞER KATAN CEVABINI YARAT (Soru/İstekse - EN KRİTİK ADIM):**
    * **Eğer Alakalı Referans Varsa:**
        * **Sıfırdan Yeniden Yaz:** Özümsediğin bilgileri kullanarak, cevabı **tamamen kendi kelimelerinle, farklı cümle yapılarıyla ve kendi anlatım tarzınla sıfırdan oluştur**. Referans metnin ifadelerinden ve yapısından **belirgin şekilde farklılaş**. Sadece kelimeleri değiştirmek yetmez, bilgiyi **işle** ve **kendi yorumunu kat**.
        * **Sadece Sorulana Odaklan:** Kullanıcı ne sorduysa **yalnızca** onu cevapla. Ekstra bilgi ekleme, kullanıcı açıkça sormadığı sürece.
        * **Sıcak Bir Kapanış Ekle:** Cevabının sonuna, açıklanan egzersizle ilgili **kısa, pozitif ve teşvik edici** bir cümle ekle (örn: "Doğru formla harika sonuçlar alacağına eminim!", "Bu hareket core bölgen için müthiş!", "Başlangıç için harika bir seçim!").
    * **Eğer Alakalı Referans Yoksa VEYA Soru Belirsizse:**
        * **KESİNLİKLE GENEL BİLGİNİ KULLANMA.**
        * **Belirsiz Soruları Yönet:** Eğer kullanıcı "bacak egzersizleri" gibi genel bir şey soruyorsa, nazikçe şunu belirt: "Harika bir istek! Ancak veri setimizde genel 'bacak egzersizleri' listesi yerine spesifik hareketler bulunuyor. Örneğin **Squat** veya **Lunge** hakkında bilgi verebilirim. Hangisini öğrenmek istersin?"
        * **Bilgi Yoksa (Spesifik Soru İçin):** Eğer spesifik bir egzersiz sorulduysa ve referansta yoksa, şunu söyle: "Üzgünüm, '[Kullanıcının Sorduğu Konu]' hakkında sağlanan bilgilerde detay bulamadım. Belki Squat, Plank veya Lunge gibi temel hareketlerden birini sormak istersin?"

4.  **FORMATLAMA VE ÜSLUP:**
    * Cevabını **basit Markdown** (örn: **kalın**, `- liste`) ile okunaklı yap.
    * Her zaman **pozitif, teşvik edici ve net** ol.
    * Tıbbi tavsiye verme. Egzersiz önerme (sadece sorulanı açıkla).

**Referans Bilgiler (Bağlam) (TEK BİLGİ KAYNAĞIN):**
{context}

**Kullanıcının Girdisi:** {input}
**SENİN ÖZGÜN, POZİTİF VE ODAKLI CEVABIN:**'''
        exercise_info_prompt = ChatPromptTemplate.from_messages([
            ("system", exercise_info_prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        history_aware_retriever_chain = create_history_aware_retriever(
            llm, retriever, history_aware_retriever_prompt
        )
        answer_generation_chain = create_stuff_documents_chain(llm, exercise_info_prompt)
        qa_chain_with_history = create_retrieval_chain(
            history_aware_retriever_chain, answer_generation_chain
        )
        print("HAFIZALI RAG zinciri (qa_chain_with_history) başarıyla oluşturuldu.")

    except Exception as e:
        print(f"HATA: RAG bileşenleri veya HAFIZALI zincir oluşturulurken sorun oluştu: {e}")
        print(traceback.format_exc())

# Flask rotaları
@app.route('/set_language/<lang_code>')
def set_language(lang_code):
    if lang_code in LANG_DATA:
        session['lang'] = lang_code
    return redirect(request.referrer or url_for('home'))

@app.route('/about')
def about():
    lang = session.get('lang', 'tr')
    lang_data = LANG_DATA[lang]
    return render_template('about.html', lang=lang, lang_data=lang_data)

@app.route('/egzersizler')
def exercise_list():
    lang = session.get('lang', 'tr')
    lang_data = LANG_DATA[lang]
    exercises = get_exercise_list()
    print(f"{len(exercises)} adet egzersiz listelendi.")
    return render_template('egzersizler.html', exercises=exercises, lang=lang, lang_data=lang_data)

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    session.pop('chat_history', None)
    print("Sohbet geçmişi temizlendi.")
    return redirect(url_for('home'))

@app.route('/', methods=['GET', 'POST'])
def home():
    global qa_chain_with_history

    lang = session.get('lang', 'tr')
    lang_data = LANG_DATA[lang]
    simple_chat_history = session.get('chat_history', [])
    question = ""
    answer_html = ""

    if request.method == 'POST':
        question = request.form.get('question', '').strip()

        if not question:
            return render_template('index.html', chat_history=simple_chat_history, lang=lang, lang_data=lang_data)

        # Kullanıcı sorusunu geçmişe eklemeden önce, botun hazır olup olmadığını kontrol et
        if not qa_chain_with_history:
            answer = lang_data['chatbot_not_ready_message']
            answer_html = convert_markdown_to_html(answer)
            simple_chat_history.append((question, answer_html))
            session['chat_history'] = simple_chat_history
            return render_template('index.html', chat_history=simple_chat_history, lang=lang, lang_data=lang_data)

        simple_chat_history.append((question, None)) # Kullanıcı sorusunu geçici olarak ekle
        session['chat_history'] = simple_chat_history

        try:
            print(f"
--- Hafızalı RAG Zinciri Çalıştırılıyor ---")
            print(f"Sorgu: '{question}'")
            langchain_chat_history = []
            for q, a_html in simple_chat_history[:-1]: # Son soruyu çıkar, sadece geçmişi gönder
                if q and a_html:
                    # HTML'i temizleyip sadece metin kısmını LLM'e gönder
                    a_text = re.sub('<[^<]+?>', '', a_html) if a_html else ""
                    langchain_chat_history.append(HumanMessage(content=q))
                    langchain_chat_history.append(AIMessage(content=a_text))
            print(f">>> LLM ÇAĞRISI (HAFIZALI QA) BAŞLIYOR...")
            start_time = time.time()
            result = qa_chain_with_history.invoke({"input": question, "chat_history": langchain_chat_history })
            end_time = time.time()
            print(f"<<< LLM ÇAĞRISI (HAFIZALI QA) BAŞARILI ({end_time - start_time:.2f} saniye).")
            raw_answer = result['answer']
            answer_html = convert_markdown_to_html(raw_answer)
            print("Cevap Markdown'dan HTML'e çevrildi.")
            
            # Son cevabı geçmişe ekle veya güncelle
            if simple_chat_history: 
                simple_chat_history[-1] = (question, answer_html) # Son eklenen sorunun cevabını güncelle
            else: # Eğer geçmiş boşsa ve yeni bir soru-cevap oluştuysa
                simple_chat_history.append((question, answer_html))
            
            session['chat_history'] = simple_chat_history

        except Exception as e:
            print(f"HATA: Hafızalı RAG zinciri çalıştırılırken hata: {e}")
            print(traceback.format_exc())
            answer = lang_data['error_message'].format(error=str(e))
            answer_html = convert_markdown_to_html(answer)
            if simple_chat_history: simple_chat_history[-1] = (question, answer_html)
            session['chat_history'] = simple_chat_history

        return render_template('index.html', chat_history=simple_chat_history, question=question, answer=answer_html, lang=lang, lang_data=lang_data)

    # GET isteği
    return render_template('index.html', chat_history=simple_chat_history, lang=lang, lang_data=lang_data)


# Uygulama başlatıldığında RAG bileşenlerini yükle
# Bu, Flask uygulamasının başlangıcında sadece bir kez çalışacak.
# __name__ == '__main__' kontrolü, uygulamanın doğrudan çalıştırıldığında bu bloğun çalışmasını sağlar.
if __name__ == '__main__':
    initialize_rag_components() # RAG bileşenlerini önceden yükle
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 7860))) # Hugging Face Spaces varsayılan portu 7860
