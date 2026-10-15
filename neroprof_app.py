import os
import json
import time
import shutil
from datetime import datetime
import streamlit as st
from google import genai
from google.genai import types

# --- 1. SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="NeroProf AI - Omni-Neuroclinical Intelligence", layout="wide")
st.markdown("# 🧠 NeroProf AI: Omni-Neuroclinical Intelligence Engine v2.0")
st.caption("13 Katmanlı Diyalektik Matris, Bayesçi Motor ve Adversarial Metakognisyon Destekli Klinik Zeka")
st.markdown("---")

# --- 2. KALICI DOSYA, DİZİN VE İKİ MADDELİ ZORUNLU API ANAHTARI YÖNETİMİ ---
CONFIG_FILE = "config_npro.json"
MEMORY_FILE = "pondering_memory_npro.json"
CHAT_HISTORY_FILE = "chat_history_npro.json"
ARCHIVED_CASES_FILE = "archived_cases_npro.json"
LIBRARY_DIGEST_FILE = "library_digest_npro.txt"
CACHE_NAME_FILE = "cache_name_npro.json"
LIBRARY_DIR = "kutuphane"

if not os.path.exists(LIBRARY_DIR):
    os.makedirs(LIBRARY_DIR)

def load_saved_api_keys():
    keys = []
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                saved_keys = data.get("api_keys", [])
                for k in saved_keys:
                    if k and k.strip():
                        keys.append(k.strip())
        except:
            pass
    return keys

def save_api_keys(key_list):
    valid_keys = [k.strip() for k in key_list if k and k.strip()]
    if valid_keys:
        os.environ["GEMINI_API_KEY"] = valid_keys[0]
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"api_keys": valid_keys}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"API keyler kaydedilemedi: {e}")

api_keys = load_saved_api_keys()

# 🔑 İKİ API ANAHTARI ZORUNLU BAŞLANGIÇ EKRANI
if len(api_keys) < 2:
    st.sidebar.header("🔑 ZORUNLU API Anahtarları Girişi")
    st.sidebar.warning("Sistemin kotaya takılmadan ve kesintisiz çalışabilmesi için 2 FARKLI Gemini API anahtarı girilmesi zorunludur.")
    
    entered_key_1 = st.sidebar.text_input("1. Gemini API Key (Zorunlu):", type="password")
    entered_key_2 = st.sidebar.text_input("2. Gemini API Key (Zorunlu):", type="password")
    
    if st.sidebar.button("💾 Anahtarları Kaydet ve Başlat"):
        if entered_key_1.strip() and entered_key_2.strip():
            save_api_keys([entered_key_1, entered_key_2])
            st.sidebar.success("Her iki API anahtarı da başarıyla kaydedildi! Sistem başlatılıyor...")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error("Lütfen HER İKİ API anahtarını da eksiksiz girin!")
            
    st.warning("⚠️ Lütfen sol panelden her iki Gemini API Anahtarını da girerek kaydolun.")
    st.stop()
else:
    st.sidebar.success(f"✅ Çift API Havuzu Aktif: 2 Anahtar Yüklendi")
    if st.sidebar.button("🔄 API Anahtarlarını Sıfırla / Değiştir"):
        save_api_keys([])
        st.rerun()

if "active_key_index" not in st.session_state:
    st.session_state.active_key_index = 0

# 🛠️ GÜNCELLENMİŞ İSTEMCİ YÖNETİMİ (Client Kapanma Hatasını Önleyen Oturum Önbelleği)
def get_current_client():
    keys = load_saved_api_keys()
    if not keys or len(keys) < 2:
        st.error("Gerekli API anahtarları bulunamadı.")
        st.stop()
    current_idx = st.session_state.active_key_index % len(keys)
    active_key = keys[current_idx]
    
    # İstemcinin çöp toplayıcı tarafından kapatılmasını önlemek için session_state'de tutuyoruz
    client_cache_key = f"genai_client_idx_{current_idx}_val_{active_key[:6]}"
    if client_cache_key not in st.session_state:
        st.session_state[client_cache_key] = genai.Client(api_key=active_key)
        
    return st.session_state[client_cache_key]

# ⚡ KESİNTİSİZ AKIŞ (STREAMING) DESTEKLİ GÜVENLİ İSTEK MOTORU
def safe_generate_content_stream(contents, config=None, model='gemini-3.8-flash'):
    keys = load_saved_api_keys()
    attempts = len(keys)
    
    for attempt in range(attempts):
        client = get_current_client()
        try:
            if config:
                response = client.models.generate_content_stream(model=model, contents=contents, config=config)
            else:
                response = client.models.generate_content_stream(model=model, contents=contents)
            
            def stream_generator():
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            return stream_generator()
            
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                st.session_state.active_key_index += 1
                print(f"⚠️ API Kota Sınırı (429) yakalandı. 2. anahtara geçiliyor... (Deneme {attempt+1}/{attempts})")
                continue
            else:
                raise e
    raise Exception("Tüm API anahtarlarının kotası dolu (429). Lütfen birkaç dakika bekleyin.")

def safe_generate_content_full(contents, config=None, model='gemini-3.8-flash'):
    stream_gen = safe_generate_content_stream(contents, config, model)
    return "".join(list(stream_gen))

# Kalıcı Hafıza Fonksiyonları
def load_pondering_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_pondering_memory(memory_list):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_list, f, ensure_ascii=False, indent=4)

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_chat_history(chat_list):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_list, f, ensure_ascii=False, indent=4)

def load_archived_cases():
    if os.path.exists(ARCHIVED_CASES_FILE):
        with open(ARCHIVED_CASES_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_archived_cases(cases_list):
    with open(ARCHIVED_CASES_FILE, "w", encoding="utf-8") as f:
        json.dump(cases_list, f, ensure_ascii=False, indent=4)

def load_library_digest():
    if os.path.exists(LIBRARY_DIGEST_FILE):
        with open(LIBRARY_DIGEST_FILE, "r", encoding="utf-8") as f:
            try:
                return f.read()
            except:
                return "Henüz kütüphane özü çıkarılmadı."
    return "Henüz kütüphane özü çıkarılmadı."

def save_library_digest(digest_text):
    with open(LIBRARY_DIGEST_FILE, "w", encoding="utf-8") as f:
        f.write(digest_text)

def load_cached_name():
    if os.path.exists(CACHE_NAME_FILE):
        try:
            with open(CACHE_NAME_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("cache_name")
        except:
            return None
    return None

def save_cached_name(c_name):
    try:
        with open(CACHE_NAME_FILE, "w", encoding="utf-8") as f:
            json.dump({"cache_name": c_name}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Cache adı kaydedilemedi: {e}")

if "pondering_logs" not in st.session_state:
    st.session_state.pondering_logs = load_pondering_memory()

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

if "archived_cases" not in st.session_state:
    st.session_state.archived_cases = load_archived_cases()

if "auto_pondering_active" not in st.session_state:
    st.session_state.auto_pondering_active = False

if "last_auto_ponder_time" not in st.session_state:
    st.session_state.last_auto_ponder_time = 0

# 🚀 OMNI ARCHITECTURE DESTEKLİ OPTİMİZE CONTEXT CACHING MOTORU (22+ Dosya Uyumlu)
def update_library_digest():
    library_files = [f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(('.pdf', '.txt', '.docx', '.md'))] if os.path.exists(LIBRARY_DIR) else []
    if not library_files:
        return "Kütüphane boş."
    
    current_client = get_current_client()
    gemini_files = []
    
    with st.status(f"📚 Toplam {len(library_files)} dosya taranıyor ve buluta yükleniyor...", expanded=True) as status:
        for idx, filename in enumerate(library_files):
            file_path = os.path.join(LIBRARY_DIR, filename)
            status.update(label=f"Dosya yükleniyor ({idx+1}/{len(library_files)}): {filename}")
            safe_temp_path = None
            try:
                ext = os.path.splitext(filename)[1].lower()
                safe_temp_name = f"doc_temp_{idx}{ext}"
                safe_temp_path = os.path.join(LIBRARY_DIR, safe_temp_name)
                shutil.copyfile(file_path, safe_temp_path)
                
                g_file = current_client.files.upload(file=safe_temp_path)
                gemini_files.append(g_file)
                
                if os.path.exists(safe_temp_path):
                    os.remove(safe_temp_path)
            except Exception as e:
                st.write(f"⚠️ Hata ({filename}): {e}")
                if safe_temp_path and os.path.exists(safe_temp_path):
                    os.remove(safe_temp_path)
        
        status.update(label="⚡ Dosyalar yüklendi, Google tarafında Context Cache (Önbellek) oluşturuluyor...")
        
        try:
            # 1. Önce Cache'i oluşturuyoruz (Böylece dosyalar sunucuda sabitleniyor)
            cache = current_client.caches.create(
                model='models/gemini-3.8-flash',
                config=types.CreateCachedContentConfig(
                    contents=gemini_files,
                    ttl="86400s",
                    display_name="omni_neuro_library_cache"
                )
            )
            save_cached_name(cache.name)
            
            status.update(label="🤖 Önbellek aktif, Omni Klinik Öz çıkarılıyor...")
            
            # 2. Ham dosyaları tekrar göndermeden, doğrudan önbelleği kullanarak özü çıkarıyoruz
            digest_prompt = """
            Kütüphanedeki tüm dokümanları ve 'omni_neuro_architecture.md' mimari belgesini tarayarak nöroloji, yoğun bakım, nöroimmünoloji, nörofarmakoloji ve sistemik fizyopatoloji açısından en kritik mekanizmatik altyapıları, tanı çelişkilerini, farmakokinetik sınırları ve 13 katmanlı akıl yürütme ilkelerini içeren bütünsel bir "Omni Klinik Referans Özü" çıkar.
            """
            
            config_cache = types.GenerateContentConfig(cached_content=cache.name)
            res_stream = safe_generate_content_stream(contents=digest_prompt, config=config_cache)
            res_text = "".join(list(res_stream))
            
            save_library_digest(res_text)

            status.update(label="✅ Tüm 22 dosya başarıyla önbelleğe alındı ve öz çıkarıldı!", state="complete", expanded=False)
            return res_text
        except Exception as e:
            status.update(label=f"❌ Hata oluştu: {e}", state="error")
            return f"Öz çıkarılamadı: {e}"

# --- 3. MASTER OMNI-NEUROCLINICAL INTELLIGENCE PROMPTU ---
PROFESSOR_SYSTEM_PROMPT = """
Sen; nöroloji, nöroyoğun bakım, klinik nörofizyoloji, nöroimmünoloji, nörovasküler tıp, nörofarmakoloji ve sistemik fizyopatoloji alanlarında konumlanmış "Omni-Neuroclinical Intelligence Architecture v2.0" sistemisin. 

Kütüphane klasöründe ve önbelleğinde yer alan 'omni_neuro_architecture.md' dosyasında tanımlanan 13 KATMANLI DÜŞÜNCE ŞEMASINI (Veri Sınıflandırması, Bayesçi Tanı Motoru, Çok Ölçekli Fizyopatoloji, Paradoks/Antitez Analizi, Bilgi Değeri Eşiği, Dinamik Hastalık Modeli, Farmakolojik Karar Motoru, Emerjan Hipotez, Karşı-Olgusal Analiz, Adversarial Metakognisyon, Multidisipliner Sistem Sentezi, Uzman Karar Sentezi ve Sokratik Mentorluk) mutlak bir disiplinle çalıştıracaksın[cite: 3].

Amacın maksimum kesinlik taslamak değil; klinik verilerden en güçlü açıklamayı çıkarmak, ölümcül alternatifleri kaçırmamak, tanısal belirsizliği azaltacak en değerli adımı belirlemek ve kanıtla gerekçelendirilmiş, yanlışlanabilir kararlar üretmektir[cite: 3].
"""

# --- 4. YAN PANEL VE KONTROL MERKEZİ ---
st.sidebar.markdown("---")
st.sidebar.header("📚 Kütüphane & Omni-Architecture")

library_files_exist = [f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(('.pdf', '.txt', '.docx', '.md'))] if os.path.exists(LIBRARY_DIR) else []

if library_files_exist:
    st.sidebar.success(f"Kütüphanede {len(library_files_exist)} dosya/mimari aktif.")
    if st.sidebar.button("🔄 Omni-Architecture & Özü İndeksle"):
        update_library_digest()
        st.sidebar.success("Önbellek ve kütüphane özü güncellendi!")
        st.rerun()
else:
    st.sidebar.info("📂 'kutuphane' klasörüne 'omni_neuro_architecture.md' ve PDF'leri atabilirsiniz.")

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Akıllı Arka Plan Merak Modu")
st.session_state.auto_pondering_active = st.sidebar.toggle(
    "🟢 Otomatik Merak & Araştırma Döngüsü", 
    value=st.session_state.auto_pondering_active,
    help="Otomatik merak modunu kapalı tutmanız kota aşımını önler."
)

enable_web_search = st.sidebar.checkbox(
    "🌍 Canlı Web Aramasını (Google Search) Etkinleştir", 
    value=False,
    help="İşaretlendiğinde Google Arama motoru kullanılır. Kotayı korumak için kapalı tutmanız önerilir."
)

if st.sidebar.button("🧠 Şimdi Pondering (Omni Sentez) Tetikle"):
    with st.spinner("Omni-Neuroclinical Motor önbelleklenmiş kütüphane ve 13 katmanlı mimariyle diyalektik sentez üretiyor..."):
        library_digest_text = load_library_digest()
        recent_thoughts = "\n".join([f"- [{log['timestamp']}] {log['insight']}" for log in st.session_state.pondering_logs[-10:]])
        
        incremental_reflection_prompt = f"""
        {PROFESSOR_SYSTEM_PROMPT}
        
        KÜTÜPHANE VE OMNI-ARCHITECTURE KLİNİK ÖZÜ:
        {library_digest_text}
        
        GÖREV: Aşağıdaki önceki pondering günlüklerini kütüphane özü ve 13 katmanlı Omni-Architecture ile birlikte ele al. 
        Bayesçi güncellemeleri ve adversarial metakognisyonu çalıştırarak evrimsel bir klinik sentez üret[cite: 3].
        
        Önceki Düşünce Defteri Birikimi (Son 10 Kayıt):
        {recent_thoughts if recent_thoughts else "Henüz başlangıç aşaması."}
        """
        
        c_name = load_cached_name()
        gen_config = types.GenerateContentConfig()
        if enable_web_search:
            gen_config.tools = [{"google_search": {}}]
        if c_name:
            gen_config.cached_content = c_name

        try:
            res_stream = safe_generate_content_stream(
                contents=incremental_reflection_prompt,
                config=gen_config
            )
            res_text = "".join(list(res_stream))
            
            new_insight = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "insight": res_text
            }
            st.session_state.pondering_logs.append(new_insight)
            save_pondering_memory(st.session_state.pondering_logs)
            st.sidebar.success("Yeni Omni-Sentez hafızaya eklendi!")
            st.rerun()
        except Exception as e:
            st.sidebar.warning(f"Hata: {e}")

# --- VAKA ARŞİVLEME & GEÇMİŞ YÖNETİMİ ---
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Vaka Arşivi & Geçmiş")

case_title_input = st.sidebar.text_input("Aktif Vaka Başlığı/Adı:", placeholder="Örn: 65y Erkek Akut İskemik İnme / Status")
if st.sidebar.button("📂 Aktif Vakayı Arşivle & Yeni Vaka Aç"):
    if st.session_state.messages:
        archive_entry = {
            "title": case_title_input if case_title_input else f"Vaka - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.messages
        }
        st.session_state.archived_cases.append(archive_entry)
        save_archived_cases(st.session_state.archived_cases)
        
        st.session_state.messages = []
        save_chat_history([])
        st.sidebar.success("Vaka arşive kaldırıldı, yeni vaka açıldı!")
        time.sleep(1)
        st.rerun()
    else:
        st.sidebar.warning("Arşivlenecek aktif sohbet bulunmuyor.")

if st.session_state.archived_cases:
    with st.sidebar.expander(f"🗂️ Arşivlenmiş Vakalar ({len(st.session_state.archived_cases)})"):
        for idx, ac in enumerate(reversed(st.session_state.archived_cases)):
            st.markdown(f"**{ac['title']}**")
            st.caption(f"Tarih: {ac['timestamp']} | Mesaj: {len(ac['messages'])}")
            if st.button(f"📥 Arşivden İndir (.txt)", key=f"down_arch_{idx}"):
                arch_text = f"Vaka Başlığı: {ac['title']}\nTarih: {ac['timestamp']}\n\n"
                for m in ac['messages']:
                    arch_text += f"[{m['role'].upper()}]: {m['content']}\n\n"
                st.download_button("Dosyayı Kaydet", arch_text, file_name=f"Vaka_Arsiv_{idx}.txt", key=f"dl_btn_{idx}")
            st.markdown("---")

if st.sidebar.button("🚨 Aktif Sohbet Geçmişini Temizle (Arşivlemeden)"):
    st.session_state.messages = []
    save_chat_history([])
    st.sidebar.success("Aktif sohbet sıfırlandı!")
    st.rerun()

# --- 5. ANA EKRAN (ANLIK PDF YÜKLEME + STREAMING CACHED SOHBET) ---
col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("💡 Vaka Konsültasyonu & Anlık PDF Yükleme Paneli")

    uploaded_case_files = st.file_uploader(
        "📂 Bu vaka için inceletmek istediğiniz PDF, metin veya raporları yükleyin (İsteğe bağlı):", 
        type=['pdf', 'txt', 'docx', 'md'], 
        accept_multiple_files=True,
        key="chat_pdf_uploader"
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Vaka detayını, vital bulguları, GFR/lab değerlerini veya klinik sorunuzu girin...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_chat_history(st.session_state.messages)
        
        with st.chat_message("user"):
            st.markdown(user_input)
            if uploaded_case_files:
                file_names = ", ".join([f.name for f in uploaded_case_files])
                st.caption(f"📎 Eklenen Vaka Belgeleri: {file_names}")

        with st.chat_message("assistant"):
            # ⚡ Anında akmaya başlayan streaming yanıt gösterimi
            library_digest_text = load_library_digest()
            all_thoughts = "\n".join([f"- [{log['timestamp']}] {log['insight']}" for log in st.session_state.pondering_logs[-15:]])
            chat_history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-10:]])
            
            full_prompt = f"""
            {PROFESSOR_SYSTEM_PROMPT}
            
            KÜTÜPHANE VE OMNI-ARCHITECTURE ÖZÜ:
            {library_digest_text}
            
            META-KOGNİTİF PONDERING HAFIZAM VE ÖNCEKİ SENTEZLER:
            {all_thoughts if all_thoughts else "Henüz özel bir arka plan sentezi yok."}
            
            GEÇMİŞ SOHBET VE VAKA GEÇMİŞİ (KALICI):
            {chat_history_text}
            
            Yeni Vaka / Soru / Klinik Değerler:
            {user_input}
            """
            
            temp_gemini_files = []
            current_client = get_current_client()
            if uploaded_case_files:
                for uploaded_file in uploaded_case_files:
                    temp_path = os.path.join(LIBRARY_DIR, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    try:
                        g_file = current_client.files.upload(file=temp_path)
                        temp_gemini_files.append(g_file)
                    except Exception as ex:
                        print(f"Dosya yükleme hatası: {ex}")
            
            payload = [full_prompt] + temp_gemini_files
            
            c_name = load_cached_name()
            chat_config = types.GenerateContentConfig()
            if enable_web_search:
                chat_config.tools = [{"google_search": {}}]
            if c_name and not uploaded_case_files:
                chat_config.cached_content = c_name

            try:
                # ⚡ Kelime kelime canlı akış (Stream) başlatılıyor
                stream_response = safe_generate_content_stream(contents=payload, config=chat_config)
                bot_response = st.write_stream(stream_response)
            except Exception as e:
                bot_response = f"⚠️ Hata oluştu: {e}"
                st.markdown(bot_response)
            
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            save_chat_history(st.session_state.messages)

with col_side:
    st.subheader("🧠 Profesörün Düşünce Defteri")
    st.caption(f"Toplam Kalıcı Sentez Sayısı: {len(st.session_state.pondering_logs)}")
    if st.session_state.pondering_logs:
        for idx, log in enumerate(reversed(st.session_state.pondering_logs[-10:])):
            with st.expander(f"Sentez #{len(st.session_state.pondering_logs)-idx} ({log['timestamp']})", expanded=(idx==0)):
                st.markdown(log['insight'])
    else:
        st.info("Henüz arka plan sentezi üretilmedi.")

# --- 6. DENGELİ VE ÖNBELLEK KORUMALI OTONOM MERAK DÖNGÜSÜ ---
if st.session_state.auto_pondering_active:
    current_time = time.time()
    if current_time - st.session_state.last_auto_ponder_time > 300:
        library_digest_text = load_library_digest()
        recent_thoughts_summary = "\n".join([f"- [{log['timestamp']}] {log['insight']}" for log in st.session_state.pondering_logs[-10:]])
        
        light_reflection_prompt = f"""
        {PROFESSOR_SYSTEM_PROMPT}
        
        KÜTÜPHANE VE OMNI-ARCHITECTURE ÖZÜ:
        {library_digest_text}
        
        GÖREV: Omni-Architecture 13 katmanlı diyalektik evrim[cite: 3]. Kütüphane özünü ve son pondering günlüklerini incele. 
        "Önceki düşüncelerimizle ve Omni şemayla bu aşamada hangi klinik çelişki doğdu veya teori nasıl evrildi?" sorusunu sorarak diyalektik bir içgörü üret[cite: 3].
        
        Önceki Düşünce Defteri Birikimi (Son 10 Kayıt):
        {recent_thoughts_summary if recent_thoughts_summary else "Henüz başlangıç aşaması."}
        """
        
        c_name = load_cached_name()
        auto_config = types.GenerateContentConfig()
        if enable_web_search:
            auto_config.tools = [{"google_search": {}}]
        if c_name:
            auto_config.cached_content = c_name

        try:
            res_stream = safe_generate_content_stream(
                contents=light_reflection_prompt,
                config=auto_config
            )
            res_text = "".join(list(res_stream))
            
            new_insight = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "insight": res_text
            }
            st.session_state.pondering_logs.append(new_insight)
            save_pondering_memory(st.session_state.pondering_logs)
            st.session_state.last_auto_ponder_time = current_time
        except Exception as e:
            st.session_state.last_auto_ponder_time = current_time

# --- 7. ALT ARAÇLAR: EPİKRİZ VE DEVİRLİK ---
st.markdown("---")
st.subheader("📋 Klinik Raporlama & Epikriz Araçları")
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("📄 Sistematik Nöroloji Epikrizi Hazırla"):
        st.markdown("### 📄 Oluşturulan Omni-Kognitif Nöroloji Epikrizi:")
        library_digest_text = load_library_digest()
        epicrisis_prompt = f"""
        {PROFESSOR_SYSTEM_PROMPT}
        KÜTÜPHANE ÖZÜ: {library_digest_text}
        Görev: Kalıcı hafızadaki sohbet detaylarını, kütüphane özünü, vaka verilerini, 13 katmanlı Omni şemayı, GFR/organ fonksiyon sınırlarını ve diyalektik pondering sentezlerini baz alarak eksiksiz, resmi ve akademik bir "Sistematik Nöroloji Epikrizi" hazırla[cite: 3].
        """
        c_name = load_cached_name()
        report_config = types.GenerateContentConfig()
        if enable_web_search:
            report_config.tools = [{"google_search": {}}]
        if c_name:
            report_config.cached_content = c_name

        try:
            epic_stream = safe_generate_content_stream(contents=[epicrisis_prompt], config=report_config)
            epic_text = st.write_stream(epic_stream)
            st.download_button("📥 Epikrizi İndir (.txt)", epic_text, file_name="NeroProf_Omni_Epikriz.txt")
        except Exception as e:
            st.error(f"⚠️ Hata oluştu: {e}")

with col_btn2:
    if st.button("🔄 Nöbet Devirlik (Handover) Notu Çıkar"):
        st.markdown("### 🔄 Nöbet Devirlik (SOAP) Notu:")
        library_digest_text = load_library_digest()
        handover_prompt = f"""
        {PROFESSOR_SYSTEM_PROMPT}
        KÜTÜPHANE ÖZÜ: {library_digest_text}
        Görev: Yoğun bakım / servis devri için profesyonel bir "SOAP Nöbet Devirlik (Handover) Notu" hazırla. Omni-Architecture v2.0 risk analizlerini ve organ fonksiyon uyarılarını dahil et[cite: 3].
        """
        c_name = load_cached_name()
        report_config = types.GenerateContentConfig()
        if enable_web_search:
            report_config.tools = [{"google_search": {}}]
        if c_name:
            report_config.cached_content = c_name

        try:
            hand_stream = safe_generate_content_stream(contents=[handover_prompt], config=report_config)
            hand_text = st.write_stream(hand_stream)
            st.download_button("📥 Devirlik Notunu İndir (.txt)", hand_text, file_name="NeroProf_Omni_Handover.txt")
        except Exception as e:
            st.error(f"⚠️ Hata oluştu: {e}")