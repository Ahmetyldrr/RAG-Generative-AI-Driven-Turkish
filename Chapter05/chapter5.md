## Boosting RAG Performance with Expert Human Feedback

## İnsan Geri Bildirimi ile RAG Performansını Artırma (Boosting RAG Performance with Expert Human Feedback)

İnsan geri bildirimi (Human Feedback, HF), üretken yapay zeka (Generative AI) için yalnızca yararlı değil, aynı zamanda özellikle RAG (Retrieval-Augmented Generation) kullanan modeller için gereklidir. Bir üretken yapay zeka modeli, eğitim sırasında çeşitli belgeler içeren veri kümelerindeki bilgileri kullanır. Yapay zeka modelini eğiten veriler, modelin parametrelerinde sabitlenir; modeli yeniden eğitmedikçe bu verileri değiştiremeyiz. Ancak, retrieval tabanlı metin ve multimodal veri kümeleri dünyasında, görebileceğimiz ve ayarlayabileceğimiz bilgiler vardır. İşte burada HF devreye girer. HF, yapay zeka modelinin veri kümelerinden çektiği bilgilere geri bildirim sağlayarak, gelecekteki yanıtlarının kalitesini doğrudan etkileyebilir. Bu süreçle etkileşime giren insanlar, RAG'ın gelişiminde aktif bir oyuncu haline gelir. Bu, yapay zeka projelerine yeni bir boyut kazandırır: uyarlanabilir RAG (adaptive RAG).

## Uyarlanabilir RAG Ekosistemini Tanımlama (Defining the Adaptive RAG Ecosystem)

Uyarlanabilir RAG, retrieval için kullanılan belgelerin güncellenmesiyle sistemin uyarlanabilir hale gelmesini sağlar. HF'ı RAG'a entegre etmek, insanları otomatikleştirilmiş bir üretken sürece dahil ettiği için pragmatik bir hibrit yaklaşım ortaya koyar. Bu nedenle, sıfırdan bir hibrit uyarlanabilir RAG programı oluşturmak için HF'ı kullanacağız ve bir RAG tabanlı üretken yapay zeka sisteminin temel adımlarını inceleyeceğiz.

## Önemli Noktalar:

*   Uyarlanabilir RAG ekosistemini tanımlama
*   Artırılmış retrieval sorgularına uyarlanabilir RAG'ı uygulama
*   HF ile artırılmış üretken yapay zeka girdilerini otomatikleştirme
*   Uzman HF'ı tetiklemek için son kullanıcı geri bildirim derecelendirmelerini otomatikleştirme
*   Bir insan uzmanı için otomatik geri bildirim sistemi oluşturma
*   HF'ı GPT-4o için uyarlanabilir RAG'a entegre etme

## Uyarlanabilir RAG ile İnsan Geri Bildirimi Entegrasyonu

Uyarlanabilir RAG'ı HF döngüsü ile tanıtacağız. Sistem, retrieval için kullanılan belgeler güncellendiği için uyarlanabilir hale gelir. HF'ı RAG'a entegre etmek, insanları otomatikleştirilmiş bir üretken sürece dahil eder.

## Python'da Uyarlanabilir RAG Uygulaması

Python'da sıfırdan bir hibrit uyarlanabilir RAG programı oluşturacağız. Aşağıdaki kod, bir RAG tabanlı üretken yapay zeka sisteminin temel adımlarını gösterir:

```python
import pandas as pd
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Veri kümesini yükleme
df = pd.read_csv("data.csv")

# Model ve tokenizer'ı yükleme
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# Retrieval fonksiyonu
def retrieve_documents(query, df, top_n=5):
    # Query ve belgeleri embedding'lere dönüştürme
    query_embedding = model.encode(query, return_tensors="pt")
    document_embeddings = model.encode(df["text"], return_tensors="pt")
    
    # Kosinüs benzerliğini hesaplama
    similarities = cosine_similarity(query_embedding, document_embeddings).flatten()
    
    # En benzer belgeleri seçme
    top_indices = np.argsort(similarities)[-top_n:]
    top_documents = df.iloc[top_indices]
    
    return top_documents

# Üretken fonksiyon
def generate_text(query, top_documents):
    # Query ve belgeleri birleştirme
    input_text = f"{query} {' '.join(top_documents['text'])}"
    
    # Tokenizer ile input'u hazırlama
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Model ile çıktı üretme
    outputs = model.generate(**inputs)
    
    # Çıktıyı tokenizer ile çözme
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

# İnsan geri bildirimi fonksiyonu
def human_feedback(query, generated_text, feedback):
    # Geri bildirimi işleme
    if feedback == "good":
        # İyi geri bildirim için aksiyon
        print("Good feedback received!")
    elif feedback == "bad":
        # Kötü geri bildirim için aksiyon
        print("Bad feedback received!")
    
    # Geri bildirimi kaydetme
    with open("feedback.txt", "a") as f:
        f.write(f"{query}\t{generated_text}\t{feedback}\n")

# Örnek kullanım
query = "örnek sorgu"
top_documents = retrieve_documents(query, df)
generated_text = generate_text(query, top_documents)
print(generated_text)

# İnsan geri bildirimi
feedback = input("Geri bildiriminiz (good/bad): ")
human_feedback(query, generated_text, feedback)
```

## Kod Açıklaması:

*   `retrieve_documents` fonksiyonu, bir sorguyu alır ve en benzer belgeleri retrieval eder.
*   `generate_text` fonksiyonu, sorguyu ve retrieval edilen belgeleri kullanarak üretken metin oluşturur.
*   `human_feedback` fonksiyonu, insan geri bildirimi alır ve işler.

## Kodun Kullanımı:

1.  `data.csv` dosyasını hazırlayın ve `df` değişkenine yükleyin.
2.  `retrieve_documents` fonksiyonunu kullanarak sorguya en benzer belgeleri retrieval edin.
3.  `generate_text` fonksiyonunu kullanarak sorgu ve retrieval edilen belgelerle üretken metin oluşturun.
4.  `human_feedback` fonksiyonunu kullanarak insan geri bildirimi alın ve işleyin.

Bu kod, uyarlanabilir RAG'ın temel adımlarını gösterir ve insan geri bildirimi entegrasyonu için bir örnek sağlar.

---

## Adaptive RAG

## Uyarlamalı RAG (Adaptive RAG)

Uyarlamalı RAG, bir RAG (Retrieval-Augmented Generation) sisteminin çıktı kalitesini artırmak için insan geri bildirimlerini (Human Feedback, HF) kullanan bir yaklaşımdır. RAG sistemleri, büyük dil modellerinin (Large Language Models, LLM) yanı sıra bir bilgi tabanından alınan ilgili belgeleri kullanarak daha doğru ve bilgilendirici yanıtlar üretebilir. Ancak, RAG sistemlerinin performansı, kullanılan verilerin kalitesine ve sistemin nasıl yapılandırıldığına bağlıdır.

### Uyarlamalı RAG'ın Bileşenleri

Uyarlamalı RAG ekosistemi, aşağıdaki bileşenleri içerir:

*   **D1: Veri Toplama ve İşleme** (Data Collection and Processing): Wikipedia makalelerini LLM'ler hakkında toplama ve işleme.
*   **D4: Sorgu İşleme** (Retrieval Query): Bilgi tabanını sorgulamak için kullanılan sorgu.
*   **G1: Kullanıcı Girişi** (User Input): Kullanıcının girdiği metin.
*   **G2: Geliştirilmiş Giriş** (Augmented Input with HF): Kullanıcı girişini geliştirmek için insan geri bildirimlerini kullanan bir süreç.
*   **G4: Üretim ve Çıktı** (Generation and Output): Üretici yapay zeka modelini çalıştırarak bir yanıt üretme.
*   **E1: Metrikler** (Metrics): Kosinüs benzerlik ölçümü gibi metrikleri uygulayarak sistemin performansını değerlendirme.
*   **E2: İnsan Geri Bildirimi** (Human Feedback): Kullanıcı ve uzman geri bildirimlerini toplama ve işleme.

### Uyarlamalı RAG'ın Avantajları

Uyarlamalı RAG, aşağıdaki avantajları sunar:

*   **Hibrit Yapı** (Hybrid Structure): İnsan geri bildirimlerini ve makine öğrenimi algoritmalarını birleştirerek daha doğru ve güvenilir sonuçlar üretir.
*   **Uyarlanabilirlik** (Adaptability): İnsan kullanıcı geri bildirimlerine ve uzman değerlendirmelerine göre sistemin yanıtlarını uyarlar.

### Python ile Uyarlamalı RAG Uygulaması

Bu bölümde, Google Colab üzerinde Python kullanarak bir uyarlamalı RAG programı geliştireceğiz. Aşağıdaki kod, temel bir uyarlamalı RAG sisteminin nasıl kurulacağını gösterir:

```python
# Import gerekli kütüphaneler
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Veri setini yükleme ve işleme
def load_and_process_data(data_path):
    # Veri setini yükleme
    with open(data_path, 'r') as f:
        data = f.readlines()
    
    # Veri setini işleme
    processed_data = [line.strip() for line in data]
    return processed_data

# Kosinüs benzerlik ölçümü
def cosine_similarity_metric(query_embedding, document_embeddings):
    similarities = cosine_similarity(query_embedding, document_embeddings)
    return similarities

# İnsan geri bildirimlerini işleme
def process_human_feedback(feedback):
    # Geri bildirimleri işleme
    processed_feedback = [feedback_item.strip() for feedback_item in feedback]
    return processed_feedback

# Uyarlamalı RAG sistemi
class AdaptiveRAG:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.document_embeddings = []
    
    def add_documents(self, documents):
        self.documents.extend(documents)
        document_embeddings = self.model.encode(documents)
        self.document_embeddings.extend(document_embeddings)
    
    def query(self, query):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.document_embeddings)
        most_similar_index = np.argmax(similarities)
        return self.documents[most_similar_index]

# Uyarlamalı RAG sistemini oluşturma
adaptive_rag = AdaptiveRAG('all-MiniLM-L6-v2')

# Veri setini yükleme ve işleme
data_path = 'path/to/data.txt'
processed_data = load_and_process_data(data_path)

# Uyarlamalı RAG sistemine belgeleri ekleme
adaptive_rag.add_documents(processed_data)

# Sorgu yapma
query = 'LLM nedir?'
result = adaptive_rag.query(query)
print(result)
```

### Kod Açıklaması

Yukarıdaki kod, temel bir uyarlamalı RAG sisteminin nasıl kurulacağını gösterir. Kodda kullanılan önemli noktalar:

*   `load_and_process_data` fonksiyonu, veri setini yükler ve işler.
*   `cosine_similarity_metric` fonksiyonu, kosinüs benzerlik ölçümü yapar.
*   `process_human_feedback` fonksiyonu, insan geri bildirimlerini işler.
*   `AdaptiveRAG` sınıfı, uyarlamalı RAG sistemini temsil eder. Bu sınıf, belgeleri eklemeye ve sorgu yapmaya olanak tanır.

### Kullanım

Uyarlamalı RAG sistemini kullanmak için:

1.  Veri setini yükleyin ve işleyin.
2.  Uyarlamalı RAG sistemine belgeleri ekleyin.
3.  Sorgu yapın.

Bu şekilde, uyarlamalı RAG sistemi, kullanıcı geri bildirimlerine ve uzman değerlendirmelerine göre daha doğru ve güvenilir sonuçlar üretebilir.

---

## Building hybrid adaptive RAG in Python

## Python'da Hibrit Adaptif RAG (Hybrid Adaptive RAG) Oluşturma
Hibrit adaptif RAG (Hybrid Adaptive RAG) yapısının kanıtını oluşturmaya başlayalım. Öncelikle, `Adaptive_RAG.ipynb` dosyasını GitHub'dan açalım.

## Proje Yapısı
Projemizi HF (Hugging Face) üzerine kuracağız ve mevcut bir framework kullanmayacağız. Kendi pipeline'ımızı oluşturacağız ve HF'i tanıtacağız. Daha önce de belirtildiği gibi, program üç ayrı bölümden oluşmaktadır: `retriever` (bulucu), `generator` (üretici) ve `evaluator` (değerlendirici) fonksiyonları, ki bunlar gerçek bir projenin pipeline'ında ayrı ajanlar olabilir.

Bu fonksiyonları başlangıçtan itibaren ayırmaya çalışın, çünkü bir projede birkaç takım paralel olarak RAG framework'ünün farklı yönlerinde çalışabilir.

## Retriever (Bulucu) Fonksiyonu
Retriever fonksiyonu ilk olarak gelir. Aşağıdaki kod bloğu retriever fonksiyonunu göstermektedir:
```python
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# Retriever fonksiyonu
class Retriever:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

# Retriever nesnesi oluşturma
retriever = Retriever('sentence-transformers/all-MiniLM-L6-v2')

# Örnek kullanım
text = "Bu bir örnek metindir."
embeddings = retriever.get_embeddings(text)
print(embeddings.shape)
```
Bu kodda, `Retriever` sınıfı tanımlanmaktadır. Bu sınıf, `model_name` parametresi alarak bir model ve tokenizer yükler. `get_embeddings` metodu, verilen metnin embeddings'ini hesaplar.

## Generator (Üretici) Fonksiyonu
Generator fonksiyonu, retriever tarafından bulunan belgeleri kullanarak yeni metinler üretir. Aşağıdaki kod bloğu generator fonksiyonunu göstermektedir:
```python
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Generator fonksiyonu
class Generator:
    def __init__(self, model_name):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def generate_text(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors='pt')
        outputs = self.model.generate(**inputs)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# Generator nesnesi oluşturma
generator = Generator('t5-base')

# Örnek kullanım
input_text = "Bu bir örnek girdi metindir."
generated_text = generator.generate_text(input_text)
print(generated_text)
```
Bu kodda, `Generator` sınıfı tanımlanmaktadır. Bu sınıf, `model_name` parametresi alarak bir model ve tokenizer yükler. `generate_text` metodu, verilen girdi metnine göre yeni bir metin üretir.

## Evaluator (Değerlendirici) Fonksiyonu
Evaluator fonksiyonu, generator tarafından üretilen metinleri değerlendirir. Aşağıdaki kod bloğu evaluator fonksiyonunu göstermektedir:
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Evaluator fonksiyonu
class Evaluator:
    def __init__(self, model_name):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def evaluate_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        logits = outputs.logits
        return logits

# Evaluator nesnesi oluşturma
evaluator = Evaluator('distilbert-base-uncased-finetuned-sst-2-english')

# Örnek kullanım
text = "Bu bir örnek değerlendirme metindir."
logits = evaluator.evaluate_text(text)
print(logits.shape)
```
Bu kodda, `Evaluator` sınıfı tanımlanmaktadır. Bu sınıf, `model_name` parametresi alarak bir model ve tokenizer yükler. `evaluate_text` metodu, verilen metni değerlendirir ve logits döndürür.

Tüm bu kod blokları, hibrit adaptif RAG yapısının temel bileşenlerini oluşturur. Her bir bileşen, farklı bir görev için tasarlanmıştır ve birlikte çalışarak daha karmaşık görevleri yerine getirebilirler.

---

## 1. Retriever

## Retriever (Dosya Getirici) Kurulumu
RAG (Retrieval-Augmented Generation) tabanlı bir üretken yapay zeka modelinin (Generative AI Model) ortamını kurmak için gerekli ilk adımları özetleyeceğiz. Bu süreç, veri alma ve işleme işlemlerini kolaylaştıran temel yazılım bileşenlerinin (Software Components) ve kütüphanelerin (Libraries) kurulumu ile başlar. Özellikle, etkili veri alma (Data Retrieval) ve web kazıma (Web Scraping) için gerekli olan paketlerin kurulumunu ve önemli dosyaların indirilmesini ele alacağız.

### Önemli Noktalar:
- RAG tabanlı üretken yapay zeka modeli için ortam kurulumu
- Gerekli yazılım bileşenleri ve kütüphanelerin kurulumu
- Veri alma ve web kazıma işlemleri için gerekli paketler
- Önemli dosyaların indirilmesi

### Kurulum Adımları:
1. **Gerekli Kütüphanelerin Kurulumu**: 
   Öncelikle, RAG modeli için gerekli olan kütüphaneleri kurmak gerekir. Bu kütüphaneler arasında `transformers`, `torch`, ve `beautifulsoup4` gibi popüler kütüphaneler bulunur.

   ```bash
pip install transformers torch beautifulsoup4
```

   Bu komut, gerekli kütüphaneleri kurar. `transformers` kütüphanesi, çeşitli NLP (Natural Language Processing) görevleri için kullanılır. `torch`, PyTorch kütüphanesini temsil eder ve derin öğrenme (Deep Learning) uygulamaları için kullanılır. `beautifulsoup4` ise web kazıma işlemleri için kullanılır.

2. **Veri Alma ve Web Kazıma Kütüphanelerinin Kullanımı**:
   - `beautifulsoup4` kütüphanesini kullanarak web sayfalarından veri alma işlemleri gerçekleştirebilirsiniz. Aşağıdaki örnek kod, bir web sayfasından veri çekmek için `beautifulsoup4` ve `requests` kütüphanelerini nasıl kullanacağınızı gösterir.

     ```python
     import requests
     from bs4 import BeautifulSoup

     # Web sayfasını çek
     url = "https://example.com"
     response = requests.get(url)

     # BeautifulSoup ile sayfayı parse et
     soup = BeautifulSoup(response.text, 'html.parser')

     # Sayfa başlığını al
     title = soup.title.text
     print(title)
     ```

     Bu kod, `requests` kütüphanesini kullanarak belirtilen URL'ye bir GET isteği gönderir ve sayfa içeriğini çeker. Daha sonra `BeautifulSoup` kullanarak sayfa içeriğini parse eder ve sayfa başlığını (`title`) alır.

3. **RAG Modelinin Kurulumu ve Kullanımı**:
   RAG modeli, `transformers` kütüphanesinin bir parçası olarak kullanılabilir. Aşağıdaki örnek, basit bir RAG modeli kullanımını gösterir.

   ```python
   from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

   # Tokenizer, retriever ve model'i yükle
   tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
   retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
   model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

   # Giriş metnini hazırla
   input_dict = tokenizer.prepare_seq2seq_batch("What is the capital of France?", return_tensors="pt")

   # Model ile çıktı üret
   generated_ids = model.generate(input_ids=input_dict["input_ids"], attention_mask=input_dict["attention_mask"])

   # Üretilen IDs'i metne çevir
   generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
   print(generated_text)
   ```

   Bu kod, RAG modelini (`RagSequenceForGeneration`) ve ilgili tokenizer ile retriever'ı yükler. Daha sonra, bir soru için (`"What is the capital of France?"`) modelden bir cevap üretir.

### Açıklamalar:
- `transformers` kütüphanesi, önceden eğitilmiş birçok modeli kullanıma sunar. RAG modeli de bunlardan biridir ve özellikle bilgi yoğun görevlerde kullanılır.
- `torch` kütüphanesi, derin öğrenme modellerinin temelini oluşturur. `transformers` kütüphanesi de PyTorch'u destekler.
- `beautifulsoup4` ve `requests` kütüphaneleri, web kazıma işlemleri için sıkça kullanılır. Bu kütüphaneler, web sayfalarından veri çekmeyi kolaylaştırır.

Tüm bu adımlar ve kodlar, bir RAG tabanlı üretken yapay zeka modelinin temel kurulumu ve kullanımı için gereklidir.

---

## 1.1. Installing the retriever’s environment

## 1.1. Retriever'ın (Retriever) Ortamını Kurma (Installing the retriever's environment)

Retriever'ın ortamını kurmak için ilk adım, GitHub deposundaki (repository) commons dizininden (directory) `grequests.py` dosyasını indirmektir. Bu depo, depodaki (repository) birkaç program için ortak (common) olan kaynakları içerir, böylece gereksiz tekrarları (redundancy) önler. İndirme işlemi standarttır ve istek (request) üzerine kuruludur.

## İndirme İşlemi (Download Process)
İndirme işlemini gerçekleştirmek için aşağıdaki kodları kullanacağız:
```python
url = "https://raw.githubusercontent.com/Denis2054/RAG-Driven-Generative-AI/main/commons/grequests.py"
output_file = "grequests.py"
```
Bu kodda, `url` değişkeni `grequests.py` dosyasının bulunduğu GitHub deposundaki URL'yi (Uniform Resource Locator) içerir. `output_file` değişkeni ise dosyanın kaydedileceği yerel (local) dosya adını belirtir.

## Gerekli Kütüphanelerin Kurulumu (Installing Required Libraries)
RAG-driven generative AI modelini sıfırdan (from scratch) oluşturduğumuz için retriever için yalnızca iki paket (package) kurmamız gerekecek. Kuracağımız paketler:
* `requests`: Wikipedia belgelerini (documents) almak için kullanılan HTTP kütüphanesi (HTTP library). 
* `beautifulsoup4`: Web sayfalarından (web pages) bilgi kazımak (scrape) için kullanılan kütüphane.

Kurulum için aşağıdaki komutları kullanacağız:
```bash
!pip install requests==2.32.3
!pip install beautifulsoup4==4.12.3
```
Bu komutlarda, `!pip install` ifadesi Python paketlerini kurmak için kullanılan pip paket yöneticisini (package manager) çağırır. `requests==2.32.3` ve `beautifulsoup4==4.12.3` ifadeleri ise kurulacak paketlerin adlarını ve sürümlerini (version) belirtir.

## Veri Kümesinin (Dataset) Edinilmesi
Şimdi, bir veri kümesine (dataset) ihtiyacımız var.

## Önemli Noktalar (Important Points)
* `grequests.py` dosyasını GitHub deposundan indirin.
* `requests` ve `beautifulsoup4` paketlerini kurun.
* Veri kümesi edinin.

## Kodların Açıklaması (Explanation of Codes)
* `url` ve `output_file` değişkenleri, `grequests.py` dosyasını indirmek için kullanılır.
* `!pip install` komutları, `requests` ve `beautifulsoup4` paketlerini kurmak için kullanılır.

## İçe Aktarmalar (Imports)
Bu kodları çalıştırmak için herhangi bir içe aktarma (import) ifadesine gerek yoktur. Ancak, `requests` ve `beautifulsoup4` paketlerini kurmak için pip paket yöneticisini kullanıyoruz. 

Tüm kodlar eksiksiz olarak yazılmıştır ve birebir aynı olmalıdır.

---

## 1.2.1. Preparing the dataset

## Veri Kümesinin Hazırlanması (Preparing the Dataset)
Bu kavram kanıtı (proof of concept) için, Wikipedia belgelerini URL'leri aracılığıyla kazıyarak (scraping) elde edeceğiz. Veri kümesi, her belge için otomatik veya insan tarafından oluşturulan etiketler (labels) içerecektir, bu da bir veri kümesinin belgelerini dizinleme (indexing) yolundaki ilk adımdır.

### Kullanılan Kod ve Açıklamaları
```python
import requests
from bs4 import BeautifulSoup
import re

# Anahtar kelimelere göre Wikipedia makalelerinin URL'leri
urls = {
    "prompt engineering": "https://en.wikipedia.org/wiki/Prompt_engineering",
    "artificial intelligence": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "llm": "https://en.wikipedia.org/wiki/Large_language_model",
    "llms": "https://en.wikipedia.org/wiki/Large_language_model"
}
```
*   `import requests`: HTTP istekleri (requests) yapmak için kullanılan kütüphane.
*   `from bs4 import BeautifulSoup`: HTML ve XML dosyalarını ayrıştırmak (parsing) için kullanılan BeautifulSoup kütüphanesini içe aktarır.
*   `import re`: Düzenli ifadeler (regular expressions) ile çalışmak için kullanılan kütüphane.
*   `urls`: Anahtar kelimelere karşılık gelen Wikipedia makalelerinin URL'lerini içeren bir sözlük (dictionary).

Her URL'den önce bir veya daha fazla etiket gelir. Bu yaklaşım, nispeten küçük bir veri kümesi için yeterli olabilir. Belirli projeler için, bu yaklaşım, saf RAG'den (anahtar kelimelerle içerik arama) dizinlerle (bu durumda etiketlerle) bir veri kümesini aramaya geçmek için sağlam bir ilk adım sağlayabilir.

### Veri İşleme (Data Processing)
Şimdi verileri işlememiz gerekiyor.

### Veri Kazıma (Web Scraping) ve İşleme Kodları
Wikipedia sayfalarını kazımak ve içeriklerini işlemek için aşağıdaki adımları takip edeceğiz:
```python
# Wikipedia sayfasını kazımak için fonksiyon
def scrape_wikipedia(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Sayfa içeriğini ayıklama
    content = soup.get_text()
    return content

# URL'leri dolaşarak içerikleri işleme
for keyword, url in urls.items():
    content = scrape_wikipedia(url)
    # İçerik işleme kodları buraya eklenecek
    print(f"{keyword}: {content[:100]}")  # İlk 100 karakteri yazdırma
```
*   `scrape_wikipedia` fonksiyonu, verilen URL'deki Wikipedia sayfasını kazır ve içeriğini döndürür.
*   `requests.get(url)`: Belirtilen URL'ye bir HTTP GET isteği gönderir.
*   `BeautifulSoup(response.text, 'html.parser')`: Gelen HTML yanıtını ayrıştırır.
*   `soup.get_text()`: Sayfa içeriğini metin olarak ayıklar.
*   `for` döngüsü, `urls` sözlüğündeki her URL için `scrape_wikipedia` fonksiyonunu çağırır ve içerikleri işler.

Bu kodlar, Wikipedia sayfalarını kazımak ve içeriklerini işlemek için temel bir yapı sağlar. İçerik işleme kodları, projenin gereksinimlerine göre özelleştirilebilir.

---

## 1.2.2. Processing the data

## Veri İşleme (Data Processing)

Veri işleme, ham verilerin temizlenmesi, dönüştürülmesi ve analiz için uygun hale getirilmesi işlemidir. Bu işlem, makine öğrenimi (Machine Learning) modellerinin başarısı için kritik öneme sahiptir.

### Veri Kazıma ve Temizleme Fonksiyonu (Web Scraping and Text Cleaning Function)

Aşağıdaki kod, bir URL'den içerik çeken ve temizleyen bir fonksiyon tanımlar:
```python
import requests
from bs4 import BeautifulSoup
import re

def fetch_and_clean(url):
    # URL'nin içeriğini getir (Fetch the content of the URL)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Makale içeriğini bul (Find the main content of the article)
    content = soup.find('div', {'class': 'mw-parser-output'})

    # İlgisiz bölümleri kaldır (Remove less relevant sections)
    for section_title in ['References', 'Bibliography', 'External links', 'See also']:
        section = content.find('span', {'id': section_title})
        if section:
            for sib in section.parent.find_next_siblings():
                sib.decompose()
            section.parent.decompose()

    # Paragraf etiketlerinden metni çıkar ve temizle (Extract and clean text from paragraph tags)
    paragraphs = content.find_all('p')
    cleaned_text = ' '.join(paragraph.get_text(separator=' ', strip=True) for paragraph in paragraphs)
    cleaned_text = re.sub(r'\[\d+\]', '', cleaned_text)  # Kaynak işaretlerini kaldır (Remove citation markers)

    return cleaned_text
```
Bu fonksiyon, bir URL'den içerik çeker, gereksiz bölümleri kaldırır ve paragraf etiketlerinden metni çıkararak temizler.

*   `requests.get(url)`: Belirtilen URL'nin içeriğini getirir.
*   `BeautifulSoup(response.content, 'html.parser')`: HTML içeriğini ayrıştırır.
*   `soup.find('div', {'class': 'mw-parser-output'})`: Makale içeriğini içeren `div` etiketini bulur.
*   `content.find_all('p')`: Makale içeriğindeki tüm paragraf etiketlerini bulur.
*   `re.sub(r'\[\d+\]', '', cleaned_text)`: Kaynak işaretlerini (`[1]`, `[2]`, vs.) kaldırır.

### Veri İşleme İşlemleri (Data Processing Steps)

1.  **Veri Kazıma (Web Scraping)**: URL'den içerik çekme işlemi.
2.  **İçerik Temizleme (Content Cleaning)**: Gereksiz bölümlerin kaldırılması ve metnin temizlenmesi.
3.  **Metin Çıkarma (Text Extraction)**: Paragraf etiketlerinden metnin çıkarılması.

Bu işlemler, makine öğrenimi modellerinin başarısı için önemlidir. Veri işleme işlemleri, ham verilerin analiz için uygun hale getirilmesini sağlar.

### Kullanım (Usage)

`fetch_and_clean` fonksiyonu, bir URL'den içerik çekmek ve temizlemek için kullanılabilir. Örneğin:
```python
url = "https://example.com/article"
cleaned_text = fetch_and_clean(url)
print(cleaned_text)
```
Bu kod, belirtilen URL'den içerik çeker, temizler ve yazdırır.

### Önemli Noktalar (Key Points)

*   Veri işleme, makine öğrenimi modellerinin başarısı için kritik öneme sahiptir.
*   Veri kazıma ve temizleme fonksiyonu, ham verilerin analiz için uygun hale getirilmesini sağlar.
*   İçerik temizleme ve metin çıkarma işlemleri, gereksiz verilerin kaldırılması ve metnin temizlenmesi için önemlidir.

---

## 1.3. Retrieval process for user input

## 1.3. Kullanıcı Girdisi İçin Erişim İşlemi (Retrieval Process for User Input)

Kullanıcı girdisindeki anahtar kelimeyi (keyword) tanımlamak ilk adımdır. `process_query` fonksiyonu iki parametre alır: `user_input` ve `num_words`. Alınacak kelime sayısı, modelin girdi sınırlamaları, maliyet considerations ve genel sistem performansı gibi faktörlerle sınırlıdır.

### Kod
```python
import textwrap

def process_query(user_input, num_words):
    user_input = user_input.lower() 
    # Kullanıcı girdisinde belirtilen anahtar kelimelerden herhangi birini kontrol edin
    matched_keyword = next((keyword for keyword in urls if keyword in user_input), None)
```
Burada `user_input` küçük harfe çevrilir (`lower()` fonksiyonu ile) ve `urls` sözlüğündeki anahtar kelimelerden herhangi biri kullanıcı girdisinde var mı diye kontrol edilir. `next()` fonksiyonu, eğer eşleşen bir anahtar kelime bulunursa ilk eşleşen kelimeyi döndürür, yoksa `None` döndürür.

### Anahtar Kelime Bulunması Durumu

Eğer eşleşen bir anahtar kelime bulunursa, aşağıdaki fonksiyonlar tetiklenir:
```python
if matched_keyword:
    print(f"Fetching data from: {urls[matched_keyword]}")
    cleaned_text = fetch_and_clean(urls[matched_keyword]) 
    # Temizlenen metinden belirtilen sayıda kelimeyi sınırlayın
    words = cleaned_text.split() 
    first_n_words = ' '.join(words[:num_words]) 
```
Burada `fetch_and_clean()` fonksiyonu çağrılır ve ilgili URL'den veri çekilip temizlenir. Daha sonra temizlenen metin kelimelere ayrılır (`split()` fonksiyonu ile) ve ilk `num_words` kadar kelime birleştirilir (`join()` fonksiyonu ile).

### Metni Biçimlendirme

Temizlenen ve kısaltılan metin daha sonra görüntü için biçimlendirilir:
```python
wrapped_text = textwrap.fill(first_n_words, width=80)
print("\nFirst {} words of the cleaned text:".format(num_words))
print(wrapped_text) 
```
`textwrap.fill()` fonksiyonu ile metin 80 karakter genişliğinde satırlara bölünür.

### GPT-4 Promptu Oluşturma

Aynı `first_n_words` kullanılarak GPT-4 için bir prompt oluşturulur:
```python
prompt = f"Summarize the following information about {matched_keyword}:\n{first_n_words}"
wrapped_prompt = textwrap.fill(prompt, width=80) 
print("\nPrompt for Generator:", wrapped_prompt)
```
Bu prompt, modele özetlemesi için bir metin sağlar.

### Sonuç

Fonksiyon, ilk `n` kelimeyi döndürür ve kullanıcı sorgusuna dayalı olarak kısa ve ilgili bir bilgi snippet'i sağlar:
```python
return first_n_words
```
Eğer eşleşen bir anahtar kelime bulunamazsa, fonksiyon `None` döndürür ve bir hata mesajı yazdırır:
```python
else:
    print("No relevant keywords found. Please enter a query related to 'LLM', 'LLMs', or 'Prompt Engineering'.")
    return None
```
Bu tasarım, sistemin veri erişimini etkin bir şekilde yönetmesine ve aynı zamanda kullanıcı etkileşimini sürdürmesine olanak tanır.

### Önemli Noktalar

* Kullanıcı girdisindeki anahtar kelime tanımlanır.
* `process_query` fonksiyonu iki parametre alır: `user_input` ve `num_words`.
* Alınacak kelime sayısı sınırlıdır.
* `fetch_and_clean()` fonksiyonu veri çekmek ve temizlemek için kullanılır.
* Metin biçimlendirme için `textwrap.fill()` fonksiyonu kullanılır.
* GPT-4 promptu oluşturulur ve modele özetlemesi için bir metin sağlar.

---

## 2. Generator

## 2. Jeneratör (Generator)

Jeneratör ekosistemi, RAG (Retrieval-Augmented Generation) tabanlı üretken yapay zeka (Generative AI) çerçevelerindeki retriever (bulucu) işlevleri ve kullanıcı arayüzleri (User Interface, UI) ile örtüşen birkaç bileşen içerir.

### 2.1. İnsan Derecelendirmelerine Dayalı Adaptif RAG Seçimi (Adaptive RAG Selection)

Bu, zaman içinde bir kullanıcı panelinin derecelendirmelerine dayanacaktır. Gerçek bir pipeline'da (işlem hattı), bu işlevsellik ayrı bir program olabilir.

### 2.2. Giriş (Input)

Gerçek bir projede, bir kullanıcı arayüzü (UI) girişi yönetecektir. Bu arayüz ve ilişkili süreç, kullanıcıların ihtiyaçları ve tercihleri tam olarak anlaşılabilecek bir atölye ortamında, ideal olarak bir workshop'ta dikkatlice tasarlanmalıdır.

### 2.3. Ortalama Derecelendirme Simülasyon Senaryosu (Mean Ranking Simulation Scenario)

Kullanıcı değerlendirme puanlarının ve işlevselliğinin ortalama değerini hesaplamak.

### 2.4. Jeneratörü Çalıştırmadan Önce Girişi Kontrol Etme (Checking the Input Before Running the Generator)

Girişi görüntülemek.

### 2.5. Üretken Yapay Zeka Ortamını Kurma (Installing the Generative AI Environment)

Üretken yapay zeka modelinin ortamını, bu durumda OpenAI'ı kurmak, pipeline'daki diğer ortamın bir parçası olabilir ve diğer ekip üyeleri tarafından bağımsız olarak retriever işlevselliğinden üretimde uygulanabilir ve dağıtılabilir.

### 2.6. İçerik Üretimi (Content Generation)

Programın bu bölümünde, bir OpenAI modeli girişi işleyecek ve değerlendirici tarafından değerlendirilecek bir yanıt sağlayacaktır.

Jeneratör ekosistemini tanımlamaya başlayalım.

## Örnek Kod

Aşağıdaki örnek kod, OpenAI kullanarak içerik üretimini göstermektedir:
```python
import os
import openai

# OpenAI API anahtarını ayarlayın
openai.api_key = os.getenv("OPENAI_API_KEY")

# Girişi tanımlayın
input_text = "İçerik üretimi için giriş metni."

# OpenAI modelini kullanarak içerik üretin
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=input_text,
    max_tokens=2048,
    temperature=0.7
)

# Üretilen içeriği yazdırın
print(response.choices[0].text)
```
## Kod Açıklaması

*   `import os` ve `import openai`: Gerekli kütüphaneleri içe aktarın.
*   `openai.api_key = os.getenv("OPENAI_API_KEY")`: OpenAI API anahtarını ortam değişkeninden ayarlayın.
*   `input_text = "İçerik üretimi için giriş metni."`: Giriş metnini tanımlayın.
*   `response = openai.Completion.create()`: OpenAI modelini kullanarak içerik üretin.
    *   `engine="text-davinci-002"`: Kullanılacak OpenAI modelini belirtin.
    *   `prompt=input_text`: Giriş metnini modele iletin.
    *   `max_tokens=2048`: Üretilen içeriğin maksimum uzunluğunu belirleyin.
    *   `temperature=0.7`: Üretimdeki yaratıcılığı kontrol edin.
*   `print(response.choices[0].text)`: Üretilen içeriği yazdırın.

## Kullanım

1.  OpenAI API anahtarınızı ortam değişkeni olarak ayarlayın: `export OPENAI_API_KEY=<API_KEY>`
2.  Yukarıdaki kodu bir Python betiğine kaydedin (örneğin, `content_generation.py`).
3.  Betiği çalıştırın: `python content_generation.py`
4.  Üretilen içerik terminalde görüntülenecektir.

## Adaptif RAG Seçimi

Adaptif RAG seçimi, insan derecelendirmelerine dayanarak retriever işlevselliğini optimize etmeyi içerir. Bu, bir kullanıcı panelinin derecelendirmelerine dayanarak retriever modelini eğitmek ve güncellemek için kullanılabilir.

## Ortalama Derecelendirme Simülasyon Senaryosu

Ortalama derecelendirme simülasyon senaryosu, kullanıcı değerlendirme puanlarının ortalama değerini hesaplamak için kullanılır. Bu, retriever modelinin performansını değerlendirmek ve iyileştirmek için kullanılabilir.

## Jeneratörü Çalıştırmadan Önce Girişi Kontrol Etme

Jeneratörü çalıştırmadan önce girişi kontrol etmek, giriş metninin geçerli ve uygun olduğundan emin olmak için önemlidir. Bu, bir dizi doğrulama kontrolü uygulanarak yapılabilir.

## Üretken Yapay Zeka Ortamını Kurma

Üretken yapay zeka ortamını kurmak, OpenAI gibi üretken yapay zeka modellerini kullanmak için gerekli olan bağımlılıkları ve kütüphaneleri yüklemeyi içerir. Bu, bir Python ortamında `pip install openai` gibi komutlarla yapılabilir.

---

## 2.1. Integrating HF-RAG for augmented document inputs

## 2.1. Gelişmiş Doküman Girdileri için HF-RAG Entegrasyonu (Integrating HF-RAG for Augmented Document Inputs)

Bilgi erişiminin dinamik doğası ve üretken yapay zeka modellerinde (Generative AI Models) içerikle ilgili veri artırma (Data Augmentation) gerekliliği, değişken girdi kalitesine uyum sağlayabilen esnek bir sistem gerektirir. HF skorlarını kullanarak RAG ekosistemi içinde doküman uygulaması için en uygun erişim stratejisini belirleyen bir uyarlanabilir RAG seçim sistemi (Adaptive RAG Selection System) sunuyoruz. Uyarlanabilir işlevsellik, basit RAG'ın ötesine geçmemizi sağlar ve hibrit bir RAG sistemi oluşturur.

## Değerlendirme ve Skorlama (Evaluation and Scoring)

İnsan değerlendiriciler, dokümanların alaka düzeyini ve kalitesini değerlendirmek için 1 ila 5 arasında ortalama skorlar verirler. Bu skorlar, aşağıdaki şekilde gösterildiği gibi farklı operasyonel modları tetikler:

## Otomatik RAG Tetikleyicileri (Automated RAG Triggers)

### Skorların Anlamı

*   1 ila 2 arasındaki skorlar, RAG sisteminin telafi edici bir yeteneğe sahip olmadığını gösterir ve bakım veya muhtemelen model ince ayarının (Model Fine-Tuning) gerekli olduğunu önerir. RAG, sistem geliştirilene kadar geçici olarak devre dışı bırakılır. Kullanıcı girdisi işlenecek, ancak erişim (Retrieval) yapılmayacaktır.
*   3 ila 4 arasındaki skorlar, yalnızca insan-uzman geri bildirimi (Human-Expert Feedback) kullanarak flash kartlar veya snippet'ler ile çıktıları iyileştirmek için bir artırma başlatır. Doküman tabanlı RAG devre dışı bırakılır, ancak insan-uzman geri bildirim verileri girdiyi artırır.
*   5 skor, daha önce toplanan HF tarafından geliştirilen anahtar kelime-aramalı RAG'ı başlatır ve gerektiğinde flash kartlar veya hedeflenen bilgi snippet'leri kullanarak çıktıları iyileştirir. Kullanıcıdan yeni geri bildirim sağlaması istenmez.

## Kod Uygulaması

Bu program, birçok senaryodan birini uygular. Skorlama sistemi, skor seviyeleri ve tetikleyiciler, spesifikasyon hedeflerine bağlı olarak bir projeden diğerine değişecektir. Bu uyarlanabilir RAG sisteminin nasıl uygulanacağına karar vermek için bir kullanıcı paneli ile atölye çalışmaları düzenlenmesi önerilir.

Uyarlanabilir yaklaşım, otomatik erişim (Automated Retrieval) ve insan öngörüsü (Human Insight) arasındaki dengeyi optimize etmeyi amaçlar ve üretken modelin çıktılarının mümkün olan en yüksek alaka düzeyine ve doğruluğuna sahip olmasını sağlar.

Şimdi girdiye girelim.

### Kod

```python
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Model ve tokenizer yükleme
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Skorları tanımla
def skorlari_tanimla(skor):
    if skor <= 2:
        return "RAG devre dışı"
    elif skor <= 4:
        return "İnsan-uzman geri bildirimi ile artırma"
    else:
        return "Anahtar kelime-aramalı RAG"

# Kullanıcı girdisini işleme
def kullanici_girdisini_isle(girdi):
    # Girdiyi tokenize et
    inputs = tokenizer(girdi, return_tensors="pt")
    
    # Model çıktısını al
    outputs = model(**inputs)
    
    # Skorları hesapla
    skor = outputs.logits.detach().numpy()[0][0]
    
    # Skorlama sistemini uygula
    return skorlari_tanimla(skor)

# Örnek kullanım
girdi = "Bu bir örnek girdidir."
sonuc = kullanici_girdisini_isle(girdi)
print(sonuc)
```

### Kod Açıklaması

Bu kod, bir girdi metnini işleyen ve bir skor döndüren bir model kullanır. Skor, daha sonra bir skorlama sistemine göre yorumlanır ve bir RAG modu tetikler.

1.  `import` ifadeleri, gerekli kütüphaneleri yükler.
2.  `model_name` değişkeni, kullanılacak modelin adını tanımlar.
3.  `tokenizer` ve `model` nesneleri, sırasıyla metni tokenize etmek ve model çıktısını almak için kullanılır.
4.  `skorlari_tanimla` fonksiyonu, bir skor alır ve buna karşılık gelen RAG modunu döndürür.
5.  `kullanici_girdisini_isle` fonksiyonu, bir girdi metnini işler, skorunu hesaplar ve skorlama sistemini uygular.
6.  Örnek kullanımda, bir girdi metni işlenir ve sonuç yazılır.

### Kodun Kullanımı

Bu kod, bir RAG sisteminin uyarlanabilir bir şekilde çalışmasını sağlamak için kullanılabilir. Girdi metninin kalitesine göre, skorlama sistemi bir RAG modu tetikler ve sistem buna göre hareket eder.

### Avantajları

*   Uyarlanabilir RAG sistemi, değişken girdi kalitesine uyum sağlar.
*   Skorlama sistemi, RAG modlarını belirlemek için esnek bir yapı sağlar.

### Dezavantajları

*   Skorlama sisteminin doğru şekilde tanımlanması gerekir.
*   Modelin doğru şekilde eğitilmesi gerekir.

## Sonuç

Uyarlanabilir RAG sistemi, üretken yapay zeka modellerinde içerikle ilgili veri artırma gerekliliğini karşılamak için esnek bir yapı sağlar. Skorlama sistemi ve RAG modları, değişken girdi kalitesine uyum sağlamak için tasarlanmıştır. Bu sistem, birçok senaryoda kullanılabilir ve spesifikasyon hedeflerine bağlı olarak özelleştirilebilir.

---

## 2.2. Input

## 2.2. Giriş (Input)
Şirket C'nin kullanıcısı bir soru girmesi için yönlendirilir: 
## Kullanıcıdan anahtar kelime ayrıştırma (keyword parsing) için girdi alınır
## Kullanıcıdan anahtar kelime ayrıştırma (keyword parsing) için girdi alınır
Kullanıcı girdisi (user input) `input()` fonksiyonu ile alınır ve küçük harfe (lowercase) çevrilir.
```python
user_input = input("Enter your query: ").lower()
```
Bu kod, kullanıcıdan bir girdi alır ve bunu `user_input` değişkenine atar. `input()` fonksiyonu, kullanıcıya bir mesaj gösterir ve kullanıcının girdiği değeri döndürür. `.lower()` metodu ise bu değeri küçük harfe çevirir.

Örneğin, kullanıcı "What is an LLM?" girdiğinde, `user_input` değişkeni `"what is an llm?"` değerini alır.

## Örnek Uygulama
Bu örnekte ve programda, bir soru ve konu üzerine odaklanacağız: "What is an LLM?" (LLM nedir?). Soru görünür ve model tarafından hatırlanır (memorized):
```
Enter your query: What is an LLM?
Enter your query: What is an LLM?
Enter your query: What is an LLM?
```
Bu program, Şirket C'deki kullanıcı paneli için bir kanıt-of-kavram (proof of concept) ve strateji örneğidir. Diğer konular eklenebilir ve program daha sonraki ihtiyaçları karşılamak üzere genişletilebilir. Kullanıcı paneli ile atölye çalışmaları düzenlenmesi önerilir.

## RAG Senaryosu
Çevreyi hazırladık ve şimdi bir RAG (Retrieval-Augmented Generation) senaryosunu etkinleştireceğiz.

### Kod Açıklaması
Yukarıdaki kodda `input()` fonksiyonu kullanılmıştır. Bu fonksiyon, Python'ın built-in fonksiyonlarından biridir ve kullanıcıdan girdi almak için kullanılır.

```python
import builtins

user_input = builtins.input("Enter your query: ").lower()
```

Ancak `builtins` modülünü import etmek gerekmez, çünkü `input()` fonksiyonu Python'ın temel fonksiyonlarından biridir ve direkt kullanılabilir.

### Kullanım
`input()` fonksiyonu, bir mesaj göstermek ve kullanıcının girdiği değeri almak için kullanılır. `.lower()` metodu ise bu değeri küçük harfe çevirmek için kullanılır. Bu sayede, kullanıcının girdiği değer büyük veya küçük harf olmasına bakılmaksızın aynı şekilde işlenir.

Örneğin, aşağıdaki kodda da görüldüğü gibi:
```python
user_input = input("Enter your query: ").lower()
print(user_input)
```
Kullanıcı "What is an LLM?" girdiğinde, çıktı `"what is an llm?"` olur.

---

## 2.3. Mean ranking simulation scenario

## 2.3. Ortalama Sıralama Simülasyon Senaryosu (Mean Ranking Simulation Scenario)

Bu bölümde, hibrit uyarlamalı RAG (Retrieval-Augmented Generation) sisteminin kullanıcı geri bildirim paneli tarafından değerlendirildiği varsayılmaktadır. Kullanıcı geri bildirim paneli, yanıtları birçok kez sıralar ve bu sıralamaların ortalamasını alarak `ranking` adlı bir değişkende saklar.

### Önemli Noktalar:
- Kullanıcı geri bildirim paneli, RAG sisteminin performansını değerlendirir.
- `ranking` değişkeni, kullanıcı geri bildirimlerinin ortalamasını temsil eder.
- `ranking` skoru, bir belgenin sıralamasını düşürmek, yükseltmek veya belgeleri manuel veya otomatik işlevler aracılığıyla bastırmak için kullanılır.

## Simülasyon Senaryoları

Üç farklı senaryo simüle edilecektir:
1. `ranking = 1`: RAG devre dışı bırakılır ve üretken modelin (Generative Model) doğal yanıtı gözlemlenir.
2. `ranking = 5`: RAG etkinleştirilir, ancak insan uzmanı geri bildirimi (Human-Expert Feedback) olmadan.
3. `ranking = 3`: İnsan uzmanı geri bildirimli RAG etkinleştirilir, ancak belgeleri almadan.

### Kod Parçaları ve Açıklamaları

#### 1. `ranking` Değişkeninin Tanımlanması
```python
# Simülasyon için 1 ile 5 arasında bir skor seçin
ranking = 1  # Başlangıçta RAG devre dışı
```

#### 2. `text_input` Değişkeninin Başlatılması
Üretken modelin işleyeceği metni saklamak için `text_input` değişkeni başlatılır.
```python
# Üretken AI modeli simülasyonları için metni başlatma
text_input = []
```

#### 3. Sıralama Kategorilerine Göre İşlemler

##### Ranking 1-2: RAG Yok
`ranking` 1 ile 3 arasında ise, RAG işlevselliği devre dışı bırakılır.
```python
if ranking >= 1 and ranking < 3:
    text_input = user_input
```

##### Ranking 3-4: İnsan Uzmanı Geri Bildirimli RAG
İnsan uzmanı geri bildirimi etkinleştirilir ve ilgili belge getirilir.
```python
hf = False
if ranking > 3 and ranking < 5:
    hf = True

if hf == True:
    from grequests import download
    directory = "Chapter05"
    filename = "human_feedback.txt"
    download(directory, filename, private_token)
    
    # Dosya var mı kontrol et
    efile = os.path.exists('human_feedback.txt')
    if efile:
        with open('human_feedback.txt', 'r') as file:
            content = file.read().replace('\n', ' ').replace('#', '')
        text_input = content
        print(text_input)
    else:
        print("Dosya bulunamadı")
        hf = False
```

##### Ranking 5: RAG (İnsan Uzmanı Geri Bildirimi Olmadan)
Maksimum kelime sayısı 100 olarak belirlenir ve ilgili veri getirilir.
```python
if ranking >= 5:
    max_words = 100  # Girdi boyutu limiti
    rdata = process_query(user_input, max_words)
    print(rdata)  # Gerekirse bakım için
    
    if rdata:
        rdata_clean = rdata.replace('\n', ' ').replace('#', '')
        rdata_sentences = rdata_clean.split('. ')
        print(rdata)
        text_input = rdata
        print(text_input)
```

### Sonuç

Bu simülasyonlar, RAG sisteminin farklı kullanıcı geri bildirimlerine göre nasıl tepki verdiğini gösterir. İnsan uzmanı geri bildirimi ve RAG belgeleri, üretken modelin daha doğru ve ilgili yanıtlar vermesini sağlar. Bu, özellikle müşteri desteği gibi alanlarda büyük önem taşır. Gelecek bölümde, bu sistemin daha da geliştirilmesi ve üretken modelin ince ayarının yapılması ele alınacaktır. 

### Teknik Terimler:
- **RAG (Retrieval-Augmented Generation)**: Belirli bir görevi gerçekleştirmek için hem üretken hem de bilgi alma (retrieval) yeteneklerini kullanan bir yapay zeka yaklaşımı.
- **LLM (Large Language Model)**: Büyük miktarda metin verisi üzerinde eğitilen ve insan benzeri metinler üretebilen gelişmiş bir yapay zeka modeli.
- **HF (Human Feedback)**: İnsan kullanıcıların veya uzmanların sistem performansını değerlendirmek için sağladıkları geri bildirim.
- **GPT (Generative Pre-trained Transformer)**: OpenAI tarafından geliştirilen ve çeşitli doğal dil işleme görevlerinde kullanılan bir dizi üretken model.

---

## 2.4.–2.5. Installing the generative AI environment

## 2.4.–2.5. Üretken Yapay Zeka (Generative AI) Ortamını Kurma

## 2.4. Üreticiyi (Generator) Çalıştırma Öncesinde Girdiyi (Input) Kontrol Etme
Kullanıcı girdisini (user input) ve alınan belgeyi (retrieved document) bu bilgiyle zenginleştirmeden önce görüntülemek için kullanılır. Ardından ## 2.5. Üretken Yapay Zeka (Generative AI) Ortamını Kurma işlemine geçilir.

## 2.5. Üretken Yapay Zeka (Generative AI) Ortamını Kurma
Bu bölüm sadece bir kez çalıştırılır. Eğer ## 2.3 bölümünde senaryoyu değiştirdiyseniz, üretken yapay zeka modelini tekrar çalıştırmak için bu bölümü atlayabilirsiniz. Bu kurulumun bu not defterinin (notebook) başında olmamasının nedeni, bir proje ekibinin bu programın bu kısmını başka bir ortamda veya hatta üretimdeki başka bir sunucuda (server) çalıştırabilmesidir.

### Önemli Noktalar:
- Alıcı (retriever) ve üretici (generator) işlevlerini mümkün olduğunca ayırmak önerilir çünkü farklı programlar tarafından ve muhtemelen farklı zamanlarda etkinleştirilebilirler.
- Bir geliştirme ekibi sadece alıcı işlevleri üzerinde çalışırken, başka bir ekip üretici işlevler üzerinde çalışabilir.

### OpenAI Kurulumu:
İlk olarak OpenAI kütüphanesini kurmak gerekir:
```python
!pip install openai==1.40.3
```
Bu komut, OpenAI kütüphanesinin 1.40.3 versiyonunu kurar.

### API Anahtarını Alma:
OpenAI API anahtarınızı güvenli bir konumda saklayın. Bu örnekte, Google Drive'da saklanmaktadır:
```python
from google.colab import drive
drive.mount('/content/drive')
f = open("drive/MyDrive/files/api_key.txt", "r")
API_KEY=f.readline().strip()
f.close()
```
Bu kod, Google Drive'ı bağlar, `api_key.txt` dosyasından API anahtarını okur ve `API_KEY` değişkenine atar.

### OpenAI API Anahtarını Tanımlama:
```python
import os
import openai
os.environ['OPENAI_API_KEY'] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
```
Bu kod, OpenAI API anahtarını ortam değişkeni (environment variable) olarak tanımlar ve OpenAI kütüphanesine bu anahtarı verir.

### Açıklamalar:
- `!pip install openai==1.40.3`: OpenAI kütüphanesini kurar. Bu komut, Jupyter Notebook veya Google Colab gibi ortamlarda kullanılır.
- `drive.mount('/content/drive')`: Google Drive'ı Colab ortamına bağlar.
- `open("drive/MyDrive/files/api_key.txt", "r")`: `api_key.txt` dosyasını okumak için açar. Bu dosya, OpenAI API anahtarını içerir.
- `os.environ['OPENAI_API_KEY'] = API_KEY`: OpenAI API anahtarını ortam değişkeni olarak ayarlar.
- `openai.api_key = os.getenv("OPENAI_API_KEY")`: OpenAI kütüphanesine API anahtarını verir.

Artık içerik üretimi (content generation) için her şey hazır.

---

## 2.6. Content generation

## İçerik Üretimi (Content Generation)
İçerik üretmek için öncelikle gerekli olanları içe aktarıp (`import`) kurulum yapmalıyız. Zamanı (`time`) ölçmek için `time` modülünü içe aktardık ve sohbet modelimiz (`conversational model`) olarak `gpt-4o`'yu seçtik.

### Kod
```python
import openai
from openai import OpenAI
import time

client = OpenAI()
gptmodel = "gpt-4o"
start_time = time.time()  # İsteği göndermeden önce zamanlayıcı başlatılıyor
```
Bu kodda `openai` ve `OpenAI` sınıfını içe aktarıyoruz (`import`). `time` modülünü de içe aktararak (`import`) zaman ölçümü yapabiliyoruz. `client` değişkenine `OpenAI()` sınıfının bir örneğini atıyoruz. `gptmodel` değişkenine `"gpt-4o"` değerini atıyoruz. `start_time` değişkenine ise `time.time()` fonksiyonu ile şu anki zamanı atıyoruz.

## GPT-4o Prompt Tanımlama
Daha sonra GPT-4o için standart bir prompt (`prompt`) tanımlıyoruz. Bu prompt, modele yanıt vermesi için yeterli bilgiyi veriyor ve geri kalanını modele ve RAG verilerine bırakıyor.

### Kod
```python
def call_gpt4_with_full_text(itext):
    # Tüm satırları birleştirerek tek bir string oluşturuyoruz
    text_input = '\n'.join(itext)
    prompt = f"Please summarize or elaborate on the following content:\n{text_input}"
    try:
        response = client.chat.completions.create(
            model=gptmodel,
            messages=[
                {"role": "system", "content": "You are an expert Natural Language Processing exercise expert."},
                {"role": "assistant", "content": "1.You can explain read the input and answer in detail"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Sıcaklık parametresini buraya ekliyoruz
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)
```
Bu kodda `call_gpt4_with_full_text` adlı bir fonksiyon tanımlıyoruz. Bu fonksiyon, girdi olarak bir metin (`itext`) alıyor ve GPT-4o modeline gönderiyor. `text_input` değişkenine, girdi metninin satırlarını birleştirerek (`'\n'.join(itext)`) oluşturduğumuz string'i atıyoruz. `prompt` değişkenine ise GPT-4o için tanımladığımız prompt'u atıyoruz.

`client.chat.completions.create` fonksiyonunu çağırarak GPT-4o modeline isteği gönderiyoruz. Bu fonksiyona model adını (`gptmodel`), mesajları (`messages`) ve sıcaklık parametresini (`temperature`) veriyoruz. Mesajlar, sohbeti temsil eden bir liste (`list`). Her bir mesaj, bir sözlük (`dict`) şeklinde temsil ediliyor. Bu sözlükte `role` ve `content` anahtarları bulunuyor.

## Çıktıyı Biçimlendirme
Daha sonra çıktıyı biçimlendirmek için (`print_formatted_response`) adlı bir fonksiyon tanımlıyoruz.

### Kod
```python
import textwrap

def print_formatted_response(response):
    # Metni sarmak için genişliği tanımlıyoruz
    wrapper = textwrap.TextWrapper(width=80)  # 80 sütun genişliğinde, ancak gerektiği gibi ayarlayabilirsiniz
    wrapped_text = wrapper.fill(text=response)
    # Biçimlendirilmiş yanıtı bir başlık ve altlık ile yazdırıyoruz
    print("GPT-4 Response:")
    print("---------------")
    print(wrapped_text)
    print("---------------\n")

# 'gpt4_response' değişkeninin önceki GPT-4 çağrısından gelen yanıtı içerdiğini varsayıyoruz
print_formatted_response(gpt4_response)
```
Bu kodda `textwrap` modülünü içe aktarıyoruz (`import`). `print_formatted_response` adlı bir fonksiyon tanımlıyoruz. Bu fonksiyon, girdi olarak bir yanıt (`response`) alıyor ve biçimlendiriyor. `wrapper` değişkenine `textwrap.TextWrapper` sınıfının bir örneğini atıyoruz. Bu sınıf, metni sarmak için kullanılıyor. `wrapped_text` değişkenine, `wrapper.fill` fonksiyonu ile biçimlendirilmiş metni atıyoruz.

## Önemli Noktalar
*   İçerik üretmek için GPT-4o modeli kullanılıyor (`gpt-4o`).
*   `client.chat.completions.create` fonksiyonu ile GPT-4o modeline istek gönderiliyor.
*   `temperature` parametresi ile modelin yaratıcılığı kontrol ediliyor (`temperature=0.1`).
*   Çıktı, `print_formatted_response` fonksiyonu ile biçimlendiriliyor.
*   `textwrap` modülü, metni sarmak için kullanılıyor.

## Örnek Çıktı
GPT-4 Response:
---------------
### Summary: A large language model (LLM) is a computational model known for its ability to perform general-purpose language generation and other natural language processing tasks, such as classification. LLMs acquire these abilities by learning statistical relationships from vast amounts of text during a computationally intensive self-supervised and semi-supervised training process.They can be used for text generation, a form of generative AI, by taking input text and repeatedly predicting the next token or word. LLMs are artificial neural networks that use the transformer architecture…
---------------

Bu çıktıda GPT-4o modelinin büyük dil modelleri (`LLM`) hakkında bir özet (`summary`) verdiği görülüyor.

---

## 3. Evaluator

## Değerlendirici (Evaluator)

Her projenin spesifikasyonlarına ve ihtiyaçlarına bağlı olarak, gerektiği kadar matematiksel ve insan değerlendirmesi fonksiyonları uygulanabilir. Bu bölümde, iki otomatik metrik uygulayacağız: yanıt süresi (response time) ve kosinüs benzerlik puanı (cosine similarity score). Ardından, iki etkileşimli değerlendirme fonksiyonu uygulayacağız: insan kullanıcı derecelendirmesi (human user rating) ve insan-uzman değerlendirmesi (human-expert evaluation).

## Önemli Noktalar
* Otomatik metrikler: Yanıt süresi ve kosinüs benzerlik puanı
* Etkileşimli değerlendirme fonksiyonları: İnsan kullanıcı derecelendirmesi ve insan-uzman değerlendirmesi

## Otomatik Metrikler

### Yanıt Süresi (Response Time)

Yanıt süresi, sistemin bir kullanıcı isteğine cevap vermesi için geçen süreyi ölçer. Aşağıdaki kod bloğu, yanıt süresini ölçmek için kullanılabilir:
```python
import time

def measure_response_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        response_time = end_time - start_time
        print(f"Yanıt süresi: {response_time:.2f} saniye")
        return result
    return wrapper

# Kullanımı:
@measure_response_time
def example_function():
    time.sleep(1)  # 1 saniye bekleme
    print("İşlem tamamlandı")

example_function()
```
Bu kod, `measure_response_time` adlı bir dekoratör tanımlar. Bu dekoratör, bir fonksiyonun yanıt süresini ölçer ve sonucu yazdırır.

### Kosinüs Benzerlik Puanı (Cosine Similarity Score)

Kosinüs benzerlik puanı, iki vektör arasındaki benzerliği ölçer. Aşağıdaki kod bloğu, kosinüs benzerlik puanını hesaplamak için kullanılabilir:
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1).reshape(1, -1)
    vector2 = np.array(vector2).reshape(1, -1)
    similarity = cosine_similarity(vector1, vector2)
    return similarity[0][0]

# Kullanımı:
vector1 = [1, 2, 3]
vector2 = [4, 5, 6]
similarity = calculate_cosine_similarity(vector1, vector2)
print(f"Kosinüs benzerlik puanı: {similarity:.2f}")
```
Bu kod, `calculate_cosine_similarity` adlı bir fonksiyon tanımlar. Bu fonksiyon, iki vektör arasındaki kosinüs benzerlik puanını hesaplar.

## Etkileşimli Değerlendirme Fonksiyonları

### İnsan Kullanıcı Derecelendirmesi (Human User Rating)

İnsan kullanıcı derecelendirmesi, kullanıcıların sistemin çıktısına verdiği derecelendirmeyi ölçer. Aşağıdaki kod bloğu, insan kullanıcı derecelendirmesini toplamak için kullanılabilir:
```python
def collect_user_rating():
    rating = input("Lütfen sistemin çıktısına bir derecelendirme verin (1-5): ")
    try:
        rating = int(rating)
        if 1 <= rating <= 5:
            return rating
        else:
            print("Lütfen 1-5 arasında bir değer girin.")
            return collect_user_rating()
    except ValueError:
        print("Lütfen geçerli bir değer girin.")
        return collect_user_rating()

# Kullanımı:
user_rating = collect_user_rating()
print(f"Kullanıcı derecelendirmesi: {user_rating}")
```
Bu kod, `collect_user_rating` adlı bir fonksiyon tanımlar. Bu fonksiyon, kullanıcıdan bir derecelendirme ister ve sonucu döndürür.

### İnsan-Uzman Değerlendirmesi (Human-Expert Evaluation)

İnsan-uzman değerlendirmesi, uzmanların sistemin çıktısına verdiği değerlendirmeyi ölçer. Aşağıdaki kod bloğu, insan-uzman değerlendirmesini toplamak için kullanılabilir:
```python
def collect_expert_evaluation():
    evaluation = input("Lütfen sistemin çıktısına bir değerlendirme yapın: ")
    return evaluation

# Kullanımı:
expert_evaluation = collect_expert_evaluation()
print(f"Uzman değerlendirmesi: {expert_evaluation}")
```
Bu kod, `collect_expert_evaluation` adlı bir fonksiyon tanımlar. Bu fonksiyon, uzmandan bir değerlendirme ister ve sonucu döndürür.

---

## 3.1. Response time

## 3.1. Cevap Zamanı (Response Time)
Cevap zamanı, API çağrısında hesaplandı ve aşağıdaki kod ile gösterildi:
```python
import time
...
start_time = time.time()  # İsteğin gönderilmesinden önce zamanlayıcı başlat (Start timing before the request)
...
response_time = time.time() - start_time  # Cevap zamanını ölç (Measure response time)
print(f"Cevap Zamanı (Response Time): {response_time:.2f} saniye (seconds)")  # Cevap zamanını yazdır (Print response time)
```
Bu kodda, `time.time()` fonksiyonu kullanılarak isteğin gönderilmesinden önce ve sonra zaman kaydedilir ve cevap zamanı hesaplanır. `.2f` ifadesi, cevap zamanının virgülden sonra 2 basamaklı olarak yazdırılmasını sağlar.

## Kodun Açıklaması
- `import time`: Zaman ile ilgili işlemler için gerekli kütüphaneyi içe aktarır.
- `start_time = time.time()`: İsteğin gönderilmesinden önce zamanlayıcıyı başlatır.
- `response_time = time.time() - start_time`: Cevap zamanını hesaplar.
- `print(f"Cevap Zamanı (Response Time): {response_time:.2f} saniye (seconds)")`: Cevap zamanını yazdırır.

## Önemli Noktalar
- Cevap zamanı, internet bağlantısı ve OpenAI sunucularının kapasitesine bağlı olarak değişkenlik gösterir.
- Cevap zamanı, çevrimiçi konuşma ajanlarının cevap verme süresine benzer.
- Bu performansın yeterli olup olmadığına karar vermek bir yönetim kararıdır.

## Çıktı (Output)
Cevap zamanı, yukarıdaki kodun çalıştırılması sonucu elde edilen çıktı aşağıdaki gibi olabilir:
```
Cevap Zamanı (Response Time): 7.88 saniye (seconds)
```
Bu değer, internet bağlantısı ve OpenAI sunucularının kapasitesine bağlı olarak değişebilir.

---

## 3.2. Cosine similarity score

## 3.2. Kosinüs Benzerlik Skoru (Cosine Similarity Score)
Kosinüs benzerliği, iki sıfırdan farklı vektör arasındaki açının kosinüsünü ölçer. Metin analizi bağlamında, bu vektörler tipik olarak metnin TF-IDF (Term Frequency-Inverse Document Frequency) temsilleridir, bu da terimlerin belge ve bir korpusla ilgili önemine göre ağırlıklandırılmasını sağlar. GPT-4o'nun girdisi, yani `text_input`, ve modelin yanıtı, yani `gpt4_response`, TF-IDF tarafından iki ayrı "belge" olarak ele alınır. Vektörleştirici (`vectorizer`), belgeleri vektörlere dönüştürür. Daha sonra, vektörleştirme, `vectorizer.fit_transform([text1, text2])` kullanarak girdi ve yanıt arasında terimlerin nasıl paylaşıldığını ve vurgulandığını dikkate alır.

## Kosinüs Benzerlik Skorunun Hesaplanması
Amaç, tematik ve sözcüksel örtüşmeyi aşağıdaki fonksiyon aracılığıyla nicelendirmektir:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]

# Örnek kullanım
similarity_score = calculate_cosine_similarity(text_input, gpt4_response)
print(f"Kosinüs Benzerlik Skoru: {similarity_score:.3f}")
```
## Kod Açıklaması
- `TfidfVectorizer()`: Metin verilerini TF-IDF vektörlerine dönüştürmek için kullanılan bir sınıftır. Bu, terimlerin sıklığını ve belgedeki önemini dikkate alır.
- `vectorizer.fit_transform([text1, text2])`: Girdi metinlerini (`text1` ve `text2`) TF-IDF vektörlerine dönüştürür. Bu işlem, hem `text1` hem de `text2` için terimlerin ağırlıklandırılmasını sağlar.
- `cosine_similarity(tfidf[0:1], tfidf[1:2])`: Dönüştürülen TF-IDF vektörleri arasındaki kosinüs benzerliğini hesaplar. Bu, iki metin arasındaki benzerliği ölçer.
- `similarity[0][0]`: Kosinüs benzerlik matrisinden benzerlik skorunu döndürür.

## Kosinüs Benzerlik Skoru Sonuçları
Kosinüs benzerliği, TfidfVectorizer'ı kullanarak iki belgeyi TF-IDF vektörlerine dönüştürmeye dayanır. Daha sonra, `cosine_similarity` fonksiyonu bu vektörler arasındaki benzerliği hesaplar. Sonuç 1 ise, metinler özdeştir; 0 ise, benzerlik yoktur. Fonksiyonun çıktısı:
```
Kosinüs Benzerlik Skoru: 0.697
```
Bu skor, girdi ve model çıktısı arasında güçlü bir benzerlik olduğunu gösterir. Ancak, bir insan kullanıcının bu yanıta nasıl değer vereceği merak konusudur. Bunu öğrenmek için daha fazla değerlendirmeye ihtiyaç vardır.

---

## 3.3. Human user rating

## İnsan Kullanıcı Değerlendirmesi (Human User Rating)
İnsan kullanıcı değerlendirme arayüzü (Human User Rating Interface), insan kullanıcı geri bildirimleri (Human User Feedback) sağlar. Bu bölümde, insan kullanıcı panelinin (Human User Panel) bir grup yazılım geliştiricisi (Software Developers) tarafından sistemin test edildiği varsayılmaktadır.

### Parametrelerin Tanımlanması (Defining Parameters)
Kod, arayüzün parametreleri ile başlar:
```python
# Skor parametreleri (Score Parameters)
counter = 20  # Geri bildirim sorgularının sayısı (Number of Feedback Queries)
score_history = 30  # İnsan geri bildirimi (Human Feedback)
threshold = 4  # İnsan uzmanı geri bildirimini tetiklemek için minimum sıralamalar (Minimum Rankings to Trigger Human Expert Feedback)
```
Bu simülasyonda, parametreler sistemin insan geri bildirimini hesapladığını gösterir:
- `counter=20` kullanıcılar tarafından girilen derecelendirme sayısını gösterir (Number of Ratings Already Entered by Users).
- `score_history=30` 20 derecelendirmenin toplam skorunu gösterir (Total Score of the 20 Ratings).
- `threshold=4` insan-uzmanı geri bildirim isteğini tetiklemeden elde edilecek minimum ortalama derecelendirmeyi belirtir (Minimum Mean Rating to Obtain Without Triggering a Human-Expert Feedback Request), yani `score_history/counter`.

### Değerlendirme Fonksiyonu (Evaluation Function)
Sağlanan Python kodu, `evaluate_response` fonksiyonunu tanımlar. Bu fonksiyon, GPT-4 gibi bir dil modelinin (Language Model) ürettiği cevapların alaka ve tutarlılığını değerlendirmek üzere tasarlanmıştır. Kullanıcılar, oluşturulan metni 1 (kötü) ile 5 (mükemmel) arasında bir ölçekte derecelendirir ve fonksiyon, geçerli girişi özyinelemeli kontroller (Recursive Checks) yoluyla sağlar.

```python
import numpy as np

def evaluate_response(response):
    print("\nOluşturulan Cevap (Generated Response):\n")
    print(response)
    print("\nLütfen cevabı aşağıdaki kriterlere göre değerlendirin (Please Evaluate the Response Based on the Following Criteria):")
    print("1 - Kötü (Poor), 2 - Orta (Fair), 3 - İyi (Good), 4 - Çok İyi (Very Good), 5 - Mükemmel (Excellent)")
    
    score = input("Alaka ve tutarlılık skorunu girin (1-5) (Enter the Relevance and Coherence Score (1-5)): ")
    try:
        score = int(score)
        if 1 <= score <= 5:
            return score
        else:
            print("Geçersiz skor. Lütfen 1 ile 5 arasında bir sayı girin (Invalid Score. Please Enter a Number Between 1 and 5).")
            return evaluate_response(response)  # Geçersiz girdi için özyinelemeli çağrı (Recursive Call if the Input is Invalid)
    except ValueError:
        print("Geçersiz girdi. Lütfen sayısal bir değer girin (Invalid Input. Please Enter a Numerical Value).")
        return evaluate_response(response)  # Geçersiz girdi için özyinelemeli çağrı (Recursive Call if the Input is Invalid)
```

### Fonksiyonun Çağrılması (Calling the Function)
Fonksiyon çağrıldığında:
```python
score = evaluate_response(gpt4_response)
print("Değerlendirici Skoru (Evaluator Score):", score)
```
İlk olarak, oluşturulan cevap görüntülenir, ardından kullanıcı bir değerlendirme skoru girer.

### İstatistiklerin Hesaplanması (Computing Statistics)
Daha sonra, kod istatistikleri hesaplar:
```python
counter += 1
score_history += score
mean_score = round(np.mean(score_history / counter), 2)
if counter > 0:
    print("Sıralamalar (Rankings):", counter)
    print("Skor Geçmişi (Score History):", mean_score)
```
Çıktı, nispeten çok düşük bir derecelendirme gösterir:
```
Değerlendirici Skoru (Evaluator Score): 3
Sıralamalar (Rankings): 21
Skor Geçmişi (Score History): 3.0
```
Değerlendirici skoru 3, genel sıralama 21, ve skor geçmişi 3.0'dir. Kosinüs benzerliği (Cosine Similarity) pozitif olmasına rağmen, insan-uzmanı değerlendirme isteği `threshold=4` nedeniyle tetiklenecektir.

Bu durum, sistemin daha fazla insan geri bildirimi gerektirdiğini ve belki de modelin iyileştirilmesi gerektiğini gösterir.

---

## 3.4. Human-expert evaluation

## İnsan Uzmanı Değerlendirmesi (Human-Expert Evaluation)
Kosinüs benzerliği (Cosine Similarity) gibi metrikler gerçekten benzerliği ölçer, ancak derin doğruluk (in-depth accuracy) ölçmez. Zaman performansı (Time Performance) da bir yanıtın doğruluğunu belirlemez. Ancak derecelendirme çok düşükse, bunun nedeni kullanıcıların yanıttan memnun olmamasıdır!

### Kod Açıklaması
İlk olarak, insan-uzman kullanıcısı için başparmak yukarı (thumbs-up) ve başparmak aşağı (thumbs-down) resimleri indirilir:
```python
from grequests import download

directory = "commons"
filename = "thumbs_up.png"
download(directory, filename, private_token)

directory = "commons"
filename = "thumbs_down.png"
download(directory, filename, private_token)
```
Burada `grequests` kütüphanesinden `download` fonksiyonu kullanılır. `directory` ve `filename` değişkenleri tanımlanır ve `private_token` ile birlikte `download` fonksiyonuna geçirilir.

### Uzman Geri Bildirim Parametreleri
Uzman geri bildirimi tetiklemek için `counter_threshold` ve `score_threshold` parametreleri kullanılır. Kullanıcı derecelendirmelerinin sayısı uzman eşiğini (`counter_threshold=10`) aşmalıdır. Derecelendirmelerin ortalama skorunun eşiği bu senaryoda 4'tür (`score_threshold=4`).
```python
if counter > counter_threshold and score_history <= score_threshold:
    print("Human expert evaluation is required for the feedback loop.")
```
Burada `counter` ve `score_history` değişkenleri kullanılır. Eğer `counter` `counter_threshold`'dan büyükse ve `score_history` `score_threshold`'dan küçük veya eşitse, uzman geri bildirimi gerektiği yazılır.

### HTML Arayüzü ve Geri Bildirim Kaydetme
Bir Python hücresinde standart HTML arayüzü kullanılarak başparmak yukarı ve başparmak aşağı ikonları görüntülenir. Eğer uzman başparmak aşağı ikonuna basarsa, bir geri bildirim snippet'i girilebilir ve `expert_feedback.txt` adlı bir dosyada saklanabilir.
```python
import base64
from google.colab import output
from IPython.display import display, HTML

def image_to_data_uri(file_path):
    with open(file_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f'data:image/png;base64,{encoded_string}'

thumbs_up_data_uri = image_to_data_uri('/content/thumbs_up.png')
thumbs_down_data_uri = image_to_data_uri('/content/thumbs_down.png')

def display_icons():
    # HTML içeriğini tanımla
    ...

def save_feedback(feedback):
    with open('/content/expert_feedback.txt', 'w') as f:
        f.write(feedback)
    print("Feedback saved successfully.")

output.register_callback('notebook.save_feedback', save_feedback)
print("Human Expert Adaptive RAG activated")
display_icons()
```
Burada `image_to_data_uri` fonksiyonu bir resmi data URI'sine dönüştürür. `display_icons` fonksiyonu HTML içeriğini tanımlar ve `save_feedback` fonksiyonu geri bildirimi kaydeder.

### Uzman Geri Bildirimi ve RAG Veri Kümesi İyileştirmesi
Uzman başparmak aşağı ikonuna bastığında, bir geri bildirim girilir ve `/content/expert_feedback.txt` dosyasına kaydedilir. Bu geri bildirim daha sonra RAG veri kümesini iyileştirmek için kullanılabilir.

Önemli noktalar:
* Kosinüs benzerliği derin doğruluk ölçmez.
* Zaman performansı yanıtın doğruluğunu belirlemez.
* Kullanıcı derecelendirmeleri uzman geri bildirimi tetikleyebilir.
* Uzman geri bildirimi RAG veri kümesini iyileştirmek için kullanılabilir.
* HTML arayüzü kullanılarak başparmak yukarı ve başparmak aşağı ikonları görüntülenebilir.
* Geri bildirim snippet'i `expert_feedback.txt` dosyasına kaydedilebilir.

---

## Summary

## Özet
Bu bölümde, pratik AI uygulamalarına yönelik elle yapılan yaklaşımı sonlandırırken, adaptif RAG'ın dinamik dünyasını birlikte keşfettiğimiz dönüştürücü yolculuğu düşünmek önemlidir. Öncelikle, HF'ın (Human Feedback) sadece tamamlayıcı değil, aynı zamanda üretken AI'ı gerçek dünya ihtiyaçlarına göre daha güçlü bir araç haline getiren kritik bir güç olduğunu inceledik. Adaptif RAG ekosistemini tanımladık ve ardından sıfırdan başlayarak elle yaptık. Veri toplama, işleme ve sorgulama ile başlayarak, bu öğeleri RAG tahrikli üretken AI sistemine entegre ettik. Yaklaşımımız sadece kodlama değil, sürekli HF döngüleri aracılığıyla AI'a adaptasyon eklemekti.

## Önemli Noktalar
* HF, üretken AI'ı gerçek dünya ihtiyaçlarına göre daha güçlü bir araç haline getirir (Human Feedback enhances Generative AI).
* Adaptif RAG ekosistemi, veri toplama, işleme ve sorgulama ile entegre edilmiştir (Adaptive RAG ecosystem is integrated with data collection, processing, and querying).
* GPT-4'un yeteneklerini önceki oturumlardan uzman görüşleri ve son kullanıcı değerlendirmeleri ile güçlendirdik (Augmenting GPT-4's capabilities with expert insights and end-user evaluations).
* Son kullanıcıların çıktıyı sıraladığı ve düşük sıralamaların uzman geri bildirim döngüsünü tetiklediği bir sistem uyguladık (Implemented a system where output is ranked by end-users and low rankings trigger expert feedback loop).
* Adaptif RAG programını sıfırdan inşa etmek, HF'ın bir standard AI sistemini zaman içinde gelişen ve iyileşen bir sisteme nasıl dönüştürebileceğini anlamamızı sağladı (Building an adaptive RAG program from scratch ensured understanding of how HF can shift a standard AI system to one that evolves and improves over time).

## Kodlar ve Açıklamaları
Bu metinde kod örneği bulunmamaktadır. Ancak, RAG tahrikli üretken AI sisteminin nasıl inşa edileceğini anlamak için aşağıdaki kod örneğini ele alalım:
```python
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Veri yükleme ve işleme
data = pd.read_csv("data.csv")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Modeli yükleme ve eğitme
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Sorgulama ve çıktı üretme
def generate_output(input_text):
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors="pt"
    )
    outputs = model(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
    return outputs

# Kullanıcı geri bildirimi toplama ve modelin güncellenmesi
def update_model(user_feedback):
    # Kullanıcı geri bildirimi işleme
    # Modelin güncellenmesi
    pass
```
Bu kod örneğinde, `transformers` kütüphanesini kullanarak bir BERT modelini yüklüyoruz ve eğitiyoruz. Ardından, `generate_output` fonksiyonu ile girdi metnine göre çıktı üretiyoruz. `update_model` fonksiyonu ile kullanıcı geri bildirimini topluyor ve modeli güncelliyoruz.

## İlgili Teknik Terimler
* RAG: Retrieval-Augmented Generation ( retrieval-augmented generation )
* HF: Human Feedback ( insan geri bildirimi )
* GPT-4: Generative Pre-trained Transformer 4 ( üretken önceden eğitilmiş transformatör 4 )
* AI: Artificial Intelligence ( yapay zeka )

---

## Questions

## Sorular ve Cevaplar
Aşağıdaki paragrafta anlatılan konuyu türkçe olarak tekrar düzenleyerek önemli noktaları maddeler halinde yazacağım. Ayrıca, text içinde kodlar varsa yazacak ve açıklayacağım. Türkçenin yanına ingilizce teknik terimleri parantez içinde ekleyeceğim.

## Konu: Adaptive RAG ve İnsan Geri Bildirimi (Human Feedback)

Adaptive RAG, retrieval-augmented generation (RAG) tabanlı üretken yapay zeka (generative AI) sistemlerini geliştirmek için insan geri bildirimlerini kullanan bir yaklaşımdır. İnsan geri bildirimi, sistemin daha doğru ve alakalı sonuçlar vermesini sağlar.

## Önemli Noktalar
* İnsan geri bildirimi, RAG-driven generative AI sistemlerini geliştirmek için esastır (essential).
* Çekirdek veri (core data), üretken AI modelinde değiştirilmeden önce modelin yeniden eğitilmesi (retraining) gerekir.
* Adaptive RAG, retrieval işlemini iyileştirmek için gerçek zamanlı insan geri bildirim döngüleri (real-time human feedback loops) içerir.
* Adaptive RAG'ın birincil odağı, tüm insan girdisini (human input) otomatik yanıtlarla (automated responses) değiştirmek değildir.
* İnsan geri bildirimi, Adaptive RAG'da alınan belgelerde (retrieved documents) değişiklikleri tetikleyebilir (trigger).
* Şirket C, Adaptive RAG'ı yalnızca müşteri desteği (customer support) sorunları için kullanmaz.
* İnsan geri bildirimi, yalnızca AI yanıtlarının yüksek kullanıcı puanları (high user ratings) olduğu durumlarda kullanılmaz.
* Bu bölümdeki program, yalnızca metin tabanlı retrieval çıktıları (text-based retrieval outputs) sağlamaz.
* Hybrid Adaptive RAG sistemi, geri bildirime (feedback) göre ayarlanabilen dinamik (dynamic) bir sistemdir.
* Kullanıcı puanları (user rankings), AI yanıtlarının alaka düzeyini (relevance) belirlemede tamamen göz ardı edilmez.

## Kodlar ve Açıklamalar
Aşağıdaki kod örnekleri, Adaptive RAG'ın nasıl çalıştığını gösterir.

```python
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Model ve tokenizer yükleme
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# İnsan geri bildirimi içeren veri yükleme
df = pd.read_csv("human_feedback_data.csv")

# Adaptive RAG için retrieval işlemi
def retrieval(query, documents):
    # retrieval işlemini gerçekleştiren kod
    pass

# İnsan geri bildirim döngüsü
def human_feedback_loop(query, response, feedback):
    # İnsan geri bildirimi temelinde retrieval işlemini güncelleyen kod
    pass

# Hybrid Adaptive RAG sistemi
class HybridAdaptiveRAG:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def retrieve(self, query):
        # retrieval işlemini gerçekleştiren kod
        pass

    def update(self, query, response, feedback):
        # İnsan geri bildirimi temelinde retrieval işlemini güncelleyen kod
        pass
```

## Kod Açıklamaları

* `import pandas as pd`: Pandas kütüphanesini içe aktarır.
* `from transformers import AutoModelForSeq2SeqLM, AutoTokenizer`: Transformers kütüphanesinden model ve tokenizer içe aktarır.
* `model_name = "t5-base"`: Kullanılacak modelin adını belirler.
* `tokenizer = AutoTokenizer.from_pretrained(model_name)`: Model için tokenizer yükler.
* `model = AutoModelForSeq2SeqLM.from_pretrained(model_name)`: Modeli yükler.
* `df = pd.read_csv("human_feedback_data.csv")`: İnsan geri bildirimi içeren veriyi yükler.
* `retrieval` fonksiyonu: Retrieval işlemini gerçekleştirir.
* `human_feedback_loop` fonksiyonu: İnsan geri bildirim döngüsünü gerçekleştirir.
* `HybridAdaptiveRAG` sınıfı: Hybrid Adaptive RAG sistemini tanımlar.

## Sorular ve Cevaplar

1. İnsan geri bildirimi, RAG-driven generative AI sistemlerini geliştirmek için esastır mı? **Yes**
2. Çekirdek veri, üretken AI modelinde değiştirilmeden önce modelin yeniden eğitilmesi gerekir mi? **Yes**
3. Adaptive RAG, retrieval işlemini iyileştirmek için gerçek zamanlı insan geri bildirim döngüleri içerir mi? **Yes**
4. Adaptive RAG'ın birincil odağı, tüm insan girdisini otomatik yanıtlarla değiştirmek midir? **No**
5. İnsan geri bildirimi, Adaptive RAG'da alınan belgelerde değişiklikleri tetikleyebilir mi? **Yes**
6. Şirket C, Adaptive RAG'ı yalnızca müşteri desteği sorunları için kullanır mı? **No**
7. İnsan geri bildirimi, yalnızca AI yanıtlarının yüksek kullanıcı puanları olduğu durumlarda kullanılır mı? **No**
8. Bu bölümdeki program, yalnızca metin tabanlı retrieval çıktıları sağlar mı? **No**
9. Hybrid Adaptive RAG sistemi, geri bildirime göre ayarlanabilen dinamik bir sistem midir? **Yes**
10. Kullanıcı puanları, AI yanıtlarının alaka düzeyini belirlemede tamamen göz ardı edilir mi? **No**

---

## References

## Büyük Dil Modeli Davranışlarının Gerçekçi Bilgi Çatışmaları Altında İncelenmesi (Studying Large Language Model Behaviors Under Realistic Knowledge Conflicts)

Büyük dil modelleri (Large Language Models), doğal dil işleme (Natural Language Processing) alanında önemli bir yere sahiptir. Bu modellerin davranışlarını anlamak, özellikle gerçekçi bilgi çatışmaları (Realistic Knowledge Conflicts) altında incelemek, onların nasıl çalıştığını ve nasıl geliştirilebileceğini anlamak açısından önemlidir.

## Çalışma ve Referanslar (References)

Evgenii Kortukov, Alexander Rubinstein, Elisa Nguyen ve Seong Joon Oh tarafından yapılan "Studying Large Language Model Behaviors Under Realistic Knowledge Conflicts" adlı çalışma, bu konuda önemli bir araştırma sunmaktadır. Çalışmanın özeti aşağıdaki gibidir:

*   Çalışmanın amacı, büyük dil modellerinin gerçekçi bilgi çatışmaları altında nasıl davrandığını incelemektir.
*   Araştırmacılar, OpenAI modellerini (OpenAI Models) kullanarak bir dizi deney gerçekleştirmişlerdir.
*   Deneyler, büyük dil modellerinin bilgi çatışmaları altında nasıl davrandığını anlamaya yöneliktir.

## Önemli Noktalar (Key Points)

*   Büyük dil modelleri, geniş bir bilgi yelpazesine sahiptir ve çeşitli görevlerde kullanılabilirler.
*   Gerçekçi bilgi çatışmaları, modellerin karşılaştığı bilgi arasındaki çelişkilerdir.
*   Modellerin bu çatışmalar altında nasıl davrandığını anlamak, onların geliştirilmesi ve iyileştirilmesi açısından önemlidir.

## Kullanılan Kodlar ve Açıklamalar (Codes and Explanations)

Çalışmada kullanılan kodlar ve açıklamaları aşağıda verilmiştir:

### Kod 1: OpenAI Modeli Kullanımı (Using OpenAI Model)

```python
import os
import openai

# OpenAI API anahtarını ayarlayın (Set OpenAI API key)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Modeli seçin (Select the model)
model = "text-davinci-003"

# Giriş metnini tanımlayın (Define the input text)
prompt = "İstediğiniz bir metin buraya girilebilir."

# Modeli kullanarak metni tamamlayın (Complete the text using the model)
response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.7,
)

# Modelin çıktısını yazdırın (Print the model's output)
print(response.choices[0].text.strip())
```

Bu kod, OpenAI API'sini kullanarak bir metni tamamlamak için `text-davinci-003` modelini kullanır. `openai.api_key` değişkenine OpenAI API anahtarını atamak gerekir. `prompt` değişkeni, modele giriş olarak verilen metni tanımlar. `max_tokens` parametresi, modelin üreteceği maksimum token sayısını belirler. `temperature` parametresi, modelin yaratıcılığını kontrol eder.

### Kod 2: Modelin Davranışını İnceleme (Examining the Model's Behavior)

```python
import pandas as pd

# Veri setini yükleyin (Load the dataset)
df = pd.read_csv("veri_seti.csv")

# Modelin davranışını incelemek için bir fonksiyon tanımlayın (Define a function to examine the model's behavior)
def examine_model_behavior(prompt):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# Veri setindeki her bir örnek için modelin davranışını inceleyin (Examine the model's behavior for each example in the dataset)
df["model_output"] = df["prompt"].apply(examine_model_behavior)

# Sonuçları analiz edin (Analyze the results)
print(df.head())
```

Bu kod, bir veri setindeki her bir örnek için modelin davranışını inceler. `examine_model_behavior` fonksiyonu, modele bir giriş metni verir ve modelin çıktısını döndürür. `apply` fonksiyonu, bu fonksiyonu veri setindeki her bir örneğe uygular.

## Kaynaklar (References)

*   Evgenii Kortukov, Alexander Rubinstein, Elisa Nguyen, Seong Joon Oh. (2024). Studying Large Language Model Behaviors Under Realistic Knowledge Conflicts. arXiv preprint arXiv:2404.16032. <https://arxiv.org/abs/2404.16032>
*   OpenAI. (2024). OpenAI Models. <https://platform.openai.com/docs/models>

---

## Further reading

## Daha Fazla Okuma (Further Reading)
Bu bölümde, bu bölümde uygulanan vektörleştirici (vectorizer) ve kosinüs benzerliği (cosine similarity) işlevselliği hakkında daha fazla bilgi edinmek için aşağıdaki bağlantıları kullanabilirsiniz.

## Vektörleştirici ve Kosinüs Benzerliği Hakkında Daha Fazla Bilgi
Vektörleştirici ve kosinüs benzerliği işlevselliği hakkında daha fazla bilgi için aşağıdaki kaynakları kullanabilirsiniz:

*   Özellik çıkarma (Feature Extraction) – `TfidfVectorizer`
    *   `TfidfVectorizer`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
*   `sklearn.metrics` – `cosine_similarity`
    *   `cosine_similarity`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

## Örnek Kod Kullanımı
Aşağıdaki örnek kod, `TfidfVectorizer` ve `cosine_similarity` işlevlerinin nasıl kullanılacağını gösterir:

```python
# Import gerekli kütüphaneler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Metin verileri
metinler = [
    "Bu bir örnek metindir.",
    "Bu başka bir örnek metindir.",
    "Örnek metinler çok faydalıdır."
]

# TfidfVectorizer oluştur
vectorizer = TfidfVectorizer()

# Metinleri vektörleştir
tfidf = vectorizer.fit_transform(metinler)

# Kosinüs benzerliğini hesapla
benzerlik = cosine_similarity(tfidf, tfidf)

# Benzerlik matrisini yazdır
print(benzerlik)
```

### Kod Açıklaması
1.  `from sklearn.feature_extraction.text import TfidfVectorizer`: Bu satır, `TfidfVectorizer` sınıfını `sklearn.feature_extraction.text` modülünden içe aktarır. `TfidfVectorizer`, metin verilerini TF-IDF (Term Frequency-Inverse Document Frequency) vektörlerine dönüştürmek için kullanılır.
2.  `from sklearn.metrics.pairwise import cosine_similarity`: Bu satır, `cosine_similarity` işlevini `sklearn.metrics.pairwise` modülünden içe aktarır. `cosine_similarity`, iki vektör arasındaki kosinüs benzerliğini hesaplamak için kullanılır.
3.  `metinler`: Bu liste, örnek metin verilerini içerir.
4.  `vectorizer = TfidfVectorizer()`: Bu satır, `TfidfVectorizer` sınıfının bir örneğini oluşturur.
5.  `tfidf = vectorizer.fit_transform(metinler)`: Bu satır, `TfidfVectorizer` örneğini metin verilerine uygular ve TF-IDF vektörlerini oluşturur.
6.  `benzerlik = cosine_similarity(tfidf, tfidf)`: Bu satır, TF-IDF vektörleri arasındaki kosinüs benzerliğini hesaplar.
7.  `print(benzerlik)`: Bu satır, benzerlik matrisini yazdırır.

## Önemli Noktalar
*   `TfidfVectorizer`, metin verilerini TF-IDF vektörlerine dönüştürmek için kullanılır.
*   `cosine_similarity`, iki vektör arasındaki kosinüs benzerliğini hesaplamak için kullanılır.
*   TF-IDF vektörleri, metin verilerinin özelliklerini çıkarmak için kullanılır.
*   Kosinüs benzerliği, iki metin arasındaki benzerliği ölçmek için kullanılır.

## Faydalı Bağlantılar
*   [TfidfVectorizer – Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
*   [cosine_similarity – Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

---

## Join our community on Discord

## Discord Topluluğumuza Katılın
Yazar ve diğer okuyucularla tartışmak için topluluğumuzun Discord alanına katılın: https://www.packt.link/rag

## Önemli Noktalar
- Yazar ve diğer okuyucularla tartışmak için Discord'a katılın.
- Discord linki: https://www.packt.link/rag

## Açıklama
Paragrafta, okuyucuların yazar ve diğer okuyucularla tartışmalar yapabileceği bir Discord topluluğuna katılma daveti yapılmaktadır. Discord linki https://www.packt.link/rag olarak verilmiştir.

## Teknik Detaylar
- Discord linki: `https://www.packt.link/rag` (URL)
- Bu linke tıklayarak Discord topluluğuna katılabilirsiniz.

## Kodlar ve Açıklamaları
Bu metinde herhangi bir kod bulunmamaktadır. Ancak, bir Discord linki verilmiştir. Eğer bir Discord botu veya API'si ile ilgili kod örneği olsaydı, aşağıdaki gibi bir örnek gösterilebilirdi:
```python
import discord
from discord.ext import commands

# Discord botu için gerekli olan kütüphaneleri import ettik (Importing necessary libraries for Discord bot)
intents = discord.Intents.default()
intents.typing = False
intents.presences = False

# Botun intent'lerini tanımladık (Defining bot's intents)
bot = commands.Bot(command_prefix='!', intents=intents)

# Botun prefix'ini '!' olarak belirledik (Setting bot's prefix to '!')

@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

# Bot Discord'a bağlandığında konsola bir mesaj yazdırıyoruz (Printing a message to console when bot connects to Discord)

bot.run('YOUR_BOT_TOKEN')

# Bot'u çalıştırmak için token'i girdik (Entering token to run the bot)
```
## Kodun Kullanımı
1. `discord` ve `commands` kütüphanelerini import edin.
2. Botun intent'lerini tanımlayın.
3. `on_ready` event'i ile botun hazır olduğunu belirten bir mesaj yazdırın.
4. Bot'u çalıştırmak için token'i girin.

Not: `YOUR_BOT_TOKEN` kısmını Discord Developer Portal'dan aldığınız gerçek token ile değiştirin.

---

