## Dynamic RAG with Chroma and Hugging Face Llama

## Dinamik RAG (Dynamic RAG) ile Chroma ve Hugging Face Llama Kullanımı
Günümüzün hızla değişen ortamında, hızlı ve bilinçli kararlar alma yeteneği her zamankinden daha önemlidir. Sağlık, bilimsel araştırma ve müşteri hizmetleri yönetimi gibi çeşitli alanlardaki karar vericiler, yalnızca kısa bir süre için geçerli olan gerçek zamanlı verilere ihtiyaç duyarlar. Bir toplantı, yalnızca geçici ama son derece hazırlıklı verilere ihtiyaç duyabilir. Bu nedenle, veri kalıcılığı kavramı değişmektedir. Tüm bilgilerin sonsuza kadar saklanması gerekmez; bunun yerine, birçok durumda odak, belirli ihtiyaçlar için özel olarak hazırlanmış kesin ve ilgili verilerin kullanılmasına kaymaktadır.

## Dinamik RAG'ın Mimarisi (Architecture of Dynamic RAG)
Dinamik RAG'ın mimarisi, geçici Chroma koleksiyonlarının (temporary Chroma collections) oluşturulması ve kullanılması üzerine kuruludur. Her sabah, o günkü toplantılar için gerekli verileri içeren yeni bir koleksiyon oluşturulur, böylece uzun vadeli veri birikimi ve yönetim yükü önlenir.

## Veri Hazırlama (Preparing a Dataset for Dynamic RAG)
Bu bölümde, bilimsel bir veri kümesi (hard science dataset) kullanarak dinamik ve verimli karar alma sürecini desteklemek için bir Python programı oluşturacağız. Bu yaklaşım, modern veri yönetiminin esnekliğini ve verimliliğini vurgulayacaktır.

### Adım 1: Gerekli Kütüphanelerin İçe Aktarılması (Importing Necessary Libraries)
```python
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from chromadb import Client
from chromadb.config import Settings
```
Bu kod, gerekli kütüphaneleri içe aktarır. `pandas` veri işleme için, `torch` ve `transformers` makine öğrenimi modelleri için, `chromadb` ise Chroma veritabanı ile etkileşim için kullanılır.

### Adım 2: SciQ Veri Kümesinin İndirilmesi (Downloading SciQ Dataset)
```python
from datasets import load_dataset
dataset = load_dataset("sciq", split="train")
```
Bu kod, Hugging Face'den SciQ veri kümesini indirir. SciQ, fizik, kimya ve biyoloji gibi konularda binlerce crowdsourced bilim sorusu içerir.

### Adım 3: Chroma Koleksiyonunun Oluşturulması (Creating a Chroma Collection)
```python
client = Client(Settings())
collection = client.create_collection("sciq_daily")
```
Bu kod, Chroma veritabanına bağlanır ve "sciq_daily" adında yeni bir koleksiyon oluşturur.

### Adım 4: Verilerin Embedlenmesi ve Chroma Koleksiyonuna Eklenmesi (Embedding and Upserting Data in a Chroma Collection)
```python
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.detach().numpy()[0]

for example in dataset:
    text = example["question"]
    embedding = embed_text(text)
    collection.add([text], [embedding], [{"source": "sciq"}])
```
Bu kod, SciQ veri kümesindeki soruları embedler ve Chroma koleksiyonuna ekler.

### Adım 5: Koleksiyonun Sorgulanması (Querying a Collection)
```python
query_text = "What is the process of mitosis?"
query_embedding = embed_text(query_text)
results = collection.query(query_embedding, n_results=5)
```
Bu kod, Chroma koleksiyonunu sorgular ve ilgili sonuçları döndürür.

### Adım 6: Hugging Face Llama Modelinin Yapılandırılması (Configuring Hugging Face's Framework for Meta Llama)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

llama_model_name = "meta-llama/Llama-2-7b-chat-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name)
```
Bu kod, Hugging Face Llama modelini yapılandırır.

### Adım 7: Sorgu Sonuçlarına Dayalı Yanıt Oluşturma (Generating a Response based on the Augmented Input)
```python
def generate_response(query_text, results):
    # Llama modeli kullanarak yanıt oluşturma
    inputs = llama_tokenizer(query_text, return_tensors="pt")
    outputs = llama_model.generate(**inputs, max_length=100)
    response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

response = generate_response(query_text, results)
print(response)
```
Bu kod, Llama modeli kullanarak sorgu sonuçlarına dayalı bir yanıt oluşturur.

Bu bölümde, dinamik RAG'ın mimarisini ve Chroma ile Hugging Face Llama kullanarak nasıl uygulanacağını gösterdik. Bu yaklaşım, çeşitli alanlardaki karar vericilere hızlı ve bilinçli kararlar alma olanağı sağlar.

---

## The architecture of dynamic RAG

## Dinamik RAG (Retrieval Augmented Generation) Mimarisi
Dinamik bir ortamda, bilginin her gün değiştiği bir durumda, her sabah 10.000'den fazla soru ve doğrulanmış cevapları toplamak ve bu bilgilere toplantılar sırasında hızlı ve etkili bir şekilde erişmek zorundasınız. Bu dinamik RAG yöntemi, uzun vadeli depolama veya karmaşık altyapılara ihtiyaç duymadan güncel bilgilere erişmenizi sağlar. Bu, verilerin kısa süreliğine ilgili olduğu ancak karar verme açısından kritik olduğu ortamlar için mükemmeldir.

## Dinamik RAG Uygulama Alanları
Bu dinamik RAG yöntemi, çeşitli alanlarda geniş uygulama alanlarına sahiptir:
*   Müşteri desteği (Customer Support): Günlük güncellenen SSS'lere gerçek zamanlı olarak erişerek müşteri sorularına hızlı cevaplar vermek.
*   Sağlık (Healthcare): Tıbbi ekiplerin toplantılar sırasında en son araştırma ve hasta verilerini kullanarak karmaşık sağlık sorularını cevaplaması.
*   Finans (Finance): Finansal analistlerin en son piyasa verilerini sorgulayarak yatırım ve stratejiler hakkında bilinçli kararlar alması.
*   Eğitim (Education): Eğitimcilerin en son eğitim kaynaklarına ve araştırmalara erişerek soruları cevaplaması ve öğrenmeyi geliştirmesi.
*   Teknik Destek (Tech Support): IT ekiplerinin güncellenen teknik belgeleri kullanarak sorunları çözmesi ve kullanıcılara etkili bir şekilde yönlendirmesi.
*   Satış ve Pazarlama (Sales and Marketing): Ekiplerin en son ürün bilgilerine ve piyasa trendlerine hızlı bir şekilde erişerek müşteri sorularını cevaplaması ve strateji oluşturması.

## Dinamik RAG Sisteminin Bileşenleri
Dinamik RAG sisteminin bileşenleri aşağıdaki gibidir:
*   Geçici Chroma koleksiyonu oluşturma (D1, D2, D3, E2): Her sabah, o günkü toplantı için özel olarak geçici bir Chroma koleksiyonu oluşturulur. Bu koleksiyon, toplantı sonrasında saklanmaz ve yalnızca o günkü ihtiyaçlara hizmet eder.
*   İlgili verileri gömme (D1, D2, D3, E2): Koleksiyon, müşteri desteği etkileşimleri, tıbbi raporlar veya bilimsel gerçekler gibi kritik verileri gömer. Bu gömme işlemi, içeriği özel olarak toplantı gündemine göre düzenler.
*   Toplantı öncesi veri doğrulama (D4): Toplantı başlamadan önce, bu geçici Chroma koleksiyonuna karşı bir dizi sorgu çalıştırılarak verilerin doğruluğu ve toplantı hedeflerine uygunluğu sağlanır.
*   Gerçek zamanlı sorgu işleme (G1, G2, G3, G4): Toplantı sırasında, sistem katılımcıların spontan sorgularını işleyecek şekilde tasarlanmıştır. Tek bir soru, belirli bilgilerin alınmasını tetikleyebilir ve bu bilgiler daha sonra Llama'nın girdisini artırmak için kullanılır.

## Chroma Kullanımı
Chroma, koleksiyonlardaki vektörleri depolamak, yönetmek ve aramak için tasarlanmış güçlü, açık kaynaklı, AI-native bir vektör veritabanıdır. Chroma, ihtiyacımız olan her şeye sahiptir ve makinemizde çalıştırılabilir. LLM'leri içeren uygulamalar için de çok uygundur.

## Dinamik RAG'ın Avantajları
Dinamik RAG mimarisi, aşağıdaki avantajlara sahiptir:
*   **Verimlilik ve maliyet etkinliği (Efficiency and cost-effectiveness)**: Chroma'yı geçici depolama ve Llama'yı cevap oluşturma için kullanmak, sistemin hafif olmasını ve devam eden depolama maliyetlerine yol açmamasını sağlar.
*   **Esneklik (Flexibility)**: Sistemin geçici yapısı, her gün yeni verilerin entegrasyonuna olanak tanır ve en güncel bilgilerin her zaman mevcut olmasını sağlar.
*   **Ölçeklenebilirlik (Scalability)**: Yaklaşım, diğer benzer veri kümelerine ölçeklenebilir, yeter ki bu veri kümeleri etkili bir şekilde gömülebilsin ve sorgulanabilsin.
*   **Kullanıcı dostu olma (User-friendliness)**: Sistemin tasarımı basit ve anlaşılırdır, bu da teknik bilgisi sınırlı olan ancak güvenilir cevaplara hızlı bir şekilde ihtiyaç duyan kullanıcılar için erişilebilir olmasını sağlar.

## Dinamik RAG Programı Oluşturma
Dinamik RAG programı oluşturmak için aşağıdaki kodları kullanabilirsiniz:
```python
# Import gerekli kütüphaneler
import chromadb
from chromadb.utils import embedding_functions

# Chroma client'ı oluştur
client = chromadb.Client()

# Embedding fonksiyonu oluştur
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Geçici Chroma koleksiyonu oluştur
collection = client.create_collection(name="temp_collection", embedding_function=embedding_func)

# Verileri koleksiyona ekle
collection.add(
    documents=["Bu bir örnek metin.", "Bu başka bir örnek metin."],
    ids=["id1", "id2"]
)

# Sorgu çalıştır
results = collection.query(
    query_texts=["Örnek metin"],
    n_results=2
)

# Sonuçları yazdır
print(results)
```
Bu kod, Chroma client'ı oluşturur, bir embedding fonksiyonu tanımlar, geçici bir Chroma koleksiyonu oluşturur, verileri koleksiyona ekler, bir sorgu çalıştırır ve sonuçları yazdırır.

## Kod Açıklaması
*   `client = chromadb.Client()`: Chroma client'ı oluşturur.
*   `embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")`: Embedding fonksiyonu oluşturur. Bu fonksiyon, metinleri vektörlere dönüştürmek için kullanılır.
*   `collection = client.create_collection(name="temp_collection", embedding_function=embedding_func)`: Geçici Chroma koleksiyonu oluşturur.
*   `collection.add(documents=["Bu bir örnek metin.", "Bu başka bir örnek metin."], ids=["id1", "id2"])`: Verileri koleksiyona ekler.
*   `results = collection.query(query_texts=["Örnek metin"], n_results=2)`: Sorgu çalıştırır ve en yakın 2 sonucu döndürür.
*   `print(results)`: Sonuçları yazdırır.

---

## Installing the environment

## Ortamın Kurulumu (Installing the Environment)
Ortam, makinemizde veya ücretsiz bir Google Colab hesabında çalıştırabileceğimiz açık kaynaklı ve ücretsiz kaynaklara odaklanmaktadır. Bu bölümde, bu kaynakları Hugging Face ve Chroma ile Google Colab'da çalıştıracağız. İlk olarak, Hugging Face'i kuracağız.

### Hugging Face Kurulumu
Hugging Face'i kurmak için aşağıdaki kodu kullanacağız:
```python
!pip install transformers
```
Bu kod, `transformers` kütüphanesini kurar. `transformers` kütüphanesi, Hugging Face tarafından geliştirilen ve doğal dil işleme (NLP) görevlerinde kullanılan bir kütüphanedir.

**Kod Açıklaması:**
- `!pip install`: Bu komut, Python paket yöneticisi `pip` kullanarak belirtilen paketi kurar.
- `transformers`: Bu, Hugging Face'in `transformers` kütüphanesinin adıdır.

### Chroma Kurulumu
Chroma'yı kurmak için aşağıdaki kodu kullanacağız:
```python
!pip install chromadb
```
Bu kod, `chromadb` kütüphanesini kurar. `chromadb`, Chroma tarafından geliştirilen ve veritabanı işlemleri için kullanılan bir kütüphanedir.

**Kod Açıklaması:**
- `!pip install`: Bu komut, Python paket yöneticisi `pip` kullanarak belirtilen paketi kurar.
- `chromadb`: Bu, Chroma'nın `chromadb` kütüphanesinin adıdır.

### Gerekli Kütüphanelerin İçe Aktarılması (Importing Required Libraries)
Gerekli kütüphaneleri içe aktarmak için aşağıdaki kodu kullanacağız:
```python
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import chromadb
```
**Kod Açıklaması:**
- `import pandas as pd`: Bu satır, `pandas` kütüphanesini `pd` takma adıyla içe aktarır. `pandas`, veri işleme ve analizinde kullanılan bir kütüphanedir.
- `import torch`: Bu satır, `torch` kütüphanesini içe aktarır. `torch`, derin öğrenme modelleri oluşturmak ve eğitmek için kullanılan bir kütüphanedir.
- `from transformers import AutoModelForSequenceClassification, AutoTokenizer`: Bu satır, `transformers` kütüphanesinden `AutoModelForSequenceClassification` ve `AutoTokenizer` sınıflarını içe aktarır. `AutoModelForSequenceClassification`, dizi sınıflandırma görevleri için kullanılan bir modeldir. `AutoTokenizer`, metni tokenlara ayırmak için kullanılan bir tokenizatördür.
- `import chromadb`: Bu satır, `chromadb` kütüphanesini içe aktarır. `chromadb`, veritabanı işlemleri için kullanılan bir kütüphanedir.

Tüm bu kurulum ve içe aktarma işlemleri, Hugging Face ve Chroma'yı Google Colab'da kullanmaya hazır hale getirir.

---

## Hugging Face

## Hugging Face Kullanımı
Hugging Face'in açık kaynaklı kaynaklarını kullanarak Llama modeli için bir veri seti indirme işlemini gerçekleştireceğiz. Bunun için öncelikle https://huggingface.co/ adresinden kayıt olarak Hugging Face API tokeninizi elde etmelisiniz.

## Hugging Face API Tokenini Alma ve Kullanma
API tokeninizi güvenli bir konumda sakladığınızdan emin olun. Google Colab kullanıyorsanız, Google Secret oluşturarak veya manuel olarak tokeninizi girebilirsiniz.

### Google Drive Kullanarak Tokeni Alma
```python
from google.colab import drive
drive.mount('/content/drive')
f = open("drive/MyDrive/files/hf_token.txt", "r")
access_token = f.readline().strip()
f.close()
```
Bu kod, Google Drive'da saklanan `hf_token.txt` dosyasından tokeni okur.

### Manuel Olarak Tokeni Girme
```python
# access_token = "[YOUR HF_TOKEN]"
```
Bu satırı uncomment ederek tokeninizi manuel olarak girebilirsiniz.

### Tokeni Ortam Değişkenine Atama
```python
import os
os.environ['HF_TOKEN'] = access_token
```
Bu kod, tokeni `HF_TOKEN` ortam değişkenine atar.

## Gerekli Kütüphanelerin Kurulumu
Hugging Face'in `datasets` kütüphanesini kurmak için:
```bash
!pip install datasets==2.20.0
```
Bu kurulum, veri setlerini indirmek için gereklidir. Ancak, Google Colab'ın önceden kurulu olan `pyarrow` kütüphanesiyle çakışabilir.

### Hugging Face Transformers Kütüphanesini Kurma
```bash
!pip install transformers==4.41.2
```
Bu kütüphane, dil modellerini kullanmak için gereklidir.

### Accelerate Kütüphanesini Kurma
```bash
!pip install accelerate==0.31.0
```
Bu kütüphane, PyTorch paketlerini GPU'larda çalıştırmak için gereklidir.

## Llama Modelini Kullanma
```python
from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
```
Bu kod, Llama modelini (`meta-llama/Llama-2-7b-chat-hf`) tokenizer olarak yükler.

### Hugging Face Pipeline'ı Oluşturma
```python
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
```
Bu kod, metin oluşturma (`text-generation`) görevi için bir pipeline oluşturur.

*   `transformers.pipeline`: Pipeline oluşturma fonksiyonu.
*   `text-generation`: Pipeline'ın görevi.
*   `model`: Kullanılacak model.
*   `torch_dtype=torch.float16`: PyTorch tensorlarının veri tipi (`float16`) olarak ayarlanır. Bu, dinamik RAG için bellek tüketimini azaltır ve özellikle yarı hassasiyetli hesaplamaları destekleyen GPU'larda işlemleri hızlandırır.
*   `device_map="auto"`: Pipeline'ın otomatik olarak en iyi cihazı (CPU, GPU, çoklu-GPU, vb.) belirlemesini sağlar.

## Özet
Hugging Face API tokeninizi aldıktan sonra, gerekli kütüphaneleri kurarak Llama modelini kullanmaya hazır hale getirdik. Hugging Face pipeline'ı oluşturarak metin oluşturma görevini gerçekleştirebilirsiniz.

---

## Chroma

## Chroma Kurulumu ve Kullanımı
Chroma, açık kaynaklı bir vektör veritabanıdır (vector database). Aşağıdaki komut ile Chroma kurulabilir:
```python
!pip install chromadb==0.5.3
```
Bu komut, Chroma'nın 0.5.3 sürümünü kurar.

## ONNX (Open Neural Network Exchange) ve ONNX Runtime
ONNX, makine öğrenimi (Machine Learning, ML) modellerinin farklı çerçeveler ve donanımlar arasında kullanılmasını sağlayan bir standart formattır. ONNX Runtime ise ONNX modellerini çalıştırmak için performans odaklı bir motordur. Bu motor, ML modelleri için çapraz platform hızlandırıcı olarak görev yapar ve donanıma özgü kütüphanelerle entegrasyon sağlar. Bu sayede modeller çeşitli donanım konfigürasyonları (CPUs, GPUs ve diğer hızlandırıcılar) için optimize edilebilir.

ONNX Runtime'ın kurulumu, Chroma kurulumu ile birlikte yapılır. ONNX Runtime, Hugging Face ile entegre çalışır ve modellerin farklı donanım ve yazılım konfigürasyonlarında uyumlu ve verimli bir şekilde çalışmasını sağlar.

## Hugging Face ve Model Kullanımı
Hugging Face, model yaşam döngüsünün her aşamasında kullanılan bir çerçeve sağlar. Bu çerçeve, önceden eğitilmiş modellere erişimden, modellere ince ayar yapmaya ve onları eğitmeye kadar çeşitli işlemleri destekler. ONNX formatına dönüştürülmüş modeller, ONNX Runtime kullanılarak geniş çapta ve optimize edilmiş bir şekilde dağıtılabilir.

## spaCy ve Doğruluk Hesaplaması
spaCy kütüphanesi kullanılarak, vektör deposuna yapılan sorgulamalara alınan cevaplar ile orijinal tamamlama metni arasındaki doğruluk hesaplanabilir. Aşağıdaki komut, spaCy'den orta büyüklükte bir İngilizce dil modeli indirir:
```python
!python -m spacy download en_core_web_md
```
Bu model, `en_core_web_md` olarak adlandırılır ve genel NLP görevleri için dengelenmiştir. Model, metin benzerliğini hesaplamak için verimlidir.

## Kurulumların Tamamlanması
Yukarıdaki kurulumlar tamamlandıktan sonra, dinamik RAG için gerekli açık kaynaklı, optimize edilmiş ve maliyet etkin kaynaklar başarıyla kurulmuş olur. Artık programın çekirdeğini çalıştırmaya hazırız.

### Kullanılan Kodlar ve Açıklamaları

1. Chroma Kurulumu:
   ```python
   !pip install chromadb==0.5.3
   ```
   Bu kod, Chroma'nın 0.5.3 sürümünü kurar.

2. spaCy Model Kurulumu:
   ```python
   !python -m spacy download en_core_web_md
   ```
   Bu kod, spaCy'den `en_core_web_md` adlı İngilizce dil modelini indirir.

### İçe Aktarılan Kütüphaneler (Import Kısımları)

Yukarıdaki metinde doğrudan içe aktarılan kütüphaneler (import) gösterilmemiştir. Ancak Chroma, ONNX Runtime ve spaCy kütüphanelerini kullanmak için ilgili Python betiklerinde aşağıdaki import komutları kullanılabilir:

```python
import chromadb
import onnxruntime
import spacy
```

Bu import komutları, ilgili kütüphaneleri Python betiğine dahil etmek için kullanılır.

### Önemli Noktalar

*   Chroma, açık kaynaklı bir vektör veritabanıdır.
*   ONNX, makine öğrenimi modellerinin farklı çerçeveler ve donanımlar arasında kullanılmasını sağlar.
*   ONNX Runtime, ONNX modellerini çalıştırmak için performans odaklı bir motordur.
*   Hugging Face, model yaşam döngüsünün her aşamasında kullanılan bir çerçeve sağlar.
*   spaCy, metin benzerliğini hesaplamak için kullanılır.

### Teknik Terimler

*   Vektör Veritabanı (Vector Database)
*   Makine Öğrenimi (Machine Learning, ML)
*   ONNX (Open Neural Network Exchange)
*   ONNX Runtime
*   Hugging Face
*   spaCy
*   NLP (Natural Language Processing, Doğal Dil İşleme)

---

## Activating session time

## Oturum Süresini Aktif Hale Getirme (Activating Session Time)

Gerçek hayattaki dinamik RAG projelerinde çalışırken, zaman çok önemlidir! Örneğin, günlük karar alma toplantısı saat 10'da ise, RAG hazırlık ekibi bu toplantıya hazırlık için saat 8'de başlamak zorunda kalabilir. Bu süre zarfında, toplantının amacı için gerekli olan verileri çevrimiçi olarak toplamak, şirket verilerini işlemek veya başka yollarla hazırlamak zorundadırlar.

İlk olarak, eğer mümkünse bir GPU (Graphics Processing Unit) etkinleştirin. Google Colab'da, örneğin, **Runtime** > **Change runtime type** > **Change runtime type** sekmesine gidin ve eğer mümkünse ve uygun bir GPU seçin. Eğer yoksa, notebook biraz daha uzun sürecek ancak CPU'da çalışacaktır.

Daha sonra, bu bölümdeki her bir hücreyi sırasıyla çalıştırarak süreci derinlemesine anlayın. Aşağıdaki kod, ortam kurulduktan sonra oturum süresinin ölçülmesini sağlar:
```python
import time

# Oturum başlangıç zamanını kaydet
session_start_time = time.time()
```
**Kod Açıklaması:** `time.time()` fonksiyonu, mevcut zamanı saniye cinsinden döndürür. Bu kod, oturum başlangıç zamanını `session_start_time` değişkenine kaydeder.

Son olarak, oturumu yeniden başlatın, **Runtime** > **Run all** sekmesine gidin ve tüm hücreleri çalıştırın. Program sona erdiğinde, **Total session time** bölümüne gidin. Burada, bir hazırlık çalışmasının ne kadar sürdüğüne dair bir tahmin göreceksiniz.

Günlük toplantıya kalan zaman içinde, verileri, sorguları ve model parametrelerini ihtiyacınıza göre birkaç kez ayarlayabilirsiniz. Bu dinamik RAG yaklaşımı, bu becerilere sahip herhangi bir takımı, hızlı hareket eden bu dünyada değerli bir varlık haline getirecektir.

## Önemli Noktalar:
* GPU etkinleştirme (GPU activation)
* Oturum süresini ölçme (Measuring session time)
* Verileri hazırlama ve model parametrelerini ayarlama (Preparing data and tweaking model parameters)
* Dinamik RAG yaklaşımı (Dynamic RAG approach)

## Kod Kullanımı:
Yukarıdaki kodu, notebook'unuzun başlangıcında, ortam kurulduktan sonra kullanın. Bu kod, oturum süresini ölçmeye başlar. Program sona erdiğinde, toplam oturum süresini hesaplamak için `time.time()` fonksiyonunu tekrar kullanabilirsiniz:
```python
session_end_time = time.time()
total_session_time = session_end_time - session_start_time
print("Toplam oturum süresi:", total_session_time, "saniye")
```
**Kod Açıklaması:** `session_end_time` değişkeni, oturum bitiş zamanını kaydeder. `total_session_time` değişkeni, oturum başlangıç ve bitiş zamanları arasındaki farkı hesaplar. Sonuç, saniye cinsinden yazılır.

---

## Downloading and preparing the dataset

## Veri Kümesinin İndirilmesi ve Hazırlanması (Downloading and Preparing the Dataset)

Bu bölümde, Welbl, Liu ve Gardner (2017) tarafından oluşturulan SciQ veri kümesini (dataset) kullanacağız. SciQ veri kümesi, çoklu seçimli (multiple-choice) fen bilimleri sorularını oluşturmak için crowdsourcing yöntemiyle oluşturulmuştur. Veri kümesi, 13.679 çoklu seçimli sorudan oluşmaktadır.

### Veri Kümesinin Yüklenmesi (Loading the Dataset)

İlk olarak, gerekli kütüphaneleri (libraries) içe aktaralım (import):
```python
from datasets import load_dataset
import pandas as pd
```
SciQ veri kümesini HuggingFace'den yükleyelim:
```python
dataset = load_dataset("sciq", split="train")
```
### Veri Kümesinin Filtrelenmesi (Filtering the Dataset)

Veri kümesini, `support` ve `correct_answer` sütunları (columns) boş olmayan (non-empty) satırları (rows) içerecek şekilde filtreleyelim:
```python
filtered_dataset = dataset.filter(lambda x: x["support"] != "" and x["correct_answer"] != "")
```
Filtrelenen satır sayısını yazdıralım:
```python
print("Number of questions with support: ", len(filtered_dataset))
```
Çıktı (output):
```
Number of questions with support: 10481
```
### Veri Kümesinin Temizlenmesi (Cleaning the Dataset)

DataFrame'i temizleyerek ihtiyacımız olan sütunlara odaklanalım. Yanlış cevapları (distractors) içeren sütunları kaldıralım:
```python
df = pd.DataFrame(filtered_dataset)
columns_to_drop = ['distractor3', 'distractor1', 'distractor2']
df.drop(columns=columns_to_drop, inplace=True)
```
### Veri Kümesinin Birleştirilmesi (Merging the Dataset)

`correct_answer` ve `support` sütunlarını birleştirerek yeni bir `completion` sütunu oluşturalım:
```python
df['completion'] = df['correct_answer'] + " because " + df['support']
```
`completion` sütununda NaN değerleri (NaN values) bulunmadığından emin olalım:
```python
df.dropna(subset=['completion'], inplace=True)
```
### Veri Kümesinin Hazır Hale Getirilmesi (Preparing the Dataset)

DataFrame'in şeklini (shape) yazdıralım:
```python
df.shape
```
Çıktı:
```
(10481, 4)
```
Sütun adlarını (column names) yazdıralım:
```python
print(df.columns)
```
Çıktı:
```
Index(['question', 'correct_answer', 'support', 'completion'], dtype='object')
```
Veri kümesi artık gömme (embedding) ve upsert işlemleri için hazırdır.

## Önemli Noktalar (Key Points)

* SciQ veri kümesi, çoklu seçimli fen bilimleri sorularını içerir.
* Veri kümesi, `support` ve `correct_answer` sütunları boş olmayan satırları içerecek şekilde filtrelenmiştir.
* Yanlış cevapları içeren sütunlar kaldırılmıştır.
* `correct_answer` ve `support` sütunları birleştirilerek yeni bir `completion` sütunu oluşturulmuştur.
* Veri kümesi, gömme ve upsert işlemleri için hazırdır.

---

## Embedding and upserting the data in a Chroma collection

## Chroma Koleksiyonunda Veri Embed Etme ve Upsert İşlemi
Chroma istemcisini (client) oluşturarak ve bir koleksiyon adı tanımlayarak başlayacağız.

## Chroma İstemcisini Oluşturma ve Koleksiyon Adı Tanımlama
Öncelikle Chroma kütüphanesini import edip, bir istemci (client) örneği oluşturacağız. Varsayılan Chroma istemcisi geçicidir (ephemeral), yani diske kaydetmez.

```python
import chromadb
client = chromadb.Client()
collection_name = "sciq_supports6"
```

Bu kod, Chroma kütüphanesini import eder ve bir `client` örneği oluşturur. `collection_name` değişkeni, oluşturulacak koleksiyonun adını tanımlar.

## Koleksiyonun Varlığını Kontrol Etme
Koleksiyonu oluşturmadan ve verileri koleksiyona upsert etmeden önce, koleksiyonun zaten var olup olmadığını doğrulamamız gerekir.

```python
collections = client.list_collections()
collection_exists = any(collection.name == collection_name for collection in collections)
print("Collection exists:", collection_exists)
```

Bu kod, mevcut koleksiyonları listeler ve tanımladığımız `collection_name` adlı koleksiyonun var olup olmadığını kontrol eder. Eğer koleksiyon varsa `True`, yoksa `False` döndürür.

## Koleksiyon Oluşturma
Eğer koleksiyon yoksa, `collection_name` ile tanımladığımız isimde yeni bir koleksiyon oluşturacağız.

```python
if collection_exists != True:
    collection = client.create_collection(collection_name)
else:
    print("Collection ", collection_name, " exists:", collection_exists)
```

Bu kod, eğer koleksiyon yoksa (`collection_exists` `False` ise), `client.create_collection()` methodunu kullanarak yeni bir koleksiyon oluşturur.

## Koleksiyon Yapısını İnceleme
Oluşturduğumuz koleksiyonun sözlük (dictionary) yapısını inceleyelim.

```python
results = collection.get()
for result in results:
    print(result)
```

Bu kod, koleksiyondaki her bir öğenin sözlük yapısını yazdırır. Çıktıda `ids`, `embeddings`, `metadatas`, `documents`, `uris`, `data`, ve `included_ids` gibi anahtarlar görünür.

## Anahtar Alanları İnceleme
Üç önemli anahtar alanını inceleyelim:
- `ids`: Koleksiyondaki her bir öğenin benzersiz tanımlayıcıları (unique identifiers).
- `embeddings`: Dokümanların gömülü vektörleri (embedded vectors).
- `documents`: Doğru cevap ve destek içeriğinin birleştirildiği `completion` sütununu ifade eder.

## Hafif ve Hızlı Bir LLM Modeline İhtiyaç
Dinamik RAG ortamımız için hafif ve hızlı bir LLM modeline ihtiyacımız var.

---

## Selecting a model

## Model Seçimi (Model Selection)
Chroma, varsayılan olarak `all-MiniLM-L6-v2` modelini kullanır. Ancak, bu modeli kullandığımızdan emin olmak için modeli initialize edelim.

## Model Tanımlama (Model Definition)
Model adını tanımlayarak başlayalım:
```python
model_name = "all-MiniLM-L6-v2"
```
Bu kod, embedding ve sorgulama (querying) işlemleri için kullanılacak modelin adını tanımlar.

## all-MiniLM-L6-v2 Modeli Hakkında (About all-MiniLM-L6-v2 Model)
`all-MiniLM-L6-v2` modeli, Wang ve diğerleri (2021) tarafından geliştirilen bir model sıkıştırma yöntemi kullanılarak tasarlanmıştır. Bu yöntem, transformer modellerinin bileşenleri arasındaki self-attention ilişkilerini distilleme üzerine odaklanır. Bu yaklaşım, öğretmen (teacher) ve öğrenci (student) modelleri arasındaki dikkat başlıkları (attention heads) sayısında esneklik sağlar ve sıkıştırma verimliliğini artırır.

## Modelin Özellikleri (Model Properties)
`all-MiniLM-L6-v2` modeli, ONNX ile Chroma'ya entegre edilmiştir. Modelin sihirli tarafı, bir öğretmen modeli ve öğrenci modeli aracılığıyla sıkıştırma ve bilgi distilasyonu (knowledge distillation) üzerine kuruludur.

### Öğretmen Modeli (Teacher Model)
Öğretmen modeli, genellikle BERT, RoBERTa ve XLM-R gibi daha büyük ve karmaşık bir modeldir. Bu model, kapsamlı bir veri seti üzerinde önceden eğitilmiştir ve yüksek doğruluk ve görevleri hakkında derin bir anlayışa sahiptir.

### Öğrenci Modeli (Student Model)
Öğrenci modeli, `all-MiniLM-L6-v2`, öğretmen modelinin davranışını taklit etmek üzere eğitilen daha küçük ve daha az karmaşık bir modeldir. Bu model, dinamik RAG mimarisi için çok etkili olacaktır.

## Kod Parçası (Code Snippet)
```python
import chroma

# Model adını tanımla
model_name = "all-MiniLM-L6-v2"

# Chroma'yı model ile initialize et
chroma.init(model_name=model_name)
```
Bu kod parçası, `all-MiniLM-L6-v2` modelini kullanarak Chroma'yı initialize eder.

## Kullanım (Usage)
`all-MiniLM-L6-v2` modeli, embedding ve sorgulama işlemlerini hızlandırmak için kullanılır. Bu model, büyük ve karmaşık modellerin (GPT-4o gibi) günlük görevlerde kullanılmasını sağlar.

## Veri Embedding İşlemi (Data Embedding Process)
Sonraki adım, verileri embedding işlemidir. Bu işlem, `all-MiniLM-L6-v2` modeli kullanılarak gerçekleştirilecektir.

---

## Embedding and storing the completions

## Embedding ve Tamamlamaların Depolanması (Embedding and Storing the Completions)
Chroma koleksiyonunda veri gömmek (embedding) ve güncellemek (upserting) sorunsuz ve özlüdür. Bu senaryoda, `df` veri setinden elde edilen `completion_list` adlı listedeki tüm tamamlamaları gömüp (embed) güncelleyeceğiz (upsert).

### Veri Hazırlama (Data Preparation)
Öncelikle, gömülecek ve depolanacak soruların sayısını belirleyelim:
```python
ldf = len(df)
nb = ldf  # gömülecek ve depolanacak soruların sayısı
```
 Ardından, `completion` sütununu string listesine dönüştürelim:
```python
completion_list = df["completion"][:nb].astype(str).tolist()
```
Bu kod, `df` veri setinin `completion` sütunundaki ilk `nb` satırını string tipine dönüştürür ve bir liste haline getirir.

### Embedding ve Depolama (Embedding and Storing)
 Koleksiyonun var olup olmadığını kontrol edelim ve verileri yalnızca bir kez yükleyelim:
```python
if collection_exists != True:
    collection.add(
        ids=[str(i) for i in range(0, nb)],  # ID'ler string olarak
        documents=completion_list,
        metadatas=[{"type": "completion"} for _ in range(0, nb)],
    )
```
Bu kod, koleksiyon yoksa (`collection_exists` False ise), `completion_list` içerisindeki metinleri gömer (embed) ve `collection` nesnesine ekler. `ids` parametresi, her bir belge için benzersiz bir kimlik (ID) sağlar. `metadatas` parametresi, her bir belge için meta veri sağlar; bu örnekte, her bir belgenin türü (`type`) "completion" olarak belirlenmiştir.

### İşlem Zamanını Ölçme (Measuring Response Time)
 İşlem zamanını ölçmek için `time` modülünü kullanalım:
```python
import time
start_time = time.time()  # İşlem öncesi zamanı kaydet
# yukarıdaki kodları çalıştır
response_time = time.time() - start_time  # İşlem sonrası zamanı hesapla
print(f"Response Time: {response_time:.2f} seconds")
```
Bu kod, işlem öncesi ve sonrası zamanı kaydederek işlem süresini hesaplar ve saniye cinsinden yazdırır.

### Sonuç (Output)
Chroma, varsayılan modelini (`all-MiniLM-L6-v2`) kullanarak embedding işlemini gerçekleştirir. İşlem sonucu, 10.000'den fazla belge için tatmin edici bir işlem zamanı gösterir:
```
Response Time: 234.25 seconds
```
Bu işlem zamanı, kullanılan donanıma (örneğin, GPU varlığı) bağlı olarak değişebilir. Erişilebilir bir GPU kullanıldığında, işlem zamanı dinamik RAG senaryoları için gereken ihtiyaçlara uygun hale gelir.

### Chroma Vektör Deposu (Chroma Vector Store)
Artık Chroma vektör deposu doldurulmuştur. Gömülü (embedded) verileri inceleyebiliriz.

---

## Displaying the embeddings

## Gömmeleri (Embeddings) Görüntüleme
Program şimdi gömmeleri (embeddings) getiriyor ve ilkini görüntülüyor:

## Kod
```python
# Gömmeleri içeren koleksiyonu getir (Fetch the collection with embeddings included)
result = collection.get(include=['embeddings'])

# Sonuçtan ilk gömme (embedding) çıkar (Extract the first embedding from the result)
first_embedding = result['embeddings'][0]

# İlk gömmenin (embedding) uzunluğu ile çalışmanız gerekirse:
embedding_length = len(first_embedding)

print("İlk gömme (First embedding):", first_embedding)
print("Gömme uzunluğu (Embedding length):", embedding_length)
```

## Açıklama
Yukarıdaki kod, `collection` nesnesinden gömmeleri (embeddings) içeren verileri getirir ve ilk gömme (embedding) değerini `first_embedding` değişkenine atar. Daha sonra bu gömmenin uzunluğunu hesaplar ve ekrana yazdırır.

## Kullanılan Kod Parçalarının Açıklaması
- `collection.get(include=['embeddings'])`: Bu satır, `collection` nesnesinden gömmeleri içeren verileri getirir. `include` parametresi, hangi verilerin dahil edileceğini belirtir. Burada sadece `'embeddings'` dahil edilmiştir.
- `result['embeddings'][0]`: Bu satır, getirilen verilerden ilk gömme (embedding) değerini çıkarır. `result` bir sözlük (dictionary) nesnesidir ve `'embeddings'` anahtarına karşılık gelen değer bir liste (list) dir. Bu listedeki ilk eleman (`[0]`) ilk gömme (embedding) değeridir.
- `len(first_embedding)`: Bu satır, ilk gömme (embedding) değerinin uzunluğunu hesaplar.

## Çıktı
Çıktı, ilk gömme (embedding) değerini ve bu değerin uzunluğunu gösterir:
```
İlk gömme (First embedding): [0.03689068928360939, -0.05881563201546669, -0.04818134009838104,…]
Gömme uzunluğu (Embedding length): 384
```

## all-MiniLM-L6-v2 Modeli Hakkında
all-MiniLM-L6-v2 modeli, metin verilerinin karmaşıklığını azaltarak cümleleri ve paragrafları 384 boyutlu (dimensional) bir uzaya (space) eşler. Bu, one-hot encoded vektörlerin tipik boyutluluğundan (örneğin OpenAI text-embedding-ada-002'nin 1,526 boyutu) çok daha düşüktür. Bu model, yoğun (dense) vektörler kullanır, yani vektör uzayının tüm boyutlarını kullanarak farklı belgeler arasında nüanslı anlamsal (semantic) ilişkiler üretir. Bu, seyrek (sparse) vektör modellerinin (örneğin Bag-of-Words (BoW) modeli) aksine, kelimelerin sırasını ve çevresini yakalar, bu da LLM'leri eğitirken metnin anlamını anlamak için çok önemlidir.

## Sonuç
Dokümanları daha küçük boyutlu bir uzayda yoğun vektörler olarak gömmüş olduk ve tatmin edici sonuçlar üreteceğiz.

---

## Querying the collection

## Koleksiyon Sorgulama (Querying the Collection)
Bu bölümde, Chroma vektör deposuna (vector store) entegre semantik arama (semantic search) işlevselliği kullanılarak bir sorgu çalıştırılır. Sorgu, başlangıç veri setindeki (initial dataset) soruların vektör temsillerini (vector representations) sorgular: `dataset["question"][:nbq]`.

### Sorgu İşlemi
Sorgu, her bir soru için en ilgili veya benzer belgeleri (`n_results=1`) ister. Her bir soru metni bir vektöre dönüştürülür. Ardından, Chroma vektör benzerlik araması (vector similarity search) yaparak gömülü vektörleri (embedded vectors) belge vektörleri veritabanımızla karşılaştırır ve vektör benzerliğine göre en yakın eşleşmeyi bulur.

```python
import time
start_time = time.time()  # İsteğin başlangıcında zamanlayıcı başlat
results = collection.query(
    query_texts=df["question"][:nb],
    n_results=1
)
response_time = time.time() - start_time  # Yanıt süresini ölç
print(f"Yanıt Süresi: {response_time:.2f} saniye")  # Yanıt süresini yazdır
```

### Yanıt Süresi
Çıktı, 10.000'den fazla sorgu için tatmin edici bir yanıt süresi gösterir:
```
Yanıt Süresi: 199.34 saniye
```

## Doğrulama ve Analiz (Validation and Analysis)
10.000'den fazla sorguyu analiz edeceğiz. SpaCy kullanarak bir sorgunun sonucunu değerlendireceğiz ve orijinal tamamlama (original completion) ile karşılaştıracağız.

### SpaCy Modelini Yükleme
```python
import spacy
import numpy as np
nlp = spacy.load('en_core_web_md')  # Önceden eğitilmiş SpaCy dil modelini yükle
```

### Benzerlik Fonksiyonu
İki metin arasındaki benzerliği hesaplayan bir fonksiyon tanımlar:
```python
def simple_text_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    vector1 = doc1.vector
    vector2 = doc2.vector
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 0.0
    else:
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return similarity
```

### Doğrulama Çalışması
10.000 sorgu üzerinde tam bir doğrulama çalışması yapar:
```python
nbqd = 100  # Görüntülenecek yanıt sayısı
acc_counter = 0
display_counter = 0
for i, q in enumerate(df['question'][:nb]):
    original_completion = df['completion'][i]
    retrieved_document = results['documents'][i][0]
    similarity_score = simple_text_similarity(original_completion, retrieved_document)
    if similarity_score > 0.7:
        acc_counter += 1
    display_counter += 1
    if display_counter <= nbqd or display_counter > nb - nbqd:
        print(i, " ", f"Soru: {q} ")
        print(f"Alınan belge: {retrieved_document} ")
        print(f"Orijinal tamamlama: {original_completion} ")
        print(f"Benzerlik Skoru: {similarity_score:.2f} ")
        print()  # Girdiler arasında daha iyi okunabilirlik için boş satır
```

### Doğruluk Hesaplaması
Tüm sonuçlar analiz edildikten sonra, 10.000'den fazla sorgu için doğruluk hesaplanır:
```python
if nb > 0:
    acc = acc_counter / nb
print(f"Belge sayısı: {nb:.2f} ")
print(f"Genel benzerlik skoru: {acc:.2f} ")
```

### Sonuç
Çıktı, tüm soruların ilgili sonuçlar döndürdüğünü gösterir:
```
Belge sayısı: 10481.00
Genel benzerlik skoru: 1.00
```

---

## Prompt and retrieval

## Prompt ve Retrieval (İstem ve Erişim) Açıklaması
Bu bölüm, gerçek zamanlı sorgulama yapılan toplantılarda kullanılan bir bölümdür. Arayüzü ihtiyaçlarınıza göre uyarlayabilirsiniz. Fonksiyonellik üzerine odaklanacağız.

## İlk Sorgu ve Varyantları
İlk istem (prompt), başlangıç veri kümesinden gelen tam metindir. Toplantıdaki bir katılımcının veya kullanıcının soruyu bu şekilde soracağı pek olası değildir. Ancak sistemin çalışıp çalışmadığını doğrulamak için kullanabiliriz.

### Kod Parçası ve Açıklaması
```python
import time
import textwrap

# Başlangıç zamanı ölçümü
start_time = time.time()

# Koleksiyonu sorgulama
results = collection.query(
    query_texts=[prompt],  # Sorgu metni listesi olarak beklendiği gibi
    n_results=1  # Alınacak sonuç sayısı
)

# Yanıt zamanı ölçümü
response_time = time.time() - start_time

# Yanıt zamanını yazdırma
print(f"Yanıt Zamanı: {response_time:.2f} saniye\n")

# Dokümanların alındığını kontrol etme
if results['documents'] and len(results['documents'][0]) > 0:
    # Daha iyi okunabilirlik için textwrap kullanma
    wrapped_question = textwrap.fill(prompt, width=70)
    wrapped_document = textwrap.fill(results['documents'][0][0], width=70)

    # Biçimlendirilmiş sonuçları yazdırma
    print(f"Soru: {wrapped_question}")
    print("\n")
    print(f"Alınan Doküman: {wrapped_document}")
    print()
else:
    print("Doküman alınamadı.")
```
Bu kod, `collection.query` metodunu kullanarak bir sorgu gerçekleştirir ve yanıt zamanını ölçer. `textwrap.fill` fonksiyonu, daha iyi okunabilirlik için metni 70 karakter genişliğinde biçimlendirir.

## Varyantlar ve Zorluk Derecesi
İlk varyant (`variant 1`), başlangıç sorusuna benzer ve sorulabilir. İkinci varyant (`variant 2`), başlangıç sorusundan sapar ve zorlayıcı olabilir. Sistemin stabilitesini ve beklendiği gibi yanıt verip vermediğini test etmek için bu varyantları kullanabilirsiniz.

## Sonuç ve İyileştirme
Sistem, sorguyu başarıyla gerçekleştirdi ve ilgili bir doküman aldı. Yanıt zamanı hızlıydı: `Yanıt Zamanı: 0.03 saniye`. Alınan doküman, sorulan soru ile alakalıydı.

## Hugging Face Llama ile NLP Özeti
Hugging Face Llama, bu yanıta dayanarak kısa bir Doğal Dil İşleme (NLP) özeti yazacaktır.

### Önemli Noktalar
* Gerçek zamanlı sorgulama için `collection.query` metodu kullanılır.
* Yanıt zamanı `time.time()` fonksiyonu ile ölçülür.
* `textwrap.fill` fonksiyonu, metni biçimlendirmek için kullanılır.
* Sistemin stabilitesi ve performansı, farklı varyantlarla test edilmelidir.
* Hugging Face Llama, alınan yanıttan bir NLP özeti oluşturur.

---

## RAG with Llama

## RAG with Llama (Retrieval-Augmented Generation ile Llama)

Bu bölümde, meta-llama/Llama-2-7b-chat-hf modelinin nasıl yapılandırılacağı ve kullanılacağı anlatılmaktadır.

### Llama Modelinin Yapılandırılması

Llama 2 modelinin davranışını yapılandırmak için bir fonksiyon oluşturmalıyız:
```python
def LLaMA2(prompt):
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=100, 
        temperature=0.5, 
        repetition_penalty=2.0, 
        truncation=True
    )
    return sequences
```
Bu fonksiyon, Llama 2 modelini kullanarak verilen `prompt` (iprompt) için bir çıktı üretir.

#### Parametrelerin Açıklaması

*   `prompt`: Modelin çıktı üretmek için kullandığı girdi metni.
*   `do_sample`: Boolean değer (True veya False). True olduğunda, model tokenleri olasılık dağılımına göre rastgele seçer, böylece daha çeşitli çıktılar üretir.
*   `top_k`: Örnekleme sürecinde dikkate alınacak en yüksek olasılıklı kelime tokenlerinin sayısını sınırlar. 10 olarak ayarlandığında, model en olası 10 token arasından seçim yapar.
*   `num_return_sequences`: Bağımsız olarak üretilen çıktıların sayısını belirtir. Burada 1 olarak ayarlanmıştır, yani fonksiyon her bir prompt için bir çıktı döndürür.
*   `eos_token_id`: Tokenize edilmiş biçimde bir dizinin sonunu işaret eden token. Bu token üretildiğinde, model daha fazla token üretmeyi durdurur.
*   `max_new_tokens`: Modelin üretebileceği yeni token sayısını sınırlar. Burada 100 olarak ayarlanmıştır, yani çıktı en fazla 100 token uzunluğunda olur.
*   `temperature`: Örnekleme sürecinde rastgeleliği kontrol eder. 0.5 gibi bir değer, modelin çıktılarını daha az rastgele ve daha odaklı hale getirir, ancak yine de bazı çeşitliliklere izin verir.
*   `repetition_penalty`: Modelin aynı tokeni tekrarlamasını engelleyen bir değiştiricidir. 2.0 gibi bir değer, daha önce kullanılan tokenlerin tekrar kullanılma olasılığını azaltır, böylece daha çeşitli ve daha az tekrarlayan metinler üretilir.
*   `truncation`: Etkinleştirildiğinde, çıktının `max_new_tokens` tarafından belirtilen maksimum uzunluğu aşmamasını sağlar, fazla tokenleri keser.

### Llama Modelinin Kullanılması

İlk olarak, `iprompt` ve `results` değişkenlerini tanımlayalım:
```python
iprompt = 'Read the following input and write a summary for beginners.'
lprompt = iprompt + " " + results['documents'][0][0]
```
Ardından, Llama modelini çağırmak için `LLaMA2` fonksiyonunu kullanırız:
```python
import time
start_time = time.time() 
response = LLaMA2(lprompt)
```
Üretilen çıktıyı ve yanıt süresini alırız:
```python
for seq in response:
    generated_part = seq['generated_text'].replace(iprompt, '') 
    response_time = time.time() - start_time 
    print(f"Response Time: {response_time:.2f} seconds") 
```
Çıktıyı daha güzel bir formatta göstermek için:
```python
import textwrap
wrapped_response = textwrap.fill(response[0]['generated_text'], width=70)
print(wrapped_response)
```
Bu, teknik olarak kabul edilebilir bir özet çıktısı verir.

### Farklı LLM'lerin Kullanılması

Gerekirse, meta-llama/Llama-2-7b-chat-hf modelini GPT-4o gibi başka bir modelle değiştirebiliriz. Dinamik RAG'de performans en önemli kuraldır.

Önemli noktalar:

*   Llama 2 modelinin yapılandırılması ve kullanılması
*   Çeşitli parametrelerin ayarlanması (`do_sample`, `top_k`, `num_return_sequences`, `eos_token_id`, `max_new_tokens`, `temperature`, `repetition_penalty`, `truncation`)
*   Llama modelinin çıktı üretmesi ve yanıt süresi ölçülmesi
*   Farklı LLM'lerin kullanılması ve performansın önemi

Tüm kod:
```python
import time
import textwrap

def LLaMA2(prompt):
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=100, 
        temperature=0.5, 
        repetition_penalty=2.0, 
        truncation=True
    )
    return sequences

iprompt = 'Read the following input and write a summary for beginners.'
lprompt = iprompt + " " + results['documents'][0][0]

start_time = time.time() 
response = LLaMA2(lprompt)

for seq in response:
    generated_part = seq['generated_text'].replace(iprompt, '') 
    response_time = time.time() - start_time 
    print(f"Response Time: {response_time:.2f} seconds") 

wrapped_response = textwrap.fill(response[0]['generated_text'], width=70)
print(wrapped_response)
```

---

## Deleting the collection

## Koleksiyonu Silme (Deleting the Collection)
Koleksiyonu manuel olarak silmek için aşağıdaki kod kullanılabilir:
```python
#client.delete_collection(collection_name)
client.delete_collection(collection_name)
```
Bu kod, `client` nesnesi üzerinden `delete_collection` metodunu çağırarak belirtilen `collection_name` adlı koleksiyonu siler.

## Oturumu Kapatma (Closing the Session)
Oturumu kapatmak da geçici dinamik RAG koleksiyonunu silmek için kullanılabilir.

## Koleksiyonun Varlığını Kontrol Etme (Checking Collection Existence)
Oluşturduğumuz koleksiyonun hala var olup olmadığını kontrol edebiliriz:
```python
# Tüm koleksiyonları listele (List all collections)
collections = client.list_collections()

# Belirli bir koleksiyonun varlığını kontrol et (Check if the specific collection exists)
collection_exists = any(collection.name == collection_name for collection in collections)
print("Koleksiyon var:", collection_exists)
```
Bu kod, `client` nesnesi üzerinden `list_collections` metodunu çağırarak tüm koleksiyonları listeler ve daha sonra `any` fonksiyonu ile belirtilen `collection_name` adlı koleksiyonun varlığını kontrol eder.

## Kullanım Senaryoları (Usage Scenarios)
Eğer bir oturumda koleksiyon üzerinde hala çalışıyorsak, cevap `True` olacaktır:
```
Koleksiyon var: True
```
Eğer koleksiyonu kod ile sildiğimizde veya oturumu kapattığımızda, cevap `False` olacaktır.

## Toplam Oturum Zamanı (Total Session Time)
Şimdi, toplam oturum zamanını inceleyelim.

### Kod Açıklamaları
* `client.delete_collection(collection_name)`: Belirtilen koleksiyonu siler. (`collection_name` parametresi silinecek koleksiyonun adıdır.)
* `client.list_collections()`: Tüm koleksiyonları listeler.
* `any(collection.name == collection_name for collection in collections)`: Belirtilen koleksiyonun varlığını kontrol eder. (`collection_name` parametresi aranan koleksiyonun adıdır.)

### Örnek Kullanım
```python
import pinecone

# Pinecone client oluşturma
client = pinecone.init(api_key="API_KEY", environment="ENVIRONMENT")

# Koleksiyon oluşturma
collection_name = "example-collection"
client.create_collection(collection_name)

# Koleksiyonu silme
client.delete_collection(collection_name)

# Koleksiyonun varlığını kontrol etme
collections = client.list_collections()
collection_exists = any(collection.name == collection_name for collection in collections)
print("Koleksiyon var:", collection_exists)
```
Bu örnekte, önce Pinecone client oluşturulur, daha sonra bir koleksiyon oluşturulur, koleksiyon silinir ve son olarak koleksiyonun varlığı kontrol edilir.

---

## Total session time

## Toplam Oturum Zamanı (Total Session Time)

Aşağıdaki kod, oturum başlangıcı ile "Installing the environment ( Ortamın Kurulumu )" bölümünden hemen sonraki zaman arasındaki farkı ölçer.

## Kod
```python
end_time = time.time() - session_start_time  # Zamanı ölç (Measure response time)
print(f"Oturum hazırlık zamanı (Session preparation time): {response_time:.2f} saniye (seconds)")
```
## Açıklama

Bu kod, `time` modülünü kullanarak oturum başlangıç zamanı (`session_start_time`) ile mevcut zaman arasındaki farkı hesaplar ve bu farkı `end_time` değişkenine atar. Daha sonra, `response_time` değişkeninin değerini `.2f` formatında yazdırır.

## Kullanılan Kod Parçalarının Ayrıntılı Açıklaması

*   `time.time()`: Bu fonksiyon, mevcut zamanı (saniye cinsinden) döndürür. 
*   `session_start_time`: Oturum başlangıç zamanını temsil eder. 
*   `end_time = time.time() - session_start_time`: Oturum başlangıç zamanı ile mevcut zaman arasındaki farkı hesaplar.
*   `print(f"Oturum hazırlık zamanı (Session preparation time): {response_time:.2f} saniye (seconds)")`: Hesaplanan zamanı `.2f` formatında (iki ondalık basamaklı) yazdırır.

## İlgili Kütüphaneler

Bu kodun çalışması için aşağıdaki kütüphanelerin import edilmesi gerekir:
```python
import time
```
## Çıktının Anlamı

Çıktı iki farklı anlam taşıyabilir:

1.  Dinamik RAG senaryosunun günlük veri seti ile Chroma koleksiyonu için hazırlanması, sorgulanması ve Llama tarafından özetlenmesi işlemleri için harcanan zamanı ölçer.
2.  Tüm not defterinin herhangi bir müdahale olmadan çalıştırılması için geçen süreyi ölçer.

## Örnek Çıktı

"Oturum hazırlık zamanı (Session preparation time): 780.35 saniye (seconds)"

Bu işlemin toplam süresi 15 dakikadan azdır ve dinamik bir RAG senaryosunda hazırlık zamanı kısıtlamalarına uyar. Bu, toplantıdan önce sistemi ayarlamak için birkaç deneme yapmaya imkan tanır. Böylece, dinamik bir RAG sürecini başarıyla tamamladık ve şimdi yolculuğumuzu özetleyeceğiz.

---

## Summary

## Özet
Hızlı değişen bir dünyada, karar verme süreçlerinde bilgi toplama hızı rekabet avantajı sağlar. Dinamik RAG (Retrieval-Augmented Generation), toplantı odalarına hızlı ve uygun maliyetli yapay zeka entegrasyonu için bir yol sunar. Zor bilim sorularına cevap bulma ihtiyacını simüle eden bir sistem kurduk. SciQ veri setini indirip hazırlayarak, günlük bir toplantıda zor bilim sorularının sorulacağı bir ortamı simüle ettik. Katılımcılar, karar verme süreçlerinde zaman kaybetmek istemezler. Bu, bir pazarlama kampanyası, makale doğrulama veya zor bilim bilgisi gerektiren diğer durumlar için geçerlidir.

## Sistemin Kurulumu
Chroma koleksiyon vektör deposu (vector store) oluşturduk. Daha sonra 10.000'den fazla belgeyi (document) gömerek (embedding) Chroma vektör deposuna veri ve vektörleri ekledik. Bu işlem, `all-MiniLM-L6-v2` modelini kullanarak gerçekleştirildi.

### Kod
```python
import chromadb
from chromadb.utils import embedding_functions

# Chroma client oluşturma
client = chromadb.Client()

# Embedding fonksiyonu oluşturma
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Koleksiyon oluşturma
collection = client.create_collection(name="sciq_collection", embedding_function=embedding_func)

# Veri ekleme
 documents = [...]  # SciQ veri setinden alınan belgeler
 collection.add(documents=documents)
```

## Veri Hazırlama
SciQ veri setini indirip hazırlayarak, günlük bir toplantıda zor bilim sorularının sorulacağı bir ortamı simüle ettik.

### Kod
```python
import pandas as pd

# SciQ veri setini indirme
sciq_data = pd.read_csv("sciq_dataset.csv")

# Veri hazırlama
questions = sciq_data["question"]
answers = sciq_data["answer"]

# Veri setini işleme
for question, answer in zip(questions, answers):
    # Veri işleme işlemleri
    pass
```

## Sorgulama ve Doğruluk Ölçümü
Oluşturduğumuz koleksiyonu sorgulayarak sistemin doğruluğunu ölçtük.

### Kod
```python
# Sorgulama
query = "zor bilim sorusu"
results = collection.query(query_texts=[query])

# Doğruluk ölçümü
accuracy = [...]  # Sonuçların doğruluğunu hesaplama
print("Doğruluk:", accuracy)
```

## Kullanıcı Promptu ve Sorgulama Fonksiyonu
Kullanıcı promptu ve sorgulama fonksiyonunu oluşturduk.

### Kod
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model ve tokenizer yükleme
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Kullanıcı promptu ve sorgulama fonksiyonu
def query_function(query):
    # Sorgulama işlemleri
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(**inputs)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Kullanıcı promptu
user_query = "zor bilim sorusu"
summary = query_function(user_query)
print("Özet:", summary)
```

## Sonuç
Dinamik RAG örneğimiz, üretim ortamına geçmeden önce daha fazla çalışma gerektirecektir. Ancak, açık kaynaklı, hafif, RAG-tabanlı üretken yapay zeka için hızlı veri toplama, gömme ve sorgulama için bir yol sunar. Eğer veri saklama ihtiyacımız varsa ve büyük vektör depoları oluşturmak istemiyorsak, veri setlerimizi OpenAI GPT-4o-mini modeline entegre edebiliriz.

---

## Questions

## Sorular ve Cevaplar
Aşağıdaki paragrafta anlatılan konuyu türkçe olarak tekrar düzenleyerek önemli noktaları maddeler halinde yazacağız. Aynı zamanda text içinde kodlar varsa yazıp açıklayacağız.

## Konu
Paragraf, bir makine öğrenimi (Machine Learning, ML) modelinin dinamik olarak geri getirme (dynamic retrieval) işlemi için Chroma veritabanını kullanan bir Jupyter Notebook'u anlatmaktadır. Notebook, Hugging Face API tokenını güvenli bir şekilde kullanmaktadır.

## Önemli Noktalar
* Hugging Face API tokenı güvenli bir şekilde kullanılmaktadır (API token is not hardcoded).
* Chroma veritabanı geçici vektör depolama için kullanılmaktadır (Temporary storage of vectors).
* Accelerate kütüphanesi kullanılmaktadır (facilitate the deployment of ML models).
* Kullanıcı kimlik doğrulaması API tokenından ayrı olarak Chroma veritabanına erişim için gerekli değildir (No separate user authentication required).
* Notebook, sorguların gerçek zamanlı hızlandırılması için GPU optimizasyonunu kullanmamaktadır (No GPU optimization).
* Notebook'un oturum zaman ölçümleri dinamik RAG işleminin optimizasyonuna yardımcı olabilir (Session time measurements).
* Script, Chroma'nın ML modelleriyle entegrasyonunu göstermektedir (Chroma's capability to integrate with ML models).
* Script, Chroma veritabanının parametrelerini oturum performans ölçütlerine göre ayarlamak için işlevsellik içermemektedir (No functionality for adjusting Chroma database parameters).

## Sorular ve Cevaplar
1. Does the script ensure that the Hugging Face API token is never hardcoded directly into the notebook for security reasons? 
## Evet (Yes)
2. Is the accelerate library used here to facilitate the deployment of ML models on cloud-based platforms? 
## Evet (Yes), Kod: `import accelerate`
   ```python
import accelerate
```
   Bu kod, accelerate kütüphanesini içe aktarmaktadır. Bu kütüphane, makine öğrenimi modellerinin bulut tabanlı platformlarda dağıtımını kolaylaştırmak için kullanılmaktadır.

3. Is user authentication separate from the API token required to access the Chroma database in this script? 
## Hayır (No)
4. Does the notebook use Chroma for temporary storage of vectors during the dynamic retrieval process? 
## Evet (Yes), Kod: `import chromadb`
   ```python
import chromadb
```
   Bu kod, chromadb kütüphanesini içe aktarmaktadır. Bu kütüphane, vektörlerin geçici olarak depolanması için kullanılmaktadır.

5. Is the notebook configured to use real-time acceleration of queries through GPU optimization? 
## Hayır (No)
6. Can this notebook’s session time measurements help in optimizing the dynamic RAG process? 
## Evet (Yes)
7. Does the script demonstrate Chroma’s capability to integrate with ML models for enhanced retrieval performance? 
## Evet (Yes)
8. Does the script include functionality for adjusting the parameters of the Chroma database based on session performance metrics? 
## Hayır (No)

Tüm kodları eksiksiz olarak yazdık ve açıkladık. Import kısımlarının tümünü ekledik.

---

## References

## Referanslar
Makale içinde bahsedilen konular ve kullanılan kaynaklar aşağıda özetlenmiştir.

## Konu Özeti
Paragrafta, çeşitli teknik makaleler ve kaynaklar hakkında bilgi verilmiştir. Bu kaynaklar, doğal dil işleme (Natural Language Processing, NLP) alanında kullanılan yöntemler ve modeller hakkında detaylı bilgi içermektedir.

## Önemli Noktalar
- Crowdsourcing Multiple Choice Science Questions makalesi, Johannes Welbl, Nelson F. Liu ve Matt Gardner tarafından yazılmıştır.
- MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers makalesi, Wenhui Wang, Hangbo Bao, Shaohan Huang, Li Dong ve Furu Wei tarafından yazılmıştır.
- Hugging Face Llama model dokümantasyonu, Llama modeli hakkında detaylı bilgi içermektedir.
- ONNX, makine öğrenimi modellerinin farklı platformlarda çalıştırılmasına olanak tanıyan bir formattır.

## Kaynaklar
- Crowdsourcing Multiple Choice Science Questions: http://arxiv.org/abs/1707.06209
- MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers: https://arxiv.org/abs/2012.15828
- Hugging Face Llama model dokümantasyonu: https://huggingface.co/docs/transformers/main/en/model_doc/llama
- ONNX: https://onnxruntime.ai/

## Kullanılan Kodlar ve Açıklamaları
Paragrafta doğrudan bir kod pasajı bulunmamaktadır. Ancak, bahsedilen kaynaklarda çeşitli kod örnekleri bulunmaktadır. Örneğin, Hugging Face Llama modelini kullanmak için aşağıdaki kod örneği kullanılabilir:
```python
from transformers import LlamaForCausalLM, LlamaTokenizer

# Model ve tokenizer'ı yükle (Load model and tokenizer)
model = LlamaForCausalLM.from_pretrained("llama-model-name")
tokenizer = LlamaTokenizer.from_pretrained("llama-model-name")

# Giriş metnini hazırla (Prepare input text)
input_text = "Örnek giriş metni (Example input text)"

# Tokenize giriş metni (Tokenize input text)
inputs = tokenizer(input_text, return_tensors="pt")

# Modeli kullanarak çıktı üret (Generate output using the model)
outputs = model.generate(**inputs)

# Çıktıyı çöz (Decode output)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(output_text)
```
Bu kod örneğinde, önceden eğitilmiş (pretrained) bir Llama modeli ve tokenizer'ı yüklenmektedir. Daha sonra, bir giriş metni tokenize edilmekte ve model kullanılarak bir çıktı üretilmektedir. Üretilen çıktı daha sonra çözülerek metin haline getirilmektedir.

## Kodun Kullanımı
Yukarıdaki kod örneği, bir doğal dil işleme görevi için Llama modelini nasıl kullanabileceğinizi göstermektedir. Kodun her bir kısmı aşağıdaki gibidir:
- `from transformers import LlamaForCausalLM, LlamaTokenizer`: Gerekli sınıfları import eder. Bu sınıflar, Llama modeli ve tokenizer'ı için kullanılır.
- `model = LlamaForCausalLM.from_pretrained("llama-model-name")`: Llama modelini yükler. `"llama-model-name"` yerine kullanmak istediğiniz modelin adını yazmalısınız.
- `tokenizer = LlamaTokenizer.from_pretrained("llama-model-name")`: Tokenizer'ı yükler. Yine, `"llama-model-name"` yerine uygun model adını kullanmalısınız.
- `inputs = tokenizer(input_text, return_tensors="pt")`: Giriş metnini tokenize eder ve PyTorch tensörleri olarak döndürür.
- `outputs = model.generate(**inputs)`: Modeli kullanarak bir çıktı üretir.
- `output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)`: Üretilen çıktıyı çözer ve özel tokenleri atlar.

## Import Kısımları
Kullanılan import kısımları aşağıdaki gibidir:
```python
from transformers import LlamaForCausalLM, LlamaTokenizer
```
Bu import ifadesi, `transformers` kütüphanesinden `LlamaForCausalLM` ve `LlamaTokenizer` sınıflarını içe aktarır.

---

## Further reading

## Daha Fazla Okuma (Further Reading)
Önceden eğitilmiş (pre-trained) Transformer modellerinin görev-agnostik (task-agnostic) sıkıştırılması için derin öz-dikkat (deep self-attention) damıtma yöntemi MiniLM, Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang ve Ming Zhou tarafından araştırılmıştır. Bu konu hakkında daha fazla bilgi için aşağıdaki makaleye başvurabilirsiniz: https://arxiv.org/abs/2002.10957.

## MiniLM: Derin Öz-Dikkat Damıtma (Deep Self-Attention Distillation)
MiniLM, önceden eğitilmiş Transformer modellerinin sıkıştırılması için bir yöntemdir. Bu yöntem, derin öz-dikkat damıtma (deep self-attention distillation) kullanır. 
Önemli noktalar:
*   Önceden eğitilmiş Transformer modellerinin sıkıştırılması (Pre-trained Transformer model compression)
*   Derin öz-dikkat damıtma (Deep self-attention distillation)
*   Görev-agnostik sıkıştırma (Task-agnostic compression)

## LLaMA: Açık ve Etkili Temel Dil Modelleri (Open and Efficient Foundation Language Models)
LLaMA, Hugo Touvron, Thibaut Lavril, Gautier Lzacard ve diğerleri tarafından geliştirilen açık ve etkili temel dil modelleridir. Daha fazla bilgi için: https://arxiv.org/abs/2302.13971.
Önemli noktalar:
*   Açık ve etkili temel dil modelleri (Open and efficient foundation language models)
*   LLaMA modelleri

## ONNX Runtime Paketi Oluşturma (Building an ONNX Runtime Package)
ONNX Runtime paketi oluşturmak için aşağıdaki adımları takip edebilirsiniz: https://onnxruntime.ai/docs/build/custom.html#custom-build-packages.
Önemli noktalar:
*   ONNX Runtime paketi oluşturma (Building an ONNX Runtime package)
*   Özel derleme paketleri (Custom build packages)

Örnek kod parçaları verilmediğinden dolayı, örnek bir kod bloğu aşağıda verilmiştir. 
```python
# Import gerekli kütüphaneler
import torch
from transformers import AutoModel, AutoTokenizer

# Model ve tokenizer yükleme
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Giriş verisini tokenleştirme
input_text = "Bu bir örnek cümledir."
inputs = tokenizer(input_text, return_tensors="pt")

# Modeli çalıştırma
outputs = model(**inputs)

# Çıktıları işleme
embeddings = outputs.last_hidden_state[:, 0, :]
print(embeddings)
```
Bu kod, `sentence-transformers/all-MiniLM-L6-v2` modelini kullanarak bir cümleyi embedding'e çevirir. 
*   `import torch`: PyTorch kütüphanesini içe aktarır.
*   `from transformers import AutoModel, AutoTokenizer`: Transformers kütüphanesinden `AutoModel` ve `AutoTokenizer` sınıflarını içe aktarır.
*   `model_name = "sentence-transformers/all-MiniLM-L6-v2"`: Kullanılacak modelin adını belirler.
*   `tokenizer = AutoTokenizer.from_pretrained(model_name)`: Belirtilen model için tokenizer'ı yükler.
*   `model = AutoModel.from_pretrained(model_name)`: Belirtilen model için modeli yükler.
*   `inputs = tokenizer(input_text, return_tensors="pt")`: Giriş metnini tokenleştirir ve PyTorch tensorları olarak döndürür.
*   `outputs = model(**inputs)`: Modeli çalıştırır ve çıktıları döndürür.
*   `embeddings = outputs.last_hidden_state[:, 0, :]`: Çıktıların son gizli katmanının ilk tokeninin embedding'ini alır.

---

## Join our community on Discord

## Discord Topluluğumuza Katılın (Join our community on Discord)

Packt topluluğumuzun Discord alanına katılarak yazar ve diğer okuyucularla tartışmalara katılabilirsiniz. Discord'a katılmak için aşağıdaki bağlantıyı kullanabilirsiniz: 
https://www.packt.link/rag

## Önemli Noktalar (Key Points)
- Yazar ve diğer okuyucularla tartışmalara katılma imkanı (Opportunity to engage in discussions with the author and other readers)
- Discord bağlantısı: https://www.packt.link/rag (Discord link: https://www.packt.link/rag)

## Kodlar ve Açıklamaları (Codes and Explanations)
Paragrafda herhangi bir kod örneği bulunmamaktadır. Ancak Discord'a katılmak için verilen bağlantı bir URL'dir ve aşağıdaki gibi kullanılabilir:
```python
import webbrowser

# Discord bağlantısını açmak için
url = "https://www.packt.link/rag"
webbrowser.open(url)
```
## Kod Açıklaması (Code Explanation)
Yukarıdaki kod, Python programlama dilinde yazılmıştır ve `webbrowser` modülünü kullanarak varsayılan web tarayıcısında belirtilen URL'yi açar.
- `import webbrowser`: Bu satır, `webbrowser` modülünü içe aktarır. Bu modül, web tarayıcısında URL açmak için kullanılır.
- `url = "https://www.packt.link/rag"`: Bu satır, Discord bağlantısını içeren URL'yi tanımlar.
- `webbrowser.open(url)`: Bu satır, tanımlanan URL'yi varsayılan web tarayıcısında açar.

## Kullanım (Usage)
Yukarıdaki kodu kullanmak için:
1. Python'ın sisteminizde kurulu olduğundan emin olun.
2. Bir Python betiği (.py) oluşturun veya bir Python IDE'si (Integrated Development Environment) açın.
3. Kodun tamamını bu betik veya IDE'ye yapıştırın.
4. Betiği çalıştırın veya IDE'de kodu execute edin.

Bu işlem, varsayılan web tarayıcınızda Discord bağlantısını açacaktır.

---

