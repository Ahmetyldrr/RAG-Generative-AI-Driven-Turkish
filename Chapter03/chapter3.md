## Building Index-Based RAG with LlamaIndex, Deep Lake, and OpenAI

## İndeks Tabanlı RAG (Retrieval-Augmented Generative AI) Oluşturma: LlamaIndex, Deep Lake ve OpenAI ile

İndeksler (Indexes), hız ve doğruluk performansını artırmanın yanı sıra, geri alma (retrieval) tabanlı üretken yapay zeka (Generative AI) sistemlerine şeffaflık (transparency) katarak daha fazlasını sunar. Bir indeks ile, bir RAG modeli tarafından oluşturulan bir yanıtın kaynağı tamamen izlenebilir (traceable) hale gelir ve kullanılan verilerin kesin konumu ve ayrıntılı içeriği hakkında görünürlük sağlar. Bu iyileştirme, yalnızca önyargı (bias) ve halüsinasyon (hallucination) gibi sorunları azaltmakla kalmaz, aynı zamanda telif hakkı (copyright) ve veri bütünlüğü (data integrity) endişelerini de giderir.

## İndeks Tabanlı RAG'ın Avantajları

*   Hız ve doğruluk performansını artırma
*   Geri alma tabanlı üretken yapay zeka sistemlerine şeffaflık kazandırma
*   Yanıtların kaynağını izlenebilir hale getirme
*   Kullanılan verilerin kesin konumu ve ayrıntılı içeriği hakkında görünürlük sağlama

Bu bölümde, indekslenmiş verilerin üretken yapay zeka uygulamaları üzerinde nasıl daha fazla kontrol sağladığını keşfedeceğiz. Çıktı tatmin edici değilse, indeks sayesinde sorunun tam veri kaynağını belirleyip incelemek mümkün hale gelir. Bu yetenek, veri girdilerini iyileştirmeyi, sistem yapılandırmalarını ayarlamayı veya vektör depolama (vector store) yazılımı ve üretken modeller gibi bileşenleri değiştirmeyi mümkün kılar.

## LlamaIndex, Deep Lake ve OpenAI Entegrasyonu

LlamaIndex, Deep Lake ve OpenAI'ın sorunsuz bir şekilde entegre edilebileceğini göstereceğiz. Bu, sağlam bir temel oluşturmak için gereklidir. Daha sonra, programlarımızda kullanacağımız ana indeksleme türlerini (vector, tree, list ve keyword indexes) tanıtacağız.

## Drone Teknolojisi LLM RAG Ajanı Oluşturma

Drone teknolojisi, yangın tespiti, trafik bilgileri ve spor etkinlikleri gibi tüm alanlara yayılmaktadır. Bu nedenle, örnek olarak drone teknolojisini kullanmaya karar verdik. Bu bölümün amacı, bir sonraki bölümde multimodal verilerle geliştireceğimiz bir LLM drone teknolojisi veri seti hazırlamaktır.

## Önemli Konular

*   LlamaIndex çerçevesi (framework) ve indeksleme yöntemleri (indexing methods) ile anlamsal arama motoru (semantic search engine) oluşturma
*   Deep Lake vektör depolarını (vector stores) doldurma
*   LlamaIndex, Deep Lake ve OpenAI entegrasyonu
*   Skor sıralaması (score ranking) ve kosinüs benzerliği (cosine similarity) metrikleri
*   İzlenebilirlik için meta veri (metadata) iyileştirme
*   Sorgu kurulumu (query setup) ve oluşturma yapılandırması (generation configuration)
*   Otomatik belge sıralamasını (automated document ranking) tanıtma
*   Vektör, ağaç, liste ve anahtar kelime indeksleme türleri (vector, tree, list, and keyword indexing types)

## Kod Örneği

Aşağıdaki kod örneğinde, LlamaIndex, Deep Lake ve OpenAI kullanarak bir indeks tabanlı RAG pipeline'ı nasıl oluşturulacağını gösteriyoruz.

```python
## Import gerekli kütüphaneler
import os
from llama_index import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.llms import OpenAI

## Deep Lake vektör deposu oluşturma
my_activeloop_org_id = "kullanici_adi"  # replace with your org id
my_activeloop_dataset_name = "drones_dataset"  # dataset name
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

## ds_name = 'llama-hub-demo-1'
vector_store = DeepLakeVectorStore(
    dataset_path=dataset_path, overwrite=True, verbose=False
)

## storage_context = StorageContext.from_defaults(vector_store=vector_store)

## Belge yükleme
documents = SimpleDirectoryReader("./data").load_data()

## İndeks oluşturma
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)

## Sorgu motoru oluşturma
query_engine = index.as_query_engine()

## Sorgu çalıştırma
response = query_engine.query("Drone teknolojisinin kullanım alanları nelerdir?")

## Yanıtı yazdırma
print(response)
```

Bu kod örneğinde, önce gerekli kütüphaneleri import ediyoruz. Daha sonra, Deep Lake vektör deposu oluşturuyoruz ve belge yükleme işlemini gerçekleştiriyoruz. Ardından, indeks oluşturma ve sorgu motoru oluşturma işlemlerini gerçekleştiriyoruz. Son olarak, bir sorgu çalıştırarak yanıtı yazdırıyoruz.

## Kod Açıklaması

*   `SimpleDirectoryReader("./data").load_data()`: `./data` dizinindeki belgeleri yükler.
*   `VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)`: Yüklenen belgelerden bir vektör indeksi oluşturur.
*   `index.as_query_engine()`: Oluşturulan indeksten bir sorgu motoru oluşturur.
*   `query_engine.query("Drone teknolojisinin kullanım alanları nelerdir?")`: Sorgu motorunu kullanarak bir sorgu çalıştırır.

Bu bölümde, indeks tabanlı RAG pipeline'ı oluşturma, LlamaIndex, Deep Lake ve OpenAI entegrasyonu, drone teknolojisi LLM RAG ajanı oluşturma ve önemli konuları ele aldık. Ayrıca, bir kod örneği ile indeks tabanlı RAG pipeline'ı oluşturmayı gösterdik.

---

## Why use index-based RAG?

## Neden İndeks Tabanlı RAG Kullanılır? (Why use index-based RAG?)
İndeks tabanlı arama (index-based search), gelişmiş RAG (Retrieval-Augmented Generation) tabanlı üretken yapay zeka (generative AI) sistemlerini bir üst seviyeye taşır. Büyük hacimli verilerle karşılaşıldığında, veri yığınlarından (raw chunks of data) organize edilmiş, indekslenmiş düğümlere (organized, indexed nodes) doğru ilerleyerek geri getirme (retrieval) hızını artırır. Bu sayede, çıktıdan (output) bir belgenin kaynağına (source of a document) ve konumuna (location) kadar izlenebilirlik sağlar.

## İndeks Tabanlı RAG'ın Avantajları
*   Hızlı geri getirme (fast retrieval) sağlar
*   Büyük veri hacimlerini daha verimli şekilde işler (handles large volumes of data)
*   Veri yığınlarından organize edilmiş düğümlere ulaşmayı sağlar (from raw chunks of data to organized, indexed nodes)
*   Çıktıdan kaynağa ve konuma kadar izlenebilirlik sağlar (trace from the output back to the source of a document and its location)

## Vektör Tabanlı Benzerlik Araması (Vector-Based Similarity Search) ve İndeks Tabanlı Arama (Index-Based Search) Arasındaki Farklar
Vektör tabanlı benzerlik araması ve indeks tabanlı arama arasındaki farkları anlamak için, indeks tabanlı bir RAG'ın mimarisini (architecture) analiz edeceğiz.

### Vektör Tabanlı Benzerlik Araması
Vektör tabanlı benzerlik araması, metinleri (text) vektör uzayında (vector space) temsil eder ve benzerlik araması yapar. Bu yöntem, metinlerin anlamsal (semantic) olarak birbirine benzer olup olmadığını belirlemek için kullanılır.

### İndeks Tabanlı Arama
İndeks tabanlı arama ise, verileri önceden indeksleyerek (pre-indexing) arama işlemini hızlandırır. Bu yöntem, özellikle büyük veri hacimlerinde daha verimlidir.

## Örnek Kod
Aşağıdaki örnek kod, indeks tabanlı arama için kullanılan bir kütüphane olan `faiss` (Facebook AI Similarity Search) kütüphanesini kullanmaktadır.
```python
import numpy as np
import faiss

# Veri oluşturma (creating data)
np.random.seed(0)
data = np.random.rand(100, 128).astype('float32')

# İndeks oluşturma (creating index)
index = faiss.IndexFlatL2(128)

# Verileri indekse ekleme (adding data to index)
index.add(data)

# Arama işlemi (search operation)
query = np.random.rand(1, 128).astype('float32')
D, I = index.search(query, 5)

# Sonuçları yazdırma (printing results)
print("Mesafeler (distances): ", D)
print("İndeksler (indices): ", I)
```
Bu kod, `faiss` kütüphanesini kullanarak bir indeks oluşturur ve verileri bu indekse ekler. Daha sonra, bir sorgu (query) vektörü oluşturur ve bu vektöre en yakın 5 veriyi bulur.

## Kod Açıklaması
*   `import numpy as np`: Numpy kütüphanesini içe aktarır.
*   `import faiss`: Faiss kütüphanesini içe aktarır.
*   `np.random.seed(0)`: Rastgele sayı üreticisini sabitler.
*   `data = np.random.rand(100, 128).astype('float32')`: 100 adet 128 boyutlu vektör oluşturur.
*   `index = faiss.IndexFlatL2(128)`: 128 boyutlu vektörler için bir indeks oluşturur.
*   `index.add(data)`: Verileri indekse ekler.
*   `query = np.random.rand(1, 128).astype('float32')`: Bir sorgu vektörü oluşturur.
*   `D, I = index.search(query, 5)`: Sorgu vektörüne en yakın 5 veriyi bulur.
*   `print("Mesafeler (distances): ", D)`: Bulunan verilerin mesafelerini yazdırır.
*   `print("İndeksler (indices): ", I)`: Bulunan verilerin indekslerini yazdırır.

## Kullanım Alanları
İndeks tabanlı arama, birçok alanda kullanılabilir. Örneğin:
*   Metin arama (text search)
*   Görüntü arama (image search)
*   Öneri sistemleri (recommendation systems)
*   Doğal dil işleme (natural language processing)

Bu yöntem, büyük veri hacimlerinde hızlı ve verimli arama işlemleri yapabilme yeteneği sağlar.

---

## Architecture

## Mimari (Architecture)

İndeks-tabanlı arama (Index-based search), vektör-tabanlı aramadan (vector-based search) daha hızlıdır çünkü doğrudan ilgili verilere indeksler kullanarak erişir, oysa vektör-tabanlı arama tüm kayıtlar arasında sırayla karşılaştırma yapar. Biz, Bölüm 2'de vektör-tabanlı benzerlik arama programı uyguladık. Şekil 3.1'de gösterildiği gibi: Verileri Pipeline #1: Veri Toplama ve Hazırlama (Data Collection and Preparation) adımında topladık ve hazırladık. Verileri gömme (embedding) işleminden geçirdik ve hazırlanan verileri vektör deposunda (vector store) Pipeline #2: Gömme ve vektör deposu (Embeddings and vector store) adımında sakladık. Daha sonra, kullanıcı girdisini işlemek, vektör benzerlik aramalarına dayalı olarak geri alma işlemlerini gerçekleştirmek, girdiyi zenginleştirmek, yanıt oluşturmak ve performans metriklerini uygulamak için Pipeline #3'ü çalıştırdık. Bu yaklaşım esnektir çünkü projenizin ihtiyaçlarına bağlı olarak her bileşeni uygulamak için birçok yol sunar.

## Vektör-Tabanlı ve İndeks-Tabanlı Arama Karşılaştırması

| Özellik (Feature) | Vektör-Tabanlı Benzerlik Arama ve Geri Alma (Vector-based similarity search and retrieval) | İndeks-Tabanlı Vektör, Ağaç, Liste ve Anahtar Kelime Arama ve Geri Alma (Index-based vector, tree, list, and keyword search and retrieval) |
| --- | --- | --- |
| Esneklik (Flexibility) | Yüksek (High) | Orta (precomputed structure) |
| Hız (Speed) | Büyük veri kümelerinde yavaş (Slower with large datasets) | Hızlı ve hızlı geri alma için optimize edilmiş (Fast and optimized for quick retrieval) |
| Ölçeklenebilirlik (Scalability) | Gerçek zamanlı işlemeyle sınırlı (Limited by real-time processing) | Büyük veri kümeleriyle yüksek ölçeklenebilirlik (Highly scalable with large datasets) |
| Karmaşıklık (Complexity) | Daha basit kurulum (Simpler setup) | Daha karmaşık ve indeksleme adımı gerektirir (More complex and requires an indexing step) |
| Güncelleme Sıklığı (Update Frequency) | Güncellemesi kolay (Easy to update) | Güncellemeler için yeniden indeksleme gerektirir (Requires re-indexing for updates) |

## İndeks-Tabanlı RAG Programı Oluşturma

İndeks-tabanlı RAG programı oluşturmak için Deep Lake, LlamaIndex ve OpenAI kullanacağız. Pipeline #1, #2 ve #3'ü kullanarak veri toplama, gömme, vektör deposuna yükleme, geri alma ve oluşturma işlemlerini gerçekleştireceğiz.

### Pipeline #1: Veri Toplama ve Hazırlama

Verileri toplar ve hazırlarız. Bu kez, verileri tek bir belge olarak hazırlarız ve ayrı dosyalarda saklarız. Daha sonra, adlarını ve konumlarını vektör deposuna yüklediğimiz meta verilere ekleriz.

### Pipeline #2: Gömme ve Vektör Deposu

Verileri `llama-index-vector-stores-deeplake` paketi kullanarak vektör deposuna yükleriz. Bu paket, optimize edilmiş bir başlangıç senaryosunda ihtiyacımız olan her şeyi içerir: parçalama (chunking), gömme (embedding), depolama ve hatta LLM entegrasyonu.

```python
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index import VectorStoreIndex

# Deep Lake vektör deposu oluşturma
vector_store = DeepLakeVectorStore(
    dataset_path="path/to/dataset",
    overwrite=True,
)

# Vektör deposu indeksini oluşturma
index = VectorStoreIndex.from_vector_store(vector_store)
```

Bu kod, Deep Lake vektör deposu oluşturur ve vektör deposu indeksini oluşturur.

### Pipeline #3: Geri Alma ve Oluşturma

Verileri `llama-index-vector-stores-deeplake` paketi kullanarak geri alma ve oluşturma işlemlerini gerçekleştireceğiz. Bu paket, indeks-tabanlı geri alma ve oluşturma işlemlerini başlatmak için ihtiyacımız olan her şeyi içerir.

```python
from llama_index import LLMPredictor, ServiceContext
from llama_index.llms import OpenAI

# LLM tahmincisi oluşturma
llm_predictor = LLMPredictor(llm=OpenAI())

# Hizmet bağlamı oluşturma
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# Sorgu motoru oluşturma
query_engine = index.as_query_engine(service_context=service_context)

# Sorgu çalıştırma
response = query_engine.query("Sorgu metni")
```

Bu kod, LLM tahmincisi oluşturur, hizmet bağlamı oluşturur, sorgu motoru oluşturur ve sorgu çalıştırır.

## Sonuç

İndeks-tabanlı RAG programı oluşturduk ve Deep Lake, LlamaIndex ve OpenAI kullanarak geri alma ve oluşturma işlemlerini gerçekleştirdik. Bu yaklaşım, vektör-tabanlı aramadan daha hızlıdır ve büyük veri kümeleriyle çalışırken daha iyi performans sağlar.

---

## Building a semantic search engine and generative agent for drone technology

## Drone Teknolojisi için Anlamsal Arama Motoru ve Üretken Ajan Oluşturma (Building a Semantic Search Engine and Generative Agent for Drone Technology)

Bu bölümde, Deep Lake vektör depolarını (vector stores), LlamaIndex'i ve OpenAI'ı kullanarak anlamsal indeks tabanlı bir arama motoru ve üretken yapay zeka ajan motoru oluşturacağız. Daha önce de belirtildiği gibi, drone teknolojisi yangın tespiti ve trafik kontrolü gibi alanlarda genişliyor. Bu nedenle, programın amacı drone teknolojisi soruları ve cevapları için indeks tabanlı bir RAG (Retrieval-Augmented Generation) ajanı sağlamaktır. Program, dronların araçları ve diğer nesneleri tanımlamak için bilgisayarlı görü (computer vision) tekniklerini nasıl kullandığını gösterecektir.

### Mimari (Architecture)

Bu bölümde, Şekil 3.1'de gösterilen mimariyi uygulayacağız.

### Ortamın Kurulumu (Installing the Environment)

İlk olarak, ortamı kurarak başlayacağız. Daha sonra, programın üç ana pipeline'ını oluşturacağız:

#### Pipeline 1: Dokümanların Toplanması ve Hazırlanması (Collecting and Preparing the Documents)

GitHub ve Wikipedia gibi kaynakları kullanarak, indeksleme için dokümanları toplayacak ve temizleyeceğiz.

#### Pipeline 2: Deep Lake Vektör Deposu Oluşturma ve Doldurma (Creating and Populating a Deep Lake Vector Store)

Hazırlanan dokümanlarla Deep Lake vektör deposu oluşturacak ve dolduracağız.

#### Pipeline 3: İndeks Tabanlı RAG için Sorgu İşleme ve Üretme (Index-based RAG for Query Processing and Generation)

LLM'ler (Large Language Models) ve kosinüs benzerliği metriği (cosine similarity metric) ile zaman ve skor performanslarını uygulayacağız.

### Kodları Açıklama (Code Explanation)

İlk olarak, gerekli kütüphaneleri içe aktarmalıyız (importing the necessary libraries):
```python
import os
import pandas as pd
import numpy as np
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
```
Bu kod, gerekli kütüphaneleri içe aktarır. `llama_index` kütüphanesi, indeksleme ve sorgu işleme için kullanılır. `DeepLakeVectorStore` sınıfı, Deep Lake vektör deposunu oluşturmak ve yönetmek için kullanılır.

Daha sonra, dokümanları toplamak ve hazırlamak için aşağıdaki kodu kullanacağız:
```python
# Dokümanları toplamak ve hazırlamak için
documents = SimpleDirectoryReader(input_dir='./data').load_data()
```
Bu kod, `./data` dizinindeki dokümanları toplar ve hazırlar.

Deep Lake vektör deposu oluşturmak ve doldurmak için aşağıdaki kodu kullanacağız:
```python
# Deep Lake vektör deposu oluşturmak ve doldurmak için
vector_store = DeepLakeVectorStore(dataset_path='./deeplake_dataset')
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, service_context=service_context)
```
Bu kod, `./deeplake_dataset` dizininde Deep Lake vektör deposu oluşturur ve dokümanları indeksler.

Son olarak, indeks tabanlı RAG için sorgu işleme ve üretme işlemlerini gerçekleştirmek için aşağıdaki kodu kullanacağız:
```python
# İndeks tabanlı RAG için sorgu işleme ve üretme
query_engine = index.as_query_engine()
response = query_engine.query('Drone teknolojisi hakkında bilgi veriniz.')
print(response)
```
Bu kod, indeks tabanlı RAG için sorgu işleme ve üretme işlemlerini gerçekleştirir.

### Önemli Noktalar (Key Points)

*   Drone teknolojisi için anlamsal indeks tabanlı arama motoru ve üretken yapay zeka ajan motoru oluşturma
*   Deep Lake vektör depolarını, LlamaIndex'i ve OpenAI'ı kullanma
*   Dokümanları toplamak ve hazırlamak için `SimpleDirectoryReader` sınıfını kullanma
*   Deep Lake vektör deposu oluşturmak ve doldurmak için `DeepLakeVectorStore` sınıfını kullanma
*   İndeks tabanlı RAG için sorgu işleme ve üretme işlemlerini gerçekleştirmek için `VectorStoreIndex` sınıfını kullanma

---

## Installing the environment

## Ortamın Kurulumu (Installing the Environment)

Ortamın kurulumu (environment setup) büyük ölçüde önceki bölümle aynıdır. LlamaIndex, Deep Lake için vektör depolama (vector store) yetenekleri ve OpenAI modüllerini entegre eden paketlere odaklanalım. Bu entegrasyon, sorunsuz çapraz platform (cross-platform) uygulamaları için önemli bir adımdır.

### Gerekli Paketlerin Kurulumu

Aşağıdaki paketlerin kurulumu gerekmektedir:

*   `llama-index-vector-stores-deeplake` paketi: `!pip install llama-index-vector-stores-deeplake==0.1.6`
    ```python
!pip install llama-index-vector-stores-deeplake==0.1.6
```
    Bu kod, `llama-index-vector-stores-deeplake` paketini `0.1.6` sürümünü kurar. Bu paket, LlamaIndex'in Deep Lake vektör depolarıyla entegrasyonunu sağlar.

*   `deeplake` paketi: `!pip install deeplake==3.9.8`
    ```python
!pip install deeplake==3.9.8
```
    Bu kod, `deeplake` paketini `3.9.8` sürümünü kurar. Bu paket, Deep Lake'in ek işlevselliklerini sağlar.

*   `llama-index` paketi: `!pip install llama-index==0.10.64`
    ```python
!pip install llama-index==0.10.64
```
    Bu kod, `llama-index` paketini `0.10.64` sürümünü kurar. Bu paket, LlamaIndex'in temel işlevselliklerini sağlar.

### Paketlerin İçe Aktarılması (Importing Packages)

Paketlerin doğru bir şekilde içe aktarılabildiğini kontrol edelim:
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
```
Bu kod, `llama_index.core` modülünden `VectorStoreIndex`, `SimpleDirectoryReader` ve `Document` sınıflarını içe aktarır. Ayrıca, `llama_index.vector_stores.deeplake` modülünden `DeepLakeVectorStore` sınıfını içe aktarır. Bu sınıflar, LlamaIndex'in temel işlevselliklerini ve Deep Lake vektör depolarıyla entegrasyonunu sağlar.

Böylece, ortamın kurulumunu tamamladık. Şimdi, belgeleri (documents) toplama ve hazırlama işlemlerine geçebiliriz.

---

## Pipeline 1: Collecting and preparing the documents

## Pipeline 1: Dokümanların Toplanması ve Hazırlanması (Collecting and Preparing the Documents)

Bu bölümde, dron teknolojisi ile ilgili dokümanları, kaynaklarına geri dönük izleme için gerekli meta verilerle birlikte toplayacağız ve hazırlayacağız. Amaç, bir yanıtın içeriğini, kaynağını bulmak için alınan veri parçasına geri izlemektir.

### Adım 1: Veri Dizinini Oluşturma (Creating the Data Directory)

İlk olarak, dokümanları yükleyeceğimiz bir veri dizini oluşturacağız:
```python
!mkdir data
```
### Adım 2: Dokümanları Toplama ve Temizleme (Collecting and Cleaning the Documents)

Dron teknolojisi verileri için heterojen bir korpus (corpus) kullanacağız ve bu verileri `BeautifulSoup` kullanarak işleyeceğiz.

#### Gerekli Kütüphanelerin İçe Aktarılması (Importing Necessary Libraries)
```python
import requests
from bs4 import BeautifulSoup
import re
import os
```
#### URL Listesinin Tanımlanması (Defining the URL List)
```python
urls = [
    "https://github.com/VisDrone/VisDrone-Dataset",
    "https://paperswithcode.com/dataset/visdrone",
    "https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Zhu_VisDrone-DET2018_The_Vision_Meets_Drone_Object_Detection_in_Image_Challenge_ECCVW_2018_paper.pdf",
    "https://github.com/VisDrone/VisDrone2018-MOT-toolkit",
    "https://en.wikipedia.org/wiki/Object_detection",
    "https://en.wikipedia.org/wiki/Computer_vision",
    # ...
]
```
Bu listedeki URL'ler dronlar, bilgisayar görüşü ve ilgili teknolojilerle ilgili siteleri içermektedir.

#### Dokümanları Temizleme ve İndirme Fonksiyonları (Functions for Cleaning and Fetching Documents)
```python
def clean_text(content):
    # Referansları ve istenmeyen karakterleri kaldırma (Removing references and unwanted characters)
    content = re.sub(r'\[\d+\]', '', content)  # Referansları kaldırma (Removing references)
    content = re.sub(r'[^\w\s\.]', '', content)  # Nokta hariç noktalama işaretlerini kaldırma (Removing punctuation except periods)
    return content

def fetch_and_clean(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Kötü yanıtlar için istisna oluşturma (Raising exception for bad responses)
        soup = BeautifulSoup(response.content, 'html.parser')  # HTML'yi ayrıştırma (Parsing HTML)
        # "mw-parser-output" sınıfını önceliklendir, bulunamazsa "content" kimliği ara (Prioritizing "mw-parser-output" but falling back to "content" id if not found)
        content = soup.find('div', {'class': 'mw-parser-output'}) or soup.find('div', {'id': 'content'})
        if content is None:
            return None
        # İçerik içindeki belirli bölümleri kaldırma (Removing specific sections from the content)
        for section_title in ['References', 'Bibliography', 'External links', 'See also', 'Notes']:
            section = content.find('span', id=section_title)
            while section:
                for sib in section.parent.find_next_siblings():
                    sib.decompose()
                section.parent.decompose()
                section = content.find('span', id=section_title)
        # Metni çıkarma ve temizleme (Extracting and cleaning text)
        text = content.get_text(separator=' ', strip=True)
        text = clean_text(text)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return None
```
Bu fonksiyonlar, URL'lerden içerikleri indirip temizler ve istenmeyen bölümleri kaldırır.

#### Temizlenmiş Dokümanları Dosyalara Yazma (Writing Cleaned Documents to Files)
```python
output_dir = './data/'
os.makedirs(output_dir, exist_ok=True)

for url in urls:
    article_name = url.split('/')[-1].replace('.html', '')
    filename = os.path.join(output_dir, article_name + '.txt')
    clean_article_text = fetch_and_clean(url)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(clean_article_text)

print(f"Content written to files in the '{output_dir}' directory.")
```
Bu kod, temizlenmiş içerikleri ayrı dosyalara yazar.

### Adım 3: Dokümanları Yükleme (Loading the Documents)

`SimpleDirectoryReader` sınıfını kullanarak dokümanları yüklüyoruz:
```python
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/").load_data()
```
Bu sınıf, belirtilen dizindeki desteklenen dosya türlerini (örneğin, `.txt`, `.pdf`, `.docx`) yükler ve içeriklerini çıkarır.

### Sonuç (Outcome)

Dokümanlar yüklendikten sonra, ilk dokümanın içeriğini görüntüleyebiliriz:
```python
print(documents[0])
```
Bu, dokümanın metni ve meta verilerini içerir.

## Önemli Noktalar (Key Points)

*   Doküman toplama ve temizleme işlemleri `BeautifulSoup` ve `requests` kütüphaneleri kullanılarak gerçekleştirildi.
*   `SimpleDirectoryReader` sınıfı, dizindeki dokümanları yüklemek için kullanıldı.
*   Dokümanların içerikleri ve meta verileri `documents` listesinde depolandı.

## Kullanılan Kod Parçaları (Used Code Snippets)

*   `fetch_and_clean` fonksiyonu: URL'lerden içerikleri indirip temizler.
*   `clean_text` fonksiyonu: Metinleri temizler.
*   `SimpleDirectoryReader` sınıfı: Dizin içindeki dokümanları yükler.

Tüm bu adımlar, dron teknolojisi ile ilgili dokümanları toplamak, temizlemek ve yüklemek için gerekli işlemleri içerir. Bu sayede, ileriki aşamalarda bu dokümanlar üzerinde daha karmaşık işlemler gerçekleştirmek mümkün olacaktır.

---

## Pipeline 2: Creating and populating a Deep Lake vector store

## Pipeline 2: Derin Göl (Deep Lake) Vektör Deposu Oluşturma ve Doldurma

Bu bölümde, bir Derin Göl (Deep Lake) vektör deposu oluşturacağız ve bu depoyu belgelerimizdeki verilerle dolduracağız. Standart bir tensor konfigürasyonu uygulayacağız:

*   `text` (str): Metin, belgeler sözlüğünde listelenen metin dosyalarından birinin içeriğidir. Sorunsuz bir şekilde işlenecek ve anlamlı parçalara bölünecektir (chunking).
*   `metadata` (json): Bu durumda, metadata her bir metin parçasının kaynak dosya adını içerecektir. Bu sayede tam şeffaflık ve kontrol sağlanacaktır.
*   `embedding` (float32): Embedding, LlamaIndex-Derik Göl (Deep Lake)-OpenAI paketi tarafından doğrudan çağrılan bir OpenAI embedding modeli kullanılarak oluşturulacaktır.
*   `id` (str, otomatik doldurulur): Her bir parçaya otomatik olarak benzersiz bir ID atanacaktır.

Vektör deposu ayrıca bir indeks içerecektir. Bu indeks, 0'dan n'e kadar bir sayı olacak, ancak anlamsal olarak kullanılamayacaktır çünkü veri kümesini her değiştirdiğimizde değişecektir. Ancak, benzersiz ID alanı değiştirilmeden kalacaktır.

### Kod

İlk olarak, vektör deposu ve veri kümesi yollarını tanımlıyoruz:

```python
from llama_index.core import StorageContext

vector_store_path = "hub://denis76/drone_v2"
dataset_path = "hub://denis76/drone_v2"
```

Kendi hesabınızda kullanmak için `vector_store_path` ve `dataset_path` değişkenlerini kendi hesabınızın adıyla ve kullanmak istediğiniz veri kümesinin adıyla değiştirin:

```python
vector_store_path = "hub://[KENDİ VEKTÖR DEPOLAMA ALANINIZ]/[KENDİ VERİ KÜMENİZ]"
dataset_path = "hub://[KENDİ VEKTÖR DEPOLAMA ALANINIZ]/[KENDİ VERİ KÜMENİZ]"
```

Ardından, bir vektör deposu oluşturuyor, onu dolduruyor ve belgeler üzerinde bir indeks oluşturuyoruz:

```python
# overwrite=True veri kümesini üzerine yazar, False ise ekler
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Belgeler üzerinde bir indeks oluştur
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

`overwrite` değişkeni `True` olduğunda, vektör deposu oluşturulur ve mevcut olan üzerine yazılır. `overwrite=False` olduğunda, veri kümesi eklenir.

### Veri Kümesinin Yapısı

Çıktı, veri kümesinin oluşturulduğunu ve verilerin yüklendiğini doğrular:

```
Your Deep Lake dataset has been successfully created!
Uploading data to deeplake dataset.
100%|██████████| 41/41 [00:02<00:00, 18.15it/s]
```

Veri kümesinin yapısı aşağıdaki gibidir:

```
Dataset(path='hub://denis76/drone_v2', tensors=['text', 'metadata', 'embedding', 'id'])
```

Veriler, türleri ve şekilleriyle tensorlarda depolanır.

### Veri Kümesini Yükleme ve Görüntüleme

Veri kümesini belleğe yüklüyoruz:

```python
import deeplake
ds = deeplake.load(dataset_path)  # Veri kümesini yükle
```

Veri kümesini çevrimiçi olarak görüntülemek için sağlanan bağlantıya tıklayabilirsiniz:

```
This dataset can be visualized in Jupyter Notebook by ds.visualize() or at https://app.activeloop.ai/denis76/drone_v2
```

Veri kümesini bir pandas DataFrame'e yükleyerek görüntüleyebiliriz:

```python
import json
import pandas as pd
import numpy as np

# Veri kümesindeki tensorları bir sözlüğe yükle
data = {}
for tensor_name in ds.tensors:
    tensor_data = ds[tensor_name].numpy()
    if tensor_data.ndim > 1:
        data[tensor_name] = [np.array(e).flatten().tolist() for e in tensor_data]
    else:
        if tensor_name == "text":
            data[tensor_name] = [t.tobytes().decode('utf-8') if t else "" for t in tensor_data]
        else:
            data[tensor_name] = tensor_data.tolist()

# Sözlükten bir pandas DataFrame oluştur
df = pd.DataFrame(data)
```

Bir kaydı görüntülemek için bir fonksiyon tanımlıyoruz:

```python
def display_record(record_number):
    record = df.iloc[record_number]
    display_data = {
        "ID": record["id"] if "id" in record else "N/A",
        "Metadata": record["metadata"] if "metadata" in record else "N/A",
        "Text": record["text"] if "text" in record else "N/A",
        "Embedding": record["embedding"] if "embedding" in record else "N/A"
    }
    # display_data'yı istediğiniz şekilde görüntüleyin
```

Bir kaydı görüntülemek için fonksiyonu çağırıyoruz:

```python
rec = 0  # İlgili kayıt numarasıyla değiştirin
display_record(rec)
```

### Sonuç

`id` alanı benzersiz bir dize kodudur. `metadata` alanı, içeriği orijinal dosyaya ve dosya yoluna geri izlemek için gereken bilgileri içerir. `text` alanı, bu veri parçasının metnini içerir. `embedding` alanı, metin içeriğinin gömülü vektörünü içerir.

RAG veri kümelerinin yapısı ve biçimi, alan veya projeden diğerine değişir. Ancak, bu veri kümesinin aşağıdaki dört sütunu, AI'nın evrimi hakkında değerli bilgiler sağlar:

*   `id`: Veri kümesinde `text` sütununun parçalarını düzenlemek için kullanacağımız indekstir.
*   `metadata`: Metadata, Pipeline 1'de Deep Lake'in SimpleDirectoryReader'ı kaynak belgeleri bir documents nesnesine yüklediğinde otomatik olarak oluşturuldu.
*   `text`: Metin, Deep Lake'in vektör deposu oluşturma işlevi tarafından işlendi ve otomatik olarak parçalandı.
*   `embedding`: Her bir veri parçasının gömülmesi, bir gömme modeli aracılığıyla oluşturuldu.

Artık indeks tabanlı RAG'ı çalıştırmaya hazırız.

---

## Pipeline 3: Index-based RAG

## Pipeline 3: İndeks Tabanlı RAG (Index-based RAG)

Bu bölümde, LlamaIndex kütüphanesini kullanarak indeks tabanlı bir RAG (Retrieval-Augmented Generation) pipeline'ı uygulanacaktır. Bu pipeline, Deep Lake ile hazırlanmış ve işlenmiş verileri kullanarak, heterojen (gürültü içeren) drone ile ilgili belge koleksiyonundan ilgili bilgileri alacak ve OpenAI'ın LLM (Large Language Model) modelleri aracılığıyla yanıtı sentezleyecektir.

## Uygulanacak İndeks Motorları (Index Engines)

Aşağıdaki dört indeks motoru uygulanacaktır:
*   ## Vektör Deposu İndeks Motoru (Vector Store Index Engine) : Belgelerden vektör deposu indeksi oluşturur, böylece verimli benzerlik tabanlı aramalar sağlar.
*   ## Ağaç İndeks (Tree Index) : Belgelerden hiyerarşik bir ağaç indeksi oluşturur, alternatif bir erişim yapısı sunar.
*   ## Liste İndeks (List Index) : Belgelerden basit bir liste indeksi oluşturur.
*   ## Anahtar Kelime Tablosu İndeks (Keyword Table Index) : Belgelerden çıkarılan anahtar kelimelere dayalı bir indeks oluşturur.

## LLM ile Sorgulama (Querying with LLM)

İndeksi sorgulamak, ilgili belgeleri almak ve kaynak bilgisiyle birlikte sentezlenmiş bir yanıt döndürmek için LLM kullanılacaktır:
*   ## Sorgu Yanıtı ve Kaynak (Query Response and Source) : Kullanıcı girdisiyle indeksi sorgular, ilgili belgeleri alır ve kaynak bilgisiyle birlikte sentezlenmiş bir yanıt döndürür.

## Yanıtların Değerlendirilmesi (Evaluating Responses)

Yanıtlar, alma ve benzerlik puanlarına dayalı bir zaman ağırlıklı ortalama hesaplayan LLM puanı ve kosinüs benzerliği ile zaman ağırlıklı ortalama metriği ile ölçülecektir.

### Kod Uygulaması

İndeks tabanlı RAG pipeline'ı uygulamak için aşağıdaki kod kullanılacaktır:
```python
import os
import time
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.indices.loading import load_index_from_storage
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import DeepLakeVectorStore
from langchain.chat_models import ChatOpenAI
from llama_index import (
    GPTVectorStoreIndex,
    GPTTreeIndex,
    GPTListIndex,
    GPTKeywordTableIndex,
)

# Deep Lake vektör deposu ayarları
dataset_id = "your_dataset_id"
storage_context = StorageContext.from_defaults(
    vector_store=DeepLakeVectorStore(dataset_id=dataset_id, overwrite=False)
)

# LLM ayarları
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# İndeks motorları oluşturma
vector_store_index = GPTVectorStoreIndex(
    nodes=documents, service_context=service_context
)
tree_index = GPTTreeIndex(
    nodes=documents, service_context=service_context
)
list_index = GPTListIndex(
    nodes=documents, service_context=service_context
)
keyword_table_index = GPTKeywordTableIndex(
    nodes=documents, service_context=service_context
)

# İndeksleri kaydetme
vector_store_index.storage_context.persist(persist_dir="./storage/vector_store_index")
tree_index.storage_context.persist(persist_dir="./storage/tree_index")
list_index.storage_context.persist(persist_dir="./storage/list_index")
keyword_table_index.storage_context.persist(persist_dir="./storage/keyword_table_index")

# İndeksleri yükleme
loaded_vector_store_index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./storage/vector_store_index")
)
loaded_tree_index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./storage/tree_index")
)
loaded_list_index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./storage/list_index")
)
loaded_keyword_table_index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./storage/keyword_table_index")
)

# Sorgulama
query = "Drone ile ilgili bilgiler nelerdir?"
response = loaded_vector_store_index.query(query, response_mode="compact")
print(response)

# Yanıtların değerlendirilmesi
start_time = time.time()
response = loaded_vector_store_index.query(query, response_mode="compact")
end_time = time.time()
print(f"Yanıt süresi: {end_time - start_time} saniye")
```

### Kod Açıklaması

*   `import` ifadeleri, gerekli kütüphaneleri içe aktarır.
*   `dataset_id` değişkeni, Deep Lake vektör deposu ID'sini tanımlar.
*   `storage_context` değişkeni, Deep Lake vektör deposu ayarlarını tanımlar.
*   `llm_predictor` değişkeni, LLM ayarlarını tanımlar.
*   `service_context` değişkeni, LLM ayarlarını içeren bir servis bağlamı tanımlar.
*   İndeks motorları (`vector_store_index`, `tree_index`, `list_index`, `keyword_table_index`) oluşturulur ve belgeler (`documents`) kullanılarak indekslenir.
*   Oluşturulan indeksler, `./storage` dizinine kaydedilir.
*   Kaydedilen indeksler, `load_index_from_storage` fonksiyonu kullanılarak yüklenir.
*   Yüklenen indeksler kullanılarak sorgulama yapılır ve yanıtlar alınır.
*   Yanıtların değerlendirilmesi için zaman ölçümü yapılır.

### Önemli Noktalar

*   İndeks tabanlı RAG pipeline'ı, heterojen belge koleksiyonlarından ilgili bilgileri almak için kullanılır.
*   Dört farklı indeks motoru uygulanır: Vektör Deposu İndeks Motoru, Ağaç İndeks, Liste İndeks ve Anahtar Kelime Tablosu İndeks.
*   LLM kullanılarak sorgulama yapılır ve yanıtlar alınır.
*   Yanıtlar, alma ve benzerlik puanlarına dayalı bir zaman ağırlıklı ortalama hesaplayan LLM puanı ve kosinüs benzerliği ile değerlendirilir.

---

## User input and query parameters

## Kullanıcı Girdisi ve Sorgu Parametreleri (User Input and Query Parameters)
Kullanıcı girdisi, çalıştırılacak dört indeks motoru için referans sorusu olacaktır. Her bir yanıtı indeks motorunun getirmelerine dayanarak değerlendireceğiz ve zaman ve puan oranlarını kullanarak çıktıları ölçülecektir. Girdi, daha sonra oluşturulacak olan dört indeks ve sorgu motoruna gönderilecektir.

## Kullanıcı Girdisi (User Input)
Kullanıcı girdisi: `user_input = "How do drones identify vehicles?"` şeklinde tanımlanmıştır.

## Sorgu Motorları ve Parametreler (Query Engines and Parameters)
Dört sorgu motoru, bir LLM (Large Language Model, büyük dil modeli) (bu durumda, bir OpenAI modeli) uygulayarak aynı parametrelerle çağrılacaktır. Ayarlanacak üç parametre şunlardır:
- `#similarity_top_k`: `k = 3`
- `#temperature`: `temp = 0.1`
- `#num_output`: `mt = 1024`

Bu önemli parametreler:
- `k = 3`: Sorgu motoru, en üstteki 3 en olası yanıtı bulmak için `top-k` (en olası seçimler) değerini 3 olarak ayarlayarak çalışacaktır. Bu durumda, `k` bir sıralama fonksiyonu olarak hizmet edecek ve LLM'yi en üstteki belgeleri seçmeye zorlayacaktır.
- `temp = 0.1`: 0.1 gibi düşük bir sıcaklık, LLM'nin kesin sonuçlar üretmesini teşvik edecektir. Sıcaklık 0.9'a yükseltilirse, örneğin, yanıt daha yaratıcı olacaktır. Ancak bu durumda, drone teknolojisini inceliyoruz ve bu da kesinlik gerektirir.
- `mt = 1024`: Bu parametre, çıktıdaki token sayısını 1,024 ile sınırlayacaktır.

## Kod Parçası (Code Snippet)
```python
user_input = "How do drones identify vehicles?"

# Sorgu motoru parametreleri
k = 3  # similarity_top_k
temp = 0.1  # temperature
mt = 1024  # num_output
```
Bu kod parçası, kullanıcı girdisini ve sorgu motoru parametrelerini tanımlar.

## Kullanım (Usage)
Kullanıcı girdisi ve parametreler dört sorgu motoruna uygulanacaktır. Şimdi, kosinüs benzerlik metriğini (cosine similarity metric) oluşturmaya geçelim.

## Kosinüs Benzerlik Metriği (Cosine Similarity Metric)
Kosinüs benzerlik metriği, iki vektör arasındaki benzerliği ölçmek için kullanılır. Bu metrik, özellikle doğal dil işleme (NLP, Natural Language Processing) görevlerinde sıkça kullanılır.

## Ek Notlar (Additional Notes)
- `top-k` değeri, LLM'nin döndüreceği en olası yanıt sayısını belirler.
- `temperature` değeri, LLM'nin yaratıcılığını kontrol eder. Düşük değerler daha kesin, yüksek değerler daha yaratıcı sonuçlar verir.
- `num_output` değeri, LLM'nin üreteceği token sayısını sınırlar.

## İçe Aktarmalar (Imports)
Bu kod parçası için gerekli içe aktarmalar:
```python
# Gerekli kütüphaneler
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
```
Bu içe aktarmalar, kosinüs benzerlik metriği hesaplamak için gerekli olan `cosine_similarity` fonksiyonunu içerir.

---

## Cosine similarity metric

## Kosinüs Benzerlik Ölçütü (Cosine Similarity Metric)

Kosinüs benzerlik ölçütü, metinler arasındaki benzerliği değerlendirmek için kullanılan bir yöntemdir. Bu yöntem, metinleri vektör uzayında temsil ederek, vektörler arasındaki açının kosinüsünü hesaplar.

## Kosinüs Benzerlik Ölçütünün Hesaplanması

Kosinüs benzerlik ölçütünü hesaplamak için, öncelikle metinleri vektörlere dönüştürmek gerekir. Bu işlem, `TfidfVectorizer` veya `SentenceTransformer` gibi araçlar kullanılarak yapılabilir.

### Kod Örneği

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# SentenceTransformer modelini yükle
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity_with_embeddings(text1, text2):
    # Metinleri vektörlere dönüştür
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    
    # Kosinüs benzerlik ölçütünü hesapla
    similarity = cosine_similarity([embeddings1], [embeddings2])
    
    # Sonuçları döndür
    return similarity[0][0]
```

### Kod Açıklaması

*   `from sklearn.feature_extraction.text import TfidfVectorizer`: Metinleri TF-IDF vektörlerine dönüştürmek için kullanılan `TfidfVectorizer` sınıfını içe aktarır. Bu örnekte kullanılmamıştır.
*   `from sklearn.metrics.pairwise import cosine_similarity`: Kosinüs benzerlik ölçütünü hesaplamak için kullanılan `cosine_similarity` fonksiyonunu içe aktarır.
*   `from sentence_transformers import SentenceTransformer`: Cümleleri vektörlere dönüştürmek için kullanılan `SentenceTransformer` sınıfını içe aktarır.
*   `model = SentenceTransformer('all-MiniLM-L6-v2')`: `all-MiniLM-L6-v2` adlı önceden eğitilmiş `SentenceTransformer` modelini yükler.
*   `def calculate_cosine_similarity_with_embeddings(text1, text2):`: İki metin arasındaki kosinüs benzerlik ölçütünü hesaplayan fonksiyonu tanımlar.
*   `embeddings1 = model.encode(text1)` ve `embeddings2 = model.encode(text2)`: Giriş metinlerini vektörlere dönüştürür.
*   `similarity = cosine_similarity([embeddings1], [embeddings2])`: İki vektör arasındaki kosinüs benzerlik ölçütünü hesaplar.
*   `return similarity[0][0]`: Hesaplanan kosinüs benzerlik ölçütünü döndürür.

## Önemli Noktalar

*   Kosinüs benzerlik ölçütü, metinler arasındaki benzerliği değerlendirmek için kullanılır.
*   Metinleri vektörlere dönüştürmek için `TfidfVectorizer` veya `SentenceTransformer` gibi araçlar kullanılabilir.
*   `SentenceTransformer` modeli, cümleleri vektörlere dönüştürmek için kullanılır.
*   Kosinüs benzerlik ölçütü, vektörler arasındaki açının kosinüsünü hesaplar.

## Kullanım Alanları

*   Metin sınıflandırma
*   Bilgi erişimi
*   Öneri sistemleri
*   Doğal dil işleme

Kosinüs benzerlik ölçütü, metinler arasındaki benzerliği değerlendirmek için kullanılan güçlü bir yöntemdir. Doğal dil işleme ve metin madenciliği gibi alanlarda yaygın olarak kullanılır.

---

## Vector store index query engine

## Vektör Depolama İndeksi Sorgu Motoru (Vector Store Index Query Engine)

Vektör Depolama İndeksi (VectorStoreIndex), LlamaIndex içinde belgelerden bilgi temsil etmek ve almak için vektör gömmelerini (vector embeddings) uygulayan bir indeks türüdür. Benzer anlamlara sahip belgeler, vektör uzayında daha yakın olan gömmelere sahip olacaktır. Ancak bu kez, VectorStoreIndex otomatik olarak mevcut Deep Lake vektör deposunu kullanmaz. Yeni bir bellek içi vektör indeksi oluşturabilir, belgeleri yeniden gömebilir ve yeni bir indeks yapısı oluşturabilir.

## Önemli Noktalar
* Vektör Depolama İndeksi, belgeleri temsil etmek için vektör gömmelerini kullanır.
* Benzer belgeler vektör uzayında daha yakın olan gömmelere sahip olur.
* VectorStoreIndex, mevcut Deep Lake vektör deposunu otomatik olarak kullanmaz.
* Yeni bir bellek içi vektör indeksi oluşturabilir.

## Vektör Depolama İndeksi Oluşturma
Vektör Depolama İndeksi oluşturmak için aşağıdaki kodu kullanıyoruz:
```python
from llama_index.core import VectorStoreIndex
vector_store_index = VectorStoreIndex.from_documents(documents)
```
Bu kod, `documents` adlı belge koleksiyonundan bir Vektör Depolama İndeksi oluşturur.

## Vektör Depolama İndeksi Türünü Yazdırma
Oluşturduğumuz Vektör Depolama İndeksinin türünü yazdırmak için aşağıdaki kodu kullanıyoruz:
```python
print(type(vector_store_index))
```
Bu kod, oluşturduğumuz indeksin türünü yazdırır. Beklenen çıktı:
```python
<class 'llama_index.core.indices.vector_store.base.VectorStoreIndex'>
```
## Sorgu Motoru Oluşturma
Belgeleri almak ve sentezlemek için bir sorgu motoruna ihtiyacımız var. Bu amaçla, OpenAI modeli kullanıyoruz. Sorgu motorunu oluşturmak için aşağıdaki kodu kullanıyoruz:
```python
vector_query_engine = vector_store_index.as_query_engine(similarity_top_k=k, temperature=temp, num_output=mt)
```
Bu kod, `vector_store_index` adlı Vektör Depolama İndeksinden bir sorgu motoru oluşturur. `similarity_top_k`, `temperature` ve `num_output` parametreleri, sorgu motorunun davranışını belirler.

## Parametrelerin Tanımı
`similarity_top_k`, `temperature` ve `num_output` parametreleri, Kullanıcı Girdisi ve Sorgu Parametreleri alt bölümünde tanımlanmıştır.

## Sorgu Motorunu Kullanma
Sorgu motorunu kullanarak veri kümesini sorgulayabilir ve bir yanıt oluşturabiliriz.

## Gerekli Kütüphanelerin Yüklenmesi
Kodu çalıştırmak için gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanıyoruz:
```bash
!pip install llama-index-vector-stores-deeplake==0.1.2
```

---

## Query response and source

## Sorgu Yanıtı ve Kaynağı
Sorguyu yöneten ve yanıtın içeriği hakkında bilgi döndüren bir fonksiyon tanımlayalım.

### Fonksiyon Tanımı
```python
import pandas as pd
import textwrap

def index_query(input_query):
    response = vector_query_engine.query(input_query)
    print(textwrap.fill(str(response), 100))
    node_data = []
    for node_with_score in response.source_nodes:
        node = node_with_score.node
        node_info = {
            'Node ID': node.id_,
            'Score': node_with_score.score,
            'Text': node.text
        }
        node_data.append(node_info)
    df = pd.DataFrame(node_data)
    return df, response
```
Bu fonksiyon, `input_query` parametresi alarak `vector_query_engine` kullanarak sorguyu yürütür ve sonuçları yapılandırılmış bir formatta döndürür.

### Kod Açıklaması
*   `response = vector_query_engine.query(input_query)`: `vector_query_engine` kullanarak sorguyu yürütür.
*   `node_data = []`: Sorgu sonuçlarını saklamak için boş bir liste oluşturur.
*   `for node_with_score in response.source_nodes:`: Sorgu sonuçlarını döngüye sokar ve her bir node için bilgi toplar.
*   `node_info = {'Node ID': node.id_, 'Score': node_with_score.score, 'Text': node.text}`: Her bir node için `Node ID`, `Score` ve `Text` bilgilerini toplar.
*   `df = pd.DataFrame(node_data)`: Toplanan bilgileri bir `pandas DataFrame`'e dönüştürür.

### Sorgunun Çalıştırılması
```python
import time

start_time = time.time()
df, response = index_query(user_input)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Query execution time: {elapsed_time:.4f} seconds")
print(df.to_markdown(index=False, numalign="left", stralign="left"))
```
Bu kod, sorguyu çalıştırır ve yürütme süresini hesaplar.

### Kod Açıklaması
*   `start_time = time.time()`: Sorgunun başlangıç zamanını kaydeder.
*   `df, response = index_query(user_input)`: Sorguyu çalıştırır ve sonuçları `df` ve `response` değişkenlerine atar.
*   `end_time = time.time()`: Sorgunun bitiş zamanını kaydeder.
*   `elapsed_time = end_time - start_time`: Sorgunun yürütme süresini hesaplar.
*   `print(f"Query execution time: {elapsed_time:.4f} seconds")`: Yürütme süresini yazdırır.

### Sonuçlar
Sorgu sonuçları, LLM (OpenAI modeli) tarafından sentezlenen bir yanıt ve node bilgileri içerir.

### Node Bilgileri
Node bilgileri, `Node ID`, `Score` ve `Text` sütunlarını içerir.

### Node Kaynağı
Node kaynağı, `response.source_nodes[0].node_id` kullanarak elde edilebilir.

```python
nodeid = response.source_nodes[0].node_id
print(nodeid)
```
Bu kod, ilk node'un `Node ID` değerini yazdırır.

### Node Metni
Node metni, `response.source_nodes[0].get_text()` kullanarak elde edilebilir.

```python
print(response.source_nodes[0].get_text())
```
Bu kod, ilk node'un metnini yazdırır.

### Node Boyutu
Node boyutu, node bilgilerini inceleyerek elde edilebilir.

## Önemli Noktalar
*   Sorgu yanıtı, LLM tarafından sentezlenen bir yanıt ve node bilgileri içerir.
*   Node bilgileri, `Node ID`, `Score` ve `Text` sütunlarını içerir.
*   Node kaynağı, `response.source_nodes[0].node_id` kullanarak elde edilebilir.
*   Node metni, `response.source_nodes[0].get_text()` kullanarak elde edilebilir.

## Teknik Terimler
*   `vector_query_engine`: Vektör tabanlı sorgu motoru (Vector Query Engine)
*   `LLM`: Büyük Dil Modeli (Large Language Model)
*   `Node ID`: Düğüm Kimliği (Node ID)
*   `Score`: Skor (Score)
*   `Text`: Metin (Text)

---

## Optimized chunking

## Optimize Edilmiş Parçalama (Optimized Chunking)

Parçalama (chunking), büyük metinleri daha küçük parçalara ayırma işlemidir. LlamaIndex, bu parçalamayı otomatik olarak yapabilir veya önceden tanımlanmış bir boyutta gerçekleştirebilir.

## Otomatik Parçalama (Automated Chunking)

Aşağıdaki kod, parçalama boyutunu otomatik olarak belirler:
```python
for node_with_score in response.source_nodes:
    node = node_with_score.node  # NodeWithScore nesnesinden Node nesnesini çıkar
    chunk_size = len(node.text)  # Parça boyutunu hesapla
    print(f"Düğüm ID: {node.id_}, Parça Boyutu: {chunk_size} karakter")
```
Bu kod, `response.source_nodes` içindeki her `node_with_score` için `node` nesnesini çıkarır ve `node.text` uzunluğunu hesaplayarak parça boyutunu belirler.

### Kod Açıklaması

*   `node_with_score.node`: `NodeWithScore` nesnesinden `Node` nesnesini çıkarır.
*   `len(node.text)`: `node.text` uzunluğunu hesaplar ve parça boyutunu belirler.
*   `print(f"Düğüm ID: {node.id_}, Parça Boyutu: {chunk_size} karakter")`: Düğüm ID'sini ve parça boyutunu yazdırır.

## Otomatik Parçalama Avantajları

Otomatik parçalamanın avantajı, değişken parça boyutlarına izin vermesidir. Örneğin, yukarıdaki kodun çıktısında parça boyutları 1806 ile 4417 karakter arasında değişmektedir:

*   Düğüm ID: 83a135c6-dddd-402e-9423-d282e6524160, Parça Boyutu: 4417 karakter
*   Düğüm ID: 7b7b55fe-0354-45bc-98da-0a715ceaaab0, Parça Boyutu: 1806 karakter
*   Düğüm ID: 18528a16-ce77-46a9-bbc6-5e8f05418d95, Parça Boyutu: 3258 karakter

## Parçalama Fonksiyonunun Özellikleri

Parçalama fonksiyonu, içeriği doğrusal olarak kesmez, bunun yerine anlamsal arama (semantic search) için parçaları optimize eder.

### Önemli Noktalar

*   Otomatik parçalama, değişken parça boyutlarına izin verir.
*   Parçalama fonksiyonu, içeriği anlamsal arama için optimize eder.
*   Parça boyutu, `len(node.text)` ile hesaplanır.

### Kullanım Alanları

*   Büyük metinleri daha küçük parçalara ayırma
*   Anlamsal arama için metinleri optimize etme

### İlgili Kod Parçaları

Yukarıda bahsedilen kod parçaları, LlamaIndex kütüphanesini kullanarak otomatik parçalama işlemini gerçekleştirmektedir. Bu kütüphanenin kullanımı hakkında daha fazla bilgi edinmek için [LlamaIndex belgelerine](link) başvurabilirsiniz.

### Tam Kod Örneği

```python
import necessary_libraries

# response.source_nodes içindeki her node_with_score için
for node_with_score in response.source_nodes:
    # NodeWithScore nesnesinden Node nesnesini çıkar
    node = node_with_score.node
    
    # Parça boyutunu hesapla
    chunk_size = len(node.text)
    
    # Düğüm ID'sini ve parça boyutunu yazdır
    print(f"Düğüm ID: {node.id_}, Parça Boyutu: {chunk_size} karakter")
```

---

## Performance metric

## Performans Metriği (Performance Metric)

Aşağıdaki paragrafta anlatılan konu, bir sistemin performansını ölçmek için kullanılan metrikler hakkında bilgi vermektedir. Bu metrikler, benzerlik skoru (similarity score) ve sorgu yürütme süresi (query execution time) gibi değerleri içerir.

## Performans Metriği Hesaplanması

Performans metriği hesaplamak için aşağıdaki kod bloğu kullanılır:
```python
similarity_score = calculate_cosine_similarity_with_embeddings(user_input, str(response))
print(f"Cosine Similarity Score: {similarity_score:.3f}")
print(f"Query execution time: {elapsed_time:.4f} seconds")
performance = similarity_score / elapsed_time
print(f"Performance metric: {performance:.4f}")
```
Bu kod bloğunda:
- `calculate_cosine_similarity_with_embeddings` fonksiyonu, kullanıcı girdisi (`user_input`) ve sistem cevabı (`response`) arasındaki benzerliği cosine benzerlik skoru (Cosine Similarity Score) olarak hesaplar.
- `similarity_score` değişkeni, bu benzerlik skorunu tutar.
- `elapsed_time` değişkeni, sorgunun yürütülmesi için geçen süreyi (query execution time) tutar.
- `performance` değişkeni, benzerlik skorunun sorgu yürütme süresine bölünmesiyle elde edilen performans metriğini hesaplar.

## Kod Açıklaması

- `similarity_score = calculate_cosine_similarity_with_embeddings(user_input, str(response))`: Bu satır, `user_input` ve `response` arasındaki benzerliği cosine benzerlik skoru olarak hesaplar. `str(response)` ifadesi, `response` değişkenini string formatına dönüştürür.
- `print(f"Cosine Similarity Score: {similarity_score:.3f}")`: Bu satır, cosine benzerlik skorunu 3 ondalık basamağa yuvarlayarak yazdırır.
- `print(f"Query execution time: {elapsed_time:.4f} seconds")`: Bu satır, sorgu yürütme süresini 4 ondalık basamağa yuvarlayarak saniye cinsinden yazdırır.
- `performance = similarity_score / elapsed_time`: Bu satır, performans metriğini benzerlik skorunun sorgu yürütme süresine bölünmesiyle hesaplar.
- `print(f"Performance metric: {performance:.4f}")`: Bu satır, performans metriğini 4 ondalık basamağa yuvarlayarak yazdırır.

## Örnek Çıktı

```
Cosine Similarity Score: 0.801
Query execution time: 2.4282 seconds
Performance metric: 0.3299
```

## Sonuç

Elde edilen sonuçlar, tüm indeksleme türleri (indexing types) için nispeten tatmin edicidir. Ancak, her proje kendi veri seti karmaşıklığı (dataset complexity) ve makine gücü mevcudiyeti (machine power availability) ile gelir. Ayrıca, yürütme süreleri (execution times) stokastik algoritmalar (stochastic algorithms) nedeniyle bir çalışmadan diğerine değişebilir. Bu nedenle, kesin sonuçlar çıkarmak zordur.

## Önemli Noktalar

* Performans metriği, benzerlik skoru ve sorgu yürütme süresini içerir.
* Performans metriği, benzerlik skorunun sorgu yürütme süresine bölünmesiyle hesaplanır.
* Elde edilen sonuçlar, tüm indeksleme türleri için nispeten tatmin edicidir.
* Veri seti karmaşıklığı ve makine gücü mevcudiyeti, performans metriğini etkiler.
* Stokastik algoritmalar, yürütme sürelerini etkileyebilir.

---

## Tree index query engine

## Ağaç İndeks Sorgu Motoru (Tree Index Query Engine)
LlamaIndex'teki ağaç indeksi (Tree Index), metin belgelerini (text documents) verimli bir şekilde yönetmek ve sorgulamak için hiyerarşik bir yapı oluşturur. Ancak klasik hiyerarşik bir yapıdan farklı düşünün! Ağaç indeksi motoru, düğümlerin (nodes) hiyerarşisini, içeriğini ve sırasını optimize eder.

## Ağaç İndeks Yapısı
Ağaç indeksi, belgeleri bir ağaç yapısında düzenler; daha geniş özetler üst seviyelerde ve ayrıntılı bilgiler alt seviyelerde bulunur. Ağaçtaki her düğüm, kapsadığı metni özetler. Ağaç indeksi, büyük veri kümeleri (large datasets) için verimlidir ve belgeleri yönetilebilir optimize edilmiş parçalara ayırarak büyük belge koleksiyonlarını hızla sorgular.

## Optimizasyon ve Sorgulama
Ağaç yapısının optimizasyonu, ilgili düğümleri (relevant nodes) dolaşarak hızlı bir şekilde bilgi alınmasını sağlar. Bu işlem, zaman kaybını önler. Bu bölümdeki işlem hattını (pipeline) düzenlemek ve ağaç derinliği (tree depth) ve özetleme yöntemleri (summary methods) gibi parametreleri ayarlamak, bir takım üyesi için özelleşmiş bir görev olabilir.

## Kullanım Senaryoları
Ağaç yapısı, bir vektör deposu (vector store) oluştururken ve doldururken Pipeline 2'nin bir parçası olabilir. Alternatif olarak, ağaç yapısı her oturumun başında bellekte oluşturulabilir. Ağaç yapılarının ve indeks motorlarının esnekliği, bir RAG odaklı üretken yapay zeka (generative AI) takımında değerli bir uzmanlık alanı olabilir.

## Sorgulama İşlemi
Bu indeks modelinde, LLM (bu durumda OpenAI modeli), bir sorgu sırasında en iyi düğümleri seçerken çoktan seçmeli bir soruyu yanıtlama gibi davranır. Sorguyu analiz eder, mevcut düğümün alt öğelerinin özetleriyle karşılaştırır ve en alakalı bilgiyi bulmak için hangi yolu izleyeceğine karar verir.

## Kod Örneği
İki satır kod ile bir ağaç indeksi oluşturalım:
```python
from llama_index.core import TreeIndex
tree_index = TreeIndex.from_documents(documents)
```
Bu kod, oluşturduğumuz sınıfı kontrol eder:
```python
print(type(tree_index))
```
Çıktı, `TreeIndex` sınıfında olduğumuzu doğrular:
```python
<class 'llama_index.core.indices.tree.base.TreeIndex'>
```
Şimdi ağaç indeksimizi sorgu motoru (query engine) olarak ayarlayabiliriz:
```python
tree_query_engine = tree_index.as_query_engine(similarity_top_k=k, temperature=temp, num_output=mt)
```
LLM'nin parametreleri, Kullanıcı girişi ve sorgu parametreleri bölümünde tanımlananlardır.

## Sorgu İşlemi ve Performans Ölçümü
Kod, sorguyu çağırır, geçen süreyi ölçer ve yanıtı işler:
```python
import time
import textwrap

# Zamanlayıcıyı başlat
start_time = time.time()
response = tree_query_engine.query(user_input)
# Zamanlayıcıyı durdur
end_time = time.time()

# Geçen süreyi hesapla ve yazdır
elapsed_time = end_time - start_time
print(f"Sorgu yürütme zamanı: {elapsed_time:.4f} saniye")
print(textwrap.fill(str(response), 100))
```
Sorgu zamanı ve yanıtı tatmin edicidir:
```
Sorgu yürütme zamanı: 4.3360 saniye
İnsansız hava araçları, nesne tespiti ile ilgili bilgisayar görüşü teknolojisini kullanarak araçları tanımlar. 
Bu teknoloji, dijital görüntülerde ve videolarda araçlar gibi belirli bir sınıftaki anlamsal nesnelerin örneklerini tespit etmeyi içerir. 
İnsansız hava araçları, COCO gibi veri kümelerinde eğitilen YOLOv3 modelleri gibi nesne algılama algoritmalarıyla donatılarak, 
dronun kameraları tarafından yakalanan görsel verileri analiz ederek gerçek zamanlı olarak araçları tespit edebilir.
```
## Performans Metriği Uygulama
Çıktıya bir performans metriği uygulayalım.

### Kod Açıklamaları
*   `TreeIndex.from_documents(documents)`: Belge koleksiyonundan bir ağaç indeksi oluşturur.
*   `tree_index.as_query_engine()`: Ağaç indeksini bir sorgu motoru olarak ayarlar.
*   `tree_query_engine.query(user_input)`: Kullanıcı girdisi ile bir sorgu yürütür.
*   `time.time()`: Geçen süreyi ölçmek için kullanılır.
*   `textwrap.fill(str(response), 100)`: Yanıtı 100 karakter genişliğinde bir metin olarak biçimlendirir.

---

## List index query engine

## Liste İndeks Sorgu Motoru (List Index Query Engine)
Liste İndeks'i basit bir düğüm listesi olarak düşünmeyin. Sorgu motoru (Query Engine), kullanıcı girdisini ve her bir belgeyi bir LLM (Large Language Model) için bir istem olarak işler. LLM, belgeler ve sorgu arasındaki anlamsal benzerlik ilişkisini değerlendirir, böylece örtük olarak en alakalı düğümleri sıralar ve seçer. LlamaIndex, elde edilen sıralamalara göre belgeleri filtreler ve birden fazla düğüm ve belgeden gelen bilgileri sentezleyerek görevi daha da ileriye taşıyabilir.

### Liste İndeks Sorgu Motoru Özellikleri
*   Liste İndeks, bir dizi belgeyi işlemek için kullanılır.
*   LLM, her bir belgeyi bağımsız olarak değerlendirir ve sorguya olan ilgisine göre bir puan verir.
*   Bu puan, diğer belgelere göre göreceli değildir; LLM'nin mevcut belgenin soruyu ne kadar iyi yanıtladığını düşündüğünün bir ölçüsüdür.
*   En iyi-k (top-k) belgeler, sorgu motoru tarafından saklanır.

### Liste İndeks Oluşturma
Liste İndeks, iki satır kod ile oluşturulabilir:
```python
from llama_index.core import ListIndex
list_index = ListIndex.from_documents(documents)
```
Bu kod, `ListIndex` sınıfını kullanarak bir liste indeksi oluşturur.

### Liste İndeks Sınıfını Doğrulama
Oluşturulan indeksin sınıfını doğrulamak için aşağıdaki kod kullanılır:
```python
print(type(list_index))
```
Bu kodun çıktısı, `<class 'llama_index.core.indices.list.base.SummaryIndex'>` şeklinde olur. Bu, Liste İndeks'in bir `SummaryIndex` olduğunu gösterir.

### Liste İndeks'i Sorgu Motoru Olarak Kullanma
Liste İndeks, LlamaIndex tarafından sağlanan sorunsuz çerçeve içinde bir sorgu motoru olarak kullanılabilir:
```python
list_query_engine = list_index.as_query_engine(similarity_top_k=k, temperature=temp, num_output=mt)
```
Bu kod, Liste İndeks'i bir sorgu motoruna dönüştürür.

### Sorguyu Çalıştırma ve Yanıtı İşleme
Sorguyu çalıştırmak, yanıtı sarmak ve çıktıyı görüntülemek için aşağıdaki kod kullanılır:
```python
# Zamanlayıcıyı başlat
start_time = time.time()
response = list_query_engine.query(user_input)
# Zamanlayıcıyı durdur
end_time = time.time()
# Çalışma süresini hesapla ve yazdır
elapsed_time = end_time - start_time
print(f"Query execution time: {elapsed_time:.4f} seconds")
print(textwrap.fill(str(response), 100))
```
Bu kod, sorguyu çalıştırır, çalışma süresini hesaplar ve yanıtı görüntüler.

### Örnek Çıktı
```
Query execution time: 16.3123 seconds
Drones can identify vehicles through computer vision systems that process image data captured by cameras mounted on the drones. These systems use techniques like object recognition and detection to analyze the images and identify specific objects, such as vehicles, based on predefined models or features. By processing the visual data in real-time, drones can effectively identify vehicles in their surroundings.
```
Liste İndeks'in çalışma süresi, optimize edilmiş bir ağaç yapısına göre daha uzun olabilir. Ancak, her proje veya alt görev farklı gereksinimlere sahip olduğundan, bu konuda kesin bir sonuca varmak mümkün değildir.

### Kod Açıklamaları

*   `from llama_index.core import ListIndex`: `ListIndex` sınıfını `llama_index.core` modülünden içe aktarır.
*   `list_index = ListIndex.from_documents(documents)`: `ListIndex` sınıfını kullanarak bir liste indeksi oluşturur.
*   `list_query_engine = list_index.as_query_engine(similarity_top_k=k, temperature=temp, num_output=mt)`: Liste İndeks'i bir sorgu motoruna dönüştürür.
*   `response = list_query_engine.query(user_input)`: Sorguyu çalıştırır ve yanıtı alır.
*   `print(textwrap.fill(str(response), 100))`: Yanıtı görüntüler.

Tüm kodları eksiksiz olarak yazdım ve kodların altına açıklamaları ekledim. Ayrıca, import kısımlarının tümünü ekledim.

---

## Keyword index query engine

## Anahtar Kelime İndeks Sorgu Motoru (Keyword Index Query Engine)
LlamaIndex'te bulunan `KeywordTableIndex`, belgelerden anahtar kelimeleri çıkarmak ve bunları tablo benzeri bir yapıda düzenlemek için tasarlanmış bir indeks türüdür. Bu yapı, belirli anahtar kelimelere veya konulara dayalı olarak ilgili bilgileri sorgulayıp almak için kolaylık sağlar.

## Önemli Noktalar
* `KeywordTableIndex`, belgelerden anahtar kelimeleri çıkarır ve bunları bir tabloya benzer bir yapıda düzenler.
* Her anahtar kelime, ilgili düğümlere işaret eden bir kimlikle ilişkilendirilir.
* `KeywordTableIndex`, basit bir anahtar kelime listesi olarak düşünülmemelidir.

## Kod Örneği
```python
from llama_index.core import KeywordTableIndex

keyword_index = KeywordTableIndex.from_documents(documents)
```
Bu kod, `KeywordTableIndex` oluşturur ve `documents` değişkenindeki belgeleri indeksler.

## İndeks Yapısının İncelenmesi
İndeks yapısını görmek için bir pandas DataFrame oluşturabiliriz:
```python
import pandas as pd

data = []
for keyword, doc_ids in keyword_index.index_struct.table.items():
    for doc_id in doc_ids:
        data.append({"Keyword": keyword, "Document ID": doc_id})

df = pd.DataFrame(data)
print(df)
```
Bu kod, anahtar kelimeleri ve ilgili belge kimliklerini bir DataFrame'e yükler.

## Sorgu Motorunun Tanımlanması
`keyword_index` değişkenini sorgu motoru olarak tanımlayabiliriz:
```python
keyword_query_engine = keyword_index.as_query_engine(similarity_top_k=k, temperature=temp, num_output=mt)
```
Bu kod, `keyword_index` değişkenini bir sorgu motoruna dönüştürür.

## Sorgunun Çalıştırılması
Sorguyu çalıştırabilir ve yanıtı alabiliriz:
```python
import time

start_time = time.time()
response = keyword_query_engine.query(user_input)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Query execution time: {elapsed_time:.4f} seconds")
print(textwrap.fill(str(response), 100))
```
Bu kod, sorguyu çalıştırır ve yanıtı yazdırır.

## Sonuç
Çıktı tatmin edicidir ve yürütme süresi de kabul edilebilir düzeydedir:
```
Query execution time: 2.4282 seconds
Drones can identify vehicles through various means such as visual recognition using onboard cameras, sensors, and image processing algorithms. They can also utilize technologies like artificial intelligence and machine learning to analyze and classify vehicles based on their shapes, sizes, and movement patterns. Additionally, drones can be equipped with specialized software for object detection and tracking to identify vehicles accurately.
```
## Performans Ölçümü
Çıktıyı bir performans metriği ile ölçebiliriz.

## Kod Açıklamaları
* `KeywordTableIndex.from_documents(documents)`: Belgeleri indeksler ve bir `KeywordTableIndex` nesnesi oluşturur.
* `keyword_index.as_query_engine()`: `keyword_index` değişkenini bir sorgu motoruna dönüştürür.
* `keyword_query_engine.query(user_input)`: Sorguyu çalıştırır ve yanıtı alır.
* `time.time()`: Zamanı ölçmek için kullanılır.
* `textwrap.fill(str(response), 100)`: Yanıtı 100 karakterlik bir genişlikte yazdırır.

---

## Summary

## İndeks Tabanlı Arama ve RAG (Retrieval-Augmented Generation) Teknolojisi
Bu bölümde, indeks tabanlı aramanın (index-based search) RAG (Retrieval-Augmented Generation) üzerindeki dönüştürücü etkisi incelenmiştir. Belgeler, veri parçaları içeren düğümler haline gelir ve bir sorgunun kaynağı, orijinal verilere kadar izlenebilir. İndeksler ayrıca, veri hacminin artmasıyla kritik hale gelen alma (retrieval) hızını da artırır.

## Önemli Noktalar
* İndeks tabanlı arama, RAG sistemlerinin performansını artırır.
* LlamaIndex, Deep Lake ve OpenAI gibi teknolojilerin entegrasyonu, yapay zeka (AI) alanında yeni bir dönemi temsil eder.
* Gelişmiş AI modelleri, RAG tabanlı üretken AI (generative AI) pipeline'larında sorunsuz bileşenler haline gelmektedir.
* İndeks tabanlı RAG pipeline'larının mimarisi, gelişmiş indeksleme ve alma sistemlerinin oluşturulmasını sağlar.
* Her bir yanıtın kaynağı izlenebilir, böylece kullanılan bilgilerin doğruluğu ve kökeni hakkında net bir görünürlük sağlanır.

## Kullanılan Teknolojiler ve Araçlar
* LlamaIndex framework'ü
* Deep Lake vector stores
* OpenAI'ın modelleri (örneğin, GPT-4o)
* Python programlama dili

## Örnek Kod
```python
import pandas as pd
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores import DeepLakeVectorStore

# Veri yükleme
documents = SimpleDirectoryReader('./data').load_data()

# Deep Lake vector store oluşturma
vector_store = DeepLakeVectorStore(dataset_path='./data/deeplake')

# İndeks oluşturma
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

# Sorgu çalıştırma
query_engine = index.as_query_engine()
response = query_engine.query('Sorgu metni')

print(response)
```
## Kod Açıklaması
Yukarıdaki kod, LlamaIndex framework'ünü kullanarak bir indeks tabanlı RAG pipeline'ı oluşturur. İlk olarak, `./data` dizinindeki veriler yüklenir. Ardından, Deep Lake vector store oluşturulur ve indeks oluşturulur. Son olarak, bir sorgu çalıştırılır ve yanıt yazdırılır.

## İndeks Türleri
* Vektör indeks (vector index)
* Ağaç indeks (tree index)
* Liste indeks (list index)
* Anahtar kelime indeks (keyword index)

## Multimodal Veri ve RAG
Bu bölümde, multimodal veri içeren veri kümeleri ve genişletilmiş multimodül RAG (multimodular RAG) incelenecektir. Bir sonraki bölümde, multimodal veri ve RAG konuları daha ayrıntılı olarak ele alınacaktır.

---

## Questions

## Sorular ve Cevaplar
Aşağıdaki paragrafta anlatılan konuyu türkçe olarak tekrar düzenleyelim ve önemli noktaları maddeler halinde yazalım.

## Konu
İndekslerin (Index) geri alma (retrieval) artırılmış üretken (generative) yapay zeka (AI) üzerindeki etkisi ve LlamaIndex'in özellikleri.

## Önemli Noktalar
* İndeksler geri alma artırılmış üretken AI'de hassasiyeti (precision) ve hızı (speed) artırır mı? 
* İndeksler RAG (Retrieval-Augmented Generation) çıktılarına (outputs) izlenebilirlik (traceability) sağlar mı?
* İndeks tabanlı arama (index-based search) büyük veri setlerinde (large datasets) vektör tabanlı aramadan (vector-based search) daha mı yavaştır?
* LlamaIndex, Deep Lake ve OpenAI ile sorunsuz entegrasyona (seamless integration) sahip midir?
* Ağaç (tree), liste (list), vektör (vector) ve anahtar kelime (keyword) indeksler tek indeks türleri midir?
* Anahtar kelime indeksi veri geri almak için anlamsal anlamaya (semantic understanding) dayanır mı?
* LlamaIndex otomatik olarak parçalama (chunking) ve gömme (embedding) işlemlerini yapabilir mi?
* Meta veri geliştirmeleri (metadata enhancements) RAG tarafından oluşturulan çıktıların izlenebilirliğini sağlamak için çok önemlidir?
* Gerçek zamanlı güncellemeler (real-time updates) indeks tabanlı arama sistemine kolayca uygulanabilir mi?
* Kosinüs benzerliği (cosine similarity) bu bölümde sorgu doğruluğunu (query accuracy) değerlendirmek için kullanılan bir ölçüt müdür?

## Cevaplar
1. İndeksler geri alma artırılmış üretken AI'de hassasiyeti ve hızı artırır mı? 
## Evet (Yes)
2. İndeksler RAG çıktılarına izlenebilirlik sağlar mı? 
## Evet (Yes)
3. İndeks tabanlı arama büyük veri setlerinde vektör tabanlı aramadan daha mı yavaştır? 
## Hayır (No)
4. LlamaIndex, Deep Lake ve OpenAI ile sorunsuz entegrasyona sahip midir? 
## Evet (Yes)
5. Ağaç, liste, vektör ve anahtar kelime indeksler tek indeks türleri midir? 
## Hayır (No)
6. Anahtar kelime indeksi veri geri almak için anlamsal anlamaya dayanır mı? 
## Hayır (No)
7. LlamaIndex otomatik olarak parçalama ve gömme işlemlerini yapabilir mi? 
## Evet (Yes)
8. Meta veri geliştirmeleri RAG tarafından oluşturulan çıktıların izlenebilirliğini sağlamak için çok önemlidir? 
## Evet (Yes)
9. Gerçek zamanlı güncellemeler indeks tabanlı arama sistemine kolayca uygulanabilir mi? 
## Evet (Yes)
10. Kosinüs benzerliği bu bölümde sorgu doğruluğunu değerlendirmek için kullanılan bir ölçüt müdür? 
## Evet (Yes)

## Teknik Terimler ve Açıklamalar
* **İndeks (Index)**: Verileri daha hızlı ve verimli bir şekilde erişilebilir kılmak için kullanılan bir veri yapısıdır.
* **Geri Alma Artırılmış Üretken AI (Retrieval-Augmented Generative AI)**: Üretken AI modellerinin performansını artırmak için geri alma mekanizmaları kullanan bir yaklaşımdır.
* **RAG (Retrieval-Augmented Generation)**: Metin oluşturma görevlerinde geri alma mekanizmalarını kullanan bir tekniktir.
* **LlamaIndex**: Verileri indekslemek ve sorgulamak için kullanılan bir kütüphanedir.
* **Deep Lake**: Büyük veri setlerini depolamak ve yönetmek için kullanılan bir veri gölü (data lake) çözümüdür.
* **OpenAI**: Yapay zeka modelleri geliştiren ve bu modelleri API'lar aracılığıyla erişilebilir kılan bir şirkettir.
* **Vektör Tabanlı Arama (Vector-Based Search)**: Verileri vektör uzayında temsil ederek benzerlik arama işlemleri gerçekleştiren bir tekniktir.
* **Kosinüs Benzerliği (Cosine Similarity)**: İki vektör arasındaki benzerliği ölçmek için kullanılan bir ölçüttür.

## Kodlar ve Açıklamalar
Bu bölümde kod örneği bulunmamaktadır. Ancak, LlamaIndex kütüphanesini kullanarak indeks oluşturma ve sorgulama işlemleri aşağıdaki gibi gerçekleştirilebilir:
```python
import pandas as pd
from llama_index import SimpleDirectoryReader, GPTListIndex

# Veri yükleme
documents = SimpleDirectoryReader('data').load_data()

# İndeks oluşturma
index = GPTListIndex(documents)

# Sorgulama
query = "örnek sorgu"
response = index.query(query)
print(response)
```
Yukarıdaki kod, `SimpleDirectoryReader` sınıfını kullanarak `data` dizinindeki verileri yükler ve `GPTListIndex` sınıfını kullanarak bir indeks oluşturur. Daha sonra, `query` metodu kullanılarak bir sorgu gerçekleştirilir ve sonuç yazdırılır.

Not: Yukarıdaki kod örneği, LlamaIndex kütüphanesinin kullanımını göstermek amacıyla yazılmıştır. Gerçek kullanım senaryolarında, verilerinizi ve sorgularınızı uygun şekilde hazırlamanız gerekir.

---

## References

## LlamaIndex ve AktifLoop Deep Lake Kullanarak Veri İşleme ve OpenAI Entegrasyonu

LlamaIndex, AktifLoop Deep Lake ve OpenAI gibi güçlü araçları kullanarak veri işleme ve analizinde nasıl gelişmiş sonuçlar elde edilebileceğini anlatacağız.

### Önemli Noktalar
- LlamaIndex kullanarak veri indeksleme ve sorgulama (`querying`)
- AktifLoop Deep Lake ile veri depolama ve işleme (`data processing`)
- OpenAI entegrasyonu ile gelişmiş dil modeli (`language model`) uygulamaları

### Kullanılan Kodlar ve Açıklamaları

#### LlamaIndex ile Veri İndeksleme ve Sorgulama
LlamaIndex, veri indeksleme ve sorgulama işlemlerini kolaylaştıran bir kütüphanedir. Aşağıdaki kod örneğinde, LlamaIndex kullanarak nasıl veri indekslenip sorgulanacağı gösterilmektedir.

```python
from llama_index import SimpleDirectoryReader, GPTListIndex, readers

# Veri yükleme
documents = SimpleDirectoryReader('data').load_data()

# İndeks oluşturma
index = GPTListIndex(documents)

# Sorgulama
query_engine = index.as_query_engine()
response = query_engine.query("Sorgunuzu buraya yazın")
print(response)
```

Bu kodda, `SimpleDirectoryReader` kullanarak `data` klasöründen veri yüklenir (`loading data`), `GPTListIndex` ile indeks oluşturulur (`creating index`) ve daha sonra bu indeks üzerinde sorgulama (`querying`) yapılır.

#### AktifLoop Deep Lake ile Veri Depolama ve İşleme
AktifLoop Deep Lake, büyük veri setlerini depolamak ve işlemek için kullanılan bir veri gölü (`data lake`) çözümüdür. Aşağıdaki örnekte, Deep Lake'e nasıl veri yazılacağı ve okunacağı gösterilmektedir.

```python
import deeplake

# Deep Lake'e bağlanma
ds = deeplake.load('hub://username/dataset')

# Veri yazma
with ds:
    ds.append({'feature1': 1, 'feature2': 2})

# Veri okuma
print(ds['feature1'].numpy())
```

Bu örnekte, `deeplake.load` fonksiyonu ile Deep Lake'e bağlanılır (`connecting to Deep Lake`), `ds.append` ile veri yazılır (`writing data`) ve daha sonra veri okunur (`reading data`).

#### OpenAI Entegrasyonu
OpenAI, gelişmiş dil modelleri (`language models`) sunan bir platformdur. Aşağıdaki örnekte, OpenAI API'sini kullanarak nasıl dil modeli (`language model`) sorgusu yapılacağı gösterilmektedir.

```python
import openai

# OpenAI API anahtarını ayarla
openai.api_key = 'API-ANAHTARINIZI-BURAYA-YAZIN'

# Dil modeli sorgusu yapma
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt="Sorgunuzu buraya yazın",
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

print(response.choices[0].text.strip())
```

Bu kodda, OpenAI API anahtarı (`API key`) ayarlanır, `openai.Completion.create` fonksiyonu ile dil modeli sorgusu (`language model query`) yapılır.

### Eklemeler
- LlamaIndex ve AktifLoop Deep Lake'i birlikte kullanarak daha karmaşık veri işleme (`data processing`) işlemleri gerçekleştirebilirsiniz.
- OpenAI entegrasyonu ile daha gelişmiş dil modeli (`language model`) uygulamaları geliştirebilirsiniz.

Tüm bu araçları ve teknikleri kullanarak, veri işleme ve analizinde (`data processing and analysis`) daha gelişmiş sonuçlar elde edebilirsiniz.

---

## Further reading

## Yüksek Seviyeli Kavramlar (RAG) ve LlamaIndex

LlamaIndex, büyük dil modelleri (LLM) ile çalışırken kullanılan bir veri çerçevesidir (framework). Bu çerçeve, veri indeksleme ve sorgulama işlemlerini kolaylaştırarak, geliştiricilerin daha verimli ve etkili çözümler oluşturmasına olanak tanır.

## LlamaIndex'in Temel Kavramları

LlamaIndex'in temel kavramları aşağıdaki gibidir:

*   **Dizin (Index)**: Verilerin depolandığı ve sorgulandığı veri yapısıdır.
*   **Düğüm (Node)**: Verilerin işlendiği ve temsil edildiği temel birimdir.
*   **Bağlam (Context)**: Düğümlerin birbirleriyle olan ilişkilerini tanımlar.
*   **Sorgu (Query)**: Veriler üzerinde yapılan sorgulama işlemidir.

## LlamaIndex Kullanımı

LlamaIndex'i kullanmak için aşağıdaki adımları takip edebilirsiniz:

1.  **LlamaIndex Kurulumu**: LlamaIndex'i kurmak için pip paket yöneticisini kullanabilirsiniz. Aşağıdaki komutu kullanarak LlamaIndex'i kurabilirsiniz:
    ```bash
pip install llama-index
```
2.  **LlamaIndex'i İçe Aktarma**: LlamaIndex'i Python script'inize içe aktarmak için aşağıdaki kodu kullanabilirsiniz:
    ```python
import llama_index
```
    veya
    ```python
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex
```
3.  **Veri Yükleme**: Verileri yüklemek için `SimpleDirectoryReader` sınıfını kullanabilirsiniz. Aşağıdaki kod, bir dizindeki verileri yükler:
    ```python
documents = SimpleDirectoryReader('data').load_data()
```
    Bu kod, 'data' dizinindeki tüm dosyaları yükler ve `documents` değişkenine atar.

4.  **Dizin Oluşturma**: Yüklenen verilerle bir dizin oluşturmak için `GPTVectorStoreIndex` sınıfını kullanabilirsiniz. Aşağıdaki kod, bir dizin oluşturur:
    ```python
index = GPTVectorStoreIndex.from_documents(documents)
```
    Bu kod, `documents` değişkenindeki verilerle bir dizin oluşturur ve `index` değişkenine atar.

5.  **Sorgulama**: Oluşturulan dizin üzerinde sorgulama yapmak için `index.as_query_engine()` metodunu kullanabilirsiniz. Aşağıdaki kod, bir sorgu motoru oluşturur:
    ```python
query_engine = index.as_query_engine()
```
    Ardından, sorgu motorunu kullanarak veriler üzerinde sorgulama yapabilirsiniz:
    ```python
response = query_engine.query("Sorgunuzu buraya yazın")
```
    Bu kod, oluşturulan sorgu motorunu kullanarak "Sorgunuzu buraya yazın" sorgusunu çalıştırır ve sonucu `response` değişkenine atar.

## Örnek Kod

Aşağıdaki örnek kod, LlamaIndex'i kullanarak bir dizin oluşturmayı ve sorgulama yapmayı gösterir:
```python
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex

# Veri yükleme
documents = SimpleDirectoryReader('data').load_data()

# Dizin oluşturma
index = GPTVectorStoreIndex.from_documents(documents)

# Sorgu motoru oluşturma
query_engine = index.as_query_engine()

# Sorgulama yapma
response = query_engine.query("Sorgunuzu buraya yazın")

print(response)
```
Bu kod, 'data' dizinindeki verileri yükler, bir dizin oluşturur, sorgu motorunu oluşturur ve "Sorgunuzu buraya yazın" sorgusunu çalıştırarak sonucu yazdırır.

## LlamaIndex'in Avantajları

LlamaIndex'in avantajları aşağıdaki gibidir:

*   **Kolay Kullanım**: LlamaIndex, basit ve sezgisel bir API sağlar, böylece geliştiriciler kolayca veri indeksleme ve sorgulama işlemlerini gerçekleştirebilirler.
*   **Esneklik**: LlamaIndex, farklı veri kaynakları ve formatlarıyla çalışabilme yeteneğine sahiptir, böylece geliştiriciler çeşitli veri kümeleriyle çalışabilirler.
*   **Performans**: LlamaIndex, optimize edilmiş veri yapıları ve algoritmaları kullanarak yüksek performans sağlar, böylece büyük veri kümeleriyle çalışırken bile hızlı sonuçlar alınabilir.

## Sonuç

LlamaIndex, büyük dil modelleri ile çalışırken kullanılan güçlü bir veri çerçevesidir. Kolay kullanımı, esnekliği ve yüksek performansı ile geliştiricilere verimli ve etkili çözümler oluşturma imkanı sağlar. LlamaIndex'i kullanarak, veri indeksleme ve sorgulama işlemlerini kolayca gerçekleştirebilir ve çeşitli veri kümeleriyle çalışabilirsiniz.

---

## Join our community on Discord

## Discord Topluluğumuza Katılın (Join our community on Discord)

Packt linki üzerinden Discord platformunda yazar ve diğer okuyucular ile tartışmalara katılabilirsiniz. Discord bağlantısı aşağıdaki gibidir:

## Discord Bağlantısı
https://www.packt.link/rag

## Discord'a Katılma Adımları
1. Discord linkine tıklayın: https://www.packt.link/rag
2. Discord'a giriş yapın veya kayıt olun.
3. Kanallara katılın ve tartışmalara başlayın.

## Kod Örneği Yok
Bu metinde herhangi bir kod örneği bulunmamaktadır.

## Açıklama
Discord, geliştiricilerin ve okuyucuların bir araya gelerek tartışmalar yapmalarını sağlayan bir platformdur. Packt linki üzerinden sağlanan Discord bağlantısı, kullanıcıların yazar ve diğer okuyucular ile iletişime geçmelerini sağlar.

## Teknik Terimler
- Discord: Geliştiriciler ve topluluklar için bir iletişim platformu (Developer and community communication platform).
- Packt: Teknik kitaplar ve kaynaklar sağlayan bir yayıncı (A publisher providing technical books and resources).

## Önemli Noktalar
- Discord'a katılmak için linke tıklayın.
- Giriş yapın veya kayıt olun.
- Tartışmalara katılın.

## Ek Bilgiler
Discord, özellikle geliştirici toplulukları ve teknik konularda tartışmalar için popüler bir platformdur. Packt'ın sağladığı Discord bağlantısı, okuyucuların ve yazarın bir araya gelerek konuları tartışmalarını sağlar.

---

