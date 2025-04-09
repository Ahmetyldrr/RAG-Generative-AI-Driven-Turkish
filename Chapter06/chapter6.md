## Scaling RAG Bank Customer Data with Pinecone

## Pinecone ile RAG Banka Müşteri Verilerini Ölçeklendirme (Scaling RAG Bank Customer Data with Pinecone)

RAG (Retrieval-Augmented Generation) belgelerini ölçeklendirmek, ister metin tabanlı ister çok modlu olsun, sadece daha fazla veri biriktirmekle ilgili değildir; temel olarak bir uygulamanın nasıl çalıştığını değiştirir. Ölçeklendirme, doğru miktarda veri bulmakla ilgilidir, sadece daha fazlasını değil. İkinci olarak, daha fazla veri ekledikçe, bir uygulamanın talepleri değişebilir - daha büyük yükü işleyebilmek için yeni özelliklere ihtiyaç duyabilir. Son olarak, maliyet izleme ve hız performansı, ölçeklendirme sırasında projelerimizi kısıtlayacaktır. Bu nedenle, bu bölüm, projelerinizde karşılaşabileceğiniz gerçek dünya ölçeklendirme zorluklarını çözmede AI'yı kullanma konusunda size en son teknikleri öğretmek üzere tasarlanmıştır.

## Önemli Noktalar
* RAG vektör depolarını ölçeklendirmenin anahtar yönleri (key aspects of scaling RAG vector stores)
* Veri hazırlama için EDA (Exploratory Data Analysis)
* Pinecone vektör depolama ile ölçeklendirme (Scaling with Pinecone vector storage)
* Müşteri banka bilgileri için chunking stratejisi (Chunking strategy for customer bank information)
* OpenAI embedding modelleri ile veri embedding (Embedding data with OpenAI embedding models)
* Verileri upsert etmek (Upserting data)
* Pinecone'u RAG için kullanmak (Using Pinecone for RAG)
* GPT-4o ile üretken AI tabanlı öneriler oluşturmak (Generative AI-driven recommendations with GPT-4o)

## İşlem Adımları
1. Kaggle banka müşteri veri setini indirin ve EDA (Exploratory Data Analysis) yapın.
2. Pinecone ve OpenAI'ın text-embedding-3-small modelini kullanarak verileri chunk ve embed edin, ardından bir Pinecone indexine upsert edin.
3. Pinecone kullanarak RAG sorguları oluşturun, kullanıcı girdisini artırın ve GPT-4o ile AI tabanlı öneriler oluşturun.

## Kod Örneği
```python
import pandas as pd
import numpy as np
from pinecone import Pinecone
from openai import OpenAI

# Kaggle veri setini indirin
df = pd.read_csv('bank_customer_data.csv')

# EDA yapın
print(df.head())
print(df.info())
print(df.describe())

# Pinecone'u başlatın
pc = Pinecone(api_key='YOUR_API_KEY')

# OpenAI'ın text-embedding-3-small modelini kullanarak verileri embed edin
client = OpenAI(api_key='YOUR_API_KEY')
embeddings = []
for text in df['text']:
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    embeddings.append(response.data[0].embedding)

# Verileri chunk edin ve Pinecone indexine upsert edin
index = pc.Index('bank_customer_data')
for i, embedding in enumerate(embeddings):
    index.upsert([(i, embedding)])

# Pinecone kullanarak RAG sorguları oluşturun
query = 'müşteri kaybını azaltmak için öneriler'
query_embedding = client.embeddings.create(
    input=query,
    model='text-embedding-3-small'
).data[0].embedding
results = index.query(
    vector=query_embedding,
    top_k=5
)

# GPT-4o ile AI tabanlı öneriler oluşturun
response = client.chat.completions.create(
    model='gpt-4o',
    messages=[
        {'role': 'user', 'content': query},
        {'role': 'assistant', 'content': 'Öneriler:' + str(results)}
    ]
)
print(response.choices[0].message.content)
```
## Kod Açıklaması
* `import` ifadeleri, gerekli kütüphaneleri içe aktarır.
* `pd.read_csv` ifadesi, Kaggle veri setini indirir.
* `Pinecone` ve `OpenAI` sınıfları, sırasıyla Pinecone ve OpenAI API'larını başlatmak için kullanılır.
* `client.embeddings.create` ifadesi, OpenAI'ın text-embedding-3-small modelini kullanarak verileri embed eder.
* `index.upsert` ifadesi, verileri Pinecone indexine upsert eder.
* `index.query` ifadesi, Pinecone kullanarak RAG sorguları oluşturur.
* `client.chat.completions.create` ifadesi, GPT-4o ile AI tabanlı öneriler oluşturur.

## Sonuç
Bu bölüm, RAG belgelerini ölçeklendirmek için Pinecone ve OpenAI teknolojilerini kullanma konusunda size bilgi vermiştir. Kaggle banka müşteri veri setini indirerek, EDA yaparak, verileri chunk ve embed ederek, Pinecone indexine upsert ederek, RAG sorguları oluşturarak ve GPT-4o ile AI tabanlı öneriler oluşturarak, banka müşteri kaybını azaltmak için öneriler oluşturabilirsiniz.

---

## Scaling with Pinecone

## Pinecone ile Ölçeklendirme (Scaling with Pinecone)

Bu bölümde, Pinecone'un yenilikçi vektör veritabanı teknolojisini (vector database technology) OpenAI'ın güçlü embedding capabilities ile birleştirerek veri işleme ve sorgulama sistemleri oluşturacağız. Amaç, müşterileri bir bankayla olan ilişkilerini sürdürmeye teşvik etmek için bir öneri sistemi (recommendation system) oluşturmaktır. Bu yaklaşımı anladıktan sonra, öneriler gerektiren herhangi bir alanda (eğlence, tıbbi veya hukuki) uygulayabileceksiniz.

## Öneri Sistemi Oluşturma

Öneri sistemi oluşturmak için, Pinecone vektör veritabanı ve OpenAI LLM modelini kullanacağız. Bir projenin özel hedeflerine bağlı olarak, bir mimari seçmek ve tasarlamak önemlidir. Projenizin ihtiyaçlarına bağlı olarak, bu metodolojiyi diğer platformlara uygulayabilirsiniz.

## Mimarinin Seçilmesi ve Tasarlanması

Mimarinin seçilmesi ve tasarlanması, bir projenin özel hedeflerine bağlıdır. Bu bölümde ve mimaride, bir vektör deposu (vector store) ve üretken bir yapay zeka modeli (generative AI model) birleştirilerek işlemleri kolaylaştırmak ve ölçeklenebilirliği sağlamak için tasarlanmıştır.

## Python'da Mimarinin Uygulanması

Python'da mimariyi uygulamak için aşağıdaki kodları kullanacağız:
```python
import pinecone
from pinecone import Pinecone, Index
import openai

# Pinecone API anahtarını ayarlayın
pinecone.init(api_key='YOUR_API_KEY', environment='us-west1-gcp')

# OpenAI API anahtarını ayarlayın
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Vektör deposu oluşturun
index_name = 'banking-recommendations'
pc = Pinecone(api_key='YOUR_API_KEY')
index = pc.Index(index_name)

# Vektörleri oluşturun
# Örneğin, müşteri verilerini embedding yapın
customer_data = [...]
embeddings = openai.Embedding.create(input=customer_data, engine='text-embedding-ada-002')

# Vektörleri vektör deposuna ekleyin
index.upsert([(id, embedding) for id, embedding in enumerate(embeddings)])

# Sorgulama yapın
query = 'müşteri önerileri'
query_embedding = openai.Embedding.create(input=query, engine='text-embedding-ada-002')
results = index.query(vectors=query_embedding, top_k=5)

# Sonuçları işleyin
for result in results:
    print(result)
```
## Kod Açıklaması

*   `pinecone.init()`: Pinecone API anahtarını ve ortamını ayarlar.
*   `openai.api_key`: OpenAI API anahtarını ayarlar.
*   `Pinecone()`: Pinecone vektör deposunu oluşturur.
*   `Index()`: Vektör deposunda bir indeks oluşturur.
*   `openai.Embedding.create()`: Müşteri verilerini embedding yapar.
*   `index.upsert()`: Vektörleri vektör deposuna ekler.
*   `index.query()`: Sorgulama yapar ve en yakın vektörleri döndürür.

## Önemli Noktalar

*   Pinecone vektör veritabanı ve OpenAI LLM modelinin birleştirilmesi, öneri sistemleri oluşturmak için güçlü bir araçtır.
*   Mimarinin seçilmesi ve tasarlanması, bir projenin özel hedeflerine bağlıdır.
*   Vektör deposu ve üretken yapay zeka modeli birleştirilerek işlemleri kolaylaştırmak ve ölçeklenebilirliği sağlamak mümkündür.
*   Python'da mimariyi uygulamak için Pinecone ve OpenAI kütüphanelerini kullanmak gerekir.

## Avantajları

*   Ölçeklenebilirlik (Scalability)
*   Yüksek performans (High Performance)
*   Kolay kullanım (Easy to Use)

## Kullanım Alanları

*   Öneri sistemleri (Recommendation Systems)
*   Doğal dil işleme (Natural Language Processing)
*   Bilgisayarlı görü (Computer Vision)

---

## Architecture

## Mimari (Architecture)
Bu bölümde, vektör tabanlı benzerlik arama (vector-based similarity search) işlevselliğini uygulayacağız. Bu işlevselliği, 2. ve 3. bölümlerde yaptığımız gibi, öneri sistemimize (recommendation system) uygulayacağız.

## Önemli Noktalar
*   **Pipeline 1: Veri Toplama ve Hazırlama (Collecting and Preparing the Dataset)**: Bu pipeline'da, standart sorgular ve k-means kümeleme (k-means clustering) ile veri seti üzerinde EDA (Exploratory Data Analysis) yapacağız.
*   **Pipeline 2: Pinecone İndeksini Ölçeklendirme (Scaling a Pinecone Index)**: Bu pipeline'da, 1.000.000'dan fazla belgeyi (document) parçalayarak (chunking), gömerek (embedding) ve Pinecone indeksine (vector store) yükleyeceğiz.
*   **Pipeline 3: RAG Üretken Yapay Zeka (RAG Generative AI)**: Bu pipeline, 1.000.000'dan fazla vektör deposunu (vector store) sorguladığımızda ve bir GPT-4o modelinin girdisini artırarak (augment) hedefli öneriler (targeted recommendations) yaptığımızda tam olarak ölçeklenmiş RAG'a ulaşmamızı sağlayacaktır.

## Uygulama Alanları
*   **Ölçeklenebilir ve Sunucusuz Altyapı (Scalable and Serverless Infrastructure)**: Pinecone'un sunucusuz mimarisini (serverless architecture) anlayarak, sunucu yönetimi ve ölçeklendirme karmaşıklıklarından kurtulacağız.
*   **Hafif ve Basitleştirilmiş Geliştirme Ortamı (Lightweight and Simplified Development Environment)**: Entegrasyon stratejimiz, harici kütüphanelerin kullanımını en aza indirerek, hafif bir geliştirme yığını (development stack) oluşturacaktır.
*   **Optimize Edilmiş Ölçeklenebilirlik ve Performans (Optimized Scalability and Performance)**: Pinecone'un vektör veritabanı (vector database), büyük ölçekli veri kümelerini (large-scale datasets) etkili bir şekilde işleyecek şekilde tasarlanmıştır.

## Kod Örneği
Pinecone ve OpenAI entegrasyonu için aşağıdaki kod örneğini kullanabilirsiniz:
```python
import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai

# Pinecone API anahtarını ayarlayın
pinecone_api_key = os.environ.get('PINECONE_API_KEY')

# OpenAI API anahtarını ayarlayın
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Pinecone'u başlatın
pc = Pinecone(api_key=pinecone_api_key)

# Pinecone indeksini oluşturun
index_name = 'my-index'
pc.create_index(
    name=index_name,
    dimension=1536,  # OpenAI embedding boyutu
    metric='cosine',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

# OpenAI embedding modelini ayarlayın
embedding_model = 'text-embedding-ada-002'

# Belgeyi parçalayarak gömün ve Pinecone indeksine yükleyin
def embed_and_upsert(document):
    # OpenAI embedding API'sini kullanarak belgeyi gömün
    response = openai.Embedding.create(
        input=document,
        model=embedding_model
    )
    embedding = response['data'][0]['embedding']

    # Pinecone indeksine belgeyi yükleyin
    index = pc.Index(index_name)
    index.upsert([(document, embedding)])

# Örnek belgeyi yükleyin
document = 'Bu bir örnek belgedir.'
embed_and_upsert(document)
```
## Kod Açıklaması
Yukarıdaki kod örneğinde, Pinecone ve OpenAI entegrasyonu için gerekli adımları gösterdik.

1.  Pinecone API anahtarını ve OpenAI API anahtarını ayarlayın.
2.  Pinecone'u başlatın ve Pinecone indeksini oluşturun.
3.  OpenAI embedding modelini ayarlayın.
4.  Belgeyi parçalayarak gömün ve Pinecone indeksine yükleyin.

Bu kod örneği, Pinecone ve OpenAI entegrasyonunun temel adımlarını göstermektedir. Gerçek dünya uygulamalarında, bu kodu projenizin gereksinimlerine göre uyarlamanız gerekebilir.

## Kullanım
Pinecone ve OpenAI entegrasyonunu kullanarak, öneri sisteminizi ölçeklendirebilir ve geliştirilmiş performans elde edebilirsiniz. Bu entegrasyon, büyük ölçekli veri kümelerini işleyerek, hedefli öneriler yapmanıza olanak tanır.

---

## Pipeline 1: Collecting and preparing the dataset

## Pipeline 1: Veri Setinin Toplanması ve Hazırlanması
Bu bölüm, Banka Müşteri Ayrılığı (Bank Customer Churn) veri setinin işlenmesi ve analizi üzerine odaklanacaktır. Çevreyi kurma, veri işleme ve makine öğrenimi (Machine Learning, ML) tekniklerini uygulama adımlarında size yol göstereceğiz. Algoritmaları araç olarak kullanmadan önce bir veri setinin "hissini" insan analizi ile anlamak önemlidir. İnsan yaratıcılığının esnekliği nedeniyle insan içgörüleri her zaman kritik olacaktır. Bu nedenle, veri toplama ve hazırlama işlemlerini Python'da üç ana adımda gerçekleştireceğiz:

## Veri Setinin Toplanması ve İşlenmesi
- Kaggle ortamını kurma ve doğrulama yapmak için Kaggle API'sini kullanma
- Banka Müşteri Ayrılığı veri setini indirme ve açma
- Gereksiz sütunları kaldırarak veri setini basitleştirme

## Keşifçi Veri Analizi (Exploratory Data Analysis, EDA)
- Veri yapısını ve türünü anlamak için ilk veri incelemelerini yapma
- Müşteri şikayetleri ile müşteri ayrılığı arasındaki ilişkileri inceleme
- Yaş ve maaş seviyelerinin müşteri ayrılığıyla nasıl ilişkili olduğunu keşfetme
- Sayısal özellikler arasındaki korelasyonları görselleştirmek için ısı haritası (heatmap) oluşturma

## Makine Öğrenimi Modelinin Eğitimi
- Makine öğrenimi için verileri hazırlama
- Müşteri davranışlarındaki desenleri keşfetmek için kümeleme (clustering) tekniklerini uygulama
- Farklı küme yapılandırmalarının etkinliğini değerlendirme

## Kaggle API'sini Kullanarak Veri Setini İndirme
Kaggle API'sini kullanmak için öncelikle Kaggle hesabınızda API token'ı oluşturmanız gerekir. Bunu yapmak için:
1. Kaggle hesabınıza giriş yapın.
2. Hesap ayarlarınızdan API bölümüne gidin ve "Create New API Token" butonuna tıklayın.
3. İndirdiğiniz `kaggle.json` dosyasını `~/.kaggle` dizinine taşıyın.

```python
import os
# Kaggle API tokenını doğrula
os.environ['KAGGLE_USERNAME'] = 'kullanici_adi'
os.environ['KAGGLE_KEY'] = 'kaggle_api_anahtari'
```

## Veri Setini İndirme ve Açma
```python
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle API'sini başlat
api = KaggleApi()
api.authenticate()

# Veri setini indir
api.dataset_download_files('veri_seti_adi', path='./data', unzip=True)

# Veri setini oku
df = pd.read_csv('./data/veri_seti.csv')

print(df.head())  # İlk 5 satırı göster
```
Bu kod, Kaggle API'sini kullanarak belirtilen veri setini indirir ve `./data` dizinine kaydeder. `unzip=True` parametresi, indirilen dosyanın otomatik olarak açılmasını sağlar.

## Gereksiz Sütunları Kaldırma
```python
# Gereksiz sütunları kaldır
df = df.drop(['sutun1', 'sutun2'], axis=1)

print(df.columns)  # Kalan sütunları göster
```
Bu kod, belirtilen sütunları (`sutun1` ve `sutun2`) veri setinden kaldırır.

## Keşifçi Veri Analizi
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Sayısal sütunlar arasındaki korelasyonu göster
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', square=True)
plt.show()

# Müşteri şikayetleri ile müşteri ayrılığı arasındaki ilişkiyi incele
print(df['sikayet_var'].value_counts())
print(df['ayrilik'].value_counts())

# Yaş ve maaş seviyelerinin müşteri ayrılığıyla ilişkisini keşfet
sns.boxplot(x='ayrilik', y='yas', data=df)
plt.show()

sns.boxplot(x='ayrilik', y='maas', data=df)
plt.show()
```
Bu kod, sayısal sütunlar arasındaki korelasyonu ısı haritası ile gösterir, müşteri şikayetleri ve müşteri ayrılığı arasındaki ilişkiyi inceler ve yaş ile maaş seviyelerinin müşteri ayrılığıyla ilişkisini boxplot ile görselleştirir.

## Makine Öğrenimi İçin Veri Hazırlama
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Veri setini eğitim ve test setlerine ayır
X = df.drop('hedef_sutun', axis=1)
y = df['hedef_sutun']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verileri standardize et
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
Bu kod, veri setini eğitim ve test setlerine ayırır ve verileri standardize eder.

## Kümeleme Tekniklerini Uygulama
```python
from sklearn.cluster import KMeans

# KMeans kümeleme modelini uygula
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_train)

# Küme merkezlerini göster
print(kmeans.cluster_centers_)
```
Bu kod, KMeans kümeleme modelini veri setine uygular ve küme merkezlerini gösterir.

Bu adımları takip ederek, Banka Müşteri Ayrılığı veri setini toplamış, işlemiş ve makine öğrenimi için hazırlamış olduk.

---

## 1. Collecting and processing the dataset

## Veri Setinin Toplanması ve İşlenmesi

Banka Müşteri Kaybı (Bank Customer Churn) veri setini Kaggle'den toplamak ve işlemek için aşağıdaki adımları izleyeceğiz: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

### Veri Seti İçeriği

Veri seti, bir bankanın 10.000 müşterisine ait kayıtları içermektedir ve müşteri kaybını (churn) etkileyen çeşitli faktörleri incelemektedir. Veri setindeki sütunların açıklamaları aşağıdaki gibidir:

*   **RowNumber**: Kayıt numarasını temsil eder ve çıktı üzerinde etkisi yoktur. (RowNumber corresponds to the record number and has no effect on the output.)
*   **CustomerId**: Rastgele değerler içerir ve müşterilerin bankadan ayrılma kararını etkilemez. (CustomerId contains random values and has no effect on customers leaving the bank.)
*   **Surname**: Müşterinin soyadını temsil eder ve bankadan ayrılma kararını etkilemez. (Surname has no impact on the customer's decision to leave the bank.)
*   **CreditScore**: Müşterinin kredi puanını temsil eder ve müşteri kaybını etkileyebilir. (CreditScore can have an effect on customer churn.)
*   **Geography**: Müşterinin konumunu temsil eder ve bankadan ayrılma kararını etkileyebilir. (Geography can affect a customer's decision to leave the bank.)
*   **Gender**: Müşterinin cinsiyetini temsil eder ve bankadan ayrılma kararını etkileyebilir. (Gender may play a role in a customer leaving the bank.)
*   **Age**: Müşterinin yaşını temsil eder ve bankadan ayrılma kararını etkileyebilir. (Age is relevant since older customers are less likely to leave their bank.)
*   **Tenure**: Müşterinin bankadaki yıllarını temsil eder ve sadakatini etkileyebilir. (Tenure refers to the number of years a customer has been a client of the bank.)
*   **Balance**: Müşterinin hesabındaki bakiyeyi temsil eder ve müşteri kaybını etkileyebilir. (Balance is a good indicator of customer churn.)
*   **NumOfProducts**: Müşterinin bankadan satın aldığı ürün sayısını temsil eder. (NumOfProducts refers to the number of products a customer has purchased through the bank.)
*   **HasCrCard**: Müşterinin kredi kartı olup olmadığını temsil eder ve müşteri kaybını etkileyebilir. (HasCrCard denotes whether a customer has a credit card.)
*   **IsActiveMember**: Müşterinin aktif olup olmadığını temsil eder ve müşteri kaybını etkileyebilir. (IsActiveMember indicates whether a customer is active.)
*   **EstimatedSalary**: Müşterinin tahmini maaşını temsil eder ve müşteri kaybını etkileyebilir. (EstimatedSalary is related to customer churn.)
*   **Exited**: Müşterinin bankadan ayrılıp ayrılmadığını temsil eder. (Exited indicates whether a customer left the bank.)
*   **Complain**: Müşterinin şikayette bulunup bulunmadığını temsil eder. (Complain indicates whether a customer has complained.)
*   **Satisfaction Score**: Müşterinin şikayet çözümünden memnuniyet derecesini temsil eder. (Satisfaction Score is the score provided by the customer for their complaint resolution.)
*   **Card Type**: Müşterinin sahip olduğu kart türünü temsil eder. (Card Type is the type of card held by the customer.)
*   **Points Earned**: Müşterinin kredi kartı kullanımından kazandığı puanları temsil eder. (Points Earned are the points earned by the customer for using a credit card.)

### Veri Setinin Toplanması ve İşlenmesi

Veri setini toplamak ve işlemek için aşağıdaki Python kodunu kullanacağız:

```python
## Import gerekli kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Veri setini yükleme
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print("Veri seti yüklenirken hata oluştu:", str(e))

## Veri setini yükleme
file_path = 'Customer-Churn-Records.csv'  # Dosya yolu
data = load_dataset(file_path)

## Veri setinin ilk 5 satırını gösterme
print(data.head())

## Veri setinin son 5 satırını gösterme
print(data.tail())

## Veri setinin boyutlarını gösterme
print(data.shape)

## Veri setinin sütunlarını gösterme
print(data.columns)

## Veri setinin özetini gösterme
print(data.info())

## Veri setinin betimsel istatistiklerini gösterme
print(data.describe())
```

Bu kod, `Customer-Churn-Records.csv` dosyasını yükler ve ilk 5 satırını, son 5 satırını, boyutlarını, sütunlarını, özetini ve betimsel istatistiklerini gösterir.

### Kod Açıklamaları

*   `import pandas as pd`: Pandas kütüphanesini içe aktarır ve `pd` takma adını verir.
*   `import numpy as np`: NumPy kütüphanesini içe aktarır ve `np` takma adını verir.
*   `import matplotlib.pyplot as plt`: Matplotlib kütüphanesini içe aktarır ve `plt` takma adını verir.
*   `import seaborn as sns`: Seaborn kütüphanesini içe aktarır ve `sns` takma adını verir.
*   `load_dataset(file_path)`: Belirtilen dosya yolundan veri setini yükler.
*   `pd.read_csv(file_path)`: CSV dosyasını okur ve bir Pandas DataFrame nesnesine dönüştürür.
*   `data.head()`: Veri setinin ilk 5 satırını gösterir.
*   `data.tail()`: Veri setinin son 5 satırını gösterir.
*   `data.shape`: Veri setinin boyutlarını (satır ve sütun sayısı) gösterir.
*   `data.columns`: Veri setinin sütunlarını gösterir.
*   `data.info()`: Veri setinin özetini gösterir (sütun isimleri, veri tipleri ve boş değer sayıları).
*   `data.describe()`: Veri setinin betimsel istatistiklerini gösterir (ortalama, standart sapma, minimum, maksimum vb.).

---

## Installing the environment for Kaggle

## Kaggle Ortamını Kurma (Installing the Environment for Kaggle)

Kaggle'den veri setlerini otomatik olarak toplamak için, https://www.kaggle.com/ adresinden kayıt olmanız ve bir API anahtarı (API Key) oluşturmanız gerekir. Bu notebook'un yazıldığı sırada, veri setlerini indirmek ücretsizdi. Kaggle API anahtarınızı saklamak ve kullanmak için talimatları takip edin. Anahtarınızı güvenli bir konumda saklayın.

### API Anahtarını Saklama (Storing the API Key)

Bu örnekte, anahtar Google Drive'da bir dosyada saklanmaktadır ve bu dosyaya erişmek için Google Drive'ı bağlamamız (mount) gerekir.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Bu kod, Google Colab'ı Google Drive'a bağlamak için kullanılır. `/content/drive` dizinine bağlanır.

### API Anahtarını Okuma ve Ortam Değişkenlerini Ayarlama (Reading the API Key and Setting Environment Variables)

Şimdi, JSON dosyasını okuyarak Kaggle kimlik doğrulaması (authentication) için ortam değişkenlerini ayarlayacağız.

```python
import os
import json

with open(os.path.expanduser("drive/MyDrive/files/kaggle.json"), "r") as f:
    kaggle_credentials = json.load(f)

kaggle_username = kaggle_credentials["username"]
kaggle_key = kaggle_credentials["key"]

os.environ["KAGGLE_USERNAME"] = kaggle_username
os.environ["KAGGLE_KEY"] = kaggle_key
```

Bu kod, `kaggle.json` dosyasını okur ve içindeki `username` ve `key` değerlerini alır. Daha sonra, bu değerleri `KAGGLE_USERNAME` ve `KAGGLE_KEY` ortam değişkenlerine atar.

### Kaggle'ı Kurma ve Kimlik Doğrulama (Installing Kaggle and Authenticating)

Kaggle'ı kurmak ve kimlik doğrulaması yapmak için aşağıdaki kodları kullanın.

```python
try:
    import kaggle
except:
    !pip install kaggle
import kaggle
kaggle.api.authenticate()
```

Bu kod, önce Kaggle kütüphanesini içe aktarmaya (import) çalışır. Eğer kütüphane yüklü değilse, `pip install kaggle` komutunu kullanarak kurar. Daha sonra, `kaggle.api.authenticate()` fonksiyonunu çağırarak kimlik doğrulamasını gerçekleştirir.

Artık Bank Customer Churn veri setini toplamak için hazırız.

---

## Collecting the dataset

## Veri Kümesinin Toplanması (Collecting the Dataset)
Veri kümesini toplamak için ilk adım, sıkıştırılmış veri kümesini indirmek, CSV dosyasını çıkarmak, bir pandas DataFrame'e yüklemek, kullanılmayacak sütunları silmek ve sonucu görüntülemektir.

### Adım 1: Sıkıştırılmış Veri Kümesini İndirme (Downloading the Zipped Dataset)
İlk olarak, sıkıştırılmış veri kümesini indirmek için aşağıdaki komutu kullanıyoruz:
```python
!kaggle datasets download -d radheshyamkollipara/bank-customer-churn
```
Bu komut, Kaggle'dan "bank-customer-churn" veri kümesini indirir.

### Adım 2: Sıkıştırılmış Dosyayı Çıkarma (Unzipping the Data)
Sıkıştırılmış dosyayı çıkarmak için `zipfile` kütüphanesini kullanıyoruz:
```python
import zipfile
with zipfile.ZipFile('/content/bank-customer-churn.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')
print("File Unzipped!")
```
Bu kod, `/content/` dizinine sıkıştırılmış dosyayı çıkarır ve "File Unzipped!" mesajını yazdırır.

### Adım 3: CSV Dosyasını pandas DataFrame'e Yükleme (Loading the CSV File into a pandas DataFrame)
CSV dosyasını pandas DataFrame'e yüklemek için `pandas` kütüphanesini kullanıyoruz:
```python
import pandas as pd
file_path = '/content/Customer-Churn-Records.csv'
data1 = pd.read_csv(file_path)
```
Bu kod, `/content/Customer-Churn-Records.csv` dosyasını `data1` adında bir pandas DataFrame'e yükler.

### Adım 4: Kullanılmayacak Sütunları Silme (Dropping Unnecessary Columns)
Kullanılmayacak sütunları silmek için `drop()` fonksiyonunu kullanıyoruz:
```python
data1.drop(columns=['RowNumber', 'Surname', 'Gender', 'Geography'], inplace=True)
```
Bu kod, `RowNumber`, `Surname`, `Gender` ve `Geography` sütunlarını siler ve DataFrame'i günceller.

### Adım 5: Sonuçları Görüntüleme (Displaying the Result)
Sonuçları görüntülemek için `data1` DataFrame'ini yazdırıyoruz:
```python
data1
```
Bu, güncellenmiş DataFrame'i görüntüler.

### Adım 6: DataFrame'i Kaydetme (Saving the DataFrame)
Güncellenmiş DataFrame'i kaydetmek için `to_csv()` fonksiyonunu kullanıyoruz:
```python
data1.to_csv('data1.csv', index=False)
!cp /content/data1.csv /content/drive/MyDrive/files/rag_c6/data1.csv
```
Bu kod, `data1` DataFrame'ini `data1.csv` dosyasına kaydeder ve daha sonra bu dosyayı Google Drive'a kopyalar.

## Önemli Noktalar (Important Points)
* Veri kümesini toplamak için Kaggle'dan "bank-customer-churn" veri kümesini indirdik.
* Sıkıştırılmış dosyayı `/content/` dizinine çıkardık.
* CSV dosyasını `data1` adında bir pandas DataFrame'e yükledik.
* Kullanılmayacak sütunları (`RowNumber`, `Surname`, `Gender` ve `Geography`) sildik.
* Güncellenmiş DataFrame'i `data1.csv` dosyasına kaydettik ve Google Drive'a kopyaladık.

## Avantajları (Advantages)
Bu yaklaşımın avantajı, Pinecone index (vector store) eklenecek verinin boyutunu optimize etmesidir. Gereksiz alanları kaldırarak veri kümesini küçültmek, sorgu performansını artırabilir ve maliyetleri azaltabilir.

---

## 2. Exploratory data analysis

## 2. Keşifçi Veri Analizi (Exploratory Data Analysis)

Bu bölümde, pandas kütüphanesinin tanımladığı ve bir bankaya ait müşteri verilerini içeren veri setini kullanarak Keşifçi Veri Analizi (EDA) gerçekleştireceğiz. EDA, veri içindeki temel desenleri ve eğilimleri anlamamıza yardımcı olduğu için, vektör depolarıyla herhangi bir RAG tekniği uygulamadan önce kritik bir adımdır.

### Önemli Noktalar:

* Müşteri şikayetleri (Complain) ile bankadan ayrılma (Exited) arasında doğrudan bir ilişki vardır.
* 50 yaş ve üzeri müşteriler, daha genç müşterilere göre bankadan ayrılma olasılıkları daha düşüktür.
* Gelir düzeyleri (özellikle $100,000 eşiği), bankadan ayrılma kararlarını önemli ölçüde etkilememektedir.

### Veri Setinin İncelenmesi

Veri setinin sütunlarını görüntülediğimizde, desenleri bulmanın zor olduğunu görürüz:
```python
import pandas as pd

# Veri setini yükleme
data1 = pd.read_csv('veri_seti.csv')

# Sütunları görüntüleme
print(data1.columns)
print(data1.head())
```
 Output:
```
 #   Column              Non-Null Count  Dtype
---  ------              --------------  -----
 0   CustomerId          10000 non-null  int64
 1   CreditScore         10000 non-null  int64
 2   Age                 10000 non-null  int64
 3   Tenure              10000 non-null  int64
 4   Balance             10000 non-null  float64
 5   NumOfProducts       10000 non-null  int64
 6   HasCrCard           10000 non-null  int64
 7   IsActiveMember      10000 non-null  int64
 8   EstimatedSalary     10000 non-null  float64
 9   Exited              10000 non-null  int64
 10  Complain            10000 non-null  int64
 11  Satisfaction Score  10000 non-null  int64
 12  Card Type           10000 non-null  object
 13  Point Earned        10000 non-null  int64
```
Yaş (Age), Tahmini Maaş (EstimatedSalary) ve Şikayet (Complain) gibi özellikler, bankadan ayrılma (Exited) ile ilişkili olabilecek belirleyici faktörlerdir.

### Şikayet ve Bankadan Ayrılma Arasındaki İlişki

Şikayet eden müşterilerin bankadan ayrılma olasılıklarının yüksek olduğunu görürüz:
```python
sum_exited = data1['Exited'].sum()
sum_complain = data1['Complain'].sum()

if sum_exited > 0:
    percentage_complain_over_exited = (sum_complain / sum_exited) * 100
else:
    percentage_complain_over_exited = 0

print(f"Sum of Exited = {sum_exited}")
print(f"Sum of Complain = {sum_complain}")
print(f"Percentage of complain over exited = {percentage_complain_over_exited:.2f}%")
```
 Output:
```
Sum of Exited = 2038
Sum of Complain = 2044
Percentage of complain over exited = 100.29%
```
Bu sonuç, şikayet eden müşterilerin büyük çoğunluğunun bankadan ayrıldığını gösterir.

### Isı Haritası (Heatmap) Oluşturma

Veri setindeki sayısal sütunlar arasındaki korelasyonu görselleştirmek için ısı haritası oluştururuz:
```python
import seaborn as sns
import matplotlib.pyplot as plt

numerical_columns = data1.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(12, 8))
sns.heatmap(data1[numerical_columns].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```
Bu ısı haritası, Şikayet (Complain) ve Bankadan Ayrılma (Exited) arasındaki yüksek korelasyonu gösterir.

### Sonuç

Keşifçi Veri Analizi (EDA) sayesinde, veri setindeki temel desenleri ve eğilimleri anlamamıza yardımcı oldu. Bu analiz, basit istatistiksel yöntemlerin ve temel veri analizi tekniklerinin, bazı durumlarda daha karmaşık makine öğrenimi modellerinden daha etkili olabileceğini gösterdi.

---

## 3. Training an ML model

## Makine Öğrenimi (ML) Modeli Eğitimi
Makine öğrenimi modeli eğitimi için kümeleme (clustering) tekniklerini, özellikle k-ortalamalar (k-means) kümelemeyi kullanarak veri setimizi daha ayrıntılı inceleyelim. Bu bölümde, veri analizi için verileri hazırlayacağız, kümeleme uygulayacağız ve sonuçları farklı metriklerle değerlendireceğiz. Bu yaklaşım, daha karmaşık derin öğrenme (deep learning) yöntemlerine başvurmadan önce içgörüler elde etmek için değerlidir.

## K-Ortalamalar (K-Means) Kümeleme Algoritması
K-ortalamalar kümeleme, bir veri setini k farklı, örtüşmeyen kümeye ayıran denetimsiz (unsupervised) bir makine öğrenimi algoritmasıdır. Algoritma, her bir kümedeki varyansı minimize ederek veri noktalarını en yakın ortalama (centroid) değerine göre k kümelerden birine atar ve her iterasyondan sonra ortalamayı yeniden hesaplar. Bu işlem, yakınsama (convergence) sağlanana kadar devam eder.

### K-Ortalamalar Kümeleme Uygulaması
K-ortalamalar kümeleme algoritmasını uygulamak için aşağıdaki adımları takip edeceğiz:
*   Veri setini hazırlama ve işleme
*   K-ortalamalar kümeleme algoritmasını uygulama
*   Sonuçları değerlendirme

### Kod Uygulaması
```python
# Import gerekli kütüphaneler
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Veri setini yükleme
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")

# Veri setini hazırlama
def prepare_data(data):
    try:
        # Gereksiz sütunları kaldırma
        data.drop(['column1', 'column2'], axis=1, inplace=True)
        
        # Eksik değerleri işleme
        data.fillna(data.mean(), inplace=True)
        
        # Ölçeklendirme (scaling)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        return scaled_data
    except Exception as e:
        print(f"Veri hazırlanırken hata oluştu: {e}")

# K-ortalamalar kümeleme algoritmasını uygulama
def apply_kmeans(data, k):
    try:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        labels = kmeans.labels_
        return labels
    except Exception as e:
        print(f"K-ortalamalar kümeleme uygulanırken hata oluştu: {e}")

# Sonuçları değerlendirme
def evaluate_results(data, labels):
    try:
        # Silhouette skoru (silhouette score)
        silhouette = silhouette_score(data, labels)
        print(f"Silhouette skoru: {silhouette}")
        
        # Calinski-Harabasz indeksi (calinski-harabasz index)
        calinski_harabasz = calinski_harabasz_score(data, labels)
        print(f"Calinski-Harabasz indeksi: {calinski_harabasz}")
        
        # Davies-Bouldin indeksi (davies-bouldin index)
        davies_bouldin = davies_bouldin_score(data, labels)
        print(f"Davies-Bouldin indeksi: {davies_bouldin}")
    except Exception as e:
        print(f"Sonuçlar değerlendirilirken hata oluştu: {e}")

# Ana fonksiyon
def main():
    file_path = 'data.csv'  # Veri seti dosya yolu
    data = load_data(file_path)
    prepared_data = prepare_data(data)
    k = 5  # Küme sayısı
    labels = apply_kmeans(prepared_data, k)
    evaluate_results(prepared_data, labels)

if __name__ == "__main__":
    main()
```

### Kod Açıklaması

*   `load_data` fonksiyonu, belirtilen dosya yolundan veri setini yükler.
*   `prepare_data` fonksiyonu, veri setini hazırlar. Gereksiz sütunları kaldırır, eksik değerleri işler ve ölçeklendirme (scaling) uygular.
*   `apply_kmeans` fonksiyonu, k-ortalamalar kümeleme algoritmasını uygular ve küme etiketlerini döndürür.
*   `evaluate_results` fonksiyonu, kümeleme sonuçlarını değerlendirir. Silhouette skoru, Calinski-Harabasz indeksi ve Davies-Bouldin indeksi gibi metrikleri hesaplar.
*   `main` fonksiyonu, ana program akışını kontrol eder. Veri setini yükler, hazırlar, k-ortalamalar kümeleme algoritmasını uygular ve sonuçları değerlendirir.

### Önemli Noktalar

*   K-ortalamalar kümeleme algoritması, denetimsiz (unsupervised) bir makine öğrenimi algoritmasıdır.
*   Algoritma, veri setini k farklı kümeye ayırır.
*   Kümeleme sonuçları, farklı metriklerle değerlendirilir.
*   Veri seti hazırlanırken, gereksiz sütunlar kaldırılır, eksik değerler işlenir ve ölçeklendirme uygulanır.

---

## Data preparation and clustering

## Veri Hazırlama ve Kümeleme (Data Preparation and Clustering)
Veri hazırlama ve kümeleme, makine öğrenimi (Machine Learning) projelerinde önemli adımlardır. Bu süreçte, veri seti üzerinde çeşitli işlemler yaparak verileri modele hazır hale getiririz.

## Veri Kopyalama (Copying Data)
İlk olarak, veri setimizi `data1`'den `data2`'ye kopyalayarak yedekliyoruz. Bu sayede, farklı makine öğrenimi modellerini denemek istediğimizde orijinal veri setine geri dönebiliriz.
```python
data2 = data1.copy()
```
Bu kod, `data1` veri setini `data2`'ye kopyalar.

## Özellik Seçimi (Feature Selection)
Veri setimizde bulunan özelliklerden (`CreditScore`, `Age`, `EstimatedSalary`, `Exited`, `Complain`, `Point Earned`) önemli olanları seçiyoruz.
```python
features = data2[['CreditScore', 'Age', 'EstimatedSalary', 'Exited', 'Complain', 'Point Earned']]
```
Bu kod, `data2` veri setinden seçilen özellikleri `features` değişkenine atar.

## Özellik Ölçekleme (Feature Scaling)
Makine öğrenimi modellerini çalıştırmadan önce, özellikleri ölçeklendirmek iyi bir uygulamadır. Bu amaçla `StandardScaler` kullanıyoruz.
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
```
Bu kod, `features` değişkenindeki özellikleri ölçeklendirir ve `features_scaled` değişkenine atar. `StandardScaler`, her bir özelliğin ortalamasını 0 ve varyansını 1'e getirir.

## Kümeleme (Clustering)
Kümeleme analizi için `KMeans` algoritmasını kullanıyoruz. Farklı sayıda küme (2, 3, 4) için denemeler yapıyoruz.
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

for n_clusters in range(2, 5):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(features_scaled)
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    db_index = davies_bouldin_score(features_scaled, cluster_labels)
    print(f'For n_clusters={n_clusters}, the silhouette score is {silhouette_avg:.4f} and the Davies-Bouldin Index is {db_index:.4f}')
```
Bu kod, `features_scaled` veri setini `n_clusters` sayısına göre kümeleyerek, silhouette skoru ve Davies-Bouldin indeksini hesaplar ve yazdırır.

## Değerlendirme Metrikleri (Evaluation Metrics)
Kümeleme performansını değerlendirmek için iki metrik kullanıyoruz: Silhouette Skoru ve Davies-Bouldin İndeksi.

*   **Silhouette Skoru**: Bu metrik, küme içi mesafelerin (intra-cluster distance) ve en yakın küme mesafelerinin (nearest cluster distance) ortalamasını hesaplar. Skor -1 ile 1 arasında değişir. Yüksek skor, iyi ayrılmış ve içsel olarak tutarlı kümeler olduğunu gösterir.
*   **Davies-Bouldin İndeksi**: Bu indeks, küme içi mesafelerin küme arası mesafelere oranını değerlendirir. Düşük indeks değerleri, daha iyi kümeleme performansını gösterir.

Bu metrikler, küme sayısına göre değişen kümeleme performansını değerlendirmek için kullanılır.

## Sonuçlar (Results)
Çıktılar, farklı küme sayıları için silhouette skoru ve Davies-Bouldin indeksini içerir. Örneğin:
```
For n_clusters=2, the silhouette score is 0.6129 and the Davies-Bouldin Index is 0.6144
For n_clusters=3, the silhouette score is 0.3391 and the Davies-Bouldin Index is 1.1511
For n_clusters=4, the silhouette score is 0.3243 and the Davies-Bouldin Index is 1.0802
```
Bu sonuçlar, 2 küme için en iyi kümeleme performansını gösterdiğini belirtir. Küme sayısı arttıkça, kümeleme performansı düşer.

---

## Implementation and evaluation of clustering

## Kümeleme (Clustering) Uygulaması ve Değerlendirmesi

Bu bölümde, veri setimiz üzerinde kümeleme (clustering) algoritması olan K-Means'ı uygulayacağız ve sonuçları değerlendireceğiz.

### K-Means Kümeleme Uygulaması

İlk olarak, K-Means algoritmasını kullanarak veri setimizi 2 kümeye ayıracağız. Bunun için `n_clusters` parametresini 2 olarak belirleyeceğiz.

```python
from sklearn.cluster import KMeans
import pandas as pd

# K-Means kümeleme modelini oluştur
kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)

# Ölçeklenmiş özellikleri kullanarak modeli eğit ve tahmin et
data2['class'] = kmeans.fit_predict(features_scaled)

# İlk birkaç satırı göstererek 'class' sütununu doğrulama
print(data2.head())
```

Bu kodda, `KMeans` sınıfını kullanarak bir K-Means modeli oluşturuyoruz. `n_clusters` parametresi ile küme sayısını 2 olarak belirtiyoruz. `n_init` parametresi ile algoritmanın farklı başlangıç noktalarıyla kaç kez çalıştırılacağını belirtiyoruz. `random_state` parametresi ile sonuçların tekrarlanabilirliğini sağlıyoruz. `fit_predict` metodu ile modeli eğitiyor ve veri setimizi kümelere ayırıyoruz.

### Sonuçların Değerlendirilmesi

Kümeleme sonuçlarını değerlendirmek için, `class` sütununa göre bazı istatistikler hesaplayacağız.

```python
# 'class' == 0 olan satırların toplamı
sum_class_0 = (data2['class'] == 0).sum()

# 'class' == 0 ve 'Complain' == 1 olan satırların toplamı
sum_class_0_complain_1 = data2[(data2['class'] == 0) & (data2['Complain'] == 1)].shape[0]

# 'class' == 0 ve 'Exited' == 1 olan satırların toplamı
sum_class_0_exited_1 = data2[(data2['class'] == 0) & (data2['Exited'] == 1)].shape[0]

print(f"Sum of 'class' == 0: {sum_class_0}")
print(f"Sum of 'class' == 0 and 'Complain' == 1: {sum_class_0_complain_1}")
print(f"Sum of 'class' == 0 and 'Exited' == 1: {sum_class_0_exited_1}")
```

Bu kodda, `class` sütununa göre bazı istatistikler hesaplıyoruz. Sonuçlar, şikayet eden müşterilerin çoğunun bankadan ayrıldığını gösteriyor.

### İkinci Kümenin Değerlendirilmesi

İkinci küme için de bazı istatistikler hesaplayacağız.

```python
# 'class' == 1 ve 'Complain' == 1 olan satırların toplamı
sum_class_1_complain_1 = data2[(data2['class'] == 1) & (data2['Complain'] == 1)].shape[0]

print(f"Sum of 'class' == 1 and 'Complain' == 1: {sum_class_1_complain_1}")
```

Bu kodda, ikinci küme için şikayet eden müşterilerin sayısını hesaplıyoruz. Sonuçlar, şikayet eden müşterilerin çoğunun bankada kaldığını gösteriyor.

### Sonuçlar

Kümeleme sonuçları, şikayet eden müşterilerin çoğunun bankadan ayrıldığını gösteriyor. Bu, müşteri kaybını önlemek için şikayetleri ele almanın önemini vurguluyor.

### Önemli Noktalar

* K-Means kümeleme algoritması kullanılarak veri seti 2 kümeye ayrıldı.
* Şikayet eden müşterilerin çoğu bankadan ayrıldı.
* İkinci kümede şikayet eden müşterilerin sayısı azdı.
* Müşteri kaybını önlemek için şikayetleri ele almak önemlidir.

### Gelecek Adımlar

* Müşteri kayıtlarını OpenAI kullanarak vektörlere dönüştürmek ve Pinecone indeksini sorgulayarak daha derin desenler bulmak.

---

## Pipeline 2: Scaling a Pinecone index (vector store)

## Pipeline 2: Pinecone İndeksini Ölçeklendirme (Vektör Deposu)

Bu bölümün amacı, veri setimizi kullanarak bir Pinecone indeksi oluşturmak ve onu 10.000 kayıtтан 1.000.000 kayda kadar ölçeklendirmektir. Önceki bölümlerde edindiğimiz bilgileri temel alarak inşa ediyor olsak da, ölçeklendirmenin özü örnek veri setlerini yönetmekten farklıdır. Bu pipeline'ın her bir sürecinin netliği aldatıcı derecede basittir: veri hazırlama (data preparation), embedding, vektör deposuna yükleme (upserting) ve belgeleri geri almak için sorgulama (querying). Bu süreçlerin her birini daha önce 2. ve 3. bölümlerde ele aldık.

Ayrıca, Deep Lake yerine Pinecone'u uygulamak ve OpenAI modellerini biraz farklı bir şekilde kullanmak dışında, vektör deposu aşaması için 2., 3. ve 4. bölümlerde yaptığımız işlevlerin aynısını gerçekleştiriyoruz:

*   ## Veri Hazırlama (Data Preparation)
    *   Veri setimizi Python kullanarak chunking için hazırlayarak başlayacağız.
*   ## Chunking ve Embedding
    *   Hazırlanan verileri chunking yapacağız ve ardından chunking yapılan verileri embed edeceğiz.
*   ## Pinecone İndeks Oluşturma (Creating the Pinecone Index)
    *   Bir Pinecone indeksi (vektör deposu) oluşturacağız.
*   ## Upserting (Yükleme)
    *   Embedded belgeleri (bu durumda, müşteri kayıtları) ve her kaydın metnini meta veri olarak yükleyeceğiz.
*   ## Pinecone İndeksi Sorgulama (Querying the Pinecone Index)
    *   Son olarak, Pipeline 3: RAG Generative AI'ı hazırlamak için ilgili belgeleri geri almak üzere bir sorgu çalıştıracağız.

Gerekirse, veri hazırlama, chunking, embedding ve sorgulama işlevleri için 2., 3. ve 4. bölümleri tekrar gözden geçirmek için tüm zamanınızı alın.

Her aşamayı nasıl uygulayacağımızı biliyoruz çünkü bunu daha önce Deep Lake ile yaptık ve Pinecone da bir tür vektör deposudur. Peki, buradaki sorun nedir? Gerçek sorun, boyutu, maliyeti ve ilgili operasyonları içeren gizli gerçek hayat proje zorluklarıdır.

### Kodları Uygulama

İlk olarak, gerekli kütüphaneleri içe aktarmak (import) gerekiyor:

```python
import pandas as pd
from pinecone import Pinecone, PodSpec
from sentence_transformers import SentenceTransformer
```

#### Veri Hazırlama

Veri setimizi hazırlamak için Python'u kullanacağız. Örneğin, `customer_support.csv` adlı bir veri setini yükleyelim:

```python
# Veri setini yükleme
df = pd.read_csv('customer_support.csv')

# Veri setinin ilk 5 satırını gösterme
print(df.head())
```

Bu kod, `customer_support.csv` dosyasını yükler ve ilk 5 satırını gösterir.

#### Chunking ve Embedding

Hazırlanan verileri chunking yapacağız ve ardından chunking yapılan verileri embed edeceğiz. Örneğin, ` SentenceTransformer` kütüphanesini kullanarak metinleri embed edebiliriz:

```python
# SentenceTransformer modelini yükleme
model = SentenceTransformer('all-MiniLM-L6-v2')

# Metinleri embedding
def embed_text(text):
    return model.encode(text)

# Veri setindeki metinleri embedding
df['embedding'] = df['text'].apply(embed_text)
```

Bu kod, ` SentenceTransformer` kütüphanesini kullanarak metinleri embed eder.

#### Pinecone İndeks Oluşturma

Bir Pinecone indeksi oluşturmak için aşağıdaki kodu kullanabiliriz:

```python
# Pinecone'u başlatma
pc = Pinecone(api_key='YOUR_API_KEY')

# Pinecone indeksini oluşturma
index_name = 'customer-support-index'
pc.create_index(
    name=index_name,
    dimension=384,  # SentenceTransformer modelinin boyutu
    metric='cosine',
    spec=PodSpec(
        environment='us-west1-gcp',
        pod_type='p1.x1'
    )
)
```

Bu kod, Pinecone'u başlatır ve belirtilen özelliklerde bir indeks oluşturur.

#### Upserting (Yükleme)

Embedded belgeleri Pinecone indeksine yüklemek için aşağıdaki kodu kullanabiliriz:

```python
# Pinecone indeksini bağlama
index = pc.Index(index_name)

# Embedded belgeleri yükleme
for index, row in df.iterrows():
    index.upsert([(str(index), row['embedding'], {'text': row['text']})])
```

Bu kod, embedded belgeleri Pinecone indeksine yükler.

#### Pinecone İndeksi Sorgulama

Pinecone indeksini sorgulamak için aşağıdaki kodu kullanabiliriz:

```python
# Sorgulama yapmak için bir metin belirleme
query_text = 'müşteri destek'

# Sorgulama metnini embedding
query_embedding = model.encode(query_text)

# Pinecone indeksini sorgulama
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

# Sonuçları gösterme
print(results)
```

Bu kod, Pinecone indeksini sorgular ve ilgili belgeleri geri alır.

Bu pipeline, veri setimizi kullanarak bir Pinecone indeksi oluşturmayı ve onu ölçeklendirmeyi amaçlamaktadır. Her bir süreç, veri hazırlama, chunking, embedding, vektör deposuna yükleme ve sorgulama gibi adımları içerir.

---

## The challenges of vector store management

## Vektör Depolama Yönetiminin Zorlukları (The Challenges of Vector Store Management)

Vektör depolama yönetimi, büyük hacimli verilerle çalışırken karşılaşılan zorlukları içerir. Bu zorlukların üstesinden gelmek için proje yönetimi kararları almak gerekir. Küçük hacimli verilerle çalışırken yapılan hatalar sınırlı sonuçlar doğurur, ancak büyük ölçekli çalışmalarda hatalar üstel olarak artar.

## Önemli Noktalar (Key Points)

*   Vektör depolama yönetimi için OpenAI ve Pinecone platformlarını kullanırken dikkat edilmesi gereken noktalar vardır.
*   OpenAI modellerinin özellikleri, hızı, maliyeti, girdi limitleri ve API çağrı oranları önemlidir.
*   Pinecone'un bulut ve bölge seçimi, kullanım oranları, depolama maliyetleri ve sınırları önemlidir.

## OpenAI Modelleri (OpenAI Models)

OpenAI, embedding için sürekli yeni modeller sunmaktadır. Bu modellerin özellikleri, hızı, maliyeti, girdi limitleri ve API çağrı oranları önemlidir. Bu bilgilere https://platform.openai.com/docs/models/embeddings adresinden ulaşabilirsiniz.

## OpenAI Modellerinin Kullanımı (Using OpenAI Models)

OpenAI modellerini kullanmak için aşağıdaki kod örneğini inceleyebilirsiniz:
```python
import os
import openai

# OpenAI API anahtarını ayarlayın
openai.api_key = os.getenv("OPENAI_API_KEY")

# Embedding modeli seçin
model = "text-embedding-ada-002"

# Metni embedding'e dönüştürün
response = openai.Embedding.create(
    input="Bu bir örnek metindir.",
    model=model
)

# Embedding'i alın
embedding = response["data"][0]["embedding"]

print(embedding)
```
Bu kod, OpenAI API'sini kullanarak bir metni embedding'e dönüştürür.

## Pinecone Kısıtları (Pinecone Constraints)

Pinecone'u kullanırken bulut ve bölge seçimi, kullanım oranları, depolama maliyetleri ve sınırları önemlidir. Bu bilgilere https://docs.pinecone.io/guides/indexes/back-up-an-index adresinden ulaşabilirsiniz.

## Pinecone'un Kullanımı (Using Pinecone)

Pinecone'u kullanmak için aşağıdaki kod örneğini inceleyebilirsiniz:
```python
import pinecone

# Pinecone API anahtarını ayarlayın
pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENVIRONMENT")

# Index oluşturun
index_name = "example-index"
pinecone.create_index(name=index_name, dimension=128, metric="cosine")

# Index'e veri ekleyin
index = pinecone.Index(index_name)
index.upsert([(1, [0.1, 0.2, 0.3])])

# Index'ten veri alın
result = index.query(vectors=[[0.1, 0.2, 0.3]], top_k=1)

print(result)
```
Bu kod, Pinecone API'sini kullanarak bir index oluşturur ve veri ekler.

## Faturalandırma Yönetimi (Billing Management)

OpenAI ve Pinecone'un faturalandırma yönetimi önemlidir. OpenAI'ın faturalandırma yönetimi için https://platform.openai.com/settings/organization/billing/overview adresini kullanabilirsiniz.

## Sınırlar (Limits)

OpenAI ve Pinecone'un sınırları önemlidir. OpenAI'ın sınırları için https://platform.openai.com/settings/organization/limits adresini kullanabilirsiniz.

## Pipeline 2 Uygulaması (Implementing Pipeline 2)

Pipeline 2'yi uygulamak için `Pipeline_2_Scaling_a_Pinecone_Index.ipynb` dosyasını GitHub deposundan açabilirsiniz. Program, ortamın kurulmasıyla başlar.

---

## Installing the environment

## Ortamın Kurulumu (Installing the Environment)

Programın Pinecone ve OpenAI ile sınırlı olması, ara yazılımlardan, platformlardan ve kısıtlamalardan kaçınmanın avantajını sağlar. API anahtarlarınızı (API Keys) güvenli bir konumda saklayın. Bu örnekte, API anahtarları Google Drive'da saklanmaktadır.

### API Anahtarlarını Saklama (Storing API Keys)

API anahtarlarınızı bir dosyada saklayın ve bu dosyayı okuyun (doğrudan not defterine yazabilirsiniz, ancak yanınızdaki biri tarafından görülebilir).

### Google Drive'ı Bağlama (Mounting Google Drive)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Bu kod, Google Colab'ı Google Drive'a bağlamak için kullanılır. `/content/drive` dizinine bağlanır.

### OpenAI ve Pinecone Kurulumu (Installing OpenAI and Pinecone)

OpenAI ve Pinecone'u kurmak için aşağıdaki komutları çalıştırın:

```bash
!pip install openai==1.40.3
!pip install pinecone-client==5.0.1
```

Bu komutlar, OpenAI ve Pinecone kütüphanelerini ilgili sürümlerle kurar.

### API Anahtarlarını Başlatma (Initializing API Keys)

Program, API anahtarlarını başlatmak için aşağıdaki adımları izler:

```python
f = open("drive/MyDrive/files/pinecone.txt", "r")
PINECONE_API_KEY = f.readline()
f.close()

f = open("drive/MyDrive/files/api_key.txt", "r")
API_KEY = f.readline()
f.close()
```

Bu kod, Pinecone ve OpenAI API anahtarlarını ilgili dosyalardan okur.

### OpenAI Anahtarını Ayarlama (Setting OpenAI Key)

```python
import os
import openai

os.environ['OPENAI_API_KEY'] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
```

Bu kod, OpenAI API anahtarını ortam değişkeni olarak ayarlar ve OpenAI kütüphanesine aktarır.

### İşlem (Processing)

Program şimdi Bank Müşteri Ayrılığı (Bank Customer Churn) veri setini işleme alır.

---

## Processing the dataset

## Veri Kümesinin İşlenmesi (Processing the Dataset)

Bu bölüm, veri kümesini gömme (embedding) için optimize edilmiş metin parçalarına (chunks) ayırmaya odaklanmaktadır. Program önce `data1.csv` veri kümesini alır ve `Pipeline 1: Collecting and preparing the dataset` bölümünde hazırlayıp kaydettiğimiz `data1.csv` dosyasını kullanır.

### Veri Kümesinin Yüklenmesi (Loading the Dataset)

İlk olarak, `data1.csv` dosyasını içeri alır:
```python
!cp /content/drive/MyDrive/files/rag_c6/data1.csv /content/data1.csv
```
Bu komut, Google Drive'da bulunan `data1.csv` dosyasını `/content/` dizinine kopyalar.

Ardından, veri kümesini pandas DataFrame'e yükler:
```python
import pandas as pd

# Dosya yolu
file_path = '/content/data1.csv'

# CSV dosyasını oku
data1 = pd.read_csv(file_path)
```
Bu kod, `pandas` kütüphanesini kullanarak `data1.csv` dosyasını okur ve `data1` adlı bir DataFrame'e yükler.

### Veri Kümesinin Kontrol Edilmesi (Verifying the Dataset)

Veri kümesinin 10.000 satır içerdiğini kontrol eder:
```python
# Satır sayısını say
number_of_lines = len(data1)
print("Number of lines: ", number_of_lines)
```
Bu kod, `data1` DataFrame'indeki satır sayısını sayar ve sonucu yazdırır.

Çıktı:
```
Number of lines: 10000
```
Bu, veri kümesinin doğru şekilde yüklendiğini doğrular.

### Müşteri Kayıtlarının Metin Parçalarına Dönüştürülmesi (Converting Customer Records to Text Chunks)

Her bir müşteri kaydı bir metin parçasına dönüştürülür:
```python
import pandas as pd

# Çıktı listesini başlat
output_lines = []

# DataFrame'deki her bir satırı dolaş
for index, row in data1.iterrows():
    # Her bir sütun için "column_name: value" şeklinde bir liste oluştur
    row_data = [f"{col}: {row[col]}" for col in data1.columns]
    # Listeyi tek bir string'e birleştir
    line = ' '.join(row_data)
    # Çıktı listesine ekle
    output_lines.append(line)

# İlk 5 satırı göster
for line in output_lines[:5]:
    print(line)
```
Bu kod, her bir müşteri kaydını bir metin parçasına dönüştürür ve `output_lines` listesine ekler.

Çıktı:
```
CustomerId: 15634602 CreditScore: 619 Age: 42 Tenure: 2 Balance: 0.0 NumOfProducts: 1 HasCrCard: 1 IsActiveMember: 1 EstimatedSalary: 101348.88 Exited: 1 Complain: 1 Satisfaction Score: 2 Card Type: DIAMOND Point Earned: 464…
```
Bu, her bir müşteri kaydının bir metin parçasına dönüştürüldüğünü gösterir.

### Veri Kümesinin Hazır Hale Getirilmesi (Preparing the Dataset for Chunking)

`output_lines` listesini `lines` listesine kopyalar:
```python
lines = output_lines.copy()
```
Ardından, `lines` listesinin satır sayısını kontrol eder:
```python
# Satır sayısını say
number_of_lines = len(lines)
print("Number of lines: ", number_of_lines)
```
Çıktı:
```
Number of lines: 10000
```
Bu, veri kümesinin doğru şekilde hazırlandığını doğrular.

Artık veri kümesi, gömme (embedding) için optimize edilmiş metin parçalarına (chunks) ayrılmaya hazırdır.

---

## Chunking and embedding the dataset

## Veri Kümesinin Parçalanması ve Gömmeleri (Chunking and Embedding the Dataset)

Bu bölümde, önceden parçalanmış (pre-chunked) verileri `lines` listesinde parçalayacağız (chunking) ve gömme (embedding) işlemi uygulayacağız. Yapılandırılmış verilerle önceden parçalanmış bir liste oluşturmak her zaman mümkün değildir, ancak mümkün olduğunda, bir modelin izlenebilirliğini (traceability), açıklığını (clarity) ve sorgulama performansını (querying performance) artırır. Parçalama işlemi (chunking process) oldukça basittir.

### Ön İşlemler (Preprocessing)

Öncelikle, gerekli kütüphaneleri içe aktarmalıyız (importing necessary libraries). Aşağıdaki kodda, `langchain` ve `sentence-transformers` kütüphanelerini kullanacağız.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
```

### Veri Kümesinin Parçalanması (Chunking the Dataset)

`RecursiveCharacterTextSplitter` sınıfını kullanarak metni parçalayacağız. Bu sınıf, metni belirli bir boyuta göre parçalamaya yarar.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # Her bir parçanın boyutu (chunk size)
    chunk_overlap=20,  # Parçalar arasındaki örtüşme boyutu (chunk overlap)
    length_function=len,  # Uzunluk fonksiyonu (length function)
)

lines = [...]  # Önceden parçalanmış metin listesi (list of pre-chunked text)

chunks = []
for line in lines:
    chunks.extend(text_splitter.split_text(line))
```

Bu kodda, `text_splitter` nesnesini oluşturuyoruz ve `chunk_size` ile `chunk_overlap` parametrelerini belirliyoruz. Daha sonra, `lines` listesindeki her bir metni parçalayarak `chunks` listesine ekliyoruz.

### Gömme İşlemi (Embedding)

Gömme işlemi için `sentence-transformers` kütüphanesini kullanacağız. Bu kütüphane, metinleri vektör uzayında temsil etmeye yarar.

```python
model = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')  # Gömme modeli (embedding model)

embeddings = model.encode(chunks, convert_to_tensor=True)  # Metinleri gömme (embedding the text)
```

Bu kodda, `SentenceTransformer` sınıfını kullanarak bir gömme modeli (embedding model) yüklüyoruz. Daha sonra, `chunks` listesindeki metinleri gömme işlemi uygulayarak `embeddings` değişkenine atıyoruz.

### Sonuç (Result)

Sonuç olarak, `chunks` listesinde parçalanmış metinleri ve `embeddings` değişkeninde bu metinlerin gömmelerini elde ettik. Bu gömmeler, daha sonraki işlemlerde kullanılmak üzere kaydedilebilir.

```python
import numpy as np

# Gömme vektörlerini numpy dizisine çevirme (converting embedding vectors to numpy array)
embeddings_array = embeddings.detach().cpu().numpy()

# Gömme vektörlerini kaydetme (saving embedding vectors)
np.save('embeddings.npy', embeddings_array)
```

Bu kodda, gömme vektörlerini numpy dizisine çevirerek `embeddings_array` değişkenine atıyoruz. Daha sonra, bu diziyi `embeddings.npy` dosyasına kaydediyoruz.

---

## Chunking

## Chunking (Parçalama)
Chunking, veri seti yönetimi için önemlidir. Önceden parçalanmış (pre-chunked) verileri chunking yapmak veri yönetiminde esneklik sağlar.

## Neden Chunking Yapılır?
Verileri doğrudan satırları kullanmak yerine chunking yapmanın birkaç nedeni vardır. 
- Veri hataları gibi sorunları gidermek için ek kalite kontrolü ve işleme gerek duyulabilir.
- Bazı chunklar, belirli bir zamanda embedding modelinin girdi sınırını (input limit) aşabilir.

## Chunking Nasıl Yapılır?
Önceden parçalanmış verileri chunking yapmak için aşağıdaki adımları takip edebilirsiniz:
### Kod
```python
# chunking için boş bir liste oluştur
chunks = []

# her bir satırı chunk olarak listeye ekle
for line in lines:
    chunks.append(line)  # her bir satır kendi chunk'ı olur

# toplam chunk sayısını yazdır
print(f"Toplam chunk sayısı: {len(chunks)}")
```
### Açıklama
Yukarıdaki kod, önceden hazırlanmış `lines` listesindeki her bir satırı `chunks` listesine ekler. Daha sonra, toplam chunk sayısını yazdırır.

## Chunkların Yapısı
Chunkların yapısını anlamak için, ilk birkaç chunk'ın uzunluğunu ve içeriğini inceleyebilirsiniz:
### Kod
```python
# ilk 3 chunk'ın uzunluğunu ve içeriğini yazdır
for i in range(3):
    print(len(chunks[i]))
    print(chunks[i])
```
### Açıklama
Bu kod, `chunks` listesindeki ilk 3 chunk'ın uzunluğunu ve içeriğini yazdırır. Bu, verilerin nasıl parçalandığını görmenizi sağlar.

## Örnek Çıktı
Yukarıdaki kodun çıktısı aşağıdaki gibi olabilir:
```
224
CustomerId: 15634602 CreditScore: 619 Age: 42 Tenure: 2 Balance: 0.0 NumOfProducts: 1 HasCrCard: 1 IsActiveMember: 1 EstimatedSalary: 101348.88 Exited: 1 Complain: 1 Satisfaction Score: 2 Card Type: DIAMOND Point Earned: 464…
224
CustomerId: 15634602 CreditScore: 619 Age: 42 Tenure: 2 Balance: 0.0 NumOfProducts: 1 HasCrCard: 1 IsActiveMember: 1 EstimatedSalary: 101348.88 Exited: 1 Complain: 1 Satisfaction Score: 2 Card Type: DIAMOND Point Earned: 464…
224
CustomerId: 15634602 CreditScore: 619 Age: 42 Tenure: 2 Balance: 0.0 NumOfProducts: 1 HasCrCard: 1 IsActiveMember: 1 EstimatedSalary: 101348.88 Exited: 1 Complain: 1 Satisfaction Score: 2 Card Type: DIAMOND Point Earned: 464…
```
## Chunkların Embedding'i
Chunklar oluşturulduktan sonra, bu chunklar embedding ( gömme ) işlemine tabi tutulabilir.

## Önemli Noktalar
- Chunking, veri seti yönetiminde önemlidir.
- Veri hataları ve girdi sınırları gibi sorunları gidermek için chunking yapılır.
- Chunkların yapısını anlamak için uzunluk ve içeriklerini incelemek gerekir.
- Chunklar embedding işlemine tabi tutulabilir. 

Tüm kodlar Python programlama dilinde yazılmıştır ve `len()` fonksiyonu ile `print()` fonksiyonu gibi temel fonksiyonları kullanır. `range()` fonksiyonu ile döngü oluşturulur ve liste elemanları işlenir.

---

## Embedding

## Embedding (Gömme)
Bu bölüm, dikkatli testler ve sorunların dikkate alınmasını gerektirecektir. Ölçeklendirme yapmanın düşündüğünden daha fazla düşünmeyi gerektirdiğini fark edeceğiz. Her proje, etkili yanıtlar sağlamak için tasarım ve test yoluyla belirli miktarda veri gerektirecektir. Ayrıca, işlem hattının her bir bileşeninin maliyetini ve faydasını dikkate almalıyız. Örneğin, embedding modelini başlatmak kolay bir görev değildir!

### Embedding Modelleri
Yazının yazıldığı sırada, OpenAI üç embedding model sunmaktadır:
- `text-embedding-3-small`
- `text-embedding-3-large`
- `text-embedding-ada-002`

Bu bölümde `text-embedding-3-small` kullanacağız. Ancak, kodu uncomment ederek diğer modelleri değerlendirebilirsiniz.

### Embedding Fonksiyonu
Embedding fonksiyonu, seçtiğiniz modeli kabul edecektir:
```python
import openai
import time

embedding_model = "text-embedding-3-small"
# embedding_model = "text-embedding-3-large"
# embedding_model = "text-embedding-ada-002"

client = openai.OpenAI()

def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return embedding
```
Bu kod, OpenAI'ın embedding API'sını kullanarak bir metnin embedding'ini alır. `get_embedding` fonksiyonu, metni ve modeli parametre olarak alır ve embedding'i döndürür.

### Embedding İşlemi
Embedding işlemini gerçekleştirmek için, parçaları (chunks) embedding'e dönüştürmeliyiz. Ancak, bu işlem stratejik seçimler gerektirir, özellikle büyük veri kümeleri ve API oran sınırlarını etkili bir şekilde yönetmek için.

```python
import openai
import time

client = openai.OpenAI()

start_time = time.time()
chunk_start = 0
chunk_end = 1000
pause_time = 3
embeddings = []
counter = 1

while chunk_end <= len(chunks):
    chunks_to_embed = chunks[chunk_start:chunk_end]
    current_embeddings = []
    for chunk in chunks_to_embed:
        embedding = get_embedding(chunk, model=embedding_model)
        current_embeddings.append(embedding)
    embeddings.extend(current_embeddings)
    chunk_start += 1000
    chunk_end += 1000
    counter += 1
    time.sleep(pause_time)

if chunk_end < len(chunks):
    remaining_chunks = chunks[chunk_end:]
    remaining_embeddings = [get_embedding(chunk, model=embedding_model) for chunk in remaining_chunks]
    embeddings.extend(remaining_embeddings)

print("All chunks processed.")
print(f"Batch {counter} embedded.")
print(f"Response Time: {time.time() - start_time} seconds")
```
Bu kod, parçaları (chunks) embedding'e dönüştürür ve embedding'leri `embeddings` listesine ekler. `while` döngüsü, parçaları 1000'erli gruplar halinde işler ve her grup arasında 3 saniye bekler.

### Embedding'lerin Kontrolü
Embedding'lerin doğru olduğunu doğrulamak için, ilk embedding'i yazdırabiliriz:
```python
print("First embedding:", embeddings[0])
```
Bu, ilk embedding'in değerlerini yazdıracaktır.

### Parçaların ve Embedding'lerin Sayısı
Parçaların (chunks) ve embedding'lerin sayısını doğrulamak için:
```python
num_chunks = len(chunks)
print(f"Number of chunks: {num_chunks}")
print(f"Number of embeddings: {len(embeddings)}")
```
Bu kod, parçaların ve embedding'lerin sayısını yazdıracaktır. İki sayı aynı olmalıdır.

Bu işlemler tamamlandıktan sonra, verileri Pinecone'a taşımaya hazırız.

---

## Duplicating data

## Veri Çoğaltma (Duplicating Data)
Veri çoğaltma, OpenAI gömmeleri (embeddings) için ödeme yapmadan hacimleri simüle etmek için parçalanmış (chunked) ve gömülü (embedded) verileri çoğaltma işlemidir. Gömme verilerinin maliyeti ve zaman performansı doğrusaldır (linear). Bu nedenle, örneğin 50.000 veri noktası içeren bir korpus (corpus) ile ölçeklendirmeyi simüle edebilir ve ihtiyaç duyduğumuz herhangi bir boyuta yanıt sürelerini ve maliyetini tahmin edebiliriz.

### Kod Açıklaması
İlk olarak, verileri kaç kez çoğaltmak istediğimizi belirleriz:
```python
# Çoğaltma boyutunu tanımla (Define the duplication size)
dsize = 5  # Deney gereksinimlerinize göre 1 ile n arasında herhangi bir değer atayabilirsiniz
total = dsize * len(chunks)
print("Toplam boyut (Total size)", total)
```
Bu kod, `dsize` değişkenine atanan değer kadar `chunks` listesini çoğaltacaktır.

### Veri Çoğaltma İşlemi
Daha sonra, parçaları ve gömmeleri çoğaltmak için yeni listeler oluştururuz:
```python
# Çoğaltılmış parçalar ve gömmeler için yeni listeler oluştur (Initialize new lists for duplicated chunks and embeddings)
duplicated_chunks = []
duplicated_embeddings = []

# Orijinal listeleri dolaş ve her girişi çoğalt (Loop through the original lists and duplicate each entry)
for i in range(len(chunks)):
    for _ in range(dsize):
        duplicated_chunks.append(chunks[i])
        duplicated_embeddings.append(embeddings[i])
```
Bu kod, `chunks` ve `embeddings` listelerindeki her elemanı `dsize` kez çoğaltır ve `duplicated_chunks` ve `duplicated_embeddings` listelerine ekler.

### Çoğaltılmış Verilerin Kontrolü
Çoğaltılmış listelerin uzunluklarını kontrol ederiz:
```python
# Çoğaltılmış listelerin uzunluklarını kontrol et (Checking the lengths of the duplicated lists)
print(f"Çoğaltılmış parça sayısı (Number of duplicated chunks): {len(duplicated_chunks)}")
print(f"Çoğaltılmış gömme sayısı (Number of duplicated embeddings): {len(duplicated_embeddings)}")
```
Bu kod, çoğaltılmış listelerin uzunluklarını yazdırır.

### Sonuç
Sonuç olarak, verileri beş kez çoğalttığımızı doğrulayan çıktı:
```
Toplam boyut (Total size) 50000
Çoğaltılmış parça sayısı (Number of duplicated chunks): 50000
Çoğaltılmış gömme sayısı (Number of duplicated embeddings): 50000
```
50.000 veri noktası, bir vektör deposunu (vector store) doldurmak için yeterli veriye sahip olmamızı sağlayan iyi bir başlangıç hacmidir.

### Önemli Noktalar
* Veri çoğaltma, OpenAI gömmeleri için ödeme yapmadan hacimleri simüle etmek için kullanılır.
* Gömme verilerinin maliyeti ve zaman performansı doğrusaldır.
* Çoğaltma boyutu (`dsize`) deney gereksinimlerine göre ayarlanabilir.
* Çoğaltılmış veriler, vektör deposunu doldurmak için kullanılabilir.

### İlgili Kod Parçaları
Yukarıdaki kod parçaları, veri çoğaltma işlemini gerçekleştirmek için kullanılır. Bu kod parçaları, `chunks` ve `embeddings` listelerini çoğaltmak için kullanılır. `dsize` değişkeni, çoğaltma boyutunu belirlemek için kullanılır.

---

## Creating the Pinecone index

## Pinecone İndeks Oluşturma (Creating the Pinecone Index)

İlk adım, API anahtarımızı (API key) tercih ettiğimiz değişken adı ile başlatmak ve ardından bir Pinecone örneği (instance) oluşturmaktır.

### Pinecone Bağlantısını Başlatma (Initializing Connection to Pinecone)

```python
import os
from pinecone import Pinecone, ServerlessSpec

# Pinecone'a bağlantıyı başlat (API anahtarını app.pinecone.io'dan al)
api_key = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
pc = Pinecone(api_key=api_key)
```

Bu kod, Pinecone'a bağlanmak için `Pinecone` sınıfını kullanır ve `api_key` değişkenini kullanarak kimlik doğrulaması yapar.

### İndeks Adını, Bulut ve Bölgeyi Seçme (Choosing Index Name, Cloud, and Region)

```python
from pinecone import ServerlessSpec

index_name = '[YOUR INDEX NAME]'  # örneğin 'bank-index-900'
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)
```

Bu kod, indeks adı, bulut sağlayıcısı ve bölgeyi belirler. `ServerlessSpec` sınıfı, sunucusuz (serverless) bir bulut örneği oluşturmak için kullanılır.

### İndeks Oluşturma (Creating the Index)

```python
import time
import pinecone

# İndeks zaten varsa (ilk kez çalıştırılıyorsa olmaması gerekir)
if index_name not in pc.list_indexes().names():
    # İndeks yoksa, oluştur
    pc.create_index(
        index_name,
        dimension=1536,  # Embedding modelinin boyutu (dimension)
        metric='cosine',  # Vektör benzerliği metriği
        spec=spec
    )
    # İndeks başlatılana kadar bekle
    time.sleep(1)
    # İndeks'e bağlan
    index = pc.Index(index_name)
    # İndeks istatistiklerini görüntüle
    index.describe_index_stats()
```

Bu kod, indeks oluşturmak için `create_index` metodunu kullanır. `dimension` parametresi, embedding modelinin boyutunu (1536) belirtir. `metric` parametresi, vektör benzerliği metriğini (`'cosine'`) belirtir.

### Önemli Noktalar

*   `dimension=1536`: Embedding modelinin boyutunu belirtir.
*   `metric='cosine'`: Vektör benzerliği metriğini belirtir. Diğer metrikler de kullanılabilir (örneğin, Euclidean uzaklığı).
*   `spec=spec`: Sunucusuz bulut örneği yapılandırmasını belirtir.

### İndeks Oluşturulduktan Sonra

İndeks oluşturulduktan sonra, program indeksin açıklamasını görüntüler:

```json
{'dimension': 1536,
 'index_fullness': 0.0,
 'namespaces': {},
 'total_vector_count': 0}
```

Vektör sayısı ve indeks doluluğu 0'dır çünkü vektör deposu henüz doldurulmamıştır. Artık upsert işlemine hazırız!

---

## Upserting

## Upserting (Veri Ekleme/Güncelleme)

Bu bölümün amacı, vektör deposunu (vector store) 50.000 gömülü vektör (embedded vector) ve ilişkili meta verileriyle (chunks) doldurmaktır. Ölçeklendirme sürecini tam olarak anlamak ve sentetik veriler kullanarak 50.000+ vektör seviyesine ulaşmaktır.

### Önemli Noktalar

*   Vektör deposunu üç alanla dolduracağız: `ids`, `embedding` ve `chunks`.
*   `ids`: Her bir chunk için benzersiz bir tanımlayıcı içerir.
*   `embedding`: Oluşturduğumuz vektörleri (embedded chunks) içerir.
*   `chunks`: Plain text olarak chunk'ları içerir, bu meta verilerdir.

### Kod Açıklamaları

#### Upsert Fonksiyonu

```python
def upsert_to_pinecone(data, batch_size):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        index.upsert(vectors=batch)
        # time.sleep(1)  # Opsiyonel: oran limitlerini aşmamak için gecikme ekleyin
```

Bu fonksiyon, veri ekleme/güncelleme işlemini toplu olarak gerçekleştirir. `data` parametresi, eklenecek/güncellenecek verileri içerir. `batch_size` parametresi, her bir toplu işlemde kaç veri öğesinin işleneceğini belirler.

*   `index.upsert(vectors=batch)`: Pinecone index'ine vektörleri ekler/günceler.

#### Toplu İşlem Boyutunu Hesaplama Fonksiyonu

```python
def get_batch_size(data, limit=4000000):
    total_size = 0
    batch_size = 0
    for item in data:
        item_size = sum([sys.getsizeof(v) for v in item.values()])
        if total_size + item_size > limit:
            break
        total_size += item_size
        batch_size += 1
    return batch_size
```

Bu fonksiyon, toplu işlem boyutunu hesaplar ve 4 MB'lık bir limite göre sınırlar.

*   `limit=4000000`: Varsayılan limit 4 MB'dir.

#### Toplu Upsert Fonksiyonu

```python
def batch_upsert(data):
    total = len(data)
    i = 0
    while i < total:
        batch_size = get_batch_size(data[i:])
        batch = data[i:i + batch_size]
        if batch:
            upsert_to_pinecone(batch, batch_size)
            i += batch_size
            print(f"Upserted {i} / {total} items...")
        else:
            break
    print("Upsert complete.")
```

Bu fonksiyon, verileri toplu olarak ekler/günceller.

*   `batch_size = get_batch_size(data[i:])`: Her bir toplu işlem için boyut hesaplar.

#### Benzersiz ID'lerin Oluşturulması

```python
ids = [str(i) for i in range(1, len(duplicated_chunks) + 1)]
```

Bu kod, veri öğeleri için benzersiz ID'ler oluşturur.

#### Meta Verilerin Hazırlanması

```python
data_for_upsert = [
    {"id": str(id), "values": emb, "metadata": {"text": chunk}}
    for id, (chunk, emb) in zip(ids, zip(duplicated_chunks, duplicated_embeddings))
]
```

Bu kod, Pinecone'a veri eklemek için meta verileri hazırlar.

*   `"id": str(id)`: Benzersiz ID'leri içerir.
*   `"values": emb`: Gömülü vektörleri içerir.
*   `"metadata": {"text": chunk}`: Plain text olarak chunk'ları içerir.

#### Toplu Upsert İşleminin Gerçekleştirilmesi

```python
batch_upsert(data_for_upsert)
```

Bu kod, hazırlanan verileri toplu olarak ekler/günceller.

#### Yanıt Süresinin Ölçülmesi

```python
import pinecone
import time
import sys

start_time = time.time()
# ...
response_time = time.time() - start_time
print(f"Upsertion response time: {response_time:.2f} seconds")
```

Bu kod, upsert işleminin yanıt süresini ölçer.

#### Pinecone Index İstatistiklerinin Görüntülenmesi

```python
print("Index stats")
print(index.describe_index_stats(include_metadata=True))
```

Bu kod, Pinecone index'inin istatistiklerini görüntüler.

*   `include_metadata=True`: Meta verileri içerir.

### Çıktılar

Upsert işleminin ilerlemesini ve yanıt süresini gösterir:

```
Upserted 316/50000 items...
Upserted 632/50000 items...
...
Upserted 50000/50000 items...
Upsert complete.
Upsertion response time: 560.66 seconds
```

Pinecone index istatistikleri:

```
Index stats
{'dimension': 1536,
 'index_fullness': 0.0,
 'namespaces': {'': {'vector_count': 50000}},
 'total_vector_count': 50000}
```

Bu çıktılar, upsert işleminin başarılı olduğunu ve 50.000 veri öğesinin eklendiğini/güncellendiğini gösterir.

---

## Querying the Pinecone index

## Pinecone İndeksini Sorgulama (Querying the Pinecone Index)

Pinecone indeksinin büyük bir veri kümesiyle yanıt verme zamanlarını doğrulamak artık görevdir. Vektör deposunu (vector store) sorgulamak ve sonuçları görüntülemek için bir fonksiyon oluşturalım.

### Sonuçları Görüntüleme Fonksiyonu

```python
def display_results(query_results):
    for match in query_results['matches']:
        print(f"ID: {match['id']}, Score: {match['score']}")
        if 'metadata' in match and 'text' in match['metadata']:
            print(f"Text: {match['metadata']['text']}")
        else:
            print("No metadata available.")
```

Bu fonksiyon, sorgu sonuçlarını (`query_results`) alır ve eşleşen (`match`) her bir vektör için ID, skor ve metadati yazdırır. Metadat içinde `text` varsa, onu da yazdırır.

### Sorgu için Gömmeyi (Embedding) Oluşturma

Sorgu için aynı gömme modelini kullanarak bir gömme fonksiyonuna ihtiyacımız var.

```python
embedding_model = "text-embedding-3-small"

def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return embedding
```

Bu fonksiyon, verilen metni (`text`) gömer ve gömme vektörünü döndürür.

### Pinecone Vektör Deposunu Sorgulama

OpenAI istemcisini (`client`) başlatalım ve sorguyu çalıştıralım.

```python
import openai
import time

# OpenAI istemcisini başlat
client = openai.OpenAI()

print("Querying vector store")
start_time = time.time()  # İsteği başlatmadan önce zamanı başlat

query_text = "Customer Robertson CreditScore 632Age 21 Tenure 2Balance 0.0NumOfProducts 1HasCrCard 1IsActiveMember 1EstimatedSalary 99000 Exited 1Complain 1Satisfaction Score 2Card Type DIAMONDPoint Earned 399"

query_embedding = get_embedding(query_text, model=embedding_model)

query_results = index.query(vector=query_embedding, top_k=1, include_metadata=True)

print("processed query results")
display_results(query_results)

response_time = time.time() - start_time  # Yanıt zamanını ölç
print(f"Querying response time: {response_time:.2f} seconds")
```

Bu kod, Pinecone vektör deposunu sorgular, sonuçları işler ve yanıt zamanını ölçer.

### Önemli Noktalar

*   Pinecone indeksini sorgulamak için `index.query()` fonksiyonu kullanılır.
*   Sorgu için gömme vektörünü oluşturmak üzere `get_embedding()` fonksiyonu kullanılır.
*   Yanıt zamanını ölçmek için `time.time()` fonksiyonu kullanılır.
*   Sorgu sonuçları `display_results()` fonksiyonu ile işlenir ve yazdırılır.

### Pinecone Konsolunda İzleme

Pinecone konsolunda (`https://app.pinecone.io/organizations/`) indeksimize giderek istatistiklerimizi izleyebilir, kullanımımızı analiz edebilir ve daha fazlasını yapabiliriz.

Bu, Pinecone indeksimizin artık girdi artırma ve içerik oluşturma için hazır olduğu anlamına gelir.

---

## Pipeline 3: RAG generative AI

## Pipeline 3: RAG Generative AI (RAG Üretken Yapay Zeka)

Bu bölümde, bankanın müşterilerine sadık kalmalarını teşvik etmek için özelleştirilmiş ve ilgi çekici bir pazarlama mesajını otomatikleştirmek için RAG (Retrieval-Augmented Generation) üretken yapay zekasını kullanacağız. Veri hazırlama ve Pinecone indeksleme programlarımızı temel alarak, gelişmiş arama işlevleri için Pinecone vektör veritabanını kullanacağız. Pinecone indeksine sorgu yapmak için bir pazar segmentini temsil eden bir hedef vektör seçeceğiz. Yanıt, en üstteki k benzer vektörleri çıkarmak için işlenecektir. Daha sonra, bu hedef pazar ile kullanıcı girdisini zenginleştireceğiz ve OpenAI'a özelleştirilmiş mesajlarla hedeflenen pazar segmentine önerilerde bulunmasını isteyeceğiz.

### Önemli Noktalar:
* RAG üretken yapay zeka kullanarak özelleştirilmiş pazarlama mesajları oluşturma
* Pinecone vektör veritabanını kullanarak gelişmiş arama işlevleri gerçekleştirme
* Hedef vektör seçerek Pinecone indeksine sorgu yapma
* En üstteki k benzer vektörleri çıkarma ve kullanıcı girdisini zenginleştirme
* OpenAI kullanarak özelleştirilmiş mesajlarla hedeflenen pazar segmentine önerilerde bulunma

### Kodlar ve Açıklamalar:

#### 1. Ortamın Kurulumu (Installing the Environment)
```python
import os
import pinecone
from pinecone import Pinecone, PodSpec
import openai

# Pinecone API anahtarını ayarla
pinecone_api_key = os.environ.get('PINECONE_API_KEY')

# OpenAI API anahtarını ayarla
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Pinecone'ı başlat
pc = Pinecone(api_key=pinecone_api_key)
```
Bu kod, Pinecone ve OpenAI API anahtarlarını ortam değişkenlerinden alır ve Pinecone'ı başlatır.

#### 2. Pinecone İndeksini Bağlama (Connecting to Pinecone Index)
```python
# Pinecone indeks adını ayarla
index_name = 'banking-index'

# Pinecone indeksini kontrol et ve bağlan
if index_name in pc.list_indexes():
    index = pc.Index(index_name)
else:
    print(f'{index_name} indeksi mevcut değil')
```
Bu kod, Pinecone indeks adını ayarlar ve indeksin mevcut olup olmadığını kontrol eder. Eğer indeks mevcutsa, bağlanır.

#### 3. Hedef Vektör Seçme ve Sorgu Yapma (Selecting Target Vector and Querying)
```python
# Hedef vektör seç
target_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

# Pinecone indeksine sorgu yap
response = index.query(
    vector=target_vector,
    top_k=5,
    include_values=True
)

# Yanıtı işle
similar_vectors = response.matches
```
Bu kod, hedef vektör seçer ve Pinecone indeksine sorgu yapar. Yanıt, en üstteki 5 benzer vektörü içerir.

#### 4. Kullanıcı Girdisini Zenginleştirme ve OpenAI Kullanma (Augmenting User Input and Using OpenAI)
```python
# Kullanıcı girdisini ayarla
user_input = 'Müşterilere özelleştirilmiş mesajlar göndermek istiyorum'

# Hedef pazar ile kullanıcı girdisini zenginleştir
augmented_input = f'{user_input} hedef pazar: {target_vector}'

# OpenAI'a gönder
response = openai.Completion.create(
    model='gpt-4o',
    prompt=augmented_input,
    max_tokens=2048
)

# Yanıtı al
recommendations = response.choices[0].text.strip()
```
Bu kod, kullanıcı girdisini ayarlar ve hedef pazar ile zenginleştirir. Daha sonra, OpenAI'a gönderir ve yanıtı alır.

### Not:
Bu kodlar, Pipeline-3_RAG_Generative AI.ipynb not defterinden alınmıştır. Kodları çalıştırmak için, gerekli API anahtarlarını ortam değişkenlerine eklemeniz gerekir.

---

## RAG with GPT-4o

## RAG ile GPT-4o Kullanımı
RAG (Retrieval-Augmented Generation) ile GPT-4o kullanarak, Pinecone vektör deposunu sorgulayacağız, kullanıcı girdisini zenginleştireceğiz ve GPT-4o ile bir yanıt oluşturacağız. Bu süreç, Chapter 3'te Deep Lake ve OpenAI üretken modeli ile yapılan işlemle benzerdir. Ancak Pinecone sorgusunun doğası ve kullanımı farklıdır.

## Önemli Noktalar
* Kullanıcı girdisi klasik anlamda bir soru değildir, bir hedef vektör (Target Vector) olarak kabul edilir ve bir pazar segmentinin profilini temsil eder.
* Kullanım amacı klasik diyalog anlamında üretken yapay zekayı zenginleştirmek değildir, GPT-4o'nun ürün ve hizmet sunmak için ilgi çekici, özelleştirilmiş bir e-posta yazması beklenir.
* Uygulama ölçeklendirilirken hız kritik öneme sahiptir, bu nedenle 1.000.000'dan fazla vektör içeren Pinecone indeksinde sorgu zamanını ölçülecektir.

## Kod Parçası
Aşağıdaki kod parçası Pinecone vektör deposunu sorgulamak, kullanıcı girdisini zenginleştirmek ve GPT-4o ile bir yanıt oluşturmak için kullanılır.
```python
import pinecone
import openai
import time

# Pinecone ayarları
pinecone.init(api_key='API_KEY', environment='ENVIRONMENT')
index_name = 'INDEX_NAME'
index = pinecone.Index(index_name)

# OpenAI ayarları
openai.api_key = 'OPENAI_API_KEY'

# Kullanıcı girdisi (hedef vektör)
user_input = [0.1, 0.2, 0.3, 0.4, 0.5]  # Örnek vektör

# Pinecone sorgusu
start_time = time.time()
query_results = index.query(vectors=[user_input], top_k=5)
end_time = time.time()
query_time = end_time - start_time

# GPT-4o ile yanıt oluşturma
prompt = "Ürün ve hizmetlerimizi sunmak için bir e-posta yazın."
response = openai.Completion.create(
    engine='gpt-4o',
    prompt=prompt,
    max_tokens=2048,
    temperature=0.7
)

# Yanıtı yazdırma
print(response.choices[0].text)
```
## Kod Açıklaması
* `pinecone.init()` fonksiyonu Pinecone'ı başlatmak için kullanılır. `api_key` ve `environment` parametreleri Pinecone hesabınıza göre ayarlanmalıdır.
* `pinecone.Index()` fonksiyonu Pinecone indeksini oluşturmak için kullanılır. `index_name` parametresi indeks adınızı temsil eder.
* `index.query()` fonksiyonu Pinecone indeksini sorgulamak için kullanılır. `vectors` parametresi sorgulanacak vektörleri, `top_k` parametresi döndürülecek en yakın vektör sayısını temsil eder.
* `openai.Completion.create()` fonksiyonu GPT-4o ile bir yanıt oluşturmak için kullanılır. `engine` parametresi GPT-4o modelini, `prompt` parametresi yanıtın oluşturulacağı metni, `max_tokens` parametresi maksimum token sayısını, `temperature` parametresi yanıtın yaratıcılığını kontrol eder.
* `time.time()` fonksiyonu sorgu zamanını ölçmek için kullanılır.

## Kullanım
Yukarıdaki kod parçası, Pinecone vektör deposunu sorgulamak, kullanıcı girdisini zenginleştirmek ve GPT-4o ile bir yanıt oluşturmak için kullanılabilir. Kullanıcı girdisi olarak bir hedef vektör (Target Vector) kullanılması ve GPT-4o'nun ürün ve hizmet sunmak için ilgi çekici, özelleştirilmiş bir e-posta yazması beklenir.

---

## Querying the dataset

## Veri Kümesini Sorgulama (Querying the Dataset)

Veri kümesini sorgulamak için bir gömme (embedding) fonksiyonuna ihtiyacımız var. Uyumluluk nedenleriyle (compatibility reasons) Pinecone indeksi (vector store) ölçeklendirme işleminde kullandığımız aynı gömme modelini kullanacağız.

### Gömme Modelini Tanımlama (Defining the Embedding Model)

Kullandığımız gömme modeli "text-embedding-3-small" olarak belirlenmiştir.

```python
import openai
import time

embedding_model = "text-embedding-3-small"
```

### OpenAI İstemcisini Başlatma (Initializing the OpenAI Client)

OpenAI istemcisini başlatmak için aşağıdaki kod bloğunu kullanıyoruz.

```python
client = openai.OpenAI()
```

### Gömme Fonksiyonunu Tanımlama (Defining the Embedding Function)

Gömme fonksiyonu `get_embedding` adlı bir fonksiyon olarak tanımlanmıştır. Bu fonksiyon, girdi olarak verilen metni gömme modelini kullanarak vektör temsiline (vector representation) dönüştürür.

```python
def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return embedding
```

*   `text.replace("\n", " ")`: Bu satır, girdi metnindeki satır sonlarını (`\n`) boşluk karakteri ile değiştirir. Bu işlem, metnin daha düzgün bir şekilde işlenmesini sağlar.
*   `client.embeddings.create(input=[text], model=model)`: Bu satır, OpenAI istemcisini kullanarak girdi metnini belirtilen gömme modeliyle vektör temsiline dönüştürür.
*   `response.data[0].embedding`: Bu satır, dönüştürülen vektör temsilini elde eder.

### Pinecone İndeksini Sorgulama (Querying the Pinecone Index)

Artık Pinecone indeksini sorgulamaya hazırız.

Pinecone indeksi sorgulanırken, yukarıda tanımlanan `get_embedding` fonksiyonu kullanılarak sorgu metni vektör temsiline dönüştürülür ve Pinecone indeksi üzerinde arama işlemi gerçekleştirilir.

Bu aşamada Pinecone indeksi sorgulama kodunu yazmak için gerekli olan diğer import işlemleri ve kod blokları gerektiğinde eklenmelidir.

Özetle önemli noktalar:

*   Gömme modeli "text-embedding-3-small" olarak belirlenmiştir.
*   OpenAI istemcisi `openai.OpenAI()` ile başlatılmıştır.
*   `get_embedding` fonksiyonu metni vektör temsiline dönüştürmek için kullanılmıştır.
*   Pinecone indeksi sorgulanmaya hazırdır.

---

## Querying a target vector

## Hedef Vektör Sorgulama (Querying a Target Vector)
Hedef vektör, bir pazarlama ekibinin müşteri sadakatini artırmak için önerilerde bulunmak üzere odaklanmak istediği bir pazar segmentini temsil eder. Bu vektör, müşteri profillerini temsil eden bir metin dizisi olarak düşünülebilir.

### Hedef Vektör Tanımlama
Pazarlama ekibi, bu vektörün tasarımına dahil olacaktır. Farklı senaryoları denemek için atölye çalışmaları düzenlenebilir. Hedef vektör, müşteri profillerini temsil eden bir dizi özelliği içerir. Örneğin, yaş (Age), tahmini maaş (EstimatedSalary), şikayet durumu (Complain) ve bankadan çıkış durumu (Exited) gibi özellikler.

### Hedef Vektör Örneği
Bu örnekte, hedef vektörümüz 42 yaşında (Age 42), 100.000'den fazla tahmini maaşı olan (EstimatedSalary 101348.88), şikayetçi olan (Complain 1) ve bankadan çıkış yapmış (Exited 1) müşterileri temsil etmektedir.

### Sorgu Metni (Query Text) Oluşturma
Sorgu metni, aradığımız müşteri profillerini temsil eden bir metin dizisidir.
```python
query_text = "Customer Henderson CreditScore 599 Age 37 Tenure 2 Balance 0.0 NumOfProducts 1 HasCrCard 1 IsActiveMember 1 EstimatedSalary 107000.88 Exited 1 Complain 1 Satisfaction Score 2 Card Type DIAMOND Point Earned 501"
```
Bu metin, bir müşteri profilini temsil etmektedir.

### Sorgu Vektörünü Embedding (Gömme) İşlemi
Sorgu metnini embedding modeli kullanarak bir vektöre dönüştürüyoruz.
```python
import time
start_time = time.time()

query_embedding = get_embedding(text=query_text, model=embedding_model)
```
Bu işlem, sorgu metnini sayısal bir vektöre çevirir.

### Sorguyu Gerçekleştirme
Embedding işleminden sonra, hedef vektöre en yakın olan üst-k (top-k) müşteri profillerini almak için sorguyu gerçekleştiriyoruz.
```python
query_results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True,
)
```
Bu sorgu, embedding vektörünü kullanarak, en yakın 5 müşteri profilini ve ilgili meta verileri döndürür.

### Sonuçları Yazdırma
Sorgu sonuçlarını ve meta verileri yazdırıyoruz.
```python
print("Query Results:")
for match in query_results['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")
    if 'metadata' in match and 'text' in match['metadata']:
        print(f"Text: {match['metadata']['text']}")
    else:
        print("No metadata available.")

response_time = time.time() - start_time
print(f"Querying response time: {response_time:.2f} seconds")
```
Bu kod, sorgu sonuçlarını, skorları ve ilgili müşteri profillerini yazdırır.

### Sonuç
Bu işlem, OpenAI generatif AI modeli kullanarak otomatik olarak sıralama (ranking), skor metriği (score metric) ve içerik (content) sağlar. Elde edilen sonuçlar, pazarlama ekibinin hedef vektörüne en yakın müşteri profillerini temsil eder.

Önemli noktalar:
* Hedef vektör tanımlama
* Sorgu metni oluşturma
* Sorgu vektörünü embedding işlemine tabi tutma
* Sorguyu gerçekleştirme
* Sonuçları yazdırma

Tüm kod:
```python
import time

start_time = time.time()

query_text = "Customer Henderson CreditScore 599 Age 37 Tenure 2 Balance 0.0 NumOfProducts 1 HasCrCard 1 IsActiveMember 1 EstimatedSalary 107000.88 Exited 1 Complain 1 Satisfaction Score 2 Card Type DIAMOND Point Earned 501"
query_embedding = get_embedding(text=query_text, model=embedding_model)

query_results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True,
)

print("Query Results:")
for match in query_results['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")
    if 'metadata' in match and 'text' in match['metadata']:
        print(f"Text: {match['metadata']['text']}")
    else:
        print("No metadata available.")

response_time = time.time() - start_time
print(f"Querying response time: {response_time:.2f} seconds")
```

---

## Extracting relevant texts

## İlgili Metinlerin Çıkarılması (Extracting Relevant Texts)
Verilen kod, üst sıralama vektörleri arasında dolaşarak (iterating), eşleşen metin meta verilerini (matching text metadata) arar ve artırma aşamasını (augmentation phase) hazırlamak için içeriği birleştirir.

## Kod
```python
relevant_texts = [match['metadata']['text'] for match in query_results['matches'] if 'metadata' in match and 'text' in match['metadata']]
combined_text = '\n'.join(relevant_texts)
print(combined_text)
```
## Kod Açıklaması
- `relevant_texts` list comprehension kullanarak oluşturulur. Bu liste, `query_results['matches']` içindeki her bir `match` için `match['metadata']['text']` değerini içerir. Ancak, bu değerin dahil edilebilmesi için `match` içinde `'metadata'` anahtarının ve `match['metadata']` içinde `'text'` anahtarının olması gerekir.
- `combined_text`, `relevant_texts` listesindeki tüm öğeleri birleştirerek tek bir string oluşturur. Bu birleştirme işlemi `\n` (yeni satır) karakteri kullanılarak yapılır, böylece okunabilirlik artırılır.
- `print(combined_text)`, birleştirilmiş metni yazdırır.

## Önemli Noktalar
- **İlgili Metinlerin Seçilmesi**: Kod, `query_results['matches']` içindeki her bir eşleşme için meta verilerde `'text'` anahtarını arar.
- **Birleştirme İşlemi**: İlgili metinler `\n` karakteri kullanılarak birleştirilir.
- **Okunabilirlik**: `\n` kullanımı, çıktının okunmasını kolaylaştırır.

## Teknik Terimler
- **List Comprehension**: Python'da liste oluşturmak için kullanılan kısa ve etkili bir yöntem.
- **Metadata**: Verinin kendisi değil, veri hakkında bilgi veren veriler.

## Çıktı
Kodun çıktısı, `combined_text` değişkeninde saklanan ve yazdırılan metindir. Bu metin, artırma (augmentation) aşamasında kullanılacak olan ilgili metinleri içerir.

## Artırma Aşaması (Augmentation Phase)
Bu aşama, AI üretimi öncesi prompt'u zenginleştirmek için kullanılır. Çıktıdaki metin, bu aşamada kullanılır.

## Örnek Çıktı
```
CustomerId: 15740160 CreditScore: 616 Age: 31 Tenure: 1 Balance: 0.0 NumOfProducts: 2 HasCrCard: 1 IsActiveMember: 1 EstimatedSalary: 54706.75 Exited: 0 Complain: 0 Satisfaction Score: 3 Card Type: DIAMOND Point Earned: 852
CustomerId: 15740160 CreditScore: 616 Age: 31 Tenure: 1 Balance: 0.0 NumOfProducts: 2 HasCrCard: 1 IsActiveMember: 1 EstimatedSalary: 54706.75 Exited: 0 Complain: 0 Satisfaction Score: 3 Card Type: DIAMOND Point Earned: 852
CustomerId: 15740160 CreditScore: 616 Age: 31 Tenure: 1 Balance: 0.0 NumOfProducts: 2 HasCrCard: 1 IsActiveMember: 1 EstimatedSalary: 54706.75 Exited: 0 Complain: 0 Satisfaction Score: 3 Card Type: DIAMOND Point Earned: 852
```
## Kullanılan Kütüphaneler
Bu kod parçasında açıkça bir kütüphane import edilmemektedir. Ancak, `query_results` değişkeninin içeriği ve yapısı, kullanılan veri yapısına ve muhtemelen bir kütüphane veya framework'e (örneğin, bir arama veya AI kütüphanesi) bağlıdır.

## İlgili Kod Parçasının Tamamı
Verilen kod parçası bağımsız olarak çalışabilir, ancak `query_results` değişkeninin tanımlı ve uygun bir yapıya sahip olması gerekir. `query_results` değişkeni, kodun çalışması için gerekli olan `matches` anahtarını içeren bir sözlük (dictionary) olmalıdır.

---

## Augmented prompt

## Gelişmiş Prompt (Augmented Prompt) Oluşturma
Gelişmiş prompt oluşturmak için üç metni birleştiriyoruz: 
- `query_prompt` (`query_prompt`): Üretken yapay zeka modeli (`Generative AI Model`) için talimatlar
- `query_text` (`query_text`): Pazarlama ekibi tarafından seçilen hedef profil içeren hedef vektör (`Target Vector`)
- `combined_context` (`combined_context`): Sorgu (`query`) tarafından seçilen benzer vektörlerin (`similar vectors`) yoğun metadata metni (`concentrated metadata text`)

## Birleştirilmiş Metin (Combined Text)
Bu üç değişkeni (`variables`) içeren `itext` (`itext`) değişkenini oluşturuyoruz.

### Kod
```python
# İlgili metinleri yeni satırlarla birleştirerek tek bir metin oluşturma
combined_context = "\n\n".join(relevant_texts)

# Prompt için sorgu metni
query_prompt = "I have this customer bank record with interesting information on age, credit score and more and similar customers. What could I suggest to keep them in my bank in an email with an url to get new advantages based on the fields for each Customer ID:"

# Birleştirilmiş metni oluşturma
itext = query_prompt + "\n\n" + query_text + "\n\n" + combined_context

# Gelişmiş girdi (Augmented Input)
print("Üretken Yapay Zeka Modeli için Prompt:", itext)
```

## Açıklama
- `combined_context = "\n\n".join(relevant_texts)`: `relevant_texts` listesindeki metinleri yeni satırlarla (`\n\n`) birleştirerek `combined_context` değişkenini oluşturur. Bu, benzer vektörlerin metadata metnini temsil eder.
- `query_prompt`: Üretken yapay zeka modeli için talimatları içeren metni temsil eder. Bu metin, modele ne yapması gerektiğini söyler.
- `itext = query_prompt + "\n\n" + query_text + "\n\n" + combined_context`: Üç metni birleştirerek `itext` değişkenini oluşturur. Bu, üretken yapay zeka modeli için gelişmiş prompttur (`Augmented Prompt`).
- `print("Üretken Yapay Zeka Modeli için Prompt:", itext)`: Oluşturulan `itext` değişkenini yazdırır. Bu, üretken yapay zeka modeli için girdi olarak kullanılır.

## Örnek Çıktı (Example Output)
Üretken Yapay Zeka Modeli için Prompt:
```
I have this customer bank record with interesting information on age, credit score and more and similar customers. What could I suggest to keep them in my bank in an email with an url to get new advantages based on the fields for each Customer ID:
...
```

## Önemli Noktalar
- Üretken yapay zeka modeli için gelişmiş prompt oluşturmak için üç metni birleştiriyoruz.
- `query_prompt`, `query_text` ve `combined_context` değişkenleri birleştirilerek `itext` oluşturulur.
- `itext`, üretken yapay zeka modeli için girdi olarak kullanılır.

## Kullanılan Teknik Terimler
- `Generative AI Model` (Üretken Yapay Zeka Modeli)
- `Target Vector` (Hedef Vektör)
- `Concentrated Metadata Text` (Yoğun Metadata Metni)
- `Augmented Prompt` (Gelişmiş Prompt)
- `Augmented Input` (Gelişmiş Girdi)

---

## Augmented generation

## Artırılmış Üretim (Augmented Generation)

Bu bölümde, artırılmış girdiyi (augmented input) bir OpenAI üretken yapay zeka modeline (OpenAI generative AI model) sunacağız. Hedef, Pinecone indeksindeki pazarlama segmentinde (marketing segment) hedef vektör (target vector) aracılığıyla elde ettiğimiz müşterilere göndermek için özelleştirilmiş bir e-posta (customized email) elde etmektir.

### OpenAI İstemcisini Oluşturma ve Model Seçimi

İlk olarak, bir OpenAI istemcisini (OpenAI client) oluşturacağız ve GPT-4o'yu üretken yapay zeka modeli olarak seçeceğiz:
```python
from openai import OpenAI
client = OpenAI()
gpt_model = "gpt-4o"
```
### Zaman Performans Ölçümü

Daha sonra, zaman performans ölçümü (time performance measurement) için bir kod ekleyeceğiz:
```python
import time
start_time = time.time()  # İsteği göndermeden önce zaman ölçümü
```
### Tamamlama İsteği Oluşturma

Şimdi, tamamlama isteğini (completion request) oluşturmaya başlayacağız:
```python
response = client.chat.completions.create(
  model=gpt_model,
  messages=[
    {
      "role": "system",
      "content": "You are the community manager can write engaging email based on the text you have. Do not use a surname but simply Dear Valued Customer instead."
    },
    {
      "role": "user",
      "content": itext  # mühendislik yapılan metin (engineered text prompt)
    }
  ],
  temperature=0,
  max_tokens=300,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```
Bu kodda:

*   `model=gpt_model`: Kullanılacak GPT modelini belirtir.
*   `messages`: İstemciye gönderilen mesajları içerir. Burada iki rol vardır: "system" ve "user".
*   "system" rolü, modele genel talimatlar verir.
*   "user" rolü, mühendislik yapılan metin istemini (engineered text prompt) içerir.
*   `temperature=0`: Düşük rastgelelik (low randomness) ile yanıt alınmasını sağlar.
*   `max_tokens=300`: Yanıt uzunluğunu 300 token ile sınırlar.
*   `top_p=1`: Tüm olası tokenleri dikkate alır; tam çeşitlilik (full diversity) sağlar.
*   `frequency_penalty=0`: Sık kullanılan kelimelerin tekrarı için ceza uygulamaz, böylece yanıtın açık kalmasını sağlar.
*   `presence_penalty=0`: Yeni konuların girişine ceza uygulamaz, böylece yanıtın istemimize fikir bulmasını sağlar.

### Yanıtı Yazdırma ve Zaman Ölçümü

İsteği gönderip yanıtı yazdıracağız:
```python
print(response.choices[0].message.content)
```
Ayrıca, yanıt alma süresini ölçüp yazdıracağız:
```python
response_time = time.time() - start_time  # Yanıt alma süresini ölçme
print(f"Querying response time: {response_time:.2f} seconds")  # Yanıt alma süresini yazdırma
```
### Önemli Noktalar

*   Artırılmış üretim (augmented generation) için OpenAI üretken yapay zeka modeli kullanılır.
*   GPT-4o modeli seçilir ve bir OpenAI istemcisi oluşturulur.
*   Zaman performans ölçümü yapılır.
*   Tamamlama isteği oluşturulur ve yanıt alınır.
*   Yanıt alma süresi ölçülür ve yazdırılır.

### Kodun Tamamı

```python
from openai import OpenAI
import time

# OpenAI istemcisini oluşturma
client = OpenAI()
gpt_model = "gpt-4o"

# Zaman performans ölçümü
start_time = time.time()

# Tamamlama isteği oluşturma
response = client.chat.completions.create(
  model=gpt_model,
  messages=[
    {
      "role": "system",
      "content": "You are the community manager can write engaging email based on the text you have. Do not use a surname but simply Dear Valued Customer instead."
    },
    {
      "role": "user",
      "content": itext  # mühendislik yapılan metin (engineered text prompt)
    }
  ],
  temperature=0,
  max_tokens=300,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

# Yanıtı yazdırma
print(response.choices[0].message.content)

# Yanıt alma süresini ölçme ve yazdırma
response_time = time.time() - start_time
print(f"Querying response time: {response_time:.2f} seconds")
```

---

## Summary

## Özet
Bu bölüm, banka müşteri kaybını azaltmaya yönelik Pinecone indeksi ve OpenAI modellerini kullanarak ölçeklenebilir bir RAG (Retrieval-Augmented Generation) tabanlı üretken yapay zeka öneri sistemi geliştirmeyi amaçlamıştır. Kaggle veri setini kullanarak, müşteri memnuniyetsizliğine ve hesap kapatmalarına yol açan faktörleri belirleme ve ele alma sürecini gösterdik. Yaklaşımımız üç ana işlem hattı (pipeline) içeriyordu.

## Anahtar Noktalar
* Müşteri kaybını azaltmak için RAG tabanlı bir öneri sistemi geliştirme
* Pinecone indeksi ve OpenAI modellerini kullanma
* Kaggle veri setini kullanma
* Üç ana işlem hattı (pipeline) oluşturma
* Veri setini temizleme ve EDA (Exploratory Data Analysis) yapma
* K-means kümeleme modelini kullanma
* RAG-driven sistemini hazırlama
* Pinecone indeksi kullanarak veri işleme
* GPT-4o kullanarak sorguları artırma

## İşlem Hattı 1 (Pipeline 1)
İlk işlem hattında, veri setini gereksiz sütunlardan arındırarak veri karmaşıklığını ve depolama maliyetlerini azalttık. EDA yoluyla, müşteri şikayetleri ile hesap kapatmaları arasında güçlü bir korelasyon keşfettik, bu da bir k-means kümeleme modeli tarafından daha da doğrulandı.

## İşlem Hattı 2 (Pipeline 2)
İkinci işlem hattında, RAG-driven sistemimizi kişiselleştirilmiş öneriler üretmeye hazırladık. OpenAI modeli kullanarak veri parçalarını işledik ve bunları bir Pinecone indeksine gömdük (embedding). Pinecone'un tutarlı upsert (upsertion) yetenekleri, hacmi ne olursa olsun verimli veri işleme sağladı.

## İşlem Hattı 3 (Pipeline 3)
Üçüncü işlem hattında, Pinecone içinde 1.000.000'dan fazla vektör kullanarak belirli pazar segmentlerini hedefleyen özel teklifler oluşturmayı amaçladık, böylece sadakati artırmayı ve kayıpları azaltmayı hedefledik. GPT-4o kullanarak sorgularımızı artırarak ikna edici öneriler ürettik.

## Kullanılan Kodlar
### Veri Setini Temizleme
```python
import pandas as pd

# Veri setini yükleme
df = pd.read_csv('veri_seti.csv')

# Gereksiz sütunları kaldırma
df = df.drop(['sütun1', 'sütun2'], axis=1)

# Veri setini temizleme
df = df.dropna()  # NaN değerleri kaldırma
df = df.reset_index(drop=True)  # İndeksi sıfırlama
```
Bu kod, veri setini yükler, gereksiz sütunları kaldırır ve NaN değerleri kaldırarak veri setini temizler.

### EDA ve K-means Kümeleme
```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# EDA yapma
plt.scatter(df['şikayet_sayısı'], df['hesap_kapatma'])
plt.show()

# K-means kümeleme modelini oluşturma
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['şikayet_sayısı', 'hesap_kapatma']])
kmeans = KMeans(n_clusters=5)
kmeans.fit(df_scaled)

# Kümeleme sonuçlarını görselleştirme
plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=kmeans.labels_)
plt.show()
```
Bu kod, EDA yapar ve k-means kümeleme modelini oluşturarak kümeleme sonuçlarını görselleştirir.

### RAG-driven Sistemini Hazırlama
```python
import pinecone
from openai import OpenAI

# Pinecone indeksini oluşturma
pinecone.init(api_key='pinecone_api_key', environment='us-west1-gcp')
index = pinecone.Index('index_name')

# OpenAI modelini oluşturma
client = OpenAI(api_key='openai_api_key')

# Veri parçalarını işleme
data_chunks = [...]
for chunk in data_chunks:
    embedding = client.embeddings.create(input=chunk, model='text-embedding-ada-002').data[0].embedding
    index.upsert([(chunk, embedding)])

# Pinecone indeksini sorgulama
query = 'sorgu'
query_embedding = client.embeddings.create(input=query, model='text-embedding-ada-002').data[0].embedding
results = index.query(query_embedding, top_k=5)

# GPT-4o kullanarak sorguları artırma
response = client.chat.completions.create(
    model='gpt-4o',
    messages=[
        {'role': 'system', 'content': 'Sistem mesajı'},
        {'role': 'user', 'content': query}
    ]
)
print(response.choices[0].message.content)
```
Bu kod, Pinecone indeksini oluşturur, OpenAI modelini kullanarak veri parçalarını işler ve Pinecone indeksini sorgular. Ayrıca GPT-4o kullanarak sorguları artırır.

## Sonuç
Bu bölümde, banka müşteri kaybını azaltmaya yönelik RAG tabanlı bir öneri sistemi geliştirdik. Üç ana işlem hattı oluşturduk ve Pinecone indeksi ile OpenAI modellerini kullanarak kişiselleştirilmiş öneriler ürettik. Gelecek bölümde, Pinecone indeksini çok modlu bir bilgi tabanına genişletmeyi planlıyoruz.

---

## Questions

## Konu
Kaggle veri kümelerinin (dataset) gerçek dünya verilerini analiz için indirilmesini ve işlenmesini içerip içermediği, Pinecone'un büyük ölçekli vektör depolama (vector storage) yetenekleri, k-means kümeleme (clustering) algoritmasının müşteri şikayetleri ve churn gibi özellikler arasındaki ilişkileri doğrulamada yardımcı olup olmadığı, bir veritabanında (database) bir milyondan fazla vektörün (vector) kullanılmasının müşteri etkileşimlerini kişiselleştirme (personalize customer interactions) yeteneğini engelleyip engellemediği, iş uygulamalarında (business applications) üretken yapay zeka (generative AI) kullanımının temel amacının karar verme süreçlerini otomatize etmek (automate decision-making processes) ve iyileştirmek olup olmadığı, hafif geliştirme ortamlarının (lightweight development environments) hızlı prototip oluşturma (rapid prototyping) ve uygulama geliştirmede (application development) avantajlı olup olmadığı, Pinecone'un mimarisinin (architecture) artan veri yüklerini (data loads) otomatik olarak ölçeklendirebilme (scale) yeteneği, üretken yapay zekanın dinamik içerik (dynamic content) ve öneriler (recommendations) oluşturmada kullanılıp kullanılmadığı, Pinecone ve OpenAI gibi yapay zeka teknolojilerinin entegrasyonunun (integration) önemli ölçüde manuel yapılandırma (manual configuration) ve bakım (maintenance) gerektirip gerektirmediği ve vektör veritabanları (vector databases) ile yapay zeka kullanan projelerin karmaşık sorguları (complex queries) ve büyük veri kümelerini (large datasets) etkili bir şekilde işleyip işleyemeyeceği soruları ele alınmaktadır.

## Önemli Noktalar
- Kaggle veri kümeleri gerçek dünya verilerini içerir ve analiz için indirilir ve işlenir (Real-world data for analysis).
- Pinecone büyük ölçekli vektör depolama için verimlidir (Efficiently managing large-scale vector storage).
- k-means kümeleme müşteri şikayetleri ve churn arasındaki ilişkileri doğrulamada yardımcı olabilir (Validate relationships between features).
- Bir veritabanında bir milyondan fazla vektör kullanmak müşteri etkileşimlerini kişiselleştirmeyi engellemez (Leveraging over a million vectors).
- Üretken yapay zeka iş uygulamalarında karar verme süreçlerini otomatize etmek ve iyileştirmek için kullanılır (Automate and improve decision-making processes).
- Hafif geliştirme ortamları hızlı prototip oluşturma ve uygulama geliştirmede avantajlıdır (Advantageous for rapid prototyping and application development).
- Pinecone'un mimarisi artan veri yüklerini otomatik olarak ölçeklendirebilir (Automatically scale to accommodate increasing data loads).
- Üretken yapay zeka dinamik içerik ve öneriler oluşturmada kullanılır (Create dynamic content and recommendations).
- Pinecone ve OpenAI entegrasyonu manuel yapılandırma ve bakım gerektirebilir (Require significant manual configuration and maintenance).
- Vektör veritabanları ve yapay zeka kullanan projeler karmaşık sorguları ve büyük veri kümelerini işleyebilir (Effectively handle complex queries and large datasets).

## Kodlar ve Açıklamalar
İlgili kodlar ve açıklamalar aşağıda verilmiştir.

### Kaggle Veri Kümelerini İndirme ve İşleme
```python
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle API'yi başlat
api = KaggleApi()
api.authenticate()

# Veri kümesini indir
api.dataset_download_files('dataset_name', path='./data', unzip=True)

# Veri kümesini oku
df = pd.read_csv('./data/data.csv')

# Veri ön işleme (data preprocessing) adımları
# ...
```
Bu kod, Kaggle API'sini kullanarak bir veri kümesini indirir ve pandas kütüphanesini kullanarak okur. Veri ön işleme adımları burada gerçekleştirilir.

### Pinecone ile Vektör Depolama
```python
import pinecone

# Pinecone'u başlat
pinecone.init(api_key='YOUR_API_KEY', environment='YOUR_ENVIRONMENT')

# Index oluştur
index_name = 'my_index'
pinecone.create_index(index_name, dimension=128, metric='cosine')

# Vektörleri index'e ekle
index = pinecone.Index(index_name)
vectors = [...]  # Vektör listesi
index.upsert(vectors)

# Vektörleri sorgula
query_vector = [...]  # Sorgulama vektörü
results = index.query(query_vector, top_k=10)
```
Bu kod, Pinecone'u kullanarak bir index oluşturur ve vektörleri bu index'e ekler. Daha sonra, bir sorgulama vektörü kullanarak benzer vektörleri sorgular.

### k-means Kümeleme
```python
from sklearn.cluster import KMeans
import numpy as np

# Veri kümesini yükle
data = np.load('data.npy')

# k-means modelini oluştur
kmeans = KMeans(n_clusters=5)

# Veriyi kümele
kmeans.fit(data)

# Küme etiketlerini al
labels = kmeans.labels_
```
Bu kod, sklearn kütüphanesini kullanarak k-means kümeleme algoritmasını uygular ve veri kümesini kümelere ayırır.

### Üretken Yapay Zeka ile İçerik Oluşturma
```python
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Model ve tokenizer'ı yükle
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Giriş metnini kodla
input_text = 'Üretken yapay zeka ile içerik oluşturma'
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Modeli kullanarak çıktı üret
output = model.generate(input_ids)

# Çıktıyı çöz
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
```
Bu kod, transformers kütüphanesini kullanarak T5 modelini yükler ve üretken yapay zeka ile içerik oluşturur.

### Pinecone ve OpenAI Entegrasyonu
```python
import pinecone
import openai

# Pinecone ve OpenAI'ı başlat
pinecone.init(api_key='YOUR_API_KEY', environment='YOUR_ENVIRONMENT')
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Pinecone index'ini oluştur
index_name = 'my_index'
pinecone.create_index(index_name, dimension=128, metric='cosine')

# OpenAI modelini kullanarak vektörleri oluştur
vectors = []
for text in texts:
    response = openai.Embedding.create(input=text, engine='text-embedding-ada-002')
    vector = response['data'][0]['embedding']
    vectors.append(vector)

# Vektörleri Pinecone index'ine ekle
index = pinecone.Index(index_name)
index.upsert(vectors)
```
Bu kod, Pinecone ve OpenAI'ı entegre ederek OpenAI modelini kullanarak vektörleri oluşturur ve Pinecone index'ine ekler.

---

## References

## Pinecone ve OpenAI ile Hızlı Başlangıç Kılavuzu (Pinecone Quickstart Guide with OpenAI)
Pinecone, yüksek performanslı bir vektör veritabanıdır (vector database). OpenAI'nin embedding ve üretken modelleri (generative models) ile entegre çalışarak çeşitli uygulamalar geliştirmek için kullanılır.

## Önemli Noktalar (Key Points)
* Pinecone, vektör veritabanı olarak kullanılır (Pinecone is used as a vector database).
* OpenAI'nin embedding modelleri ile metinleri vektörlere dönüştürmek mümkündür (Text can be converted to vectors using OpenAI's embedding models).
* Pinecone, bu vektörleri depolayarak hızlı benzerlik araması (similarity search) yapmayı sağlar.

## Kullanılan Kodlar (Used Codes)
Aşağıdaki kod örneği, Pinecone ve OpenAI'nin embedding modellerini kullanarak metinleri vektörlere dönüştürmeyi ve Pinecone'de depolamayı gösterir.
```python
import os
import pinecone
from sentence_transformers import SentenceTransformer
import openai

# OpenAI API anahtarını ayarlayın (Set OpenAI API key)
openai.api_key = os.environ["OPENAI_API_KEY"]

# Pinecone API anahtarını ve ortamını ayarlayın (Set Pinecone API key and environment)
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_environment = os.environ["PINECONE_ENVIRONMENT"]

# Pinecone'ı başlatın (Initialize Pinecone)
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

# Index oluşturun (Create index)
index_name = "quickstart-index"
pinecone.create_index(name=index_name, dimension=1536, metric="cosine")

# OpenAI embedding modelini kullanarak metinleri vektörlere dönüştürün (Convert text to vectors using OpenAI embedding model)
model_name = "text-embedding-ada-002"
text = "Örnek metin"
response = openai.Embedding.create(input=text, model=model_name)
vector = response["data"][0]["embedding"]

# Vektörü Pinecone'de depolayın (Upsert vector to Pinecone)
index = pinecone.Index(index_name)
index.upsert([(1, vector)])

# Benzerlik araması yapın (Perform similarity search)
query_vector = vector
result = index.query(vectors=query_vector, top_k=1)

# Sonuçları yazdırın (Print results)
print(result)
```
## Kod Açıklamaları (Code Explanations)
* `import os`: İşletim sistemine özgü işlevleri kullanmak için os modülünü içe aktarır.
* `import pinecone`: Pinecone kütüphanesini içe aktarır.
* `from sentence_transformers import SentenceTransformer`: Cümle dönüştürücüleri kullanmak için SentenceTransformer kütüphanesini içe aktarır (bu kodda kullanılmamıştır).
* `import openai`: OpenAI kütüphanesini içe aktarır.
* `openai.api_key = os.environ["OPENAI_API_KEY"]`: OpenAI API anahtarını ortam değişkeninden ayarlar.
* `pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)`: Pinecone'ı API anahtarı ve ortamıyla başlatır.
* `pinecone.create_index(name=index_name, dimension=1536, metric="cosine")`: Belirtilen isim, boyut ve metriğe göre bir index oluşturur.
* `openai.Embedding.create(input=text, model=model_name)`: OpenAI embedding modelini kullanarak metni vektöre dönüştürür.
* `index.upsert([(1, vector)])`: Vektörü Pinecone'de depolar.
* `index.query(vectors=query_vector, top_k=1)`: Benzerlik araması yapar ve en benzer ilk `top_k` sonucu döndürür.

## Kullanılan Kütüphaneler (Used Libraries)
* `pinecone`: Pinecone kütüphanesi
* `openai`: OpenAI kütüphanesi
* `os`: İşletim sistemine özgü işlevler için

## Referanslar (References)
* Pinecone documentation: https://docs.pinecone.io/guides/get-started/quickstart
* OpenAI embedding and generative models: https://platform.openai.com/docs/models

---

## Further reading

## Vektör Veritabanları Üzerine Kapsamlı Bir İnceleme: Depolama ve Erişim Teknikleri, Zorluklar (A Comprehensive Survey on Vector Database: Storage and Retrieval Technique, Challenge)

Vektör veritabanları (Vector Database), yüksek boyutlu vektörlerin depolanması ve bu vektörler üzerinde hızlı ve etkin arama yapılmasını sağlayan veritabanlarıdır. Bu veritabanları, özellikle makine öğrenimi (Machine Learning) ve derin öğrenme (Deep Learning) gibi alanlarda kullanılan benzerlik tabanlı arama (Similarity Search) işlemlerinde büyük önem taşımaktadır.

## Vektör Veritabanlarının Önemi

- Yüksek boyutlu veri (High-Dimensional Data) işleme kapasitesi
- Benzerlik tabanlı arama (Similarity Search) işlemlerini desteklemesi
- Büyük veri (Big Data) setlerinde hızlı ve etkin arama yapılabilmesi

## Vektör Veritabanlarının Kullanım Alanları

- Görüntü ve video işleme (Image and Video Processing)
- Doğal dil işleme (Natural Language Processing)
- Öneri sistemleri (Recommendation Systems)

## Vektör Veritabanlarında Kullanılan Teknikler

Vektör veritabanlarında kullanılan bazı teknikler şunlardır:

- **Yakın Komşu Arama (Approximate Nearest Neighbor - ANN)**: Yüksek boyutlu vektörler arasında en yakın komşuyu bulmak için kullanılan bir tekniktir.
- **Vektör Quantization (VQ)**: Vektörlerin sıkıştırılması ve temsil edilmesi için kullanılan bir tekniktir.
- **Product Quantization (PQ)**: Vektör quantization'ın bir varyantıdır ve daha etkin bir sıkıştırma sağlar.

## Örnek Kod

Aşağıdaki örnek kod, Python'da Faiss kütüphanesini kullanarak bir vektör veritabanı oluşturmayı ve benzerlik tabanlı arama yapmayı göstermektedir.

```python
import numpy as np
import faiss

# Rastgele 100 adet 128 boyutlu vektör oluştur
vectors = np.random.rand(100, 128).astype('float32')

# Faiss index oluştur
index = faiss.IndexFlatL2(128)

# Vektörleri index'e ekle
index.add(vectors)

# Arama vektörü oluştur
query_vector = np.random.rand(1, 128).astype('float32')

# Benzerlik tabanlı arama yap
D, I = index.search(query_vector, k=5)

# Sonuçları yazdır
print("Mesafeler:", D)
print("Indexler:", I)
```

Bu kodda:

- `np.random.rand(100, 128).astype('float32')` ifadesi, 100 adet 128 boyutlu rastgele vektör oluşturur.
- `faiss.IndexFlatL2(128)` ifadesi, Faiss kütüphanesini kullanarak bir index oluşturur. Bu index, L2 mesafesine ( Euclidean Distance ) göre arama yapacaktır.
- `index.add(vectors)` ifadesi, oluşturulan vektörleri index'e ekler.
- `index.search(query_vector, k=5)` ifadesi, benzerlik tabanlı arama yapar ve en yakın 5 komşuyu bulur.

## Vektör Veritabanlarındaki Zorluklar

- Yüksek boyutlu veri işleme
- Büyük veri setlerinde etkin arama yapılabilmesi
- Vektörlerin temsil edilmesi ve sıkıştırılması

## Sonuç

Vektör veritabanları, yüksek boyutlu vektörlerin depolanması ve bu vektörler üzerinde hızlı ve etkin arama yapılmasını sağlayan önemli bir teknolojidir. Bu veritabanları, makine öğrenimi ve derin öğrenme gibi alanlarda kullanılan benzerlik tabanlı arama işlemlerinde büyük önem taşımaktadır.

---

## Join our community on Discord

## Discord Topluluğumuza Katılın
Packt linki üzerinden Discord platformunda yazar ve diğer okuyucularla tartışmalara katılmak için topluluğumuza katılın.

## Önemli Noktalar
- Yazar ve diğer okuyucularla tartışmalara katılmak
- Discord platformunda etkileşimde bulunmak
- Packt linki üzerinden katılmak

## Discord Katılma Linki
https://www.packt.link/rag

## Kod Örneği Yok
Bu metinde herhangi bir kod örneği bulunmamaktadır.

## Açıklama
Discord topluluğumuza katılmak için verilen linki kullanarak https://www.packt.link/rag adresine gidin. Burada yazar ve diğer okuyucularla tartışmalara katılabilirsiniz (Join our community's Discord space for discussions with the author and other readers). Bu link dört kez tekrarlanmıştır, ancak işlevi aynıdır.

## Teknik Terimler
- Discord: Topluluklarla etkileşimde bulunmak için kullanılan bir platformdur (Community interaction platform).
- Packt: Teknik kitaplar ve kaynaklar sunan bir yayıncıdır (Technical books and resources publisher).

## Ek Bilgiler
Discord, özellikle geliştiriciler (developers) ve teknoloji (technology) meraklıları arasında popüler bir platformdur. Bu topluluğa katılarak, çeşitli konularda tartışmalara katılabilir (participate in discussions), sorularınızı sorabilir (ask questions) ve diğer üyelerden öğrenebilirsiniz (learn from other members).

---

