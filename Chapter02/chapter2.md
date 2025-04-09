## RAG Embedding Vector Stores with Deep Lake and OpenAI

## RAG Embedding Vector Stores with Deep Lake ve OpenAI

Bu bölümde, RAG (Retrieval-Augmented Generation) tabanlı üretken yapay zeka (Generative AI) uygulamalarında karşılaşılan karmaşıklıklar ve bu karmaşıklıkları aşmak için kullanılan yöntemler ele alınacaktır. 
Özellikle, büyük hacimli yapılandırılmış veya yapılandırılmamış metinlerin (unstructured texts) embedding vektörlerine dönüştürülmesi ve bu vektörlerin depolanması konusu işlenecektir.

### Önemli Noktalar:
*   Embedding vektörleri, metinlerin anlamsal özünü (semantic essence) yakalayan yüksek boyutlu vektörlerdir (high-dimensional vectors).
*   Embedding vektörleri, daha hızlı ve etkin bilgi erişimini sağlar.
*   Anahtar kelimeler (keywords) yerine embedding vektörlerinin kullanılması, daha derin anlamsal anlamları yakalar ve daha iyi sonuçlar verir.
*   RAG pipeline'ı bağımsız bileşenlere ayrılarak, birden fazla takımın aynı anda çalışması sağlanır.

### RAG Pipeline'ı Oluşturma

RAG pipeline'ını oluşturmak için aşağıdaki adımlar izlenecektir:
1.  Ham verilerden (raw data) Activeloop Deep Lake vektör deposuna (vector store) kadar olan süreç ele alınacaktır.
2.  OpenAI embedding modelleri yüklenerek, embedding vektörleri oluşturulacaktır.
3.  Çapraz platform paketleri (cross-platform packages) ve bağımlılıkları (dependencies) yönetilecektir.
4.  RAG pipeline'ı üç bileşene ayrılarak, Python'da sıfırdan oluşturulacaktır.

### Kodlama Yolculuğu

Bu kodlama yolculuğunda, aşağıdaki konular ele alınacaktır:
*   Çapraz platform ortam sorunları (cross-platform environment issues) ve paket bağımlılıkları.
*   Veri parçalama (chunking data), embedding vektörleri oluşturma ve vektör depolarına yükleme.
*   GPT-4o modeli ile retrieval sorguları kullanılarak, sağlam çıktılar (solid outputs) üretilmesi.

### Kullanılan Kütüphaneler ve Kurulum

Gerekli kütüphaneleri kurmak için aşağıdaki komutları kullanabilirsiniz:
```bash
pip install deeplake openai
```
### Kod Örneği

Aşağıdaki kod örneğinde, OpenAI embedding modeli kullanılarak, metinlerin embedding vektörlerine dönüştürülmesi gösterilmektedir:
```python
import os
import deeplake
from deeplake.util import Dataset
import openai

# OpenAI API anahtarını ayarlayın
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# OpenAI embedding modelini yükleyin
openai.api_key = os.environ["OPENAI_API_KEY"]

# Metni embedding vektörüne dönüştürün
def text_to_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Metni embedding vektörüne dönüştürün
text = "Bu bir örnek metindir."
embedding = text_to_embedding(text)
print(embedding)

# Activeloop Deep Lake vektör deposunu oluşturun
ds = deeplake.dataset("hub://YOUR_USERNAME/YOUR_DATASET_NAME")

# Embedding vektörünü vektör deposuna ekleyin
ds.append({"text": text, "embedding": embedding})
```
Yukarıdaki kodda, `text_to_embedding` fonksiyonu, OpenAI embedding modeli kullanarak metni embedding vektörüne dönüştürür. 
Ardından, Activeloop Deep Lake vektör deposu oluşturulur ve embedding vektörü bu depoya eklenir.

### Vektör Deposunu Sorgulama

Vektör deposunu sorgulamak için aşağıdaki kod örneğini kullanabilirsiniz:
```python
# Vektör deposunu sorgulayın
query = "örnek metin"
query_embedding = text_to_embedding(query)

# Vektör deposunda benzerlik arama yapın
results = ds.search(query_embedding, "embedding", k=5)

# Sonuçları yazdırın
for result in results:
    print(result["text"])
```
Yukarıdaki kodda, `ds.search` fonksiyonu, vektör deposunda benzerlik arama yapar ve en benzer 5 sonucu döndürür.

### GPT-4o Modeli ile Retrieval Sorguları Kullanma

GPT-4o modeli ile retrieval sorguları kullanmak için aşağıdaki kod örneğini kullanabilirsiniz:
```python
# GPT-4o modelini kullanarak retrieval sorgusu yapın
def retrieval_query(query):
    query_embedding = text_to_embedding(query)
    results = ds.search(query_embedding, "embedding", k=5)
    context = ""
    for result in results:
        context += result["text"] + "\n"
    prompt = f"{query}\n{context}"
    response = openai.Completion.create(
        model="gpt-4o",
        prompt=prompt,
        max_tokens=1024
    )
    return response.choices[0].text.strip()

# Retrieval sorgusu yapın
query = "örnek metin"
response = retrieval_query(query)
print(response)
```
Yukarıdaki kodda, `retrieval_query` fonksiyonu, GPT-4o modeli kullanarak retrieval sorgusu yapar ve sonucu döndürür.

Bu bölümde, RAG embedding vektör depoları ve OpenAI ile ilgili konular ele alındı. 
Embedding vektörlerinin oluşturulması, vektör depolarına yüklenmesi ve sorgulanması gibi konular işlendi. 
Ayrıca, GPT-4o modeli ile retrieval sorguları kullanılarak, sağlam çıktılar üretilmesi gösterildi.

---

## From raw data to embeddings in vector stores

## Ham Verilerden Vektör Depolamada Gömmelere (Embeddings)

Gömmeler (embeddings), her türlü veriyi (metin, resim veya ses) gerçek sayılara dönüştürür. Böylece, bir belge bir vektöre dönüştürülür. Belgelerin bu matematiksel temsilleri, belgeler arasındaki mesafeleri hesaplamamıza ve benzer verileri almamıza olanak tanır. Ham veri (kitaplar, makaleler, bloglar, resimler veya şarkılar) önce toplanır ve gürültüyü gidermek için temizlenir. Hazırlanan veriler daha sonra OpenAI `text-embedding-3-small` gibi bir modele beslenir ve bu model verileri gömer (embedder). Activeloop Deep Lake gibi bazı sistemler, bir metni belirli sayıda karakter tarafından tanımlanan önceden tanımlanmış parçalara (chunks) böler. Bir parçanın boyutu örneğin 1.000 karakter olabilir. Bu parçaları sistem tarafından optimize edilebilir. Bu parçalar, büyük miktarda veriyi işlemeyi kolaylaştırır ve bir belgenin daha ayrıntılı gömmelerini sağlar.

## Vektör Depolama ve Erişim

Ham verilerden gömmelere kadar olan süreç, Retrieval-Augmented Generation (RAG) sistemlerinin temelini oluşturur. RAG, parametrik modellerin siyah kutu (black box) sistemlerine karşı bir oyun değiştiricisidir, çünkü içerik tamamen izlenebilir durumdadır. 
- Sol taraf (Metin): RAG çerçevelerinde, oluşturulan her içerik, kaynak verilere geri izlenebilir, böylece çıktının şeffaflığı sağlanır. OpenAI üretken modeli, artırılmış girdiyi dikkate alarak yanıt verecektir.
- Sağ taraf (Gömmeler): Veri gömmeleri doğrudan görünür ve metne bağlıdır, parametrik modellerin veri kaynaklarının model parametreleri içinde kodlandığı durumun aksine.

## Vektör Mağazaları (Vector Stores)

Vektör mağazaları, yüksek boyutlu verileri (gömmeler gibi) işlemek için tasarlanmış özel veritabanlarıdır. Activeloop gibi sunucusuz platformlarda veri kümeleri oluşturabiliriz. Bunları kodda bir API aracılığıyla oluşturabilir ve erişebiliriz. Vektör mağazalarının bir diğer özelliği de, optimize edilmiş yöntemlerle veri alabilmeleridir. Vektör mağazaları, güçlü indeksleme yöntemleri ile oluşturulmuştur. Bu alma kapasitesi, bir RAG modelinin oluşturma aşamasında en ilgili gömmeleri hızlı bir şekilde bulmasına ve almasına, kullanıcı girdilerini artırmasına ve modelin yüksek kaliteli çıktı üretme yeteneğini artırmasına olanak tanır.

## Örnek Kod Parçası
Aşağıdaki kod, metin verilerini gömmek ve bir vektör mağazasına kaydetmek için kullanılabilir:
```python
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

# OpenAI API anahtarını ayarlayın
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"

# Metin verilerini tanımlayın
texts = ["Bu bir örnek metin.", "Bu başka bir örnek metin."]

# OpenAI gömme modelini tanımlayın
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Metinleri gömün
query_result = embeddings.embed_query(texts[0])

# Deep Lake vektör mağazasını tanımlayın
vector_store = DeepLake(dataset_path="hub://YOUR_USERNAME/YOUR_DATASET_NAME")

# Metinleri vektör mağazasına ekleyin
vector_store.add_texts(texts)

# Benzer metinleri sorgulayın
similar_docs = vector_store.similarity_search("örnek metin")

# Sonuçları yazdırın
print(similar_docs)
```
Bu kod, OpenAI `text-embedding-3-small` modelini kullanarak metin verilerini gömer ve Activeloop Deep Lake vektör mağazasına kaydeder. Daha sonra, benzer metinleri sorgulamak için vektör mağazasını kullanır.

## Kodun Açıklaması
- `OpenAIEmbeddings` sınıfı, OpenAI gömme modelini tanımlamak için kullanılır.
- `DeepLake` sınıfı, Activeloop Deep Lake vektör mağazasını tanımlamak için kullanılır.
- `embed_query` yöntemi, bir metni gömmek için kullanılır.
- `add_texts` yöntemi, metinleri vektör mağazasına eklemek için kullanılır.
- `similarity_search` yöntemi, benzer metinleri sorgulamak için kullanılır.

## Kullanım
Bu kod, RAG sistemlerinde metin verilerini gömmek ve vektör mağazasına kaydetmek için kullanılabilir. Ayrıca, benzer metinleri sorgulamak için de kullanılabilir.

---

## Organizing RAG in a pipeline

## RAG (Retrieval-Augmented Generation) Pipeline'ı Düzenleme
Bir RAG pipeline'ı genellikle verileri toplar ve temizler, örneğin dökümanları parçalara ayırır (chunking), gömme (embedding) işlemini uygular ve bir vektör deposu (vector store) veri setinde saklar. Vektör veri seti daha sonra bir üretken yapay zeka (generative AI) modelinin girdiğini zenginleştirmek için sorgulanır ve bir çıktı üretilir. Ancak, bir vektör deposu kullanırken bu RAG sırasını tek bir programda çalıştırmamak önerilir.

## RAG Pipeline'ını Üç Bileşene Ayırma
En azından süreci üç bileşene ayırmak gerekir:
- Veri toplama ve hazırlama (Data collection and preparation)
- Veri gömme ve vektör deposu veri setine yükleme (Data embedding and loading into the dataset of a vector store)
- Vektörleştirilmiş veri setini sorgulayarak üretken bir yapay zeka modelinin girdisini zenginleştirmek ve bir yanıt üretmek (Querying the vectorized dataset to augment the input of a generative AI model to produce a response)

## Bu Bileşen Yaklaşımının Ana Nedenleri
- **Uzmanlaşma (Specialization)**: Her takım üyesinin en iyi olduğu şeyi yapmasına olanak tanır.
- **Ölçeklenebilirlik (Scalability)**: Teknoloji geliştikçe ayrı bileşenleri yükseltmeyi ve farklı bileşenleri özel yöntemlerle ölçeklendirmeyi kolaylaştırır.
- **Paralel Geliştirme (Parallel development)**: Her takımın diğerlerini beklemeden kendi hızında ilerlemesine olanak tanır.
- **Bakım (Maintenance)**: Bileşenler bağımsızdır, bir takım bir bileşen üzerinde çalışırken diğer bileşenleri etkilemez.
- **Güvenlik (Security)**: Her takımın ayrı ayrı çalışması ve her bileşen için özel yetkilendirme, erişim ve roller olması güvenlik endişelerini ve gizlilik sorunlarını en aza indirir.

## RAG Pipeline'ı Python'da Uygulama
RAG pipeline'ını Python'da oluşturmak için önce bileşenleri tanımlamak gerekir. Aşağıdaki kod örnekleri, RAG pipeline'ının nasıl uygulanacağını gösterir.

### Veri Toplama ve Hazırlama
```python
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Veri toplama
data = pd.read_csv("data.csv")

# Metni parçalara ayırma
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = text_splitter.split_text(data["text"][0])
```

### Veri Gömme ve Vektör Deposuna Yükleme
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from pinecone import Pinecone

# Veri gömme modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

# Vektör deposu
pc = Pinecone(api_key="YOUR_API_KEY")
index = pc.Index("quickstart")

# Veri gömme ve vektör deposuna yükleme
vectors = model.encode(chunks)
index.upsert(vectors=[{"id": str(i), "values": vector} for i, vector in enumerate(vectors)])
```

### Vektörleştirilmiş Veri Setini Sorgulama
```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Üretken yapay zeka modeli
llm = OpenAI(api_key="YOUR_API_KEY")

# Vektörleştirilmiş veri setini sorgulama
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.as_retriever(),
)

query = "Sorgu metni"
result = qa.run(query)
print(result)
```

Bu kod örnekleri, RAG pipeline'ının her bir bileşenini nasıl uygulanacağını gösterir. İlk olarak, veriler toplanır ve hazırlanır. Daha sonra, veriler gömülür ve bir vektör deposuna yüklenir. Son olarak, vektörleştirilmiş veri seti sorgulanarak üretken bir yapay zeka modelinin girdisi zenginleştirilir ve bir yanıt üretilir.

---

## A RAG-driven generative AI pipeline

## RAG (Retrieval-Augmented Generation) Tabanlı Üretken Yapay Zeka Pipeline'ı
Bir RAG pipeline'ının gerçek hayattaki görünümü hakkında bilgi edinelim. Birkaç hafta içinde tüm sistemi teslim etmemiz gereken bir takım olduğumuzu düşünün. İlk başta, 다음과 gibi sorularla bombardımana tutuluruz: Kim tüm verileri toplayacak ve temizleyecek? Kim OpenAI'ın embedding (gömme) modelini kuracak? Kim bu gömme işlemlerini çalıştıracak kodu yazacak ve vektör deposunu (vector store) yönetecek? Kim GPT-4'ü uygulayacak ve ürettiği çıktıları yönetecek?

## Proje Organizasyonu
Birkaç dakika içinde herkes endişelenmeye başlar. İşin tamamı ezici görünür. Bu nedenle, Şekil 2.3'te gösterildiği gibi pipeline'ın farklı bölümlerini üstlenerek üç gruba ayrılırız.

## RAG Pipeline Bileşenleri
Her üç grubun da uygulayacağı bir bileşen vardır:
*   ## Veri Toplama ve Hazırlama (D1 ve D2) (Data Collection and Prep): Bir takım veri toplama ve temizleme işini üstlenir.
*   ## Veri Gömme ve Depolama (D2 ve D3) (Data Embedding and Storage): Başka bir takım, verileri OpenAI'ın embedding modelinden geçirerek elde edilen vektörleri Activeloop Deep Lake veri kümesinde depolar.
*   ## Gelişmiş Üretim (D4, G1-G4 ve E1) (Augmented Generation): Son takım, kullanıcı girdisine ve retrieval sorgularına dayalı içerik üretme işini üstlenir. GPT-4'ü kullanarak bu işlemi gerçekleştirirler.

## Proje Organizasyonunun Faydaları
Proje organizasyonu, Şekil 2.3'te temsil edildiği gibi, RAG ekosisteminin çerçevesinin bir varyantıdır. Bu sayede, proje artık o kadar korkutucu görünmez. Herkesin odaklanacağı bir parçası vardır ve diğer takımların dikkatini dağıtmasına gerek kalmadan çalışabilirler. Bu şekilde, işi daha hızlı bir şekilde tamamlayabilirler.

## RAG Pipeline'ı Oluşturma
Şimdi bir RAG pipeline'ı oluşturmaya başlayabiliriz.

### Kod Örneği
Aşağıdaki kod örneğinde, RAG pipeline'ının nasıl oluşturulacağı gösterilmektedir:
```python
# Import gerekli kütüphaneler
import os
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

# OpenAI API anahtarını ayarlayın
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Veri kümesini yükleyin
data = pd.read_csv("data.csv")

# Embedding modelini oluşturun
embeddings = OpenAIEmbeddings()

# Vektör deposunu oluşturun
vectorstore = DeepLake(
    dataset_path="hub://your_username/your_dataset",
    embedding_function=embeddings,
)

# Verileri vektör deposuna ekleyin
for index, row in data.iterrows():
    vectorstore.add_texts([row["text"]])

# Benzerlik araması yapın
query = "örnek sorgu"
docs = vectorstore.similarity_search(query)

# GPT-4 kullanarak içerik üretin
from langchain.llms import OpenAI

llm = OpenAI(model_name="gpt-4")
generated_text = llm.generate(docs)

print(generated_text)
```
### Kod Açıklaması
Yukarıdaki kod örneğinde, aşağıdaki adımlar gerçekleştirilmektedir:
1.  Gerekli kütüphaneler import edilir.
2.  OpenAI API anahtarı ayarlanır.
3.  Veri kümesi yüklenir.
4.  Embedding modeli oluşturulur.
5.  Vektör deposu oluşturulur.
6.  Veriler vektör deposuna eklenir.
7.  Benzerlik araması yapılır.
8.  GPT-4 kullanarak içerik üretilir.

### Kodun Kullanımı
Bu kod, RAG pipeline'ının temel bileşenlerini oluşturmak için kullanılabilir. Veri toplama ve hazırlama, veri gömme ve depolama, gelişmiş üretim gibi adımları içerir. Gerçek hayattaki bir projede, bu kodun uyarlanması ve genişletilmesi gerekebilir.

---

## Building a RAG pipeline

## RAG Pipeline'ı Oluşturma (Building a RAG Pipeline)
Bir RAG pipeline'ı oluşturmak için, önceki bölümde anlatılan ve Şekil 2.3'te gösterilen pipeline'ı uygulayacağız. 
Üç takımın (Takım #1, Takım #2 ve Takım #3) paralel olarak çalıştığını varsayarak üç bileşeni uygulayacağız:
- Veri toplama ve hazırlama (Data Collection and Preparation) Takım #1 tarafından
- Veri gömme (Data Embedding) ve depolama Takım #2 tarafından
- Artırılmış üretim (Augmented Generation) Takım #3 tarafından

## İlk Adım: Ortamı Hazırlama (Setting Up the Environment)
İlk adım, bu bileşenler için ortamı hazırlamaktır.

### Gerekli Kütüphanelerin İçe Aktarılması (Importing Necessary Libraries)
```python
import os
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
```
Bu kod, gerekli kütüphaneleri içe aktarır. 
- `os`: İşletim sistemi ile etkileşim için kullanılır.
- `pandas`: Veri manipülasyonu ve analizi için kullanılır.
- `numpy`: Sayısal işlemler için kullanılır.
- `transformers`: Önceden eğitilmiş dil modellerini kullanmak için kullanılır. (`AutoModel`, `AutoTokenizer`)
- `torch`: Derin öğrenme modellemesi için kullanılır.
- `sklearn.metrics.pairwise`: Benzerlik ölçümü için kullanılır. (`cosine_similarity`)

### Veri Toplama ve Hazırlama (Data Collection and Preparation)
Takım #1, veri toplama ve hazırlama işlemlerini gerçekleştirecektir.
```python
# Veri setini yükleme
data = pd.read_csv("data.csv")

# Veri ön işleme
data = data.dropna()  # Eksik değerleri kaldırma
data = data.drop_duplicates()  # Yinelenen satırları kaldırma
```
Bu kod, veri setini yükler ve ön işleme yapar. 
- `pd.read_csv`: CSV dosyasını yükler.
- `dropna`: Eksik değerleri kaldırır.
- `drop_duplicates`: Yinelenen satırları kaldırır.

### Veri Gömme ve Depolama (Data Embedding and Storage)
Takım #2, veri gömme ve depolama işlemlerini gerçekleştirecektir.
```python
# Model ve tokenizer'ı yükleme
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Veri gömme
inputs = tokenizer(data["text"], return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
```
Bu kod, veri gömme işlemini gerçekleştirir. 
- `AutoTokenizer.from_pretrained`: Tokenizer'ı yükler.
- `AutoModel.from_pretrained`: Modeli yükler.
- `tokenizer`: Metni tokenlara böler ve modele girdi olarak hazırlar.
- `model`: Tokenlara bölünmüş metni gömer.
- `outputs.last_hidden_state`: Modelin son katmanının çıktısını alır.
- `detach().numpy()`: Çıktıyı numpy dizisine çevirir.

### Artırılmış Üretim (Augmented Generation)
Takım #3, artırılmış üretim işlemlerini gerçekleştirecektir.
```python
# Benzerlik ölçümü
query_embedding = model(**tokenizer("query", return_tensors="pt", padding=True, truncation=True))
query_embedding = query_embedding.last_hidden_state[:, 0, :].detach().numpy()
similarities = cosine_similarity(query_embedding, embeddings)

# En benzer metni bulma
most_similar_idx = np.argmax(similarities)
most_similar_text = data["text"][most_similar_idx]
```
Bu kod, artırılmış üretim işlemini gerçekleştirir. 
- `cosine_similarity`: Sorgu gömmeleri ile veri gömmeleri arasındaki benzerliği ölçer.
- `np.argmax`: En benzer metnin indeksini bulur.
- `data["text"][most_similar_idx]`: En benzer metni alır.

## Sonuç
RAG pipeline'ı oluşturmak için üç takımın paralel olarak çalıştığını varsayarak üç bileşeni uyguladık. 
Veri toplama ve hazırlama, veri gömme ve depolama, artırılmış üretim işlemlerini gerçekleştirdik. 
Gerekli kütüphaneleri içe aktardık, veri setini yükledik ve ön işleme yaptık, veri gömme işlemini gerçekleştirdik ve benzerlik ölçümü yaparak en benzer metni bulduk.

---

## Setting up the environment

## Ortamın Kurulumu (Setting up the Environment)

Çapraz platform (cross-platform) ve çapraz kütüphane (cross-library) paketlerinin bağımlılıkları ile birlikte kurulması oldukça zorlayıcı olabilir! Bu karmaşıklığı dikkate almak ve ortamın doğru çalışması için hazırlıklı olmak önemlidir. Her paketin uyumsuz sürümlere sahip olabilecek bağımlılıkları vardır. Sürümleri uyarlasak bile, bir uygulama beklendiği gibi çalışmayabilir. Bu nedenle, paketlerin ve bağımlılıkların doğru sürümlerini kurmak için zaman ayırın. Biz bu bölümü üç bileşen için de yalnızca bir kez tanımlayacağız ve gerektiğinde bu bölüme atıfta bulunacağız.

## Önemli Noktalar
- Çapraz platform ve çapraz kütüphane paketlerinin kurulumu zor olabilir.
- Bağımlılıklar uyumsuz sürümlere sahip olabilir.
- Uygulamanın doğru çalışması için doğru sürümlerin kurulumu önemlidir.

## Ortam Kurulumu için Adımlar
1. Gerekli paketleri ve bağımlılıklarını belirleyin.
2. Uyumlu sürümleri belirlemek için paketlerin dökümanlarını inceleyin.
3. Paketleri ve bağımlılıklarını kurun.

## Örnek Kod
```python
# import gerekli kütüphaneler
import numpy as np
import pandas as pd

# numpy kütüphanesini kurmak için
# pip install numpy

# pandas kütüphanesini kurmak için
# pip install pandas

# örnek kullanım
data = {'Name': ['John', 'Anna', 'Peter', 'Linda'],
        'Age': [28, 24, 35, 32],
        'Country': ['USA', 'UK', 'Australia', 'Germany']}
df = pd.DataFrame(data)
print(df)
```

## Kod Açıklaması
- `import numpy as np`: numpy kütüphanesini `np` takma adı ile içe aktarır.
- `import pandas as pd`: pandas kütüphanesini `pd` takma adı ile içe aktarır.
- `pip install numpy` ve `pip install pandas`: sırasıyla numpy ve pandas kütüphanelerini kurmak için kullanılan komutlardır. Bu komutlar terminal veya komut istemcisinde çalıştırılır.
- `pd.DataFrame(data)`: pandas kütüphanesini kullanarak bir DataFrame oluşturur.

## Not
- Paketlerin ve bağımlılıkların doğru sürümlerini kurmak için `pip install package_name==version` komutunu kullanabilirsiniz. Örneğin, `pip install numpy==1.20.0`.

---

## The installation packages and libraries

## Kurulum Paketleri ve Kütüphaneler (Installation Packages and Libraries)
Bu bölümde, RAG (Retrieval-Augmented Generation) pipeline'ını oluşturmak için gerekli paketlere ve kütüphanelere ihtiyacımız olacak. Bağımlılık çakışmalarını (dependency conflicts) ve kütüphanelerin fonksiyonlarıyla ilgili sorunları önlemek için paket sürümlerini dondurmak (freeze) gerekecektir.

### Bağımlılık Çakışmalarının Önemi
Bağımlılık çakışmaları aşağıdaki gibi sorunlara neden olabilir:
* Bağımlılık sürümlerinin birbirleriyle çakışması (Possible conflicts between the versions of the dependencies)
* Uygulamanın çalışması için kütüphanelerden birinin güncellenmesi gerektiğinde çakışmalar (Possible conflicts when one of the libraries needs to be updated for an application to run)
* Sürümler uzun süre dondurulduğunda işlevlerin kullanım dışı kalması (Possible deprecations if the versions remain frozen for too long)
* Sürümler uzun süre dondurulduğunda hataların düzeltilmemesi (Possible issues if the versions are frozen for too long and bugs are not corrected by upgrades)

Örneğin, Ağustos 2024'te Deep Lake kütüphanesini kurmak için Pillow kütüphanesinin 10.x.x sürümüne ihtiyaç duyuluyordu, ancak Google Colab'ın Pillow sürümü 9.x.x idi. Bu nedenle, Deep Lake'i kurmadan önce Pillow'u kaldırmak ve yeniden kurmak gerekiyordu.

### Sürüm Dondurma (Freezing Versions)
Sürüm dondurma, uygulamanın bir süre stabil kalmasını sağlar, ancak uzun vadede sorunlara neden olabilir. Diğer taraftan, sürümleri çok hızlı güncelleme de diğer kütüphanelerin çalışmamasına neden olabilir. Bu nedenle, sürekli bir kalite kontrol süreci (continual quality control process) gereklidir.

Bu bölümde, programımız için sürümleri donduracağız. Şimdi, pipeline'ımız için ortamı oluşturmak üzere kurulum adımlarını inceleyelim.

### Kurulum Adımları
Kurulum için gerekli kodları aşağıda görebilirsiniz:
```python
# gerekli kütüphaneleri import etme
import pip

# kütüphaneleri kurma
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn

# Deep Lake kütüphanesini kurma
pip uninstall Pillow
pip install Pillow==10.0.0
pip install deeplake

# diğer kütüphaneleri kurma
pip install transformers torch
```
Yukarıdaki kodda:
* `pip install --upgrade pip` komutu, pip'in son sürümüne güncellenmesini sağlar.
* `pip install numpy pandas matplotlib scikit-learn` komutu, gerekli kütüphaneleri kurar.
* `pip uninstall Pillow` ve `pip install Pillow==10.0.0` komutları, Pillow kütüphanesini doğru sürüme kurar.
* `pip install deeplake` komutu, Deep Lake kütüphanesini kurar.
* `pip install transformers torch` komutu, Transformers ve Torch kütüphanelerini kurar.

Bu kodları çalıştırmak için bir Python ortamına ihtiyacınız olacaktır. Örneğin, Google Colab veya yerel bir Python kurulumu kullanabilirsiniz.

---

## The components involved in the installation process

## Kurulum Sürecinde Yer Alan Bileşenler (Components Involved in the Installation Process)

Kurulum sürecinde yer alan bileşenleri açıklayarak başlayalım. Bu bileşenler her not defterinde (notebook) mutlaka kurulu olmayabilir; bu bölüm bir paket envanteri niteliğindedir.

### Ortamın Kurulumu (Installing the Environment)

İlk olarak, `beautifulsoup4` ve `requests` kütüphanelerini kurmamız gerekiyor:
```python
!pip install beautifulsoup4==4.12.3
!pip install requests==2.31.0
```
Bu kütüphaneler veri toplama ve hazırlama işlemleri için kullanılıyor.

### Google Drive Bağlama (Mounting a Drive)

Google Colab'da Google Drive'ı bağlamak için aşağıdaki kodu kullanıyoruz:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Bu sayede API anahtarlarını ve tokenları güvenli bir şekilde saklayabiliriz.

### GitHub'dan Dosya İndirme (Creating a Subprocess to Download Files from GitHub)

GitHub'dan `grequests.py` dosyasını indirmek için bir fonksiyon yazıyoruz:
```python
import subprocess

url = "https://raw.githubusercontent.com/Denis2054/RAG-Driven-Generative-AI/main/commons/grequests.py"
output_file = "grequests.py"

curl_command = ["curl", "-o", output_file, url]

try:
    subprocess.run(curl_command, check=True)
    print("Download successful.")
except subprocess.CalledProcessError:
    print("Failed to download the file.")
```
Bu fonksiyon `curl` komutunu kullanarak dosyayı indiriyor.

### Gereksinimlerin Kurulumu (Installing Requirements)

Activeloop Deep Lake ve OpenAI ile çalışmak için gerekli kütüphaneleri kuruyoruz:
```python
!pip install deeplake==3.9.18
!pip install openai==1.40.3
```
Ayrıca, Activeloop için bir DNS sunucusu ayarlıyoruz:
```python
with open('/etc/resolv.conf', 'w') as file:
    file.write("nameserver 8.8.8.8")
```
Bu, Activeloop'un düzgün çalışması için gerekli.

### Kimlik Doğrulama Süreci (Authentication Process)

OpenAI API anahtarını ve Activeloop API tokenını ayarlıyoruz:
```python
# OpenAI API anahtarı
f = open("drive/MyDrive/files/api_key.txt", "r")
API_KEY = f.readline().strip()
f.close()

import os
import openai
os.environ['OPENAI_API_KEY'] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

# Activeloop API token
f = open("drive/MyDrive/files/activeloop.txt", "r")
API_token = f.readline().strip()
f.close()

ACTIVELOOP_TOKEN = API_token
os.environ['ACTIVELOOP_TOKEN'] = ACTIVELOOP_TOKEN
```
Bu sayede OpenAI ve Activeloop hizmetlerine erişebiliriz.

### Önemli Noktalar

*   Kurulum sürecinde `beautifulsoup4`, `requests`, `deeplake` ve `openai` kütüphaneleri kullanılıyor.
*   Google Drive bağlanarak API anahtarları ve tokenlar güvenli bir şekilde saklanıyor.
*   GitHub'dan `grequests.py` dosyası indiriliyor.
*   Activeloop Deep Lake ve OpenAI için gerekli kütüphaneler kuruluyor ve DNS sunucusu ayarlanıyor.
*   OpenAI API anahtarı ve Activeloop API tokenı ayarlanıyor.

Tüm bu adımlar tamamlandıktan sonra, kurulum hücrelerini gizleyerek pipeline bileşenlerine odaklanabiliriz.

---

## 1. Data collection and preparation

## 1. Veri Toplama ve Hazırlama (Data Collection and Preparation)

Veri toplama ve hazırlama, bu bölümde daha önce anlatıldığı gibi ilk ardışık düzen (pipeline) bileşenidir. Takım #1, Şekil 2.6'da gösterildiği gibi yalnızca kendi bileşenlerine odaklanacaktır.

## Şekil 2.6: Ardışık düzen bileşeni #1: Veri toplama ve hazırlama

Takım #1'e yardım etmek için işin içine giriyoruz. Görevimiz açıkça tanımlandı, böylece bileşeni uygulamak için geçen zamandan keyif alabiliriz. Uzay keşfinin çeşitli yönlerine kapsamlı bir bakış sağlayan 10 Wikipedia makalesini alacağız ve işleyeceğiz:

*   Uzay Keşfi (Space Exploration): Uzay keşfinin tarihi, teknolojileri, görevleri ve planları hakkında genel bakış
*   Apollo Programı (Apollo Program): İlk insanları Ay'a indiren NASA programı ve önemli görevleri hakkında detaylar
*   Hubble Uzay Teleskobu (Hubble Space Telescope): Birçok astronomik keşifte önemli rol oynayan teleskoplardan biri hakkında bilgi
*   Mars Gezgini (Mars Rover): Mars'ın yüzeyini ve çevresini incelemek için gönderilen gezginler hakkında bilgi
*   Uluslararası Uzay İstasyonu (ISS) (International Space Station): ISS hakkında detaylar, inşaatı, uluslararası işbirliği ve uzay araştırmalarındaki rolü
*   SpaceX: En etkili özel uzay uçuş şirketlerinden birinin tarihi, başarıları ve hedefleri hakkında bilgi
*   Juno (Uzay Aracı) (Juno (Spacecraft)): Jüpiter'i, yapısını ve uydularını inceleyen NASA uzay sondası hakkında bilgi
*   Voyager Programı (Voyager Program): Voyager görevleri hakkında detaylar, dış güneş sistemini ve yıldızlararası uzayı anlamamıza katkıları
*   Galileo (Uzay Aracı) (Galileo (Spacecraft)): Jüpiter'i ve uydularını inceleyen görev hakkında genel bakış, gaz devi ve sistemi hakkında değerli veriler sağladı
*   Kepler Uzay Teleskobu (Kepler Space Telescope): Diğer yıldızların yörüngesinde Dünya büyüklüğünde gezegenler keşfetmek için tasarlanan uzay teleskobu hakkında bilgi

Bu makaleler, tarihsel programlardan modern teknolojik gelişmelere ve görevlere kadar uzay keşfindeki geniş bir yelpazedeki konuları kapsıyor.

Şimdi, GitHub deposundaki `1-Data_collection_preparation.ipynb` dosyasını açın. Öncelikle verileri toplayacağız.

## Veri Toplama

Verileri toplamak için `wikipedia` kütüphanesini kullanacağız. Aşağıdaki kod, Wikipedia makalelerini almak için kullanılır:

```python
import wikipedia

# Makalelerin başlıkları
titles = [
    "Space exploration",
    "Apollo program",
    "Hubble Space Telescope",
    "Mars rover",
    "International Space Station",
    "SpaceX",
    "Juno (spacecraft)",
    "Voyager program",
    "Galileo (spacecraft)",
    "Kepler space telescope"
]

# Makaleleri al ve içeriğini sakla
articles = []
for title in titles:
    try:
        article = wikipedia.page(title)
        articles.append((title, article.content))
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"DisambiguationError: {e}")
    except wikipedia.exceptions.PageError as e:
        print(f"PageError: {e}")
```

Bu kod, `wikipedia` kütüphanesini kullanarak belirtilen başlıklara sahip Wikipedia makalelerini alır ve içeriklerini `articles` listesinde saklar.

*   `wikipedia.page(title)` fonksiyonu, belirtilen başlığa sahip Wikipedia makalesini alır.
*   `article.content` özelliği, makalenin içeriğini sağlar.

## Veri Hazırlama

Verileri topladıktan sonra, içeriklerini işleyerek temizlememiz gerekir. Aşağıdaki kod, makale içeriklerini temizlemek için kullanılır:

```python
import re

# Makale içeriklerini temizle
clean_articles = []
for title, content in articles:
    # HTML etiketlerini kaldır
    content = re.sub(r'<.*?>', '', content)
    # Özel karakterleri kaldır
    content = re.sub(r'[^a-zA-Z0-9\s]', '', content)
    # Çoklu boşlukları tek boşluğa çevir
    content = re.sub(r'\s+', ' ', content)
    clean_articles.append((title, content))
```

Bu kod, makale içeriklerini temizler:

*   `re.sub(r'<.*?>', '', content)` HTML etiketlerini kaldırır.
*   `re.sub(r'[^a-zA-Z0-9\s]', '', content)` özel karakterleri kaldırır.
*   `re.sub(r'\s+', ' ', content)` çoklu boşlukları tek boşluğa çevirir.

Temizlendikten sonra, veriler `clean_articles` listesinde saklanır.

## Sonuç

Bu bölümde, uzay keşfi ile ilgili Wikipedia makalelerini topladık ve içeriklerini temizledik. Toplanan veriler, ileriki adımlarda kullanılmak üzere hazırlandı.

---

## Collecting the data

## Veri Toplama (Collecting the Data)
Veri toplama işlemine başlamak için gerekli olan kütüphaneleri (`libraries`) içe aktarmalıyız (`import`). HTTP istekleri (`HTTP requests`) için `requests` kütüphanesini, HTML ayrıştırma (`HTML parsing`) için `BeautifulSoup` kütüphanesini ve düzenli ifadeler (`regular expressions`) için `re` kütüphanesini kullanacağız.

### Kullanılan Kütüphaneler
```python
import requests
from bs4 import BeautifulSoup
import re
```
Bu kütüphaneler sırasıyla aşağıdaki amaçlar için kullanılmaktadır:
- `requests`: Belirtilen URL'lere HTTP istekleri göndermek için kullanılır.
- `BeautifulSoup`: HTML ve XML dosyalarını ayrıştırmak için kullanılır.
- `re`: Düzenli ifadeler kullanarak metin içinde arama ve değiştirme işlemleri yapmak için kullanılır.

## URL'lerin Seçilmesi
İncelenecek Wikipedia makalelerinin URL'lerini bir liste içinde saklayacağız.
```python
# Wikipedia makalelerinin URL'leri
urls = [
    "https://en.wikipedia.org/wiki/Space_exploration",
    "https://en.wikipedia.org/wiki/Apollo_program",
    "https://en.wikipedia.org/wiki/Hubble_Space_Telescope",
    "https://en.wikipedia.org/wiki/Mars_over",
    "https://en.wikipedia.org/wiki/International_Space_Station",
    "https://en.wikipedia.org/wiki/SpaceX",
    "https://en.wikipedia.org/wiki/Juno_(spacecraft)",
    "https://en.wikipedia.org/wiki/Voyager_program",
    "https://en.wikipedia.org/wiki/Galileo_(spacecraft)",
    "https://en.wikipedia.org/wiki/Kepler_Space_Telescope"
]
```
Bu liste, kod içinde tanımlanmıştır. Ancak, URL'ler bir veritabanında (`database`), dosyada (`file`) veya JSON gibi başka bir formatta da saklanabilir.

### Önemli Noktalar
- Veri toplama işlemi için gerekli kütüphaneler içe aktarılmıştır.
- İncelenecek Wikipedia makalelerinin URL'leri bir liste içinde tanımlanmıştır.
- URL'ler farklı formatlarda saklanabilir.

### Kullanılan Kodların Açıklaması
- `import requests`: HTTP istekleri göndermek için `requests` kütüphanesini içe aktarır.
  - Kullanımı: `requests.get(url)` şeklinde kullanılır. Örneğin: `response = requests.get("https://en.wikipedia.org/wiki/Space_exploration")`
- `from bs4 import BeautifulSoup`: HTML ayrıştırma için `BeautifulSoup` kütüphanesini içe aktarır.
  - Kullanımı: `soup = BeautifulSoup(html_content, 'html.parser')` şeklinde kullanılır.
- `import re`: Düzenli ifadelerle çalışmak için `re` kütüphanesini içe aktarır.
  - Kullanımı: `re.search(pattern, string)` veya `re.findall(pattern, string)` şeklinde kullanılır.

Bu kod parçaları, veri toplama ve işleme sürecinin başlangıcını oluşturur. Sonraki adımlarda, bu URL'lere HTTP istekleri gönderilerek içeriklerin nasıl işleneceği ve analiz edileceği üzerine çalışılacaktır.

---

## Preparing the data

## Veri Hazırlama (Preparing the Data)
Veri hazırlama aşamasında ilk olarak bir temizleme fonksiyonu (`clean_text` fonksiyonu) yazılır. Bu fonksiyon, verilen bir metin dizgisinden sayısal referansları (örneğin [1], [2] gibi) düzenli ifadeler (`regular expressions`) kullanarak kaldırır ve temizlenmiş metni döndürür.

### Temizleme Fonksiyonu
```python
import re

def clean_text(content):
    # Sayısal referansları kaldır (Remove references that usually appear as [1], [2], etc.)
    content = re.sub(r'\[\d+\]', '', content)
    return content
```
Bu kod, `re.sub` fonksiyonunu kullanarak metin içerisindeki sayısal referansları kaldırır. `r'\[\d+\]'` düzenli ifadesi, köşeli parantez içerisinde bir veya daha fazla rakam (`\d+`) olan dizgileri eşleştirir.

## Veri Çekme ve Temizleme Fonksiyonu
Daha sonra, `fetch_and_clean` adlı klasik bir veri çekme ve temizleme fonksiyonu yazılır. Bu fonksiyon, belgelerden ihtiyacımız olan içeriği çıkararak temiz bir metin döndürür.

```python
import requests
from bs4 import BeautifulSoup

def fetch_and_clean(url):
    # URL'nin içeriğini getir (Fetch the content of the URL)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Ana içeriği bul (Find the main content of the article)
    content = soup.find('div', {'class': 'mw-parser-output'})
    
    # Kaynakça bölümünü kaldır (Remove the bibliography section)
    for section_title in ['References', 'Bibliography', 'External links', 'See also']:
        section = content.find('span', id=section_title)
        if section:
            # Bu bölümden belge sonuna kadar olan tüm içeriği kaldır (Remove all content from this section to the end of the document)
            for sib in section.parent.find_next_siblings():
                sib.decompose()
            section.parent.decompose()
    
    # Metni çıkar ve temizle (Extract and clean the text)
    text = content.get_text(separator=' ', strip=True)
    text = clean_text(text)
    return text
```
Bu fonksiyon, verilen bir URL'den içerikleri çeker, ana içeriği bulur, kaynakça bölümünü kaldırır ve metni temizler.

## Temizlenmiş Veriyi Dosyaya Yazma
Son olarak, temizlenmiş içerik `llm.txt` dosyasına yazılır.

```python
# Temiz metni yazmak için dosya (File to write the clean text)
with open('llm.txt', 'w', encoding='utf-8') as file:
    for url in urls:
        clean_article_text = fetch_and_clean(url)
        file.write(clean_article_text + '\n')
print("Content written to llm.txt")
```
Bu kod, `fetch_and_clean` fonksiyonu tarafından döndürülen temiz metni `llm.txt` dosyasına yazar.

## Dosyayı Doğrulama
Dosyanın doğru yazıldığını doğrulamak için ilk 20 satır okunur.

```python
# Dosyayı aç ve ilk 20 satırı oku (Open the file and read the first 20 lines)
with open('llm.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    
# İlk 20 satırı yazdır (Print the first 20 lines)
for line in lines[:20]:
    print(line.strip())
```
Bu kod, `llm.txt` dosyasını okur ve ilk 20 satırı yazdırır.

## Önemli Noktalar
*   Temizleme fonksiyonu (`clean_text`) sayısal referansları kaldırır.
*   Veri çekme ve temizleme fonksiyonu (`fetch_and_clean`) içerikleri çeker ve temizler.
*   Temizlenmiş içerik `llm.txt` dosyasına yazılır.
*   Dosya doğrulanması için ilk 20 satır okunur.

Bu işlemler, RAG (`Retrieval-Augmented Generation`) çerçevesinin temelini oluşturan belgelerin işlenmesi için önemlidir. İkinci takım (`Team #2`), belgeleri gömme (`embedding`) ve depolama işlemleri üzerinde çalışabilir.

---

## 2. Data embedding and storage

## 2. Veri Gömmeleri ve Depolama (Data Embedding and Storage)

Takım #2'nin görevi, boru hattının (pipeline) ikinci bileşeni (component) üzerinde çalışmaktır. Hazır hale getirilmiş veri (data) gruplarını (batches) çalışmak üzere alacaklardır. Veri (data) alma konusunda endişelenmelerine gerek yoktur, çünkü Takım #1 veri toplama ve hazırlama bileşeni ile onların sırtlarını kollamaktadır.

## Boru Hattı Bileşeni #2: Veri Gömmeleri ve Depolama

Şimdi, Takım #2'nin işini yapmasına yardım etmek için dalalım. GitHub Depo'sunda (Repository) bulunan `2-Embeddings_vector_store.ipynb` dosyasını açacağız. Takım #1 tarafından sağlanan verileri gömerek (embedding) ve depolayarak (storage) çalışmak üzere bir veri grubunu (batch of documents) alacağız.

### Kodları Açıklama

Aşağıdaki kod, `2-Embeddings_vector_store.ipynb` dosyasındaki kodların bir kısmını içermektedir.

```python
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Veri yükleme (Data Loading)
data = pd.read_csv('data.csv')

# Cümleleri gömme (Sentence Embedding)
model = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')
embeddings = model.encode(data['text'], convert_to_tensor=True)

# Benzerlik hesaplama (Similarity Calculation)
similarities = cosine_similarity(embeddings, embeddings)

# Sonuçları gösterme (Displaying Results)
print(similarities)
```

### Kod Açıklamaları

*   `import numpy as np`: Numpy kütüphanesini (library) `np` takma adı (alias) ile içe aktarır (import). Numpy, sayısal işlemler (numerical computations) için kullanılır.
*   `import pandas as pd`: Pandas kütüphanesini `pd` takma adı ile içe aktarır. Pandas, veri işleme (data manipulation) ve analiz (analysis) için kullanılır.
*   `from sentence_transformers import SentenceTransformer`: SentenceTransformers kütüphanesinden `SentenceTransformer` sınıfını içe aktarır. Bu sınıf, cümleleri vektörlere (vectors) dönüştürmek (transform) için kullanılır.
*   `model = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')`: `SentenceTransformer` modelini 'distilbert-multilingual-nli-stsb-quora-ranking' modeli ile başlatır (initialize). Bu model, çok dilli (multilingual) metinleri gömmek (embedding) için kullanılır.
*   `embeddings = model.encode(data['text'], convert_to_tensor=True)`: `data` veri çerçevesindeki (DataFrame) 'text' sütunundaki metinleri gömerek (embedding) vektörlere dönüştürür. `convert_to_tensor=True` parametresi, sonuçları tensor (PyTorch tensor) olarak döndürür.
*   `similarities = cosine_similarity(embeddings, embeddings)`: Gömülen vektörler arasındaki benzerliği (similarity) kosinüs benzerliği (cosine similarity) kullanarak hesaplar.
*   `print(similarities)`: Hesaplanan benzerlikleri yazdırır (print).

### Kullanım

Yukarıdaki kod, `2-Embeddings_vector_store.ipynb` dosyasındaki kodların bir kısmını içermektedir. Bu kod, veri gömmeleri (data embedding) ve benzerlik hesaplama (similarity calculation) işlemlerini gerçekleştirmek için kullanılır. Veri, `data.csv` dosyasından yüklenir (load) ve `SentenceTransformer` modeli kullanılarak gömülür (embedding). Daha sonra, gömülen vektörler arasındaki benzerlik kosinüs benzerliği kullanılarak hesaplanır ve sonuçlar yazdırılır.

## Önemli Noktalar

*   Veri gömmeleri (Data Embedding), metinleri sayısal vektörlere (numerical vectors) dönüştürme işlemidir.
*   `SentenceTransformer` modeli, cümleleri vektörlere dönüştürmek için kullanılır.
*   Kosinüs benzerliği (Cosine Similarity), iki vektör arasındaki benzerliği ölçmek için kullanılır.
*   Veri depolama (Data Storage), gömülen vektörleri depolamak için kullanılır.

## Eklemeler

*   Veri gömmeleri için farklı modeller (models) kullanılabilir, örneğin `BERT`, `RoBERTa`, `DistilBERT` gibi.
*   Benzerlik hesaplama için farklı yöntemler (methods) kullanılabilir, örneğin ` Euclidean similarity`, `Jaccard similarity` gibi.
*   Veri depolama için farklı yöntemler kullanılabilir, örneğin `Vector Database`, ` Relational Database` gibi.

---

## Retrieving a batch of prepared documents

## Hazır Belgeleri Toplu Olarak Alma (Retrieving a Batch of Prepared Documents)

İlk olarak, Team #1 tarafından sağlanan ve gelen belgelerin ilk akışı olan sunucudaki kullanılabilir bir belge grubunu indiriyoruz. Bu durumda, bunun uzay keşfi (space exploration) dosyası olduğunu varsayıyoruz.

### Belge İndirme (Downloading the Document)

Aşağıdaki kod, `grequests` kütüphanesini kullanarak bir belge indirir:
```python
from grequests import download

source_text = "llm.txt"
directory = "Chapter02"
filename = "llm.txt"
download(directory, filename)
```
Burada `source_text` değişkeni, daha sonra vektör deposuna (vector store) veri ekleyecek fonksiyon tarafından kullanılacaktır.

### Belge Doğrulama (Verifying the Document)

İndirilen belgeyi kısaca kontrol ediyoruz:
```python
with open('llm.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

for line in lines[:20]:
    print(line.strip())
```
Bu kod, `llm.txt` dosyasını açar, ilk 20 satırını okur ve yazdırır.

### Veri Parçalama (Chunking the Data)

Verileri parçalayarak (chunking) işleme hazır hale getiriyoruz. Parça boyutu (chunk size), karakter sayısına göre belirlenir. Bu örnekte, `CHUNK_SIZE = 1000` olarak belirlenmiştir:
```python
with open(source_text, 'r') as f:
    text = f.read()

CHUNK_SIZE = 1000
chunked_text = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
```
Bu kod, `source_text` dosyasını okur, `CHUNK_SIZE` boyutunda parçalara böler ve `chunked_text` listesine kaydeder.

### Vektör Deposu Oluşturma (Creating a Vector Store)

Artık verileri vektörleştirmek (vectorize) veya mevcut bir vektör deposuna eklemek için hazırız.

Önemli noktalar:

*   Belgeleri indirmek için `grequests` kütüphanesi kullanılır.
*   İndirilen belgeler doğrulanır.
*   Veriler parçalanarak (chunking) işleme hazır hale getirilir.
*   Parça boyutu (chunk size) karakter sayısına göre belirlenir.
*   Vektör deposu (vector store) oluşturmak için veriler hazır hale getirilir.

Ek bilgiler:

*   `grequests` kütüphanesi, asynchronous HTTP requests yapmak için kullanılır.
*   `CHUNK_SIZE` değişkeni, parça boyutunu belirler.
*   Vektör deposu, verileri vektörleştirmek (vectorize) veya mevcut bir vektör deposuna eklemek için kullanılır.

Kod açıklamaları:

*   `download(directory, filename)`: Belge indirir.
*   `with open('llm.txt', 'r', encoding='utf-8') as file:`: Dosyayı açar ve okur.
*   `chunked_text = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]`: Verileri parçalar.

Tüm kod:
```python
from grequests import download

# Belge indirme
source_text = "llm.txt"
directory = "Chapter02"
filename = "llm.txt"
download(directory, filename)

# Belge doğrulama
with open('llm.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

for line in lines[:20]:
    print(line.strip())

# Veri parçalama
with open(source_text, 'r') as f:
    text = f.read()

CHUNK_SIZE = 1000
chunked_text = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
```

---

## Verifying if the vector store exists and creating it if not

## Vektör Deposunun (Vector Store) Varlığının Doğrulaması ve Oluşturulması

İlk olarak, Activeloop vektör deposu (vector store) yolunu tanımlamamız gerekir, veri setimizin (dataset) var olup olmadığına bakılmaksızın:
```python
vector_store_path = "hub://denis76/space_exploration_v1"
```
"hub://denis76/space_exploration_v1" ifadesini kendi organizasyonunuz ve veri seti adınızla değiştirdiğinizden emin olun.

Ardından, vektör deposunu yüklemeye çalışacak veya yoksa otomatik olarak oluşturacak bir fonksiyon yazarız:
```python
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
import deeplake.util

try:
    # Vektör deposunu yüklemeye çalış
    vector_store = VectorStore(path=vector_store_path)
    print("Vektör deposu (Vector Store) mevcut")
except FileNotFoundError:
    print("Vektör deposu (Vector Store) mevcut değil. Oluşturabilirsiniz.")
    # Vektör deposunu oluşturma kodu buraya gelecek
    create_vector_store = True
```
Kodun açıklaması:
- `try` bloğu içinde, `VectorStore` sınıfını kullanarak `vector_store_path` ile belirtilen yoldaki vektör deposunu yüklemeye çalışıyoruz.
- Eğer vektör deposu mevcutsa, `print` ifadesi "Vektör deposu (Vector Store) mevcut" mesajını yazdırır.
- Eğer vektör deposu mevcut değilse, `FileNotFoundError` hatası oluşur ve `except` bloğu devreye girer.
- `except` bloğu içinde, "Vektör deposu (Vector Store) mevcut değil. Oluşturabilirsiniz." mesajını yazdırır ve `create_vector_store` değişkenini `True` olarak ayarlar.

Çıktı (Output), vektör deposunun başarıyla oluşturulduğunu doğrular:
```
Your Deep Lake dataset has been successfully created!
Vektör deposu (Vector Store) mevcut
```
veya
```
Your Deep Lake dataset has been successfully created!
Vektör deposu (Vector Store) mevcut
```
Şimdi, bir embedding fonksiyonu oluşturmamız gerekiyor.

## Önemli Noktalar:
* Vektör deposu yolunu (`vector_store_path`) doğru bir şekilde tanımlayın.
* `VectorStore` sınıfını kullanarak vektör deposunu yüklemeye çalışın.
* Vektör deposu mevcut değilse, `FileNotFoundError` hatası oluşur ve `except` bloğu devreye girer.
* Vektör deposunu oluşturmak için gerekli kodu `except` bloğu içine ekleyin.
* Embedding fonksiyonunu oluşturun.

---

## The embedding function

## Gömme Fonksiyonu (Embedding Function)
Gömme fonksiyonu, oluşturduğumuz veri parçalarını vektör tabanlı arama yapabilmek için vektörlere dönüştürür. Bu programda, belgeleri gömmek (embed) için "text-embedding-3-small" modelini kullanacağız.

## Neden "text-embedding-3-small"?
OpenAI'in diğer gömme modelleri de mevcuttur: https://platform.openai.com/docs/models/embeddings. 
6. Bölümde, Pinecone ile RAG Banka Müşteri Verilerini Ölçeklendirme (Scaling RAG Bank Customer Data with Pinecone) adlı bölümde, gömme modelleri için alternatif kodlar sağlanmaktadır. 
Her durumda, üretimde birini seçmeden önce gömme modellerini değerlendirmek önerilir. 
Her bir gömme modelinin özelliklerini, OpenAI tarafından açıklanan uzunluk ve kapasitelerine odaklanarak inceleyin.

## Gömme Fonksiyonu Kodları
```python
def embedding_function(texts, model="text-embedding-3-small"):
    if isinstance(texts, str):
        texts = [texts]
    texts = [t.replace("\n", " ") for t in texts]
    return [data.embedding for data in openai.embeddings.create(input=texts, model=model).data]
```
### Kod Açıklaması
- `def embedding_function(texts, model="text-embedding-3-small"):` Bu satır, `embedding_function` adlı bir fonksiyon tanımlar. Bu fonksiyon, `texts` ve `model` adlı iki parametre alır. `model` parametresi varsayılan olarak "text-embedding-3-small" değerini alır.
- `if isinstance(texts, str): texts = [texts]` Bu satır, eğer `texts` bir string ise, onu bir liste haline getirir.
- `texts = [t.replace("\n", " ") for t in texts]` Bu satır, `texts` listesindeki her bir metin içinde `\n` karakterini boşluk karakteri ile değiştirir. Bu, metinlerin daha düzgün bir şekilde işlenmesini sağlar.
- `return [data.embedding for data in openai.embeddings.create(input=texts, model=model).data]` Bu satır, OpenAI'in `embeddings.create` metodunu kullanarak `texts` listesindeki metinleri gömer ve elde edilen vektörleri döndürür.

## OpenAI Embeddings
OpenAI'in "text-embedding-3-small" modeli, genellikle sınırlı sayıda boyut kullanarak gömme yapar. Bu, büyük hesaplama iş yükleri ve depolama alanı ile yeterli detay elde etmek arasında bir denge kurar. Kodları çalıştırmadan önce model sayfasını ve fiyatlandırma bilgilerini kontrol edin: https://platform.openai.com/docs/guides/embeddings/embedding-models.

## Gerekli Import Kütüphaneleri
```python
import openai
```
Bu kütüphane, OpenAI'in API'sine erişmek için kullanılır.

## Vektör Store'u Doldurma
Artık vektör store'unu doldurmaya başlamaya hazırız.

---

## Adding data to the vector store

## Vektör Deposuna Veri Ekleme (Adding Data to the Vector Store)

Vektör deposuna veri eklemek için öncelikle `add_to_vector_store` adlı bir bayrağın (flag) `True` olarak ayarlanması gerekir. 
## Veri Ekleme İşlemi
Veri ekleme işlemi aşağıdaki kod bloğu ile gerçekleştirilir:
```python
if add_to_vector_store == True:
    with open(source_text, 'r') as f:
        text = f.read()
        CHUNK_SIZE = 1000
        chunked_text = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    vector_store.add(text=chunked_text,
                      embedding_function=embedding_function,
                      embedding_data=chunked_text,
                      metadata=[{"source": source_text}]*len(chunked_text))
```
Bu kod bloğunda:
- `source_text` değişkeni, okunacak dosyanın adını içerir (`"llm.txt"`).
- `open` fonksiyonu ile dosya okunur ve içeriği `text` değişkenine atanır.
- `CHUNK_SIZE` değişkeni, metnin parçalanacağı boyutu belirler (`1000` karakter).
- `chunked_text` list comprehension ile metin, `CHUNK_SIZE` boyutunda parçalara ayrılır.
- `vector_store.add` metodu ile parçalanmış metin, vektör deposuna eklenir.
  - `text` parametresi: Parçalanmış metin (`chunked_text`).
  - `embedding_function` parametresi: Kullanılacak embedding fonksiyonu.
  - `embedding_data` parametresi: Embedding işlemi için kullanılacak veri (`chunked_text`).
  - `metadata` parametresi: Her bir parçaya ait meta veri. Bu örnekte, kaynağın (`source_text`) adı meta veri olarak eklenir.

## Vektör Deposu Özeti (Vector Store Summary)
Vektör deposuna veri eklendikten sonra, özeti aşağıdaki gibidir:
```
Dataset(path='hub://denis76/space_exploration_v1', tensors=['text', 'metadata', 'embedding', 'id'])
  tensor      htype       shape      dtype  compression
  -------    -------     -------    -------  -------
   text       text      (839, 1)      str     None  
 metadata     json      (839, 1)      str     None  
 embedding  embedding  (839, 1536)  float32   None  
    id        text      (839, 1)      str     None
```
Bu özet, vektör deposundaki veri yapısını gösterir:
- `text`: Metin parçaları.
- `metadata`: Meta veri (kaynak bilgisi gibi).
- `embedding`: Embedding vektörleri.
- `id`: Her bir parçaya ait benzersiz kimlik.

## Vektör Deposu Özetini Yazdırma
Vektör deposu özetini yazdırmak için aşağıdaki kod kullanılır:
```python
print(vector_store.summary())
```
Bu kod, vektör deposunun yapısını ve içeriğini özetler.

## Önemli Noktalar
- Vektör deposuna veri eklemek için `add_to_vector_store` bayrağı `True` olmalıdır.
- Veri, `CHUNK_SIZE` boyutunda parçalara ayrılır ve her bir parça için embedding işlemi yapılır.
- Meta veri olarak kaynağın adı (`source_text`) eklenir.
- Vektör deposu özeti, veri yapısını ve içeriğini gösterir.

---

## Vector store information

## Vektör Depolama Bilgisi (Vector Store Information)
Activeloop'un API referansı, veri kümelerimizi (datasets) yönetmek için ihtiyacımız olan tüm bilgileri sağlar. Bu bilgilere https://docs.deeplake.ai/en/latest/ adresinden ulaşılabilir.

## Veri Kümelerini Görselleştirme (Visualizing Datasets)
Veri kümelerimizi https://app.activeloop.ai/datasets/mydatasets/ adresinden oturum açtığımızda görselleştirebiliriz. Ayrıca veri kümesini tek satır kod ile yükleyebiliriz:
```python
ds = deeplake.load(vector_store_path)
```
Bu kod, `vector_store_path` değişkeninde belirtilen yoldaki veri kümesini yükler.

## Veri Kümesini Görselleştirme (Visualizing the Dataset)
Yüklenen veri kümesini Jupyter Notebook içinde `ds.visualize()` komutu ile veya https://app.activeloop.ai/denis76/space_exploration_v1 adresinden görselleştirebiliriz.
```python
ds.visualize()
```
Bu komut, veri kümesini Jupyter Notebook içinde görselleştirir.

## Veri Kümesi Bilgisi (Dataset Information)
Veri kümesinin tahmini boyutunu (estimated size) bytes cinsinden hesaplayabiliriz:
```python
ds_size = ds.size_approx()
```
Daha sonra bu boyutu megabyte ve gigabyte cinsine çevirebiliriz:
```python
ds_size_mb = ds_size / 1048576
print(f"Dataset size in megabytes: {ds_size_mb:.5f} MB")

ds_size_gb = ds_size / 1073741824
print(f"Dataset size in gigabytes: {ds_size_gb:.5f} GB")
```
Bu kod, veri kümesinin boyutunu megabyte ve gigabyte cinsinden hesaplar ve yazdırır.

## Örnek Çıktı (Example Output)
```
Dataset size in megabytes: 55.31311 MB
Dataset size in gigabytes: 0.05402 GB
```
Bu çıktı, veri kümesinin boyutunu megabyte ve gigabyte cinsinden gösterir.

## Kullanılan Kodların Açıklaması (Explanation of Used Codes)
* `deeplake.load(vector_store_path)`: `vector_store_path` değişkeninde belirtilen yoldaki veri kümesini yükler.
* `ds.visualize()`: Yüklenen veri kümesini Jupyter Notebook içinde görselleştirir.
* `ds.size_approx()`: Veri kümesinin tahmini boyutunu bytes cinsinden hesaplar.
* `ds_size_mb = ds_size / 1048576`: Veri kümesinin boyutunu megabyte cinsine çevirir.
* `ds_size_gb = ds_size / 1073741824`: Veri kümesinin boyutunu gigabyte cinsine çevirir.

## İçe Aktarılan Kütüphaneler (Imported Libraries)
```python
import deeplake
```
Bu kütüphane, veri kümelerini yüklemek ve yönetmek için kullanılır.

## Kullanılan Teknik Terimler (Used Technical Terms)
* Veri kümesi (Dataset)
* Vektör depolama (Vector Store)
* Activeloop API
* Deep Lake
* Jupyter Notebook

## Eklemeler (Additions)
Veri kümelerini yönetmek ve görselleştirmek için Activeloop'un sağladığı araçlar ve API'ler oldukça kullanışlıdır. Ayrıca veri kümesinin boyutunu hesaplamak ve çevirmek için kullanılan kodlar da oldukça pratiktir.

---

## 3. Augmented input generation

## Artırılmış Giriş Üretimi (Augmented Input Generation)

Artırılmış üretim (Augmented Generation), üçüncü boru hattı bileşenidir (pipeline component). Kullanıcı girdisini artırmak için aldığımız verileri kullanacağız. Bu bileşen, kullanıcı girdisini işler, vektör deposunu (vector store) sorgular, girdiyi artırır ve `gpt-4-turbo`'yu çağırır.

## Boru Hattı Bileşeni #3: Artırılmış Giriş Üretimi

Şekil 2.9'da gösterildiği gibi, boru hattı bileşeni #3, `Retrieval Augmented Generation` (RAG) adını tamamen hak etmektedir. Ancak, bu bileşeni çalıştırmak, Team #1 ve Team #2'nin artırılmış giriş içeriği oluşturmak için gerekli bilgileri sağlamak için yaptıkları iş olmadan imkansız olurdu.

### Kodları İnceleyelim

İlk olarak, `vector_store_path` değişkenini tanımlıyoruz:
```python
vector_store_path = "hub://denis76/space_exploration_v1"
```
Bu, vektör deposunun (vector store) yolunu belirtir.

Daha sonra, `deeplake` kütüphanesini kullanarak vektör deposunu yüklüyoruz:
```python
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
import deeplake.util

ds = deeplake.load(vector_store_path)
```
Bu kod, `vector_store_path` değişkeninde belirtilen yoldaki vektör deposunu yükler.

Vektör deposunun varlığını doğrulamak için bir mesaj yazdırıyoruz:
```python
vector_store = VectorStore(path=vector_store_path)
```
Bu kod, vektör deposunu `VectorStore` nesnesi olarak yükler.

Çıktı, vektör deposunun varlığını ve yüklendiğini doğrular:
```
Deep Lake Dataset in hub://denis76/space_exploration_v1 already exists, loading from the storage
```
Bu, Team #2'nin daha önce vektör deposunu oluşturduğunu ve düzgün çalıştığını varsayar.

### Kullanıcı Girdisini İşleme

Şimdi, kullanıcı girdisini işleyebiliriz.

Önemli noktalar:

* Vektör deposunu (vector store) tanımlamak ve yüklemek
* `deeplake` kütüphanesini kullanarak vektör deposunu yüklemek
* Vektör deposunun varlığını doğrulamak
* Kullanıcı girdisini işlemek

Kullanılan kodlar:

* `vector_store_path = "hub://denis76/space_exploration_v1"`
* `from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore`
* `import deeplake.util`
* `ds = deeplake.load(vector_store_path)`
* `vector_store = VectorStore(path=vector_store_path)`

---

## Input and query retrieval

## Kullanıcı Girdisi ve Sorgu Erişimi (Input and Query Retrieval)
Kullanıcı girdisini gömme (embedding) işlevini kullanarak, kullanıcıdan alınan girdiyi işleriz. Bu işlem için kullanılan kod aşağıdaki gibidir:

```python
def embedding_function(texts, model="text-embedding-3-small"):
    if isinstance(texts, str):
        texts = [texts]
    texts = [t.replace("\n", " ") for t in texts]
    return [data.embedding for data in openai.embeddings.create(input=texts, model=model).data]
```

Bu kod, `texts` parametresi olarak verilen metinleri gömer. Eğer `texts` bir dize (string) ise, onu bir liste içinde saklar. Daha sonra, metinlerdeki satır sonlarını (`\n`) boşluk karakteri ile değiştirir. Son olarak, `openai.embeddings.create` işlevini kullanarak metinleri gömer ve gömme sonuçlarını döndürür.

### Kullanım Şekli
Bu işlev, kullanıcı girdisini gömmek için kullanılır. Örneğin, bir kullanıcı arayüzünden alınan girdiyi işlemek için kullanılabilir.

## Kullanıcı Girdisini Alma (Getting User Input)
Kullanıcıdan bir girdi almak için aşağıdaki işlev kullanılır:

```python
def get_user_prompt():
    # Kullanıcıdan sorgu girdisini ister
    return input("Enter your search query: ")
```

Bu işlev, kullanıcıdan bir sorgu girdisi ister ve bu girdiyi döndürür.

### Kullanım Şekli
Bu işlev, bir kullanıcı arayüzünde kullanılabilir. Örneğin:

```python
user_prompt = get_user_prompt()
```

## Sorgu İşleme (Processing the Query)
Kullanıcı girdisini aldıktan sonra, bu girdiyi sorgu olarak işleriz. Aşağıdaki kod, bu işlemi gerçekleştirir:

```python
search_results = vector_store.search(embedding_data=user_prompt, embedding_function=embedding_function)
```

Bu kod, `vector_store` nesnesinin `search` işlevini kullanarak, `user_prompt` girdisini gömme işlevi ile birlikte sorgular ve sonuçları `search_results` değişkeninde saklar.

## Sonuçları Biçimlendirme (Formatting the Results)
Sorgu sonuçlarını biçimlendirmek için aşağıdaki kod kullanılır:

```python
def wrap_text(text, width=80):
    lines = []
    while len(text) > width:
        split_index = text.rfind(' ', 0, width)
        if split_index == -1:
            split_index = width
        lines.append(text[:split_index])
        text = text[split_index:].strip()
    lines.append(text)
    return '\n'.join(lines)
```

Bu kod, bir metni belirli bir genişlikte (`width`) satırlara böler.

### Kullanım Şekli
Bu işlev, sorgu sonuçlarını biçimlendirmek için kullanılır. Örneğin:

```python
top_text = search_results['text'][0].strip()
print(wrap_text(top_text))
```

## Önemli Noktalar
* Kullanıcı girdisini gömme işlevi (`embedding_function`) kullanarak işleriz.
* Kullanıcı girdisini almak için `get_user_prompt` işlevini kullanırız.
* Sorgu sonuçlarını `vector_store.search` işlevi ile alırız.
* Sonuçları biçimlendirmek için `wrap_text` işlevini kullanırız.

### Kodların Açıklaması
* `embedding_function`: Kullanıcı girdisini gömer.
 + `texts`: Gömülecek metinler.
 + `model`: Kullanılacak gömme modeli (`default="text-embedding-3-small"`).
* `get_user_prompt`: Kullanıcıdan sorgu girdisini ister.
* `vector_store.search`: Sorgu sonuçlarını alır.
 + `embedding_data`: Gömülecek kullanıcı girdisi.
 + `embedding_function`: Gömme işlevi.
* `wrap_text`: Metni belirli bir genişlikte satırlara böler.
 + `text`: Biçimlendirilecek metin.
 + `width`: Satır genişliği (`default=80`).

---

## Augmented input

## Artırılmış Giriş (Augmented Input)
Program, alınan en üst metni kullanıcı girdisine ekler: 
```python
augmented_input = user_prompt + " " + top_text
print(augmented_input)
```
Bu kod, `user_prompt` ve `top_text` değişkenlerini birleştirerek `augmented_input` adlı yeni bir değişken oluşturur ve bunu yazdırır (`print`).

## Çıktı (Output)
Çıktı, artırılmış girişi görüntüler:
```
Tell me about space exploration on the Moon and Mars. Exploration of space, planets …
```
## GPT-4o Modeli ile İçerik Üretme (Content Generation with GPT-4o Model)
GPT-4o modeli, artırılmış girişi işleyerek içerik üretebilir:
```python
from openai import OpenAI
import time

client = OpenAI()
gpt_model = "gpt-4o"
start_time = time.time()  # İsteği göndermeden önce zamanı başlat

def call_gpt4_with_full_text(itext):
    # Tüm satırları tek bir dizeye birleştir
    text_input = '\n'.join(itext)
    prompt = f"Please summarize or elaborate on the following content:\n{text_input}"
    try:
        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are a space exploration expert."},
                {"role": "assistant", "content": "You can read the input and answer in detail."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # İhtiyaç duyulduğunda parametreleri ince ayar yapın
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)

gpt4_response = call_gpt4_with_full_text(augmented_input)
response_time = time.time() - start_time  # Yanıt süresini ölç
print(f"Response Time: {response_time:.2f} seconds")  # Yanıt süresini yazdır
print(gpt_model, "Response:", gpt4_response)
```
Bu kod, `call_gpt4_with_full_text` adlı bir fonksiyon tanımlar. Bu fonksiyon, GPT-4o modelini kullanarak artırılmış girişi işler ve bir yanıt üretir. Yanıt süresi de ölçülür ve yazdırılır.

## Yanıtı Biçimlendirme (Formatting the Response)
Yanıtı biçimlendirmek için `textwrap` ve `markdown` kullanılır:
```python
import textwrap
import re
from IPython.display import display, Markdown, HTML
import markdown

def print_formatted_response(response):
    # Markdown desenlerini kontrol et
    markdown_patterns = [
        r"^#+\s",  # Başlıklar (Headers)
        r"^\*+",  # Madde işaretleri (Bullet points)
        r"\*\*",  # Kalın metin (Bold)
        r"_",  # İtalik metin (Italics)
        r"\[.+\]\(.+\)",  # Bağlantılar (Links)
        r"-\s",  # Liste işaretleri (Dashes used for lists)
        r"\`\`\`"  # Kod blokları (Code blocks)
    ]
    
    # Desenlerden herhangi biri eşleşirse, yanıtın markdown içerdiğini varsay
    if any(re.search(pattern, response, re.MULTILINE) for pattern in markdown_patterns):
        # Markdown algılandı, HTML'e dönüştürerek daha iyi görüntüle
        html_output = markdown.markdown(response)
        display(HTML(html_output))  # Colab'da HTML görüntülemek için display(HTML()) kullan
    else:
        # Markdown algılanmadı, düz metin olarak sar ve yazdır
        wrapper = textwrap.TextWrapper(width=80)
        wrapped_text = wrapper.fill(text=response)
        print("Text Response:")
        print("--------------------")
        print(wrapped_text)
        print("--------------------\n")

print_formatted_response(gpt4_response)
```
Bu kod, `print_formatted_response` adlı bir fonksiyon tanımlar. Bu fonksiyon, yanıtta markdown desenlerini kontrol eder ve buna göre yanıtı biçimlendirir. Markdown içeriyorsa HTML'e dönüştürerek görüntüler, yoksa düz metin olarak sarar ve yazdırır.

## Önemli Noktalar
*   Artırılmış giriş (`augmented_input`), kullanıcı girdisi (`user_prompt`) ve en üst metin (`top_text`) birleştirilerek oluşturulur.
*   GPT-4o modeli, artırılmış girişi işleyerek içerik üretir.
*   Yanıt süresi ölçülür ve yazdırılır.
*   Yanıt, markdown desenlerine göre biçimlendirilir ve görüntülenir.

## Kullanılan Kod Parçaları ve Açıklamaları
### 1. `augmented_input = user_prompt + " " + top_text`
*   Kullanıcı girdisi (`user_prompt`) ve en üst metni (`top_text`) birleştirerek `augmented_input` oluşturur.

### 2. `call_gpt4_with_full_text(itext)` Fonksiyonu
*   GPT-4o modelini kullanarak artırılmış girişi işler ve bir yanıt üretir.
*   Yanıtı döndürmeden önce gereksiz boşlukları temizler (`strip()`).

### 3. `print_formatted_response(response)` Fonksiyonu
*   Yanıtta markdown desenlerini kontrol eder.
*   Markdown içeriyorsa HTML'e dönüştürerek görüntüler.
*   Markdown içermiyorsa düz metin olarak sarar ve yazdırır.

Tüm kod parçaları, artırılmış giriş oluşturma, GPT-4o modeli ile içerik üretme, yanıtı biçimlendirme ve görüntülemeyi kapsar.

---

## Evaluating the output with cosine similarity

## Cosine Similarity ile Çıktı Değerlendirme (Evaluating the Output with Cosine Similarity)

Bu bölümde, kullanıcı girdisi (user input) ve üretken yapay zeka modelinin çıktısı (generative AI model's output) arasındaki benzerliği ölçmek için cosine similarity kullanacağız. Ayrıca, genişletilmiş kullanıcı girdisi (augmented user input) ile üretken yapay zeka modelinin çıktısı arasındaki benzerliği de ölçelim.

### Cosine Similarity Fonksiyonu Tanımlama

İlk olarak, bir cosine similarity fonksiyonu tanımlayalım:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]
```
Bu kod, `TfidfVectorizer` kullanarak iki metni vektörleştirir ve ardından `cosine_similarity` fonksiyonu ile bu vektörler arasındaki benzerliği hesaplar.

### Kullanıcı Girdisi ve GPT-4 Çıktısı Arasındaki Benzerlik

Şimdi, kullanıcı girdisi ve GPT-4 çıktısı arasındaki benzerliği hesaplayalım:
```python
similarity_score = calculate_cosine_similarity(user_prompt, gpt4_response)
print(f"Cosine Similarity Score: {similarity_score:.3f}")
```
Bu kod, `calculate_cosine_similarity` fonksiyonunu kullanarak kullanıcı girdisi ve GPT-4 çıktısı arasındaki benzerliği hesaplar ve sonucu yazdırır.

### Genişletilmiş Kullanıcı Girdisi ve GPT-4 Çıktısı Arasındaki Benzerlik

Şimdi, genişletilmiş kullanıcı girdisi ve GPT-4 çıktısı arasındaki benzerliği hesaplayalım:
```python
similarity_score = calculate_cosine_similarity(augmented_input, gpt4_response)
print(f"Cosine Similarity Score: {similarity_score:.3f}")
```
Bu kod, `calculate_cosine_similarity` fonksiyonunu kullanarak genişletilmiş kullanıcı girdisi ve GPT-4 çıktısı arasındaki benzerliği hesaplar ve sonucu yazdırır.

### Sentence Transformers ile Benzerlik Hesaplama

Cosine similarity, TF-IDF (Term Frequency-Inverse Document Frequency) kullanırken, tam kelime örtüşmesine bağlıdır ve önemli dil özelliklerini dikkate alır. Ancak, Sentence Transformers kullanarak benzerlik hesaplamak, kelimeler ve cümleler arasındaki daha derin anlamsal ilişkileri yakalar. Bu yaklaşım, metinler arasındaki anlamsal ve kavramsal benzerliği daha iyi tanır.

İlk olarak, `sentence-transformers` kütüphanesini kurmak gerekir:
```bash
!pip install sentence-transformers
```
Ardından, `all-MiniLM-L6-v2` modelini kullanarak benzerlik hesaplama fonksiyonunu tanımlayalım:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity_with_embeddings(text1, text2):
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    similarity = cosine_similarity([embeddings1], [embeddings2])
    return similarity[0][0]
```
Bu kod, `SentenceTransformer` modelini kullanarak iki metni vektörleştirir ve ardından `cosine_similarity` fonksiyonu ile bu vektörler arasındaki benzerliği hesaplar.

### Genişletilmiş Kullanıcı Girdisi ve GPT-4 Çıktısı Arasındaki Benzerlik (Sentence Transformers)

Şimdi, genişletilmiş kullanıcı girdisi ve GPT-4 çıktısı arasındaki benzerliği Sentence Transformers kullanarak hesaplayalım:
```python
similarity_score = calculate_cosine_similarity_with_embeddings(augmented_input, gpt4_response)
print(f"Cosine Similarity Score: {similarity_score:.3f}")
```
Bu kod, `calculate_cosine_similarity_with_embeddings` fonksiyonunu kullanarak genişletilmiş kullanıcı girdisi ve GPT-4 çıktısı arasındaki benzerliği hesaplar ve sonucu yazdırır.

## Önemli Noktalar

* Cosine similarity, metinler arasındaki benzerliği ölçmek için kullanılır.
* TF-IDF, tam kelime örtüşmesine bağlıdır ve önemli dil özelliklerini dikkate alır.
* Sentence Transformers, kelimeler ve cümleler arasındaki daha derin anlamsal ilişkileri yakalar.
* `all-MiniLM-L6-v2` modeli, benzerlik hesaplama için kullanılır.
* `sentence-transformers` kütüphanesi, Sentence Transformers modellerini kullanmak için gereklidir.

---

## Summary

## RAG (Retrieval-Augmented Generation) ile Gelişmiş Yapay Zeka Uygulamaları
Bu bölümde, büyük veri kümeleriyle çalışırken belge gömmelerin (document embeddings) temel rolünü vurgulayarak, RAG ile çalışan üretken yapay zekanın (generative AI) karmaşıklıklarını ele aldık. Ham metinlerden gömme oluşturma ve bunları vektör depolarında (vector stores) saklama sürecini inceledik. Activeloop gibi vektör depoları, parametrik üretken yapay zeka modellerinin aksine, gömülü metinleri her an görmemizi sağlayan API araçları ve görsel arayüzler sunar.

## Önemli Noktalar
*   Büyük veri kümeleriyle çalışırken belge gömmelerin önemi
*   Ham metinlerden gömme oluşturma ve vektör depolarında saklama
*   Activeloop gibi vektör depolarının sağladığı API araçları ve görsel arayüzler
*   RAG pipeline ile OpenAI gömmelerin Activeloop Deep Lake vektör depolarına entegrasyonu
*   RAG pipeline'ın farklı bileşenlere ayrılması ve ekiplerin aynı anda çalışabilmesi
*   Python fonksiyonları, Activeloop Deep Lake, OpenAI gömme modeli ve OpenAI GPT-4o üretken yapay zeka modelinin kullanılması

## RAG Pipeline ile OpenAI Gömme Entegrasyonu
RAG pipeline, OpenAI gömmelerin Activeloop Deep Lake vektör depolarına entegrasyonunu sağlayan bir süreçtir. Bu pipeline, farklı projelerde değişebilen farklı bileşenlere ayrılmıştır. Bu ayrım, ekiplerin aynı anda bağımlılık olmadan çalışabilmesini sağlar, geliştirme sürecini hızlandırır ve uzmanlaşmayı kolaylaştırır.

## RAG Pipeline Bileşenleri
1.  **Veri Toplama (Data Collection)**: Ham verilerin toplanması ve işlenmesi
2.  **Gömme İşleme (Embedding Processing)**: Ham verilerin gömme oluşturma ve vektör depolarında saklanması
3.  **Sorgu Üretme (Query Generation)**: Gömme oluşturulan veriler üzerinde sorgu üretilmesi ve artırılmış üretim yapay zeka süreci için kullanılması

## Python ile RAG Pipeline Uygulaması
Python kullanarak üç bileşenli bir RAG pipeline oluşturduk. Bu pipeline, ortamın ayarlanması, bağımlılıkların işlenmesi ve veri parçalama (data chunking) ve vektör deposu entegrasyonu gibi uygulama zorluklarının üstesinden gelmeyi içerir.

### Gerekli Kütüphaneler
```python
import os
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
```

### OpenAI Gömme Modeli ve Activeloop Deep Lake Vektör Deposu
```python
# OpenAI gömme modeli
embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

# Activeloop Deep Lake vektör deposu
vector_store = DeepLake(
    dataset_path="hub://your-username/your-dataset",
    embedding_function=embeddings,
    token=os.environ["ACTIVELOOP_TOKEN"],
)
```

### Veri Parçalama ve Gömme Oluşturma
```python
# Veri parçalama
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(your_text_data)

# Gömme oluşturma
docs = [Document(page_content=text) for text in texts]
vector_store.add_documents(docs)
```

### Sorgu Üretme ve Artırılmış Üretim Yapay Zeka Süreci
```python
# Sorgu üretme
query = "your_query_text"
results = vector_store.similarity_search(query)

# Artırılmış üretim yapay zeka süreci
# OpenAI GPT-4o üretken yapay zeka modeli
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

response = openai.Completion.create(
    model="gpt-4o",
    prompt=query,
    max_tokens=1024,
    temperature=0.7,
)
```

## Sonuç
Bu bölümde, RAG ile çalışan üretken yapay zekanın karmaşıklıklarını ele aldık ve belge gömmelerin önemini vurguladık. RAG pipeline ile OpenAI gömmelerin Activeloop Deep Lake vektör depolarına entegrasyonunu sağladık. Python kullanarak üç bileşenli bir RAG pipeline oluşturduk ve veri parçalama, gömme oluşturma ve sorgu üretme gibi uygulama zorluklarının üstesinden geldik. Bir sonraki bölümde, gelişmiş indeksleme yöntemlerini tanıtacağız ve girdi artırma ve alma işlemlerini daha da geliştireceğiz.

---

## Questions

## Sorular ve Cevaplar
Aşağıdaki paragrafta anlatılan konuyu türkçe olarak tekrar düzenleyerek önemli noktaları maddeler halinde yazacağız.

## Konu: RAG (Retrieval-Augmented Generation) ve Embeddingler
RAG, metin oluşturma (text generation) görevlerinde kullanılan bir tekniktir. Bu teknik, metinleri yüksek boyutlu vektörlere (high-dimensional vectors) dönüştürerek daha hızlı erişim sağlar.

## Önemli Noktalar:
* RAG, metinleri embedding adı verilen yüksek boyutlu vektörlere dönüştürür (convert text into high-dimensional vectors).
* Embeddingler, metinlerin anlamsal içeriğini (semantic content) temsil eder.
* RAG pipeline'ı bağımsız bileşenlere (independent components) ayırmak önerilir.
* RAG pipeline'ı iki ana bileşenden oluşur: embedding oluşturma ve vektör depolama (vector storage).
* Activeloop Deep Lake, hem embedding hem de vektör depolama işlemlerini gerçekleştirebilir.
* text-embedding-3-small modeli, OpenAI tarafından geliştirilen bir embedding modelidir.
* Veri embeddingleri, RAG-tabanlı sistemlerde görünür ve doğrudan izlenebilir değildir (not visible and directly traceable).
* Büyük metinleri daha küçük parçalara ayırmak (chunking), embedding ve depolama için gereklidir.
* Cosine similarity metriği, erişilen bilgilerin (retrieved information) uygunluğunu değerlendirmek için kullanılır.

## Sorular ve Cevaplar:
1. ## Do embeddings convert text into high-dimensional vectors for faster retrieval in RAG? 
Cevap: ## Yes
Açıklama: Embeddingler, metinleri yüksek boyutlu vektörlere dönüştürerek RAG'de daha hızlı erişim sağlar.

2. ## Are keyword searches more effective than embeddings in retrieving detailed semantic content? 
Cevap: ## No
Açıklama: Embeddingler, anahtar kelime aramalarına (keyword searches) göre daha ayrıntılı anlamsal içerik (detailed semantic content) erişiminde daha etkilidir.

3. ## Is it recommended to separate RAG pipelines into independent components? 
Cevap: ## Yes
Açıklama: RAG pipeline'larını bağımsız bileşenlere ayırmak önerilir.

4. ## Does the RAG pipeline consist of only two main components? 
Cevap: ## No
Açıklama: RAG pipeline'ı en az iki ana bileşene sahiptir: embedding oluşturma ve vektör depolama. Ancak daha fazla bileşene sahip olabilir.

5. ## Can Activeloop Deep Lake handle both embedding and vector storage? 
Cevap: ## Yes
Açıklama: Activeloop Deep Lake, hem embedding hem de vektör depolama işlemlerini gerçekleştirebilir.

6. ## Is the text-embedding-3-small model from OpenAI used to generate embeddings in this chapter? 
Cevap: ## Yes
Açıklama: text-embedding-3-small modeli, OpenAI tarafından geliştirilen bir embedding modelidir ve bu bölümde embedding oluşturmak için kullanılır.

7. ## Are data embeddings visible and directly traceable in an RAG-driven system? 
Cevap: ## No
Açıklama: Veri embeddingleri, RAG-tabanlı sistemlerde görünür ve doğrudan izlenebilir değildir.

8. ## Can a RAG pipeline run smoothly without splitting into separate components? 
Cevap: ## No
Açıklama: RAG pipeline'ları bağımsız bileşenlere ayrılmazsa düzgün çalışmayabilir.

9. ## Is chunking large texts into smaller parts necessary for embedding and storage? 
Cevap: ## Yes
Açıklama: Büyük metinleri daha küçük parçalara ayırmak (chunking), embedding ve depolama için gereklidir.

10. ## Are cosine similarity metrics used to evaluate the relevance of retrieved information? 
Cevap: ## Yes
Açıklama: Cosine similarity metriği, erişilen bilgilerin (retrieved information) uygunluğunu değerlendirmek için kullanılır.

## Örnek Kod:
```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Cümleleri embedding'e dönüştürmek için model yükleme
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Örnek cümleler
sentences = ['Bu bir örnek cümledir.', 'Bu başka bir örnek cümledir.']

# Cümleleri embedding'e dönüştürme
sentence_embeddings = model.encode(sentences)

# Embedding'ler arasındaki cosine similarity'yi hesaplama
cosine_similarity = np.dot(sentence_embeddings[0], sentence_embeddings[1]) / (np.linalg.norm(sentence_embeddings[0]) * np.linalg.norm(sentence_embeddings[1]))

print("Cosine Similarity:", cosine_similarity)
```
Bu kod, `sentence-transformers` kütüphanesini kullanarak cümleleri embeddinglere dönüştürür ve bu embeddingler arasındaki cosine similarity'yi hesaplar.

## Kod Açıklaması:
* `SentenceTransformer` modeli yüklenir. Bu model, cümleleri embeddinglere dönüştürmek için kullanılır.
* Örnek cümleler tanımlanır.
* `model.encode()` fonksiyonu kullanılarak cümleler embeddinglere dönüştürülür.
* İki embedding arasındaki cosine similarity, numpy kütüphanesindeki `dot()` ve `linalg.norm()` fonksiyonları kullanılarak hesaplanır.
* Sonuç ekrana yazılır.

## Kullanılan Kütüphaneler:
* `numpy`: Sayısal işlemler için kullanılır.
* `sentence_transformers`: Cümleleri embeddinglere dönüştürmek için kullanılır.

---

## References

## Referanslar
Paragrafta bahsedilen konular, OpenAI Ada dokümantasyonu, OpenAI GPT dokümantasyonu, Activeloop API dokümantasyonu ve MiniLM model referansları hakkındadır. Bu konuları aşağıda maddeler halinde özetleyebiliriz.

## Önemli Noktalar
- OpenAI Ada dokümantasyonu, embedding (gömme) modelleri hakkında bilgi verir.
- OpenAI GPT dokümantasyonu, içerik üretimi için GPT-4 Turbo ve GPT-4 modellerini açıklar.
- Activeloop API dokümantasyonu, veri işleme ve depolama ile ilgili detayları içerir.
- MiniLM model referansı, metinleri dönüştürmek için kullanılan bir modeldir.

## Kullanılan Kodlar ve Açıklamaları
Paragraf içerisinde doğrudan kod verilmemekle birlikte, bahsi geçen konularla ilgili olarak kullanılabilecek bazı kod örneklerini ve açıklamalarını aşağıda bulabilirsiniz.

### OpenAI Ada Embedding Modelleri
OpenAI'ın embedding modellerini kullanmak için aşağıdaki gibi bir kod örneği yazılabilir:
```python
import openai

# OpenAI API anahtarını tanımlayın
openai.api_key = "API-ANAHTARINIZ"

# Embedding modeli kullanarak metni gömme
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"  # Kullanılacak embedding modeli
    )
    return response['data'][0]['embedding']

text = "Örnek metin"
embedding = get_embedding(text)
print(embedding)
```
Bu kod, OpenAI'ın `text-embedding-ada-002` modelini kullanarak verilen bir metnin embedding'ini (gömme) hesaplar.

### OpenAI GPT-4 İçerik Üretimi
GPT-4 kullanarak içerik üretmek için aşağıdaki kod örneği kullanılabilir:
```python
import openai

# OpenAI API anahtarını tanımlayın
openai.api_key = "API-ANAHTARINIZ"

# GPT-4 modeli ile içerik üretme
def generate_content(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Kullanılacak GPT modeli
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']

prompt = "İçerik üretmek için bir prompt"
content = generate_content(prompt)
print(content)
```
Bu kod, verilen bir prompt'a (ipucu) dayanarak GPT-4 modeli ile içerik üretir.

### Activeloop API Kullanımı
Activeloop API'sini kullanmak için öncelikle Activeloop kütüphanesini yüklemeniz gerekir. Aşağıdaki örnek, Activeloop ile nasıl veri okunacağını gösterir:
```python
import deeplake

# Veri setini yükleyin
ds = deeplake.load("hub://activeloop/dataset-name")

# Veri setinden veri okuma
for sample in ds:
    print(sample)
```
Bu kod, Activeloop'tan bir veri setini yükler ve içindeki verileri okur.

### MiniLM Modeli Kullanımı
MiniLM modelini kullanmak için `sentence-transformers` kütüphanesini kullanabilirsiniz:
```python
from sentence_transformers import SentenceTransformer

# Modeli yükleyin
model = SentenceTransformer('all-MiniLM-L6-v2')

# Metinleri gömme
sentences = ["Bu bir örnek cümledir.", "Bu başka bir örnek cümledir."]
embeddings = model.encode(sentences)

print(embeddings)
```
Bu kod, `all-MiniLM-L6-v2` modelini kullanarak verilen cümlelerin embedding'lerini hesaplar.

## Teknik Terimler
- **Embedding (Gömme):** Metin, resim gibi verileri sayısal vektörlere dönüştürme işlemi.
- **GPT (Generative Pre-trained Transformer):** İçerik üretmek için kullanılan bir yapay zeka modeli.
- **API (Application Programming Interface):** Uygulamalar arasında veri alışverişini sağlayan arayüz.

Tüm bu kod örneklerinde, ilgili kütüphanelerin (`openai`, `deeplake`, `sentence-transformers`) kurulu olduğu varsayılmıştır. Kütüphaneleri kurmak için `pip install <kütüphane-adı>` komutunu kullanabilirsiniz.

---

## Further reading

## Daha Fazla Okuma (Further Reading)

OpenAI ve Activeloop belgeleri, embeddings (gömme) konusunda daha detaylı bilgi edinmek için önemli kaynaklardır. Aşağıda bu konudaki önemli noktalar maddeler halinde listelenmiştir.

## Önemli Noktalar (Key Points)

*   OpenAI'ın embeddings (gömme) konusunda sağladığı dökümantasyon, bu konudaki temel kaynaklardan biridir.
*   Activeloop'un dökümantasyonu, embeddings (gömme) konusunda daha spesifik bilgi edinmek için kullanılabilir.

## Kullanılan Kodlar (Used Codes)

Embeddings (gömme) konusunda kullanılan bazı kod örnekleri aşağıda verilmiştir.

### OpenAI Embeddings (Gömme) Örneği

```python
import os
import openai

# OpenAI API anahtarını (API key) ayarlayın
openai.api_key = os.getenv("OPENAI_API_KEY")

# Embedding (gömme) oluşturmak için kullanılan kod
response = openai.Embedding.create(
    input="Your text here",
    model="text-embedding-ada-002"
)

# Oluşturulan embedding (gömme) değerini alın
embedding_value = response['data'][0]['embedding']

print(embedding_value)
```

Bu kod, OpenAI API'sini kullanarak bir metin için embedding (gömme) oluşturur. `openai.Embedding.create` methodu, belirtilen metin için embedding (gömme) oluşturur. Oluşturulan embedding (gömme) değeri, `response` değişkeninde döner.

*   `openai.api_key = os.getenv("OPENAI_API_KEY")` : OpenAI API anahtarını (API key) ayarlar. Bu anahtar, OpenAI hesabınızda oluşturulabilir.
*   `openai.Embedding.create` : Embedding (gömme) oluşturmak için kullanılan methoddur.
*   `input="Your text here"` : Embedding (gömme) oluşturulacak metni belirtir.
*   `model="text-embedding-ada-002"` : Kullanılacak modeli belirtir. Bu örnekte, "text-embedding-ada-002" modeli kullanılmıştır.

### Activeloop Örneği

Activeloop'u kullanmak için gerekli olan kod örnekleri Activeloop dökümantasyonunda bulunabilir.

```python
import activeloop

# Activeloop'u başlatın
activeloop.start()

# Veri kümesini (dataset) oluşturun
dataset = activeloop.dataset("hub://your-dataset")

# Embedding (gömme) değerlerini içeren bir tensor oluşturun
embedding_tensor = dataset['embedding']

# Embedding (gömme) değerlerini alın
embedding_values = embedding_tensor.numpy()

print(embedding_values)
```

Bu kod, Activeloop'u kullanarak bir veri kümesinden (dataset) embedding (gömme) değerlerini alır.

*   `activeloop.start()` : Activeloop'u başlatır.
*   `activeloop.dataset("hub://your-dataset")` : Belirtilen veri kümesini (dataset) yükler.
*   `dataset['embedding']` : Veri kümesindeki embedding (gömme) değerlerini içeren tensor'u alır.
*   `embedding_tensor.numpy()` : Embedding (gömme) değerlerini numpy formatında alır.

## Kaynaklar (Resources)

*   [OpenAI Embeddings Dökümantasyonu](https://platform.openai.com/docs/guides/embeddings)
*   [Activeloop Dökümantasyonu](https://docs.activeloop.ai/)

---

## Join our community on Discord

## Discord Topluluğumuza Katılın (Join our community on Discord)

Packt topluluğumuzun Discord alanına katılarak yazar ve diğer okuyucularla tartışmalara katılabilirsiniz. Discord'a katılmak için aşağıdaki bağlantıyı kullanabilirsiniz: 
https://www.packt.link/rag

## Önemli Noktalar
- Yazar ve diğer okuyucularla tartışmalara katılma imkanı
- Discord bağlantısı: https://www.packt.link/rag

## Discord'a Katılma Adımları
1. Discord bağlantısına gidin: https://www.packt.link/rag
2. Discord'a giriş yapın veya üye olun
3. Packt topluluğumuzun Discord alanına katılın

## Kod Kullanımı
Bu metinde herhangi bir kod örneği bulunmamaktadır. Ancak Discord'a katılmak için herhangi bir kod kullanmanıza gerek yoktur, sadece bağlantıya tıklayarak katılabilirsiniz.

## Ek Bilgiler
Discord, özellikle geliştiriciler ve teknoloji meraklıları arasında popüler olan bir iletişim platformudur. Bu platforma katılarak diğer kullanıcılarla metin, ses veya video üzerinden iletişim kurabilirsiniz. 

## Teknik Terimler
- Discord: Bir iletişim platformu (Communication Platform)
- Packt: Bir yayıncılık şirketi (Publishing Company)

## İlgili Bağlantılar
- https://www.packt.link/rag (Discord Davet Bağlantısı / Discord Invite Link)

---

